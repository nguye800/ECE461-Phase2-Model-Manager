"""AWS Lambda entrypoint for scoring Hugging Face models backed by DynamoDB."""

from __future__ import annotations

import base64
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError

from config import ConfigContract, ModelPaths, ModelURLs, generate_model_paths
from download_manager import DownloadManager
from metric import BaseMetric
from workflow import MetricStager, run_workflow

from metrics.bus_factor import BusFactorMetric
from metrics.code_quality import CodeQualityMetric
from metrics.dataset_and_code import DatasetAndCodeScoreMetric
from metrics.dataset_quality import DatasetQualityMetric
from metrics.license import LicenseMetric
from metrics.performance_claims import PerformanceClaimsMetric
from metrics.ramp_up_time import RampUpMetric
from metrics.reproducibility import ReproducibilityMetric
from metrics.reviewedness import ReviewednessMetric
from metrics.size_metric import SizeMetric
from metrics.tree_score import TreeScoreMetric

LOG_LEVEL = os.getenv("RATE_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

OPENAPI_METRIC_FIELDS: Tuple[str, ...] = (
    "ramp_up_time",
    "bus_factor",
    "performance_claims",
    "license",
    "dataset_and_code",
    "dataset_quality",
    "code_quality",
    "reproducibility",
    "reviewedness",
)


@dataclass
class MetricSpec:
    """Declarative description of a metric and its prerequisites."""

    name: str
    factory: Callable[[], BaseMetric]
    priority: int
    required_urls: Set[str] = field(default_factory=set)
    requires_local: Set[str] = field(default_factory=set)


class RateException(Exception):
    """Raised when the request cannot be fulfilled."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def handler(event: Any, context: Any) -> Dict[str, Any]:
    """Lambda handler."""
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except json.JSONDecodeError:
            event = {}
    elif event is None:
        event = {}

    try:
        # _require_auth(event) # skipping auth since we haven't set up tokens yet
        model_id = _extract_model_id(event)
    except RateException as exc:
        return _json_response(exc.status_code, {"message": exc.message})

    table_name = os.getenv("MODELS_TABLE")
    if not table_name:
        logger.error("Environment variable MODELS_TABLE is required.")
        return _json_response(
            500, {"message": "configuration error: MODELS_TABLE not set"}
        )

    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(table_name)

    item: Optional[Dict[str, Any]] = None
    key: Optional[Dict[str, str]] = None

    try:
        item, key = _fetch_model_item(table, model_id)
        config = _build_config()
        specs = _build_metric_specs()
        model_urls = _build_model_urls(item)

        active_specs, skipped_reasons = _filter_specs_by_urls(model_urls, specs)
        active_specs, download_skips, model_paths = _ensure_local_resources(
            model_urls, active_specs, config
        )
        skipped_reasons.update(download_skips)

        analyzer_output = None
        executed_metrics: Dict[str, BaseMetric] = {}
        net_score_latency_ms = 0
        analysis_start: Optional[float] = None

        if active_specs:
            analysis_start = time.time()
            stager = MetricStager(config)
            for spec in active_specs:
                stager.attach_metric(spec.factory(), spec.priority)
            analyzer_output = run_workflow(stager, model_urls, model_paths, config)
            if analysis_start is not None:
                net_score_latency_ms = int((time.time() - analysis_start) * 1000)
            for metric in analyzer_output.metrics:
                executed_metrics[metric.metric_name] = metric

        total_specs = len(specs)
        executed_count = len(executed_metrics)
        coverage = executed_count / total_specs if total_specs else 0.0
        net_score = analyzer_output.score if analyzer_output else 0.0

        breakdown: Dict[str, Dict[str, Any]] = {}
        for spec in specs:
            metric_data = executed_metrics.get(spec.name)
            if metric_data:
                value, details = _normalize_metric_score(metric_data.score)
                entry: Dict[str, Any] = {
                    "value": value,
                    "available": True,
                    "latency_ms": int(metric_data.runtime * 1000),
                }
                if details:
                    entry["details"] = details
                breakdown[spec.name] = entry
            else:
                entry = {
                    "value": 0.0,
                    "available": False,
                    "latency_ms": 0,
                }
                reason = skipped_reasons.get(spec.name)
                if reason:
                    entry["reason"] = reason
                breakdown[spec.name] = entry

        threshold = float(os.getenv("MIN_EVIDENCE_COVERAGE", "0.5"))
        coverage_met = coverage >= threshold and executed_count > 0
        missing_metrics = [name for name, data in breakdown.items() if not data["available"]]
        if coverage_met:
            eligibility_reason: Optional[str] = None
        elif missing_metrics:
            eligibility_reason = f"Missing evidence for metrics: {', '.join(missing_metrics)}"
        else:
            eligibility_reason = "Insufficient evidence coverage"

        now_iso = _utc_now_iso()
        metrics_payload = {
            "average": net_score,
            "evidence_coverage": coverage,
            "breakdown": breakdown,
        }

        existing_scoring = item.get("scoring", {})
        score_version = os.getenv(
            "SCORE_VERSION", existing_scoring.get("score_version", "v1")
        )
        scorer_build = os.getenv(
            "SCORER_BUILD", existing_scoring.get("scorer_build")
        )
        status = "COMPLETED" if executed_count > 0 else "SKIPPED"
        scoring_payload: Dict[str, Any] = {
            "status": status,
            "last_scored_at": now_iso,
            "score_version": score_version,
        }
        if scorer_build:
            scoring_payload["scorer_build"] = scorer_build

        eligibility_payload: Dict[str, Any] = {
            "minimum_evidence_met": coverage_met,
        }
        if eligibility_reason:
            eligibility_payload["reason"] = eligibility_reason

        record = deepcopy(item)
        record["metrics"] = _decimalize(metrics_payload)
        record["scoring"] = _decimalize(scoring_payload)
        record["eligibility"] = _decimalize(eligibility_payload)
        record["updated_at"] = now_iso

        table.put_item(Item=record)

        response_payload = _build_openapi_response(
            item=item,
            model_id=model_id,
            net_score=net_score,
            net_score_latency=net_score_latency_ms,
            breakdown=breakdown,
        )
        return _json_response(200, response_payload)

    except RateException as exc:
        if item is not None and key is not None:
            _record_failure(table, item, exc.message)
        return _json_response(exc.status_code, {"message": exc.message})
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Rating failed for %s", model_id)
        if item is not None and key is not None:
            _record_failure(table, item, str(exc))
        return _json_response(500, {"message": "rating system failure"})


def _require_auth(event: Dict[str, Any]) -> None:
    expected_token = os.getenv("RATE_API_TOKEN")
    if not expected_token:
        return

    headers = event.get("headers") or {}
    provided = headers.get("Authorization") or headers.get("authorization")
    if not provided:
        raise RateException(403, "authentication token missing")

    token = provided.split(" ", 1)[1] if " " in provided else provided
    if token != expected_token:
        raise RateException(403, "authentication failed")


def _extract_model_id(event: Dict[str, Any]) -> str:
    candidates: List[Optional[str]] = []
    path_params = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}
    for source in (path_params, query_params):
        candidates.extend(
            [
                source.get("model_id"),
                source.get("modelId"),
                source.get("id"),
            ]
        )

    body = event.get("body")
    if body and not any(candidates):
        try:
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode("utf-8")
            payload = json.loads(body)
            candidates.extend(
                [
                    payload.get("model_id"),
                    payload.get("modelId"),
                    payload.get("id"),
                ]
            )
        except (ValueError, json.JSONDecodeError):
            pass

    for candidate in candidates:
        if candidate:
            candidate_str = str(candidate).strip()
            if candidate_str:
                return candidate_str
    raise RateException(400, "missing or invalid model id")


def _build_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            name="ramp_up_time",
            factory=RampUpMetric,
            priority=4,
            required_urls={"model"},
        ),
        MetricSpec(
            name="bus_factor",
            factory=BusFactorMetric,
            priority=2,
            required_urls={"codebase"},
        ),
        MetricSpec(
            name="performance_claims",
            factory=PerformanceClaimsMetric,
            priority=2,
            required_urls={"model", "codebase", "dataset"},
            requires_local={"model", "codebase", "dataset"},
        ),
        MetricSpec(
            name="license",
            factory=LicenseMetric,
            priority=1,
            required_urls={"model"},
            requires_local={"model"},
        ),
        MetricSpec(
            name="size_score",
            factory=SizeMetric,
            priority=3,
            required_urls={"model"},
        ),
        MetricSpec(
            name="dataset_and_code_score",
            factory=DatasetAndCodeScoreMetric,
            priority=2,
            required_urls={"model"},
            requires_local={"model"},
        ),
        MetricSpec(
            name="dataset_quality",
            factory=DatasetQualityMetric,
            priority=1,
            required_urls={"dataset"},
        ),
        MetricSpec(
            name="code_quality",
            factory=CodeQualityMetric,
            priority=1,
            required_urls={"codebase"},
            requires_local={"codebase"},
        ),
        MetricSpec(
            name="reproducibility",
            factory=ReproducibilityMetric,
            priority=1,
            required_urls={"model"},
            requires_local={"model"},
        ),
        MetricSpec(
            name="reviewedness",
            factory=ReviewednessMetric,
            priority=1,
            required_urls={"codebase"},
        ),
        MetricSpec(
            name="tree_score",
            factory=TreeScoreMetric,
            priority=1,
            required_urls={"model"},
        ),
    ]


def _build_config() -> ConfigContract:
    local_root = os.getenv("LOCAL_STORAGE_DIR", "/tmp/model-manager")
    num_processes = int(os.getenv("RATE_NUM_PROCESSES", "1"))
    run_multi = os.getenv("RATE_RUN_MULTI", "false").lower() in {"true", "1", "yes"}
    priority_function = os.getenv("RATE_PRIORITY_FUNCTION", "PFReciprocal")
    target_platform = os.getenv("RATE_TARGET_PLATFORM", "")

    return ConfigContract(
        num_processes=num_processes,
        run_multi=run_multi,
        priority_function=priority_function,
        target_platform=target_platform,
        local_storage_directory=local_root,
        model_path_name=os.getenv("RATE_MODEL_DIR_NAME", "models"),
        code_path_name=os.getenv("RATE_CODE_DIR_NAME", "codebases"),
        dataset_path_name=os.getenv("RATE_DATASET_DIR_NAME", "datasets"),
    )


def _build_model_urls(item: Dict[str, Any]) -> ModelURLs:
    code_info = item.get("codebase") or {}
    dataset_info = item.get("database") or item.get("dataset") or {}

    return ModelURLs(
        model=item.get("model_url"),
        codebase=code_info.get("url") or item.get("code_url"),
        dataset=dataset_info.get("url") or item.get("dataset_url"),
    )


def _filter_specs_by_urls(
    model_urls: ModelURLs, specs: List[MetricSpec]
) -> Tuple[List[MetricSpec], Dict[str, str]]:
    active: List[MetricSpec] = []
    skipped: Dict[str, str] = {}

    for spec in specs:
        missing = [
            field for field in spec.required_urls if getattr(model_urls, field, None) is None
        ]
        if missing:
            skipped[spec.name] = f"missing required urls: {', '.join(sorted(missing))}"
        else:
            active.append(spec)

    return active, skipped


def _ensure_local_resources(
    model_urls: ModelURLs, specs: List[MetricSpec], config: ConfigContract
) -> Tuple[List[MetricSpec], Dict[str, str], ModelPaths]:
    required: Set[str] = set()
    for spec in specs:
        required.update(spec.requires_local)

    download_errors: Dict[str, str] = {}
    if required:
        storage_root = Path(config.local_storage_directory)
        manager = DownloadManager(
            str(storage_root / config.model_path_name),
            str(storage_root / config.code_path_name),
            str(storage_root / config.dataset_path_name),
        )
        if "model" in required and model_urls.model:
            try:
                manager.download_model(model_urls.model)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Model download failed: %s", exc, exc_info=True)
                download_errors["model"] = str(exc)
        if "codebase" in required and model_urls.codebase:
            try:
                manager.download_codebase(model_urls.codebase)
            except Exception as exc:  # pragma: no cover
                logger.warning("Codebase download failed: %s", exc, exc_info=True)
                download_errors["codebase"] = str(exc)
        if "dataset" in required and model_urls.dataset:
            try:
                manager.download_dataset(model_urls.dataset)
            except Exception as exc:  # pragma: no cover
                logger.warning("Dataset download failed: %s", exc, exc_info=True)
                download_errors["dataset"] = str(exc)

    filtered_specs = []
    download_skips: Dict[str, str] = {}
    if download_errors:
        for spec in specs:
            failed = spec.requires_local & download_errors.keys()
            if failed:
                download_skips[spec.name] = (
                    "download failed for " + ", ".join(sorted(failed))
                )
            else:
                filtered_specs.append(spec)
    else:
        filtered_specs = specs

    model_paths = generate_model_paths(config, model_urls)
    return filtered_specs, download_skips, model_paths


def _normalize_metric_score(
    score: Any,
) -> Tuple[float, Optional[Dict[str, float]]]:
    if isinstance(score, dict):
        values = list(score.values())
        average = sum(values) / len(values) if values else 0.0
        return float(average), score
    try:
        return float(score), None
    except (TypeError, ValueError):
        return 0.0, None


def _fetch_model_item(
    table: Any, model_id: str
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    key = _build_dynamo_key(model_id)
    try:
        response = table.get_item(Key=key)
    except ClientError as exc:
        logger.exception("DynamoDB get_item failed")
        raise RateException(500, "failed to access metadata") from exc

    item = response.get("Item")
    if not item:
        raise RateException(404, f"model {model_id} not found")
    return item, key


def _build_dynamo_key(model_id: str) -> Dict[str, str]:
    pk_field = os.getenv("RATE_PK_FIELD", "pk")
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    pk_prefix = os.getenv("RATE_PK_PREFIX", "MODEL#")
    sk_value = os.getenv("RATE_META_SK", "META")
    return {
        pk_field: f"{pk_prefix}{model_id}",
        sk_field: sk_value,
    }


def _record_failure(table: Any, item: Dict[str, Any], message: str) -> None:
    try:
        now_iso = _utc_now_iso()
        record = deepcopy(item)
        existing_scoring = record.get("scoring", {})
        score_version = os.getenv(
            "SCORE_VERSION", existing_scoring.get("score_version", "v1")
        )
        scorer_build = os.getenv(
            "SCORER_BUILD", existing_scoring.get("scorer_build")
        )
        scoring_payload: Dict[str, Any] = {
            "status": "FAILED",
            "last_scored_at": now_iso,
            "score_version": score_version,
        }
        if scorer_build:
            scoring_payload["scorer_build"] = scorer_build

        failure_reason = f"Scoring run failed: {message}"
        eligibility_payload = {
            "minimum_evidence_met": False,
            "reason": failure_reason[:512],
        }

        record["scoring"] = _decimalize(scoring_payload)
        record["eligibility"] = _decimalize(eligibility_payload)
        record["updated_at"] = now_iso
        table.put_item(Item=record)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to record failure details in DynamoDB")


def _decimalize(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(round(value, 6)))
    if isinstance(value, dict):
        new_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            new_dict[key] = _decimalize(item)
        return new_dict
    if isinstance(value, list):
        return [_decimalize(item) for item in value if item is not None]
    return value


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, default=_json_default),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, set):
        return list(value)
    return value


def _build_openapi_response(
    item: Dict[str, Any],
    model_id: str,
    net_score: float,
    net_score_latency: int,
    breakdown: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Shape the Lambda response to match the published OpenAPI contract."""

    name = item.get("name") or item.get("model_id") or model_id
    category = item.get("category") or item.get("type") or "MODEL"

    response: Dict[str, Any] = {
        "name": str(name),
        "category": str(category),
        "net_score": float(net_score),
        "net_score_latency": int(net_score_latency),
    }

    for metric_name in OPENAPI_METRIC_FIELDS:
        metric_entry = breakdown.get(metric_name, {})
        response[metric_name] = float(metric_entry.get("value", 0.0))
        response[f"{metric_name}_latency"] = int(metric_entry.get("latency_ms", 0))

    return response
