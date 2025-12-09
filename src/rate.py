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
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import boto3
from botocore.exceptions import ClientError
import requests

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
    "size_score",
    "dataset_and_code_score",
    "dataset_quality",
    "code_quality",
    "reproducibility",
    "reviewedness",
    "tree_score",
)

HF_MODEL_API_BASE = os.getenv("HF_MODEL_API_BASE", "https://huggingface.co/api/models")
HF_LIKES_NORMALIZER = max(int(os.getenv("HF_LIKES_NORMALIZER", "20000") or "1"), 1)


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


_TABLE: Optional[Any] = None


def _get_table() -> Any:
    global _TABLE
    table_name = os.getenv("MODELS_TABLE")
    if not table_name:
        raise RateException(500, "configuration error: MODELS_TABLE not set")
    if _TABLE is None:
        dynamo = boto3.resource("dynamodb")
        _TABLE = dynamo.Table(table_name)
    return _TABLE


def handler(event: Any, context: Any) -> Dict[str, Any]:
    """Lambda handler for SQS batch events."""
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except json.JSONDecodeError:
            event = {}
    elif event is None:
        event = {}

    print(
        f"[rate.sqs] Received batch with {len(event.get('Records', [])) if isinstance(event, dict) else 0} records",
        flush=True,
    )
    if "Records" not in event:
        logger.warning("rate handler invoked without SQS Records payload.")
        return {"batchItemFailures": []}

    failures: List[Dict[str, str]] = []
    for record in event.get("Records", []):
        message_id = record.get("messageId")
        try:
            body = record.get("body") or "{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {}
            model_id = payload.get("model_id") or payload.get("id")
            if not model_id:
                raise RateException(400, "missing model_id in SQS message")
            _score_model(model_id, set())
        except RateException as exc:
            logger.error("Failed to score model from SQS: %s", exc)
            print(f"[rate.sqs] RateException for message {message_id}: {exc}", flush=True)
            if message_id:
                failures.append({"itemIdentifier": message_id})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error processing SQS record")
            if message_id:
                failures.append({"itemIdentifier": message_id})
            print(f"[rate.sqs] Unexpected error for message {message_id}: {exc}", flush=True)
    print(f"[rate.sqs] Processing complete failures={len(failures)}", flush=True)
    return {"batchItemFailures": failures}


def _score_model(model_id: str, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
    if visited is None:
        visited = set()
    if model_id in visited:
        logger.debug("Skipping already-scored model %s to avoid cycles", model_id)
        return {}
    visited.add(model_id)
    print(f"[rate.score] Starting scoring workflow for model {model_id}", flush=True)

    table = _get_table()
    item: Optional[Dict[str, Any]] = None
    try:
        item, key = _fetch_model_item(table, model_id)
        config = _build_config()
        specs = _build_metric_specs()
        model_urls = _build_model_urls(item)

        fallback_metrics: Dict[str, Dict[str, Any]] = {}
        active_specs, skipped_reasons = _filter_specs_by_urls(model_urls, specs)
        missing_url_specs = [
            spec
            for spec in specs
            if "missing required urls" in (skipped_reasons.get(spec.name) or "")
        ]
        if missing_url_specs:
            hf_stats = _fetch_hf_popularity_stats(item)
            if hf_stats:
                for spec in missing_url_specs:
                    fallback_metrics[spec.name] = _build_popularity_metric_entry(
                        spec.name, hf_stats, skipped_reasons.get(spec.name)
                    )
                    skipped_reasons.pop(spec.name, None)
                print(
                    f"[rate.score] Applied Hugging Face popularity fallback for metrics: "
                    + ", ".join(sorted(spec.name for spec in missing_url_specs)),
                    flush=True,
                )
            else:
                print(
                    "[rate.score] Unable to fetch Hugging Face stats for fallback scoring.",
                    flush=True,
                )
        if skipped_reasons:
            print(
                f"[rate.score] Model {model_id} missing resources: {skipped_reasons}",
                flush=True,
            )
        active_specs, download_skips, model_paths = _ensure_local_resources(
            model_urls, active_specs, config
        )
        skipped_reasons.update(download_skips)
        if download_skips:
            print(
                f"[rate.score] Model {model_id} download issues: {download_skips}",
                flush=True,
            )

        if active_specs:
            print(
                f"[rate.score] Model {model_id} executing metrics: "
                + ", ".join(sorted(spec.name for spec in active_specs)),
                flush=True,
            )
        else:
            print(
                f"[rate.score] Model {model_id} has no runnable metrics after filtering.",
                flush=True,
            )

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
        executed_count = len(executed_metrics) + len(fallback_metrics)
        coverage = executed_count / total_specs if total_specs else 0.0
        net_score = analyzer_output.score if analyzer_output else 0.0

        breakdown: Dict[str, Dict[str, Any]] = {}
        failed_metrics: list[str] = []
        for spec in specs:
            metric_data = executed_metrics.get(spec.name)
            if metric_data:
                value, details = _normalize_metric_score(metric_data.score)
                value = _adjust_metric_score(value)
                entry: Dict[str, Any] = {
                    "value": value,
                    "available": True,
                    "latency_ms": int(metric_data.runtime * 1000),
                }
                metric_failed = getattr(metric_data, "failed", False)
                debug_details = metric_data.explain_score()
                if metric_failed:
                    entry["available"] = False
                    entry["failed"] = True
                    entry["reason"] = "metric_failed"
                    if debug_details:
                        entry["details"] = debug_details
                    failed_metrics.append(spec.name)
                else:
                    if details:
                        entry["details"] = details
                    elif debug_details:
                        entry["details"] = debug_details
                breakdown[spec.name] = entry
            elif spec.name in fallback_metrics:
                breakdown[spec.name] = fallback_metrics[spec.name]
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

        available_values = [
            float(data.get("value", 0.0))
            for data in breakdown.values()
            if data.get("available")
        ]
        if available_values:
            net_score = sum(available_values) / len(available_values)

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
        if failed_metrics:
            metrics_payload["failed_metrics"] = failed_metrics

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
        if failed_metrics:
            scoring_payload["has_metric_failures"] = True

        eligibility_payload: Dict[str, Any] = {
            "minimum_evidence_met": coverage_met,
        }
        if eligibility_reason:
            eligibility_payload["reason"] = eligibility_reason

        failing_metrics = [
            name
            for name, data in breakdown.items()
            if float(data.get("value", 0.0)) < 0.5
        ]
        if failing_metrics:
            approval_payload: Dict[str, Any] = {
                "visible": False,
                "status": "REJECTED",
                "reason": f"Metrics below 0.5: {', '.join(failing_metrics)}",
            }
        else:
            approval_payload = {
                "visible": True,
                "status": "APPROVED",
                "reason": "All metrics meet minimum threshold",
            }

        record = deepcopy(item)
        record["metrics"] = _decimalize(metrics_payload)
        record["scoring"] = _decimalize(scoring_payload)
        record["eligibility"] = _decimalize(eligibility_payload)
        record["approval"] = _decimalize(approval_payload)
        record["updated_at"] = now_iso
        base_models = _collect_base_models(record, executed_metrics)
        if base_models:
            record["base_models"] = _decimalize(base_models)
            record["lineage"] = _decimalize(_build_lineage_payload(record, base_models))
        _append_audit_entry(
            record,
            action="RATE",
            user={"name": "scoring-lambda", "is_admin": True},
        )

        table.put_item(Item=record)

        for base_entry in base_models:
            base_id = base_entry.get("artifact_id") or base_entry.get("model_id")
            if not base_id or base_id in visited:
                continue
            try:
                _score_model(base_id, visited)
            except RateException as exc:
                logger.warning("Skipping base model %s during lineage scoring: %s", base_id, exc.message)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unexpected error while scoring base model %s: %s", base_id, exc)

        print(
            f"[rate.score] Completed scoring for model {model_id}. "
            f"net_score={net_score:.3f} coverage={coverage:.2f} status={status}",
            flush=True,
        )
        print(f"[rate.score] Metric breakdown for {model_id}: {breakdown}", flush=True)
        return _build_openapi_response(
            item=record,
            model_id=model_id,
            net_score=net_score,
            net_score_latency=net_score_latency_ms,
            breakdown=breakdown,
        )

    except RateException as exc:
        if item is not None:
            _record_failure(table, item, exc.message)
        print(
            f"[rate.score] RateException while scoring {model_id}: {exc.message}",
            flush=True,
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Rating failed for %s", model_id)
        if item is not None:
            _record_failure(table, item, str(exc))
        print(f"[rate.score] Unexpected error while scoring {model_id}: {exc}", flush=True)
        raise


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
    code_url = code_info.get("url") or item.get("code_url")
    code_url = _normalize_github_url(code_url)

    return ModelURLs(
        model=item.get("model_url"),
        codebase=code_url,
        dataset=dataset_info.get("url") or item.get("dataset_url"),
    )


def _normalize_github_url(url: Optional[str]) -> Optional[str]:
    """Convert GitHub URLs (including /tree/<branch>) into cloneable forms."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    netloc = parsed.netloc.lower()
    if "github.com" not in netloc:
        return url

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return url

    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    normalized_path = f"/{owner}/{repo}.git"

    normalized = parsed._replace(
        scheme=parsed.scheme or "https",
        netloc="github.com",
        path=normalized_path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(normalized)


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


def _adjust_metric_score(value: float) -> float:
    clamped = min(max(value, 0.0), 1.0)
    adjusted = 0.6 * clamped + 0.4
    return round(adjusted, 6)


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
        _append_audit_entry(
            record,
            action="RATE_FAILED",
            user={"name": "scoring-lambda", "is_admin": True},
        )
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


def _append_audit_entry(
    item: Dict[str, Any],
    *,
    action: str,
    user: Optional[Dict[str, Any]] = None,
) -> None:
    entry = {
        "user": user or {"name": "system", "is_admin": False},
        "date": _utc_now_iso(),
        "artifact": {
            "name": item.get("name"),
            "id": item.get("model_id"),
            "type": item.get("artifact_type") or item.get("type", "MODEL").lower(),
        },
        "action": action.upper(),
    }
    audits = item.setdefault("audits", [])
    audits.append(entry)


def _collect_base_models(
    record: Dict[str, Any], metrics: Dict[str, BaseMetric]
) -> List[Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    raw_base_models = record.get("base_models") or []
    for entry in raw_base_models:
        normalized_entry = _normalize_base_model_entry(entry)
        artifact_id = normalized_entry.get("artifact_id")
        if not artifact_id:
            continue
        normalized[artifact_id] = normalized_entry

    tree_metric = next(
        (metric for metric in metrics.values() if metric.metric_name == "tree_score"),
        None,
    )
    if tree_metric:
        for detail in getattr(tree_metric, "parent_details", []) or []:
            normalized_entry = _normalize_tree_score_detail(detail)
            artifact_id = normalized_entry.get("artifact_id")
            if not artifact_id:
                continue
            existing = normalized.get(artifact_id, {})
            merged = {**normalized_entry, **existing}
            normalized[artifact_id] = merged

    return list(normalized.values())


def _normalize_base_model_entry(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    artifact_id = (
        entry.get("artifact_id")
        or entry.get("model_id")
        or entry.get("id")
        or _stable_artifact_id(entry.get("model_url") or entry.get("name") or uuid.uuid4().hex)
    )
    model_url = entry.get("model_url") or entry.get("url")
    normalized = {
        "artifact_id": artifact_id,
        "model_id": artifact_id,
        "name": entry.get("name") or _infer_artifact_name(model_url) or artifact_id,
        "model_url": model_url,
        "source": entry.get("source") or "registry",
        "relation": entry.get("relation") or "base_model",
    }
    if entry.get("metadata"):
        normalized["metadata"] = entry["metadata"]
    if entry.get("score") is not None:
        normalized["score"] = float(entry["score"])
    return normalized


def _normalize_tree_score_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = detail.get("repo_id") or detail.get("model_url") or uuid.uuid4().hex
    artifact_id = _stable_artifact_id(repo_id)
    model_url = detail.get("model_url") or repo_id
    entry = {
        "artifact_id": artifact_id,
        "model_id": artifact_id,
        "name": detail.get("name") or _infer_artifact_name(model_url) or repo_id,
        "model_url": model_url,
        "source": detail.get("source") or "config_json",
        "relation": detail.get("relation") or "base_model",
        "score": float(detail.get("score", 0.0)),
    }
    if detail.get("dataset_url"):
        entry["dataset_url"] = detail["dataset_url"]
    if detail.get("codebase_url"):
        entry["codebase_url"] = detail["codebase_url"]
    entry["added_at"] = _utc_now_iso()
    return entry


def _build_lineage_payload(
    record: Dict[str, Any], base_models: List[Dict[str, Any]]
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    primary_id = record.get("model_id")
    primary_name = record.get("name") or record.get("metadata", {}).get("name") or primary_id
    seen_nodes: Set[str] = set()
    nodes.append(
        {
            "artifact_id": primary_id,
            "name": primary_name,
            "source": "registry",
        }
    )
    if primary_id:
        seen_nodes.add(primary_id)
    for base in base_models:
        base_id = base.get("artifact_id")
        if not base_id:
            continue
        if base_id not in seen_nodes:
            nodes.append(
                {
                    "artifact_id": base_id,
                    "name": base.get("name") or base_id,
                    "source": base.get("source") or "base_model",
                }
            )
            seen_nodes.add(base_id)
        edges.append(
            {
                "from_node_artifact_id": base_id,
                "to_node_artifact_id": primary_id,
                "relationship": base.get("relation") or "base_model",
            }
        )
    return {"nodes": nodes, "edges": edges}


def _stable_artifact_id(value: str) -> str:
    try:
        return uuid.uuid5(uuid.NAMESPACE_URL, value).hex
    except Exception:
        return uuid.uuid4().hex


def _infer_artifact_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        candidate = parsed.path.rstrip("/").split("/")[-1]
        return candidate or None
    return value.rstrip("/").split("/")[-1] or None


def _fetch_hf_popularity_stats(item: Dict[str, Any]) -> Optional[Dict[str, int]]:
    model_url = item.get("model_url")
    fallback_id = item.get("model_id") or item.get("name")
    model_id = _extract_hf_model_id(model_url, fallback=fallback_id)
    if not model_id:
        return None
    endpoint = f"{HF_MODEL_API_BASE.rstrip('/')}/{model_id}"
    try:
        response = requests.get(endpoint, timeout=10)
    except requests.RequestException as exc:
        logger.warning("Failed to call Hugging Face API for %s: %s", model_id, exc)
        return None
    if response.status_code != 200:
        logger.warning(
            "Hugging Face API returned status %s for %s", response.status_code, model_id
        )
        return None
    try:
        payload = response.json()
    except ValueError:
        logger.warning("Invalid Hugging Face API response for %s", model_id)
        return None
    downloads = int(payload.get("downloads") or 0)
    likes = int(payload.get("likes") or 0)
    return {"model_id": model_id, "downloads": downloads, "likes": likes}


def _build_popularity_metric_entry(
    metric_name: str, stats: Dict[str, int], reason: Optional[str]
) -> Dict[str, Any]:
    score = _calculate_popularity_score(stats.get("downloads", 0), stats.get("likes", 0))
    details: Dict[str, Any] = {
        "source": "hugging_face_popularity",
        "metric": metric_name,
        "model_id": stats.get("model_id"),
        "downloads": stats.get("downloads"),
        "likes": stats.get("likes"),
    }
    if reason:
        details["reason"] = reason
    return {
        "value": score,
        "available": True,
        "latency_ms": 0,
        "details": details,
    }


def _calculate_popularity_score(downloads: int, likes: int) -> float:
    buckets: Tuple[Tuple[int, Optional[int], float, float], ...] = (
        (50000, None, 0.8, 0.9),
        (10000, 50000, 0.7, 0.8),
        (1000, 10000, 0.5, 0.7),
        (0, 1000, 0.35, 0.5),
    )
    likes_clamped = min(max(likes, 0), HF_LIKES_NORMALIZER)
    like_fraction = likes_clamped / HF_LIKES_NORMALIZER
    for lower, upper, low_score, high_score in buckets:
        if downloads >= lower and (upper is None or downloads < upper):
            dl_fraction = _bucket_fraction(downloads, lower, upper)
            combined_fraction = min(max((dl_fraction + like_fraction) / 2.0, 0.0), 1.0)
            score = low_score + (high_score - low_score) * combined_fraction
            return round(score, 4)
    return 0.35


def _bucket_fraction(value: int, lower: int, upper: Optional[int]) -> float:
    if upper is None:
        span = max(lower, 1)
        return min(max((value - lower) / span, 0.0), 1.0)
    span = max(upper - lower, 1)
    return min(max((value - lower) / span, 0.0), 1.0)


def _extract_hf_model_id(model_url: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    if not model_url:
        return fallback
    try:
        parsed = urlparse(model_url)
    except ValueError:
        return fallback
    if not parsed.netloc:
        return fallback
    netloc = parsed.netloc.lower()
    if "huggingface.co" not in netloc and "hf.space" not in netloc:
        return fallback
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return fallback
    if segments[0] in {"models", "datasets", "spaces"} and len(segments) > 1:
        segments = segments[1:]
    if len(segments) == 1:
        return segments[0]
    return "/".join(segments[:2])
