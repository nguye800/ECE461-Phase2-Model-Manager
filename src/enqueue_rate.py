"""HTTP Lambda entry point for retrieving model ratings from DynamoDB."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from rate import (
    OPENAPI_METRIC_FIELDS,
    RateException,
    _build_openapi_response,
    _build_popularity_metric_entry,
    _fetch_hf_popularity_stats,
    _score_model,
)

LOG_LEVEL = os.getenv("RATE_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

dynamo = boto3.resource("dynamodb")


class EnqueueException(Exception):
    """Raised when the request cannot be fulfilled."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def handler(event: Any, context: Any) -> Dict[str, Any]:
    """HTTP Lambda entrypoint that returns the latest model rating."""
    event_dict = _coerce_event(event)
    method = (
        event_dict.get("requestContext", {})
        .get("http", {})
        .get("method", event_dict.get("httpMethod"))
    )
    path = event_dict.get("rawPath") or event_dict.get("path")
    print(
        f"[rate.http] Received {method or 'UNKNOWN'} {path or '/'} body={event_dict.get('body')}",
        flush=True,
    )

    try:
        model_id = _extract_model_id(event_dict)
    except EnqueueException as exc:
        return _json_response(exc.status_code, {"message": exc.message})

    table_name = os.getenv("MODELS_TABLE")
    if not table_name:
        logger.error("Environment variable MODELS_TABLE is required.")
        return _json_response(
            500, {"message": "configuration error: MODELS_TABLE not set"}
        )

    table = dynamo.Table(table_name)

    try:
        response = table.get_item(Key=_build_dynamo_key(model_id))
    except ClientError as exc:  # pragma: no cover - network dependent
        logger.exception("DynamoDB get_item failed")
        print(f"[rate.http] DynamoDB get_item failed: {exc}", flush=True)
        return _json_response(500, {"message": "failed to access metadata"})

    item = response.get("Item")
    if not item:
        print(f"[rate.http] Model {model_id} not found", flush=True)
        return _json_response(404, {"message": f"model {model_id} not found"})

    scoring = item.get("scoring", {})
    eligibility = item.get("eligibility", {})
    metrics_blob = item.get("metrics") or {}

    breakdown = metrics_blob.get("breakdown", {}) or {}
    net_score = float(metrics_blob.get("average", 0.0))
    net_score_latency = int(scoring.get("net_score_latency", 0))

    if _needs_rescore(scoring, metrics_blob, breakdown):
        refreshed = _attempt_score_refresh(table, model_id)
        if refreshed:
            item = refreshed
            scoring = item.get("scoring", {})
            eligibility = item.get("eligibility", {})
            metrics_blob = item.get("metrics") or {}
            breakdown = metrics_blob.get("breakdown", {}) or {}
            net_score = float(metrics_blob.get("average", 0.0))
            net_score_latency = int(scoring.get("net_score_latency", 0))

    fallback_needed = _detect_missing_metrics(breakdown)
    if fallback_needed:
        hf_stats = _fetch_hf_popularity_stats(item)
        if hf_stats:
            for metric_name in fallback_needed:
                breakdown[metric_name] = _build_popularity_metric_entry(
                    metric_name,
                    hf_stats,
                    reason="popularity_fallback_missing_metric",
                )
            metrics_blob["breakdown"] = breakdown
            available_values = [
                float(entry.get("value", 0.0))
                for entry in breakdown.values()
                if isinstance(entry, dict) and entry.get("available")
            ]
            if available_values:
                net_score = sum(available_values) / len(available_values)
                metrics_blob["average"] = net_score
                net_score_latency = net_score_latency or 0

    for metric_name in OPENAPI_METRIC_FIELDS:
        entry = breakdown.get(metric_name, {})
        value = entry.get("value") if isinstance(entry, dict) else entry
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = 0.0
        latency = int(entry.get("latency_ms", 0)) if isinstance(entry, dict) else 0
        availability = entry.get("available") if isinstance(entry, dict) else False
        print(
            f"[rate.http] Metric {metric_name} model={model_id} value={value_float:.3f} latency={latency}ms available={availability}",
            flush=True,
        )

    response_payload = _build_openapi_response(
        item=item,
        model_id=model_id,
        net_score=net_score,
        net_score_latency=net_score_latency,
        breakdown=breakdown,
    )
    _apply_size_score_format(response_payload, breakdown.get("size_score") or {})
    print(f"[rate.http] Returning rating for model {model_id}", flush=True)
    return _json_response(200, response_payload)


def _coerce_event(event: Any) -> Dict[str, Any]:
    if isinstance(event, str):
        try:
            return json.loads(event)
        except json.JSONDecodeError:
            return {}
    if event is None:
        return {}
    return event


def _extract_model_id(event: Dict[str, Any]) -> str:
    candidates: list[Optional[str]] = []
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
            if not candidate_str:
                continue
            if not _is_valid_model_id(candidate_str):
                raise EnqueueException(
                    400, "model id must be alphanumeric with dashes/underscores only"
                )
            return candidate_str
    raise EnqueueException(400, "missing or invalid model id")


def _is_valid_model_id(value: str) -> bool:
    """Allow alphanumerics plus dash/underscore to match DynamoDB key conventions."""
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", value))


def _json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[rate.http] Responding with status {status_code}", flush=True)
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _apply_size_score_format(response_payload: Dict[str, Any], size_entry: Dict[str, Any]) -> None:
    device_defaults = {
        "raspberry_pi": 0.0,
        "jetson_nano": 0.0,
        "desktop_pc": 0.0,
        "aws_server": 0.0,
    }
    details = size_entry.get("details")
    if isinstance(details, dict):
        formatted = {
            device: float(details.get(device, default))
            for device, default in device_defaults.items()
        }
    else:
        formatted = dict(device_defaults)
    response_payload["size_score"] = formatted
    response_payload["size_score_latency"] = int(size_entry.get("latency_ms", 0))


def _build_dynamo_key(model_id: str) -> Dict[str, str]:
    pk_field = os.getenv("RATE_PK_FIELD", "pk")
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    pk_prefix = os.getenv("RATE_PK_PREFIX", "MODEL#")
    sk_value = os.getenv("RATE_META_SK", "META")
    return {
        pk_field: f"{pk_prefix}{model_id}",
        sk_field: sk_value,
    }


def _needs_rescore(
    scoring: Dict[str, Any], metrics_blob: Dict[str, Any], breakdown: Dict[str, Any]
) -> bool:
    if scoring.get("status") != "COMPLETED":
        return True
    if not metrics_blob or not breakdown:
        return True
    return bool(_detect_missing_metrics(breakdown))


def _attempt_score_refresh(table: Any, model_id: str) -> Optional[Dict[str, Any]]:
    try:
        _score_model(model_id, set())
    except RateException as exc:
        logger.warning("rate retry failed for %s: %s", model_id, exc.message)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected error during rate retry for %s: %s", model_id, exc)
        return None
    try:
        refreshed = table.get_item(Key=_build_dynamo_key(model_id)).get("Item")
    except ClientError as exc:  # pragma: no cover
        logger.warning("Failed to refetch model %s after scoring: %s", model_id, exc)
        return None
    return refreshed


def _detect_missing_metrics(breakdown: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for metric_name in OPENAPI_METRIC_FIELDS:
        entry = breakdown.get(metric_name)
        if not isinstance(entry, dict):
            missing.append(metric_name)
            continue
        if not entry.get("available"):
            missing.append(metric_name)
            continue
        try:
            value = float(entry.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0.0:
            missing.append(metric_name)
    return missing
