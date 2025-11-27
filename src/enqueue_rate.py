"""HTTP Lambda entry point for retrieving model ratings from DynamoDB."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from rate import _build_openapi_response

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
    metrics_blob = item.get("metrics")

    if scoring.get("status") != "COMPLETED":
        reason = eligibility.get("reason") or scoring.get("status") or "rating pending"
        print(f"[rate.http] Rating pending reason={reason}", flush=True)
        return _json_response(500, {"message": reason})

    if eligibility.get("minimum_evidence_met") is False:
        reason = eligibility.get("reason") or "insufficient evidence coverage"
        print(f"[rate.http] Eligibility failure: {reason}", flush=True)
        return _json_response(500, {"message": reason})

    if not metrics_blob:
        print("[rate.http] Metrics blob missing", flush=True)
        return _json_response(500, {"message": "metrics unavailable"})

    breakdown = metrics_blob.get("breakdown", {})
    net_score = float(metrics_blob.get("average", 0.0))
    net_score_latency = int(scoring.get("net_score_latency", 0))

    response_payload = _build_openapi_response(
        item=item,
        model_id=model_id,
        net_score=net_score,
        net_score_latency=net_score_latency,
        breakdown=breakdown,
    )
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
            if candidate_str:
                return candidate_str
    raise EnqueueException(400, "missing or invalid model id")


def _json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _build_dynamo_key(model_id: str) -> Dict[str, str]:
    pk_field = os.getenv("RATE_PK_FIELD", "pk")
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    pk_prefix = os.getenv("RATE_PK_PREFIX", "MODEL#")
    sk_value = os.getenv("RATE_META_SK", "META")
    return {
        pk_field: f"{pk_prefix}{model_id}",
        sk_field: sk_value,
    }
