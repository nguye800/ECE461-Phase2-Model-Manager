"""Lightweight Lambda entry point that validates a rate request and enqueues it for asynchronous processing."""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

LOG_LEVEL = os.getenv("RATE_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

_SQS_CLIENT = boto3.client("sqs")
_QUEUE_ENV = "RATE_JOBS_QUEUE_URL"


class EnqueueException(Exception):
    """Raised when the request cannot be accepted."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def handler(event: Any, context: Any) -> Dict[str, Any]:
    """HTTP Lambda entrypoint that enqueues rate jobs."""
    event_dict = _coerce_event(event)

    try:
        # _require_auth(event_dict)
        model_id = _extract_model_id(event_dict)
    except EnqueueException as exc:
        return _json_response(exc.status_code, {"message": exc.message})

    queue_url = os.getenv(_QUEUE_ENV)
    if not queue_url:
        logger.error("%s environment variable must be set", _QUEUE_ENV)
        return _json_response(
            500, {"message": f"configuration error: {_QUEUE_ENV} not set"}
        )

    job_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    request_ctx = event_dict.get("requestContext") or {}
    payload = {
        "job_id": job_id,
        "model_id": model_id,
        "requested_at": now_iso,
        "request_context": {
            "request_id": request_ctx.get("requestId"),
            "connection_id": request_ctx.get("connectionId"),
            "stage": request_ctx.get("stage"),
            "domain_name": request_ctx.get("domainName"),
            "api_id": request_ctx.get("apiId"),
        },
        "pathParameters": event_dict.get("pathParameters"),
        "queryStringParameters": event_dict.get("queryStringParameters"),
        "headers": event_dict.get("headers"),
        "source": "enqueue_rate",
    }

    try:
        _SQS_CLIENT.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload),
        )
    except ClientError as exc:  # pragma: no cover - network dependent
        logger.exception("Failed to enqueue job %s", job_id)
        return _json_response(502, {"message": "failed to enqueue request"})

    logger.info("Enqueued job %s for model %s", job_id, model_id)
    return _json_response(
        202,
        {
            "job_id": job_id,
            "model_id": model_id,
            "status": "ENQUEUED",
            "requested_at": now_iso,
        },
    )


def _coerce_event(event: Any) -> Dict[str, Any]:
    if isinstance(event, str):
        try:
            return json.loads(event)
        except json.JSONDecodeError:
            return {}
    if event is None:
        return {}
    return event


def _require_auth(event: Dict[str, Any]) -> None:
    expected_token = os.getenv("RATE_API_TOKEN")
    if not expected_token:
        return

    headers = event.get("headers") or {}
    provided = headers.get("Authorization") or headers.get("authorization")
    if not provided:
        raise EnqueueException(403, "authentication token missing")

    token = provided.split(" ", 1)[1] if " " in provided else provided
    if token != expected_token:
        raise EnqueueException(403, "authentication failed")


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
