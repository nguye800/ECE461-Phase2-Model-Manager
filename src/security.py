"""
Security track Lambda handlers.

Endpoints:
    GET /tracks        -> Returns the list of security track plans.
    PUT /authenticate  -> Issues a stateless access token (mock implementation).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
from typing import Any


DEFAULT_TRACK_MESSAGE = (
    "Planned security track: input validation, encryption at rest, audit logging."
)
TOKEN_TTL_SECONDS = 3600


def lambda_handler(event: dict, context: Any) -> dict:
    method = _extract_method(event)
    path = (event.get("path") or "").rstrip("/") or "/"

    if method == "GET" and path == "/tracks":
        return _handle_get_tracks(event)
    if method == "PUT" and path == "/authenticate":
        return _handle_put_authenticate(event)

    return _response(404, {"error": f"Unsupported route: {method} {path}"})


def _handle_get_tracks(event: dict) -> dict:
    message = os.environ.get("SECURITY_TRACK_MESSAGE", DEFAULT_TRACK_MESSAGE)
    return _response(200, {"tracks": message})


def _handle_put_authenticate(event: dict) -> dict:
    payload = _parse_body(event)
    subject = payload.get("subject") or "anonymous"
    issued_at = int(time.time())
    expires_at = issued_at + TOKEN_TTL_SECONDS

    token_payload = {
        "sub": subject,
        "iat": issued_at,
        "exp": expires_at,
        "jti": secrets.token_urlsafe(16),
    }
    token = base64.urlsafe_b64encode(json.dumps(token_payload).encode("utf-8")).decode(
        "utf-8"
    )

    return _response(201, {"access_token": token, "expires_at": expires_at})


def _parse_body(event: dict) -> dict:
    body = event.get("body")
    if not body:
        return {}

    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _extract_method(event: dict) -> str:
    method = event.get("httpMethod")
    if method:
        return method.upper()
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method")
    if method:
        return method.upper()
    return "GET"


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
