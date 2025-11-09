"""
Lambda entry point for searching model artifacts stored in DynamoDB.

Supported routes (API Gateway compatible):
    POST /artifacts
        Searches for model artifacts matching one or more query objects. The
        response includes `x-next-offset` (a pagination token) when additional
        results are available.

    GET /artifact/byName/{name}
        Returns every artifact that matches the provided name (case-insensitive)
        using the NameIndex global secondary index.

    POST /artifact/byRegEx
        Performs a regex match over canonical names (`name_lc`) and README text.
        Pagination uses DynamoDB's LastEvaluatedKey encoded inside the same
        `x-next-offset` header used by `/artifacts`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence
from urllib.parse import unquote_plus

import boto3
from boto3.dynamodb.conditions import Attr, Key

LOGGER = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = os.environ.get("MODEL_REGISTRY_TABLE", "model_registry")
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
SCAN_BATCH_SIZE = 25


class RepositoryError(RuntimeError):
    """Raised when the backing artifact repository cannot be queried."""


@dataclass
class PaginationParams:
    start_key: dict[str, Any] | None = None
    skip: int = 0


def lambda_handler(event: dict, context: Any) -> dict:
    """
    AWS Lambda entry point. Delegates to `handle_search` for backwards
    compatibility with other modules in this repo.
    """

    return handle_search(event, context)


def handle_search(event: dict, _context: Any) -> dict:
    """
    Route incoming API Gateway events to the appropriate handler.
    """

    method = _extract_method(event)
    path = _normalize_path(event)

    if method == "POST" and path == "/artifacts":
        return _handle_post_artifacts(event)
    if method == "GET" and path.startswith("/artifact/byName"):
        return _handle_get_artifact_by_name(event, path)
    if method == "POST" and path == "/artifact/byRegEx":
        return _handle_post_regex(event)

    return _error_response(
        404,
        f"Unsupported route: {method} {path}",
    )


def _handle_post_artifacts(event: dict) -> dict:
    try:
        payload = _parse_json_body(event)
    except ValueError as exc:
        return _error_response(400, str(exc))

    queries = payload.get("artifact_queries") or payload.get("queries") or []
    if not queries:
        queries = [{"name": "*"}]
    if not isinstance(queries, list):
        return _error_response(400, "`artifact_queries` must be a list")

    limit = _clamp_limit(
        payload.get("limit")
        or _get_query_param(event, "limit")
        or _get_header(event, "x-limit")
    )
    pagination = _extract_pagination_params(payload, event)
    offset_input = (
        payload.get("offset")
        or _get_query_param(event, "offset")
        or _get_header(event, "x-offset")
    )
    target = min(limit + pagination.skip, MAX_LIMIT + pagination.skip)

    repo = _get_repository()
    try:
        artifacts, next_key = repo.search(queries, target, pagination.start_key)
    except RepositoryError as exc:
        LOGGER.error("Failed to execute artifact search: %s", exc)
        return _error_response(500, "Unable to execute search")

    artifacts = artifacts[pagination.skip : pagination.skip + limit]
    plain_artifacts = [_dynamo_to_plain(a) for a in artifacts]

    headers = {}
    if next_key:
        headers["x-next-offset"] = _encode_pagination_token(next_key)

    normalized_offset = pagination.skip
    if normalized_offset == 0 and isinstance(offset_input, str) and offset_input.strip():
        normalized_offset = offset_input

    body = {
        "artifacts": plain_artifacts,
        "page": {
            "offset": normalized_offset,
            "limit": limit,
            "returned": len(plain_artifacts),
            "has_more": bool(next_key),
        },
    }
    return _success_response(body, headers=headers)


def _handle_get_artifact_by_name(event: dict, path: str) -> dict:
    name = _extract_name_parameter(event, path)
    if not name:
        return _error_response(400, "Missing artifact name in path")

    repo = _get_repository()
    try:
        records = repo.fetch_by_name(name)
    except RepositoryError as exc:
        LOGGER.error("Failed to fetch artifact by name %s: %s", name, exc)
        return _error_response(500, "Unable to query artifact metadata")

    if not records:
        return _error_response(404, f"No artifact found with name '{name}'")

    return _success_response({"artifacts": [_dynamo_to_plain(r) for r in records]})


def _handle_post_regex(event: dict) -> dict:
    try:
        payload = _parse_json_body(event)
    except ValueError as exc:
        return _error_response(400, str(exc))

    pattern = payload.get("pattern")
    if not pattern:
        return _error_response(400, "`pattern` is required for regex search")

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return _error_response(400, f"Invalid regular expression: {exc}")

    limit = _clamp_limit(
        payload.get("limit")
        or _get_query_param(event, "limit")
        or _get_header(event, "x-limit")
    )
    pagination = _extract_pagination_params(payload, event)
    offset_input = (
        payload.get("offset")
        or _get_query_param(event, "offset")
        or _get_header(event, "x-offset")
    )
    target = min(limit + pagination.skip, MAX_LIMIT + pagination.skip)

    repo = _get_repository()
    try:
        artifacts, next_key = repo.regex_search(compiled, target, pagination.start_key)
    except RepositoryError as exc:
        LOGGER.error("Failed to run regex search: %s", exc)
        return _error_response(500, "Unable to execute regex search")

    artifacts = artifacts[pagination.skip : pagination.skip + limit]
    plain_artifacts = [_dynamo_to_plain(a) for a in artifacts]

    headers = {}
    if next_key:
        headers["x-next-offset"] = _encode_pagination_token(next_key)

    normalized_offset = pagination.skip
    if normalized_offset == 0 and isinstance(offset_input, str) and offset_input.strip():
        normalized_offset = offset_input

    body = {
        "artifacts": plain_artifacts,
        "page": {
            "offset": normalized_offset,
            "limit": limit,
            "returned": len(plain_artifacts),
            "has_more": bool(next_key),
        },
    }
    return _success_response(body, headers=headers)


# -----------------------------------------------------------------------------
# Helper utilities


def _parse_json_body(event: dict) -> dict:
    body = event.get("body")
    if body is None:
        return {}

    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError:
        raise ValueError("Request body must be valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _extract_name_parameter(event: dict, normalized_path: str) -> str | None:
    path_params = event.get("pathParameters") or {}
    if "name" in path_params and path_params["name"]:
        return unquote_plus(path_params["name"]).strip()

    suffix = normalized_path[len("/artifact/byName") :].lstrip("/")
    if suffix:
        return unquote_plus(suffix).strip()

    return None


def _extract_method(event: dict) -> str:
    if "requestContext" in event:
        http_info = event["requestContext"].get("http") or {}
        method = http_info.get("method")
        if method:
            return method.upper()
    if "httpMethod" in event:
        return str(event["httpMethod"]).upper()
    return "GET"


def _normalize_path(event: dict) -> str:
    path = event.get("rawPath") or event.get("path") or "/"
    stage = ""
    if "requestContext" in event:
        stage = event["requestContext"].get("stage") or ""

    if stage:
        stage_prefix = f"/{stage}"
        if path.startswith(stage_prefix):
            path = path[len(stage_prefix) :]
            if not path.startswith("/"):
                path = f"/{path}"
    return path or "/"


def _clamp_limit(candidate: Any) -> int:
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


def _extract_pagination_params(payload: dict, event: dict) -> PaginationParams:
    raw = (
        payload.get("offset")
        or _get_query_param(event, "offset")
        or _get_header(event, "x-offset")
    )
    if raw is None:
        return PaginationParams()
    if isinstance(raw, (int, float)):
        return PaginationParams(skip=max(int(raw), 0))
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return PaginationParams()
        if raw.isdigit():
            return PaginationParams(skip=max(int(raw), 0))
        try:
            return PaginationParams(start_key=_decode_pagination_token(raw))
        except ValueError:
            LOGGER.debug("Ignoring invalid pagination token")
            return PaginationParams()
    try:
        return PaginationParams(skip=max(int(raw), 0))
    except (TypeError, ValueError):
        return PaginationParams()


def _decode_pagination_token(token: str) -> dict[str, Any]:
    padding = "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8")
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid pagination token") from exc
    if not isinstance(data, dict):
        raise ValueError("Invalid pagination token")
    return data


def _encode_pagination_token(key: dict[str, Any]) -> str:
    payload = json.dumps(key)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8").rstrip("=")


def _get_query_param(event: dict, key: str) -> str | None:
    params = event.get("queryStringParameters") or {}
    if key in params and params[key] is not None:
        return params[key]

    multi = event.get("multiValueQueryStringParameters") or {}
    values = multi.get(key)
    if values:
        return values[0]
    return None


def _get_header(event: dict, key: str) -> str | None:
    headers = event.get("headers") or {}
    key_lower = key.lower()
    for header_key, header_value in headers.items():
        if header_key.lower() == key_lower and header_value is not None:
            return header_value
    return None


def _success_response(body: dict, headers: dict | None = None) -> dict:
    _headers = {"Content-Type": "application/json"}
    if headers:
        _headers.update(headers)
    return {"statusCode": 200, "headers": _headers, "body": json.dumps(body)}


def _error_response(status_code: int, message: str) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message, "status": status_code}),
    }


def _dynamo_to_plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [_dynamo_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _dynamo_to_plain(v) for k, v in value.items()}
    return value


# -----------------------------------------------------------------------------
# Repository implementation

_REPOSITORY: ArtifactRepository | None = None


def _get_repository() -> "ArtifactRepository":
    global _REPOSITORY
    if _REPOSITORY is None:
        table_name = os.environ.get("MODEL_REGISTRY_TABLE", DEFAULT_TABLE_NAME)
        _REPOSITORY = ArtifactRepository(table_name)
    return _REPOSITORY


class ArtifactRepository:
    def __init__(self, table_name: str):
        dynamodb = boto3.resource("dynamodb")
        self.table = dynamodb.Table(table_name)

    def search(
        self,
        queries: Sequence[dict],
        total_needed: int,
        start_key: dict[str, Any] | None = None,
    ) -> tuple[list[dict], dict[str, Any] | None]:
        normalized_queries = queries or [{"name": "*"}]
        predicate = lambda item: _matches_any_query(item, normalized_queries)
        return self._scan_models(total_needed, start_key, predicate)

    def fetch_by_name(self, name: str) -> list[dict]:
        try:
            response = self.table.query(
                IndexName="NameIndex",
                KeyConditionExpression=Key("name_lc").eq(name.lower()),
                FilterExpression=Attr("type").eq("MODEL") & Attr("sk").eq("META"),
            )
        except Exception as exc:  # pragma: no cover - boto3 specific failure path
            raise RepositoryError(str(exc)) from exc
        return response.get("Items", [])

    def regex_search(
        self,
        pattern: re.Pattern,
        total_needed: int,
        start_key: dict[str, Any] | None = None,
    ) -> tuple[list[dict], dict[str, Any] | None]:
        def predicate(item: dict) -> bool:
            haystacks = [
                item.get("name_lc") or "",
                item.get("name") or "",
                item.get("readme_text") or "",
            ]
            return any(
                isinstance(text, str) and pattern.search(text)
                for text in haystacks
            )

        return self._scan_models(total_needed, start_key, predicate)

    def _scan_models(
        self,
        total_needed: int,
        start_key: dict[str, Any] | None,
        predicate,
    ) -> tuple[list[dict], dict[str, Any] | None]:
        matched: list[dict] = []
        exclusive_start = start_key
        next_key: dict[str, Any] | None = None
        try:
            while len(matched) < total_needed:
                scan_limit = max(total_needed - len(matched), SCAN_BATCH_SIZE)
                scan_kwargs: dict[str, Any] = {
                    "FilterExpression": Attr("type").eq("MODEL") & Attr("sk").eq("META"),
                    "Limit": scan_limit,
                }
                if exclusive_start:
                    scan_kwargs["ExclusiveStartKey"] = exclusive_start

                response = self.table.scan(**scan_kwargs)
                items = response.get("Items", [])
                last_consumed_key: dict[str, Any] | None = None
                for item in items:
                    if "pk" in item and "sk" in item:
                        last_consumed_key = {"pk": item["pk"], "sk": item["sk"]}
                    if predicate(item):
                        matched.append(item)
                        if len(matched) >= total_needed:
                            next_key = response.get("LastEvaluatedKey") or last_consumed_key
                            break

                if len(matched) >= total_needed:
                    break

                exclusive_start = response.get("LastEvaluatedKey")
                if not exclusive_start:
                    next_key = None
                    break

            else:
                next_key = response.get("LastEvaluatedKey")
        except Exception as exc:  # pragma: no cover - boto3 specific failure path
            raise RepositoryError(str(exc)) from exc

        return matched, next_key


def _matches_any_query(item: dict, queries: Sequence[dict]) -> bool:
    return any(_matches_single_query(item, query or {}) for query in queries)


def _matches_single_query(item: dict, query: dict) -> bool:
    if item.get("type") != "MODEL" or item.get("sk") != "META":
        return False

    name_filter = str(query.get("name") or "*").strip()
    if name_filter not in ("", "*"):
        needle = name_filter.lower()
        haystack = (item.get("name_lc") or "").lower()
        if needle not in haystack:
            return False

    filters = query.get("filters") or {}
    license_filter = filters.get("license") or query.get("license")
    if license_filter:
        if (item.get("license") or "").lower() != str(license_filter).lower():
            return False

    average_filter = filters.get("metrics.average")
    if average_filter is not None:
        if not _value_meets_numeric_filter(
            _decimal_to_float(item.get("metrics", {}).get("average")),
            average_filter,
        ):
            return False

    base_model_filter = (
        filters.get("base_models[].model_id")
        or filters.get("base_models.model_id")
        or filters.get("base_model_id")
    )
    if base_model_filter is not None:
        if not _base_models_match(item.get("base_models") or [], base_model_filter):
            return False

    eligibility_filter = filters.get("eligibility.minimum_evidence_met")
    if eligibility_filter is not None:
        expected = bool(eligibility_filter)
        actual = bool(item.get("eligibility", {}).get("minimum_evidence_met"))
        if expected != actual:
            return False

    return True


def _base_models_match(base_models: list[dict], needle: Any) -> bool:
    if not base_models:
        return False
    if isinstance(needle, (list, tuple, set)):
        targets = {str(v).lower() for v in needle}
    else:
        targets = {str(needle).lower()}
    for entry in base_models:
        model_id = str(entry.get("model_id") or entry.get("modelId") or "").lower()
        if model_id and model_id in targets:
            return True
    return False


def _value_meets_numeric_filter(value: Any, expected: Any) -> bool:
    if value is None:
        return False
    if isinstance(expected, dict):
        min_value = _decimal_to_float(
            expected.get("min")
            or expected.get("gte")
            or expected.get("minimum")
        )
        max_value = _decimal_to_float(
            expected.get("max")
            or expected.get("lte")
            or expected.get("maximum")
        )
    else:
        parsed = _decimal_to_float(expected)
        min_value = parsed
        max_value = parsed

    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def _decimal_to_float(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Retain backwards-compatible name in case other modules import this symbol.
search = handle_search
