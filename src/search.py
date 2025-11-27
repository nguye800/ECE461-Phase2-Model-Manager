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


_DDB_ARTIFACT_TYPES = ["MODEL", "DATASET", "CODE"]

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
    method = (event.get("requestContext", {}).get("http", {}) or {}).get("method") or event.get("httpMethod")
    path = event.get("rawPath") or event.get("path")
    print(
        f"[search.lambda] Received {method or 'UNKNOWN'} {path or '/'} body={event.get('body')}",
        flush=True,
    )
    return handle_search(event, context)


def handle_search(event: dict, _context: Any) -> dict:
    """
    Route incoming API Gateway events to the appropriate handler.
    """

    method = _extract_method(event)
    path = _normalize_path(event)
    print(f"[search.handle] Routing {method} {path}", flush=True)

    if method == "POST" and path == "/artifacts":
        print("[search.handle] dispatching to _handle_post_artifacts", flush=True)
        return _handle_post_artifacts(event)
    if method == "GET" and path.startswith("/artifact/byName"):
        print("[search.handle] dispatching to _handle_get_artifact_by_name", flush=True)
        return _handle_get_artifact_by_name(event, path)
    if method == "POST" and path == "/artifact/byRegEx":
        print("[search.handle] dispatching to _handle_post_regex", flush=True)
        return _handle_post_regex(event)

    print("[search.handle] unsupported route", flush=True)
    return _error_response(
        404,
        f"Unsupported route: {method} {path}",
        log_prefix="[search.handle]",
    )


def _handle_post_artifacts(event: dict) -> dict:
    log_prefix = "[search.post_artifacts]"
    try:
        queries_input = _parse_json_body(event, expected_type=list)
    except ValueError as exc:
        return _error_response(400, str(exc), log_prefix=log_prefix)
    print(f"[search.post_artifacts] payload={queries_input}", flush=True)

    queries = queries_input or [{"name": "*"}]
    if not queries:
        queries = [{"name": "*"}]

    normalized_queries: list[dict] = []
    for query in queries:
        if not isinstance(query, dict):
            return _error_response(
                400, "Each artifact_query must be a JSON object", log_prefix=log_prefix
            )
        normalized_query = dict(query)
        normalized_query["name"] = str(query.get("name") or "*").strip() or "*"
        original_types = query.get("types")
        try:
            normalized_types = _normalize_type_filters(original_types)
        except ValueError as exc:
            return _error_response(400, str(exc), log_prefix=log_prefix)
        if normalized_types is not None:
            normalized_query["types"] = normalized_types
        else:
            normalized_query.pop("types", None)
            if isinstance(original_types, list) and not original_types:
                print(
                    "[search.post_artifacts] treating blank 'types' filter as wildcard",
                    flush=True,
                )
        normalized_queries.append(normalized_query)

    limit = _clamp_limit(
        _get_query_param(event, "limit")
        or _get_header(event, "x-limit")
    )
    pagination = _extract_pagination_params(None, event)
    target = min(limit + pagination.skip, MAX_LIMIT + pagination.skip)

    repo = _get_repository()
    try:
        artifacts, next_key = repo.search(normalized_queries, target, pagination.start_key)
    except RepositoryError as exc:
        LOGGER.error("Failed to execute artifact search: %s", exc)
        return _error_response(500, "Unable to execute search", log_prefix=log_prefix)

    artifacts = artifacts[pagination.skip : pagination.skip + limit]
    plain_artifacts = [_artifact_metadata(a) for a in artifacts]

    if len(plain_artifacts) > DEFAULT_LIMIT:
        print(
            f"[search.post_artifacts] rejecting {len(plain_artifacts)} artifacts (> {DEFAULT_LIMIT})",
            flush=True,
        )
        return _error_response(413, "too many artifacts returned.", log_prefix=log_prefix)

    headers = _build_offset_header(next_key)

    print(
        f"[search.post_artifacts] returning {len(plain_artifacts)} artifacts next_key={bool(next_key)}",
        flush=True,
    )
    return _success_response(plain_artifacts, headers=headers, log_prefix=log_prefix)


def _handle_get_artifact_by_name(event: dict, path: str) -> dict:
    log_prefix = "[search.byName]"
    name = _extract_name_parameter(event, path)
    if not name:
        print(f"[search.byName] Missing artifact name in path", flush=True)
        return _error_response(400, "Missing artifact name in path", log_prefix=log_prefix)
    print(f"[search.byName] looking up name={name}", flush=True)

    repo = _get_repository()
    try:
        records = repo.fetch_by_name(name)
    except RepositoryError as exc:
        LOGGER.error("Failed to fetch artifact by name %s: %s", name, exc)
        return _error_response(500, "Unable to query artifact metadata", log_prefix=log_prefix)

    if not records:
        return _error_response(404, f"No artifact found with name '{name}'", log_prefix=log_prefix)

    print(f"[search.byName] found {len(records)} records", flush=True)
    return _success_response([_artifact_metadata(r) for r in records], log_prefix=log_prefix)


def _handle_post_regex(event: dict) -> dict:
    log_prefix = "[search.regex]"
    try:
        payload = _parse_json_body(event, expected_type=dict)
    except ValueError as exc:
        return _error_response(400, str(exc), log_prefix=log_prefix)

    pattern = payload.get("regex")
    if pattern is None:
        pattern = payload.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return _error_response(400, "`regex` is required for regex search", log_prefix=log_prefix)

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return _error_response(400, f"Invalid regular expression: {exc}", log_prefix=log_prefix)

    limit = _clamp_limit(
        payload.get("limit")
        or _get_query_param(event, "limit")
        or _get_header(event, "x-limit")
    )
    pagination = _extract_pagination_params(payload, event)
    target = min(limit + pagination.skip, MAX_LIMIT + pagination.skip)

    repo = _get_repository()
    try:
        artifacts, next_key = repo.regex_search(compiled, target, pagination.start_key)
    except RepositoryError as exc:
        LOGGER.error("Failed to run regex search: %s", exc)
        return _error_response(500, "Unable to execute regex search", log_prefix=log_prefix)

    artifacts = artifacts[pagination.skip : pagination.skip + limit]
    plain_artifacts = [_artifact_metadata(a) for a in artifacts]
    if not plain_artifacts:
        print("[search.regex] no artifacts matched regex", flush=True)
        return _error_response(404, "No artifact found under this regex.", log_prefix=log_prefix)

    headers = _build_offset_header(next_key)

    print(
        f"[search.regex] returning {len(plain_artifacts)} artifacts next={bool(next_key)}",
        flush=True,
    )
    return _success_response(plain_artifacts, headers=headers, log_prefix=log_prefix)


# -----------------------------------------------------------------------------
# Helper utilities


def _parse_json_body(event: dict, *, expected_type: type | tuple[type, ...] = dict):
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required.")

    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError:
        raise ValueError("Request body must be valid JSON") from None
    if expected_type is not None and not isinstance(payload, expected_type):
        expected_name = (
            " or ".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ValueError(f"Request body must be a JSON {expected_name}")
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


def _extract_pagination_params(payload: Any, event: dict) -> PaginationParams:
    body_offset = payload.get("offset") if isinstance(payload, dict) else None
    raw = (
        body_offset
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


def _build_offset_header(next_key: dict[str, Any] | None) -> dict:
    if not next_key:
        return {}
    return {"offset": _encode_pagination_token(next_key)}


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


def _log_response(prefix: str, status_code: int) -> None:
    print(f"[{prefix}] Responding with status {status_code}", flush=True)


def _success_response(
    body: dict, headers: dict | None = None, log_prefix: str = "search"
) -> dict:
    _headers = {"Content-Type": "application/json"}
    if headers:
        _headers.update(headers)
    _log_response(log_prefix, 200)
    return {"statusCode": 200, "headers": _headers, "body": json.dumps(body)}


def _error_response(status_code: int, message: str, log_prefix: str = "search") -> dict:
    _log_response(log_prefix, status_code)
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


def _normalize_type_filters(types_value: Any) -> list[str] | None:
    if types_value is None:
        return None
    if not isinstance(types_value, list):
        raise ValueError("`types` must be an array of artifact types")
    normalized: list[str] = []
    for entry in types_value:
        normalized_entry = _normalize_user_type(entry)
        if normalized_entry:
            normalized.append(normalized_entry)
    if not normalized:
        return None
    return normalized


def _normalize_user_type(value: Any) -> str:
    if value is None:
        raise ValueError("`types` entries must be non-empty strings")
    normalized = str(value).strip().lower()
    allowed = {"model", "dataset", "code"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported artifact type '{value}'")
    return normalized


def _artifact_metadata(item: dict) -> dict:
    plain = _dynamo_to_plain(item)
    metadata_section = (
        plain.get("metadata") if isinstance(plain, dict) else None
    )
    def _first(*keys):
        for key in keys:
            if isinstance(key, tuple):
                current = metadata_section if key[0] == "metadata" else plain
                attribute = current.get(key[1]) if current else None
            else:
                attribute = plain.get(key) if isinstance(plain, dict) else None
            if attribute:
                return attribute
        return None

    name = _first("name", ("metadata", "name"))
    artifact_id = _first("id", "artifact_id", ("metadata", "id"))
    artifact_type = _normalize_response_type(
        _first("artifact_type", "type", ("metadata", "type"))
    )

    return {
        "name": name,
        "id": artifact_id,
        "type": artifact_type,
    }


def _normalize_response_type(value: Any) -> str:
    if value is None:
        return "model"
    normalized = str(value).strip().lower()
    mapping = {"model", "dataset", "code"}
    if normalized in mapping:
        return normalized
    return normalized or "model"


def _extract_item_type(item: dict) -> str:
    raw_value = (
        item.get("artifact_type")
        or item.get("type")
        or item.get("entity_type")
    )
    if raw_value is None:
        return "model"
    normalized = str(raw_value).strip().lower()
    if normalized in {"model", "dataset", "code"}:
        return normalized
    if normalized == "model_meta":
        return "model"
    return normalized or "model"


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
        matched: list[dict] = []
        lower_name = name.lower()
        try:
            for artifact_type in _DDB_ARTIFACT_TYPES:
                response = self.table.query(
                    IndexName="GSI_ALPHABET_LISTING",
                    KeyConditionExpression=Key("type").eq(artifact_type)
                    & Key("name_lc").eq(lower_name),
                )
                matched.extend(response.get("Items", []))
        except Exception as exc:
            raise RepositoryError(str(exc)) from exc

        return matched

    

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
                    "FilterExpression": Attr("sk").eq("META")
                    & Attr("type").is_in(_DDB_ARTIFACT_TYPES),
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
    if item.get("sk") != "META":
        return False

    artifact_type = _extract_item_type(item)

    type_filters = query.get("types")
    if type_filters:
        if artifact_type not in type_filters:
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
