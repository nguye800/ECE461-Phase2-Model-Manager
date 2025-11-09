"""
Lambda entry point for searching model artifacts stored in the registry.

Supported routes (API Gateway compatible):
    POST /artifacts
        Accepts a JSON body with `artifact_queries`, pagination controls, and
        optional filters. Returns artifacts that satisfy any of the supplied
        queries. The response header `x-next-offset` indicates how to fetch the
        next page.

    GET /artifact/byName/{name}
        Returns every artifact that matches the provided name (case-insensitive).

    POST /artifact/byRegEx
        Accepts JSON with `pattern` and returns artifacts whose name or README
        text matches the provided regular expression.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote_plus


LOGGER = logging.getLogger(__name__)

MODEL_REGISTRY_TABLE = os.environ.get("MODEL_REGISTRY_TABLE", "model_registry")
DB_PATH_ENV_VAR = "MODEL_REGISTRY_DB_PATH"
DEFAULT_DB_PATH = os.environ.get(
    DB_PATH_ENV_VAR, str(Path(__file__).resolve().parent.parent / "models.db")
)
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
READ_ME_CANDIDATE_COLUMNS = (
    "readme",
    "readme_text",
    "readme_blob",
    "readme_markdown",
)


class RepositoryError(RuntimeError):
    """Raised when the backing artifact repository cannot be queried."""


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
        # Per spec, callers can enumerate everything with {"name": "*"}
        queries = [{"name": "*"}]
    if not isinstance(queries, list):
        return _error_response(400, "`artifact_queries` must be a list")

    limit = _clamp_limit(
        payload.get("limit")
        or _get_query_param(event, "limit")
        or _get_header(event, "x-limit")
    )
    offset = _clamp_offset(
        payload.get("offset")
        or _get_query_param(event, "offset")
        or _get_header(event, "x-offset")
    )

    repo = _get_repository()
    try:
        artifacts, has_more = repo.search(queries, limit, offset)
    except RepositoryError as exc:
        LOGGER.error("Failed to execute artifact search: %s", exc)
        return _error_response(500, "Unable to execute search")

    headers = {}
    if has_more:
        headers["x-next-offset"] = str(offset + limit)

    body = {
        "artifacts": artifacts,
        "page": {
            "offset": offset,
            "limit": limit,
            "returned": len(artifacts),
            "has_more": has_more,
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

    return _success_response({"artifacts": records})


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
    offset = _clamp_offset(
        payload.get("offset")
        or _get_query_param(event, "offset")
        or _get_header(event, "x-offset")
    )

    repo = _get_repository()
    try:
        artifacts, has_more = repo.regex_search(compiled, limit, offset)
    except RepositoryError as exc:
        LOGGER.error("Failed to run regex search: %s", exc)
        return _error_response(500, "Unable to execute regex search")

    headers = {}
    if has_more:
        headers["x-next-offset"] = str(offset + limit)

    body = {
        "artifacts": artifacts,
        "page": {
            "offset": offset,
            "limit": limit,
            "returned": len(artifacts),
            "has_more": has_more,
        },
    }
    return _success_response(body, headers=headers)


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


def _clamp_offset(candidate: Any) -> int:
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


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


_REPOSITORY: ArtifactRepository | None = None


def _get_repository() -> ArtifactRepository:
    global _REPOSITORY
    if _REPOSITORY is None:
        _REPOSITORY = ArtifactRepository(
            os.environ.get(DB_PATH_ENV_VAR, DEFAULT_DB_PATH),
            MODEL_REGISTRY_TABLE,
        )
    return _REPOSITORY


@dataclass
class ArtifactRepository:
    db_path: str
    table_name: str = MODEL_REGISTRY_TABLE

    def search(
        self,
        queries: Sequence[dict],
        limit: int,
        offset: int,
    ) -> tuple[list[dict], bool]:
        where_sql, params = self._compile_where_clause(queries)
        limit_for_fetch = min(limit, MAX_LIMIT) + 1
        sql = f"""
            SELECT *
            FROM {self.table_name}
            WHERE {where_sql}
            ORDER BY repo_id COLLATE NOCASE, model_id
            LIMIT ?
            OFFSET ?
        """
        params.extend([limit_for_fetch, offset])
        rows = self._fetch_rows(sql, params)
        has_more = len(rows) > limit
        artifacts = [_normalize_row(row) for row in rows[:limit]]
        return artifacts, has_more

    def fetch_by_name(self, name: str) -> list[dict]:
        sql = f"""
            SELECT *
            FROM {self.table_name}
            WHERE LOWER(repo_id) = ?
            ORDER BY model_id
        """
        rows = self._fetch_rows(sql, [name.lower()])
        return [_normalize_row(row) for row in rows]

    def regex_search(
        self,
        pattern: re.Pattern,
        limit: int,
        offset: int,
    ) -> tuple[list[dict], bool]:
        sql = f"SELECT * FROM {self.table_name}"
        rows = self._fetch_rows(sql, [])
        matches = []
        for row in rows:
            haystacks = [row.get("repo_id") or ""]
            readme_content = _extract_readme(row)
            if readme_content:
                haystacks.append(readme_content)

            if any(pattern.search(text) for text in haystacks if text):
                matches.append(_normalize_row(row))

        limited = matches[offset : offset + limit]
        has_more = len(matches) > offset + len(limited)
        return limited, has_more

    def _compile_where_clause(
        self, queries: Sequence[dict]
    ) -> tuple[str, list[Any]]:
        if not queries:
            return "1=1", []

        clauses: list[str] = []
        params: list[Any] = []
        for query in queries:
            query = query or {}
            filters = query.get("filters") or {}
            merged = {**query, **filters}
            conds: list[str] = []
            cond_params: list[Any] = []

            name = merged.get("name")
            if name and name != "*":
                conds.append("LOWER(repo_id) LIKE ?")
                cond_params.append(f"%{str(name).lower()}%")

            license_filter = merged.get("license")
            if license_filter:
                conds.append("LOWER(license) = ?")
                cond_params.append(str(license_filter).lower())

            dataset_name = merged.get("dataset_name") or merged.get("dataset")
            if dataset_name:
                conds.append("LOWER(dataset_name) LIKE ?")
                cond_params.append(f"%{str(dataset_name).lower()}%")

            dataset_link = merged.get("dataset_link")
            if dataset_link:
                conds.append("LOWER(dataset_link) LIKE ?")
                cond_params.append(f"%{str(dataset_link).lower()}%")

            base_model = merged.get("base_models_modelID") or merged.get("base_model")
            if base_model:
                conds.append("LOWER(base_models_modelID) LIKE ?")
                cond_params.append(f"%{str(base_model).lower()}%")

            github_link = merged.get("github_link")
            if github_link:
                conds.append("LOWER(github_link) LIKE ?")
                cond_params.append(f"%{str(github_link).lower()}%")

            _append_range_filter(
                merged,
                "likes",
                "min_likes",
                "max_likes",
                conds,
                cond_params,
            )
            _append_range_filter(
                merged,
                "downloads",
                "min_downloads",
                "max_downloads",
                conds,
                cond_params,
            )
            _append_range_filter(
                merged,
                "parameter_number",
                "min_parameter_number",
                "max_parameter_number",
                conds,
                cond_params,
            )
            _append_range_filter(
                merged,
                "gb_size_of_model",
                "min_model_gb",
                "max_model_gb",
                conds,
                cond_params,
            )

            if conds:
                clauses.append(f"({' AND '.join(conds)})")
                params.extend(cond_params)
            else:
                clauses.append("(1=1)")

        return " OR ".join(clauses), params

    def _fetch_rows(self, sql: str, params: Sequence[Any]) -> list[dict]:
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise RepositoryError(str(exc)) from exc


def _append_range_filter(
    merged: dict,
    column_name: str,
    min_key: str,
    max_key: str,
    conds: list[str],
    params: list[Any],
):
    min_value = _coerce_number(merged.get(min_key))
    if min_value is not None:
        conds.append(f"{column_name} >= ?")
        params.append(min_value)

    max_value = _coerce_number(merged.get(max_key))
    if max_value is not None:
        conds.append(f"{column_name} <= ?")
        params.append(max_value)

    exact_value = merged.get(column_name)
    if exact_value is not None and min_value is None and max_value is None:
        coerced = _coerce_number(exact_value)
        if coerced is not None:
            conds.append(f"{column_name} = ?")
            params.append(coerced)


def _coerce_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return value
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_row(row: dict) -> dict:
    normalized = dict(row)
    rename_map = {
        "repo_id": "name",
        "gb_size_of_model": "model_size_gb",
        "parameter_number": "parameter_count",
    }
    for src, dest in rename_map.items():
        if src in normalized:
            normalized[dest] = normalized.pop(src)

    readme_value = _extract_readme(normalized)
    if readme_value is not None:
        normalized["readme_text"] = readme_value
    return normalized


def _extract_readme(row: dict) -> str | None:
    for column in READ_ME_CANDIDATE_COLUMNS:
        if column in row and row[column]:
            value = row[column]
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")
            return str(value)
    return None


# Retain backwards-compatible name in case other modules import this symbol.
search = handle_search
