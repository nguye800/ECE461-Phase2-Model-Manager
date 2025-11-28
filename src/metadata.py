"""Metadata Lambda consolidating artifact metadata + download endpoints."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

ARTIFACTS_TABLE_NAME = os.environ.get("ARTIFACTS_DDB_TABLE", "model-metadata")
ARTIFACTS_DDB_REGION = os.environ.get("ARTIFACTS_DDB_REGION", "us-east-1")
DEFAULT_MODEL_BUCKET = os.environ.get("MODEL_BUCKET_NAME", "modelzip-logs-artifacts")
S3_CLIENT = boto3.client("s3")

BYTES_IN_MB = 1024 * 1024


class MissingDownloadLocation(Exception):
    """Raised when an artifact does not include an S3 download reference."""


class ObjectSizeResolutionError(Exception):
    """Raised when an S3 object's size cannot be determined."""


def _dynamodb():
    if ARTIFACTS_DDB_REGION:
        return boto3.resource("dynamodb", region_name=ARTIFACTS_DDB_REGION)
    return boto3.resource("dynamodb")


def _table():
    return _dynamodb().Table(ARTIFACTS_TABLE_NAME)


# Backwards-compat aliases for tests/legacy imports
def _get_table():  # pragma: no cover - shim for tests
    return _table()


def _response(status: int, body: Dict[str, Any], prefix: str = "[metadata]"):
    print(f"{prefix} Responding with status {status}", flush=True)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _convert(value):
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value


def _parse_path(path: str) -> Tuple[str, ...]:
    return tuple(unquote(part) for part in path.strip("/").split("/") if part)


def _not_found(artifact_type: str, artifact_id: str):
    return _response(
        404,
        {"message": f"Artifact '{artifact_type}/{artifact_id}' not found."},
        prefix="[metadata.not_found]",
    )


def _get_artifact(artifact_type: str, artifact_id: str) -> Optional[Dict[str, Any]]:
    pk = f"{artifact_type.upper()}#{artifact_id}"
    result = _get_table().get_item(Key={"pk": pk, "sk": "META"})
    item = result.get("Item")
    return _convert(item) if item else None


def _get_audits(artifact_type: str, artifact_id: str):
    pk = f"{artifact_type.upper()}#{artifact_id}"
    response = _get_table().query(
        KeyConditionExpression=Key("pk").eq(pk) & Key("sk").begins_with("AUDIT#"),
        ScanIndexForward=False,
    )
    return _convert(response.get("Items", []))


def _evaluate_license_text(license_text: Optional[str]) -> Dict[str, Any]:
    license_text = (license_text or "").lower()
    permissive = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}
    restricted = {"gpl-3.0", "agpl-3.0"}


    if license_text in {"", "unknown"}:
        return {
            "fine_tune_allowed": False,
            "inference_allowed": False,
            "reason": "License is unknown",
        }


    if any(term in license_text for term in restricted):
        return {
            "fine_tune_allowed": False,
            "inference_allowed": license_text not in {"agpl-3.0"},
            "reason": f"License '{license_text}' restricts fine-tuning",
        }

    allowed = license_text in permissive or "apache" in license_text
    return {
        "fine_tune_allowed": allowed,
        "inference_allowed": allowed,
        "reason": f"License '{license_text}' treated as {'permissive' if allowed else 'unknown'}",
    }


def _evaluate_license(metadata: Dict[str, Any]):
    return _evaluate_license_text(metadata.get("license"))


def _fetch_github_license_identifier(github_url: str) -> Optional[str]:
    try:
        parsed = urlparse(github_url)
    except ValueError:
        return None
    if "github.com" not in parsed.netloc.lower():
        return None
    path_parts = [segment for segment in parsed.path.split("/") if segment]
    if len(path_parts) < 2:
        return None
    owner, repo = path_parts[0], path_parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "metadata-lambda",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None

    license_info = payload.get("license") or {}
    spdx = license_info.get("spdx_id")
    if not spdx or spdx.upper() == "NOASSERTION":
        return "unknown"
    return spdx.lower()


def _parse_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "t", "yes", "y"}


def _parse_s3_uri(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None
    parsed = urlparse(value)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    return DEFAULT_MODEL_BUCKET, value


def _resolve_artifact_location(
    metadata: Dict[str, Any], artifact_type: str, *, strict: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    assets = metadata.get("assets") or {}
    bucket = assets.get("bucket")

    def _candidate_keys() -> List[str]:
        base_keys = [
            f"{artifact_type}_key",
            f"{artifact_type}_zip_key",
        ]
        if artifact_type == "model":
            return ["model_zip_key", "key"] + base_keys
        if artifact_type == "code":
            return base_keys + ["codebase_key"]
        if artifact_type == "codebase":
            return ["codebase_key", "code_key"]
        return base_keys

    key = None
    for candidate in _candidate_keys():
        candidate_value = assets.get(candidate)
        if candidate_value:
            key = candidate_value
            break

    if not key and artifact_type == "model":
        key = metadata.get("s3_location_of_model_zip")
    if not key:
        fallback_fields = [
            f"{artifact_type}_s3_key",
            f"{artifact_type}_location",
        ]
        for field in fallback_fields:
            value = metadata.get(field)
            if value:
                key = value
                break

    bucket_override, normalized_key = _parse_s3_uri(key)
    bucket = bucket or bucket_override
    if strict and (not bucket or not normalized_key):
        raise MissingDownloadLocation(
            f"Artifact '{metadata.get('model_id')}' is missing an S3 download reference for '{artifact_type}'."
        )
    return bucket, normalized_key


def _build_download_url(bucket: Optional[str], key: Optional[str]) -> Optional[str]:
    if not bucket or not key:
        return None
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _build_artifact_envelope(artifact_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    metadata_block = item.get("metadata") or {}
    artifact_id = metadata_block.get("id") or item.get("model_id")
    name = metadata_block.get("name") or item.get("name") or artifact_id
    type_value = metadata_block.get("type") or artifact_type

    if not artifact_id:
        raise ValueError("Artifact does not contain a model identifier.")

    metadata_payload = {
        "name": name,
        "id": artifact_id,
        "type": type_value,
    }

    data_block = item.get("data") or {}
    source_url = data_block.get("url") or item.get("model_url")
    download_url = data_block.get("download_url")

    bucket, key = _resolve_artifact_location(item, artifact_type)
    download_url = download_url or _build_download_url(bucket, key)

    if not source_url:
        raise ValueError("Artifact does not contain a source url.")
    if not download_url:
        raise ValueError("Artifact does not have a download url.")

    data_payload = {"url": source_url, "download_url": download_url}

    return {"metadata": metadata_payload, "data": data_payload}


def _object_size_mb(bucket: str, key: str) -> float:
    if not bucket or not key:
        raise MissingDownloadLocation("Missing S3 location for artifact component.")
    try:
        response = S3_CLIENT.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise ObjectSizeResolutionError(
            f"Unable to determine size for s3://{bucket}/{key}"
        ) from exc
    content_length = response.get("ContentLength")
    if content_length is None:
        raise ObjectSizeResolutionError(
            f"S3 object s3://{bucket}/{key} is missing ContentLength metadata."
        )
    return round(content_length / float(BYTES_IN_MB), 4)


def _parse_json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required.")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body)
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if not body:
        raise ValueError("Request body is required.")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _build_lineage_graph(
    artifact_id: str, artifact_type: str, metadata_item: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    nodes[artifact_id] = {
        "artifact_id": artifact_id,
        "name": metadata_item.get("name") or metadata_item.get("metadata", {}).get("name"),
        "source": "registry",
    }

    base_models = metadata_item.get("base_models") or []
    for index, base in enumerate(base_models):
        base_id = (
            base.get("artifact_id")
            or base.get("model_id")
            or base.get("id")
            or f"{artifact_id}-base-{index}"
        )
        if base_id not in nodes:
            nodes[base_id] = {
                "artifact_id": base_id,
                "name": base.get("name") or base.get("model_id") or base_id,
                "source": base.get("source") or "base_model",
                "metadata": {
                    k: v
                    for k, v in base.items()
                    if k not in {"artifact_id", "model_id", "name", "source"}
                }
                or None,
            }
        edges.append(
            {
                "from_node_artifact_id": base_id,
                "to_node_artifact_id": artifact_id,
                "relationship": base.get("relation") or "base_model",
            }
        )

    return {"nodes": list(nodes.values()), "edges": edges}


def _build_cost_payload(
    artifact_id: str, artifact_type: str, metadata_item: Dict[str, Any], include_dependency: bool
) -> Dict[str, Any]:
    bucket, key = _resolve_artifact_location(metadata_item, artifact_type, strict=True)
    base_cost = _object_size_mb(bucket, key)
    total_cost = base_cost
    entry: Dict[str, Any] = {"total_cost": round(total_cost, 4)}

    if include_dependency:
        entry["standalone_cost"] = round(base_cost, 4)
        for dependency_type in ("dataset", "code", "codebase"):
            dep_bucket, dep_key = _resolve_artifact_location(metadata_item, dependency_type)
            if dep_bucket and dep_key:
                total_cost += _object_size_mb(dep_bucket, dep_key)
        entry["total_cost"] = round(total_cost, 4)

    return {artifact_id: entry}


def _format_audit_entries(
    entries: List[Dict[str, Any]], artifact_type: str, artifact_id: str
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for entry in entries:
        user = entry.get("user") or {}
        if not user and entry.get("user_name"):
            user = {"name": entry.get("user_name"), "is_admin": entry.get("user_is_admin", False)}
        artifact = entry.get("artifact") or {
            "name": entry.get("artifact_name") or entry.get("name"),
            "id": entry.get("artifact_id") or artifact_id,
            "type": entry.get("artifact_type") or artifact_type,
        }
        formatted.append(
            {
                "user": user or {"name": "unknown", "is_admin": False},
                "date": entry.get("date") or entry.get("timestamp") or entry.get("created_at"),
                "artifact": artifact,
                "action": entry.get("action") or entry.get("event") or "AUDIT",
            }
        )
    return formatted


def lambda_handler(event: Dict[str, Any], context):  # noqa: D401
    method = (
        event.get("requestContext", {})
        .get("http", {})
        .get("method", event.get("httpMethod"))
    )
    path = event.get("rawPath") or event.get("path", "")
    print(
        f"[metadata.lambda] Received {method or 'UNKNOWN'} {path or '/'} body={event.get('body')}",
        flush=True,
    )
    segments = _parse_path(path)
    query = event.get("queryStringParameters") or {}

    if method == "GET" and len(segments) >= 3 and segments[0] == "artifacts":
        artifact_type, artifact_id = segments[1], segments[2]
        item = _get_artifact(artifact_type, artifact_id)
        if not item:
            return _not_found(artifact_type, artifact_id)
        try:
            payload = _build_artifact_envelope(artifact_type, item)
        except ValueError as exc:
            return _response(400, {"message": str(exc)})
        print(
            f"[metadata.lambda] Returning artifact envelope for {artifact_type}/{artifact_id}",
            flush=True,
        )
        return _response(200, payload)

    if len(segments) >= 4 and segments[0] == "artifact":
        artifact_type, artifact_id = segments[1], segments[2]
        action = segments[3]
        metadata_item = _get_artifact(artifact_type, artifact_id)

        if method == "GET" and action == "cost":
            if artifact_type not in {"model", "dataset", "code"}:
                return _response(
                    400,
                    {"message": f"Unsupported artifact type '{artifact_type}'."},
                )
            if not metadata_item:
                return _not_found(artifact_type, artifact_id)
            dependency_flag = _parse_bool(query.get("dependency"), False)
            try:
                payload = _build_cost_payload(
                    artifact_id, artifact_type, metadata_item, dependency_flag
                )
            except MissingDownloadLocation as exc:
                return _response(404, {"message": str(exc)})
            except ObjectSizeResolutionError as exc:
                return _response(500, {"message": str(exc)})
            print(
                f"[metadata.lambda] Returning cost info for {artifact_type}/{artifact_id} dependency={dependency_flag}",
                flush=True,
            )
            return _response(200, payload)

        if not metadata_item:
            return _not_found(artifact_type, artifact_id)

        if method == "GET" and action == "lineage":
            graph = _build_lineage_graph(artifact_id, artifact_type, metadata_item)
            print(
                f"[metadata.lambda] Returning lineage for {artifact_type}/{artifact_id}",
                flush=True,
            )
            return _response(200, graph)

        if method == "GET" and action == "audit":
            audits = metadata_item.get("audits")
            if not audits:
                audits = _get_audits(artifact_type, artifact_id)
            formatted = _format_audit_entries(audits, artifact_type, artifact_id)
            print(
                f"[metadata.lambda] Returning {len(formatted)} audit entries for {artifact_type}/{artifact_id}",
                flush=True,
            )
            return _response(200, formatted)

        if method == "POST" and action == "license-check" and artifact_type == "model":
            try:
                payload = _parse_json_body(event)
            except ValueError as exc:
                return _response(400, {"message": str(exc)})

            github_url = payload.get("github_url")
            if not github_url or "github.com" not in github_url:
                return _response(400, {"message": "Field 'github_url' must be a GitHub URL."})

            github_license = _fetch_github_license_identifier(github_url)
            #if not github_license:
            #    return _response(502, {"message": "External license information could not be retrieved."})

            evaluation = _evaluate_license_text(github_license)
            hugging_face = "huggingface.co" in str(metadata_item.get("model_url", "")).lower()
            compatible = bool(
                evaluation.get("fine_tune_allowed")
                and evaluation.get("inference_allowed")
                and hugging_face
            )
            metadata_license = (metadata_item.get("license") or "").lower()
            if metadata_license and metadata_license != github_license:
                compatible = False

            print("[license-check] github_url =", github_url)
            print("[license-check] github_license =", github_license)
            print("[license-check] metadata_license =", metadata_item.get("license"))
            print("[license-check] evaluation =", evaluation)
            print("[license-check] hugging_face =", hugging_face)
            print("[license-check] FINAL compatible =", compatible, flush=True)

            print(
                f"[metadata.lambda] license-check result compatible={compatible}",
                flush=True,
            )
            return _response(200, compatible)

    return _response(
        400,
        {"message": "Unsupported route", "method": method, "path": path},
    )


lambda_handler.__name__ = "lambda_handler"