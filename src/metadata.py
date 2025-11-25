"""Metadata Lambda consolidating artifact metadata + download endpoints."""
from __future__ import annotations

import base64
import json
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import boto3
from boto3.dynamodb.conditions import Key

ARTIFACTS_TABLE_NAME = os.environ.get("ARTIFACTS_DDB_TABLE", "ModelArtifacts")
ARTIFACTS_DDB_REGION = os.environ.get("ARTIFACTS_DDB_REGION")
COST_PER_GB = float(os.environ.get("ARTIFACT_COST_PER_GB", "0.12"))
DEFAULT_MODEL_BUCKET = os.environ.get("MODEL_BUCKET_NAME")


def _dynamodb():
    if ARTIFACTS_DDB_REGION:
        return boto3.resource("dynamodb", region_name=ARTIFACTS_DDB_REGION)
    return boto3.resource("dynamodb")


def _table():
    return _dynamodb().Table(ARTIFACTS_TABLE_NAME)


# Backwards-compat aliases for tests/legacy imports
def _get_table():  # pragma: no cover - shim for tests
    return _table()


def _response(status: int, body: Dict[str, Any]):
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


def _calculate_cost(metadata: Dict[str, Any]):
    size_gb = float(metadata.get("gb_size_of_model") or 0)
    return round(size_gb * COST_PER_GB, 4)


def _evaluate_license(metadata: Dict[str, Any]):
    license_text = (metadata.get("license") or "").lower()
    permissive = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}
    restricted = {"gpl-3.0", "agpl-3.0"}

    if not license_text:
        return {
            "fine_tune_allowed": False,
            "inference_allowed": False,
            "reason": "No license information available",
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


def _resolve_artifact_location(metadata: Dict[str, Any], artifact_type: str):
    assets = metadata.get("assets") or {}
    if artifact_type == "model":
        key = assets.get("model_zip_key") or metadata.get("s3_location_of_model_zip")
    else:
        key = assets.get(f"{artifact_type}_key")
    bucket = assets.get("bucket")
    b, k = _parse_s3_uri(key)
    bucket = bucket or b
    return bucket, k


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
    cost = _calculate_cost(metadata_item)
    entry: Dict[str, Any] = {"total_cost": cost}
    if include_dependency:
        entry["standalone_cost"] = cost
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
        return _response(200, payload)

    if len(segments) >= 4 and segments[0] == "artifact":
        artifact_type, artifact_id = segments[1], segments[2]
        action = segments[3]
        metadata_item = _get_artifact(artifact_type, artifact_id)
        if not metadata_item:
            return _not_found(artifact_type, artifact_id)

        if method == "GET" and action == "lineage":
            graph = _build_lineage_graph(artifact_id, artifact_type, metadata_item)
            return _response(200, graph)

        if method == "GET" and action == "cost":
            dependency_flag = _parse_bool(query.get("dependency"), False)
            payload = _build_cost_payload(
                artifact_id, artifact_type, metadata_item, dependency_flag
            )
            return _response(200, payload)

        if method == "GET" and action == "audit":
            audits = metadata_item.get("audits")
            if not audits:
                audits = _get_audits(artifact_type, artifact_id)
            formatted = _format_audit_entries(audits, artifact_type, artifact_id)
            return _response(200, formatted)

        if method == "POST" and action == "license-check" and artifact_type == "model":
            try:
                payload = _parse_json_body(event)
            except ValueError as exc:
                return _response(400, {"message": str(exc)})

            github_url = payload.get("github_url")
            if not github_url or "github.com" not in github_url:
                return _response(400, {"message": "Field 'github_url' must be a GitHub URL."})

            evaluation = _evaluate_license(metadata_item)
            hugging_face = "huggingface.co" in str(metadata_item.get("model_url", "")).lower()
            compatible = bool(
                evaluation.get("fine_tune_allowed")
                and evaluation.get("inference_allowed")
                and hugging_face
            )
            return _response(200, compatible)

    return _response(
        400,
        {"message": "Unsupported route", "method": method, "path": path},
    )


lambda_handler.__name__ = "lambda_handler"
