"""Metadata Lambda consolidating artifact metadata + download endpoints."""
from __future__ import annotations

import base64
import json
import os
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

import boto3
from boto3.dynamodb.conditions import Key

ARTIFACTS_TABLE_NAME = os.environ.get("ARTIFACTS_DDB_TABLE", "ModelArtifacts")
COST_PER_GB = float(os.environ.get("ARTIFACT_COST_PER_GB", "0.12"))
DEFAULT_PRESIGN_TTL = int(os.environ.get("DOWNLOAD_PRESIGN_EXPIRATION", "3600"))
DEFAULT_MODEL_BUCKET = os.environ.get("MODEL_BUCKET_NAME")


def _dynamodb():
    return boto3.resource("dynamodb")


def _table():
    return _dynamodb().Table(ARTIFACTS_TABLE_NAME)


# Backwards-compat aliases for tests/legacy imports
def _get_table():  # pragma: no cover - shim for tests
    return _table()


def _s3_client():
    return boto3.client("s3")


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


def _sanitize_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(item)
    sanitized.pop("assets", None)
    sanitized.pop("s3_location_of_model_zip", None)
    sanitized.pop("s3_location_of_cloudwatch_log_for_database_entry", None)
    return sanitized


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


def _build_artifact_response(
    artifact_type: str,
    artifact_id: str,
    metadata: Dict[str, Any],
    *,
    include_logs: bool,
    inline: bool,
    presign_ttl: int,
) -> Dict[str, Any]:
    bucket, key = _resolve_artifact_location(metadata, artifact_type)
    if not bucket or not key:
        raise ValueError("Artifact does not have an associated object key")

    s3 = _s3_client()
    head = s3.head_object(Bucket=bucket, Key=key)
    size_bytes = head.get("ContentLength", 0)
    presigned = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=presign_ttl,
    )

    artifact_entry: Dict[str, Any] = {
        "type": artifact_type,
        "bucket": bucket,
        "key": key,
        "s3_url": f"s3://{bucket}/{key}",
        "presigned_url": presigned,
        "size_bytes": size_bytes,
    }

    if inline:
        obj = s3.get_object(Bucket=bucket, Key=key)
        artifact_entry["data"] = base64.b64encode(obj["Body"].read()).decode("utf-8")

    entries = [artifact_entry]

    if include_logs:
        log_bucket, log_key = _parse_s3_uri(
            metadata.get("assets", {}).get("log_key")
            or metadata.get("s3_location_of_cloudwatch_log_for_database_entry")
        )
        if log_bucket and log_key:
            log_presigned = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": log_bucket, "Key": log_key},
                ExpiresIn=presign_ttl,
            )
            entries.append(
                {
                    "type": "log",
                    "bucket": log_bucket,
                    "key": log_key,
                    "s3_url": f"s3://{log_bucket}/{log_key}",
                    "presigned_url": log_presigned,
                }
            )

    sanitized_metadata = _sanitize_metadata(metadata)
    return {
        "status": "success",
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "metadata": sanitized_metadata,
        "artifacts": entries,
        "total_artifact_size_bytes": sum(e.get("size_bytes", 0) for e in entries),
    }


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
            payload = _build_artifact_response(
                artifact_type,
                artifact_id,
                item,
                include_logs=_parse_bool(query.get("include_logs"), True),
                inline=_parse_bool(query.get("inline"), False),
                presign_ttl=int(query.get("presign_ttl", DEFAULT_PRESIGN_TTL)),
            )
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
            return _response(
                200,
                {"model_id": artifact_id, "base_models": metadata_item.get("base_models", [])},
            )

        if method == "GET" and action == "cost":
            cost = _calculate_cost(metadata_item)
            return _response(
                200,
                {
                    "model_id": artifact_id,
                    "artifact_type": artifact_type,
                    "estimated_cost_usd": cost,
                    "unit_cost_per_gb": COST_PER_GB,
                },
            )

        if method == "GET" and action == "audit":
            audits = _get_audits(artifact_type, artifact_id)
            return _response(200, {"entries": audits})

        if method == "POST" and action == "license-check" and artifact_type == "model":
            evaluation = _evaluate_license(metadata_item)
            return _response(200, {"model_id": artifact_id, "result": evaluation})

    return _response(
        400,
        {"message": "Unsupported route", "method": method, "path": path},
    )


lambda_handler.__name__ = "lambda_handler"
