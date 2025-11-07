import json
import os
from decimal import Decimal
from typing import Any, Dict, Tuple
from urllib.parse import unquote

import boto3
from boto3.dynamodb.conditions import Key


ARTIFACTS_TABLE_NAME = os.environ.get("ARTIFACTS_DDB_TABLE", "ModelArtifacts")
COST_PER_GB = float(os.environ.get("ARTIFACT_COST_PER_GB", "0.12"))


def _dynamodb_resource():
    return boto3.resource("dynamodb")


def _get_table():
    return _dynamodb_resource().Table(ARTIFACTS_TABLE_NAME)


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


def _get_artifact(artifact_type: str, artifact_id: str):
    table = _get_table()
    pk = f"{artifact_type.upper()}#{artifact_id}"
    result = table.get_item(Key={"pk": pk, "sk": "META"})
    item = result.get("Item")
    return _convert(item) if item else None


def _get_audits(artifact_type: str, artifact_id: str):
    table = _get_table()
    pk = f"{artifact_type.upper()}#{artifact_id}"
    result = table.query(
        KeyConditionExpression=Key("pk").eq(pk) & Key("sk").begins_with("AUDIT#"),
        ScanIndexForward=False,
    )
    return _convert(result.get("Items", []))


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


def lambda_handler(event: Dict[str, Any], context):
    method = (
        event.get("requestContext", {})
        .get("http", {})
        .get("method", event.get("httpMethod"))
    )
    path = event.get("rawPath") or event.get("path", "")
    segments = _parse_path(path)

    if method == "GET" and len(segments) >= 3 and segments[0] == "artifacts":
        artifact_type, artifact_id = segments[1], segments[2]
        item = _get_artifact(artifact_type, artifact_id)
        if not item:
            return _not_found(artifact_type, artifact_id)
        return _response(200, item)

    if len(segments) >= 4 and segments[0] == "artifact":
        artifact_type, artifact_id = segments[1], segments[2]
        action = segments[3]
        metadata = _get_artifact(artifact_type, artifact_id)
        if not metadata:
            return _not_found(artifact_type, artifact_id)

        if method == "GET" and action == "lineage":
            return _response(200, {"model_id": artifact_id, "base_models": metadata.get("base_models", [])})

        if method == "GET" and action == "cost":
            cost = _calculate_cost(metadata)
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
            evaluation = _evaluate_license(metadata)
            return _response(
                200,
                {
                    "model_id": artifact_id,
                    "result": evaluation,
                },
            )

    return _response(
        400,
        {
            "message": "Unsupported route",
            "method": method,
            "path": path,
        },
    )


lambda_handler.__name__ = "lambda_handler"
