import os
import re
import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.environ.get("ARTIFACTS_TABLE_NAME", "model-metadata")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")

ARTIFACT_TYPE_VALUES = {"model", "dataset", "code"}
ARTIFACT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-]+$")


def _build_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _get_header(headers: Dict[str, str], key: str) -> str | None:
    if not headers:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    return lower.get(key.lower())


def handler(event, context):
    """
    DELETE /artifacts/{artifact_type}/{id}

    Maps to DynamoDB:
      pk = ARTIFACTTYPE_UPPER + "#" + id
      sk = any (we delete all rows with that pk)
    """
    # --- Auth ---
    method = event.get("httpMethod")
    path = event.get("path") or event.get("rawPath")
    print(
        f"[delete.lambda] Received {method or 'UNKNOWN'} {path or '/'} body={event.get('body')}",
        flush=True,
    )
    token = _get_header(event.get("headers", {}) or {}, "X-Authorization")
    if not token or (AUTH_TOKEN is not None and token != AUTH_TOKEN):
        print("[delete.lambda] Authentication failed", flush=True)
        return _build_response(
            403,
            {"error": "Authentication failed due to invalid or missing AuthenticationToken."},
        )

    path_params = event.get("pathParameters") or {}
    artifact_type = path_params.get("artifact_type")
    artifact_id = path_params.get("id")

    # --- Validate inputs ---
    if artifact_type not in ARTIFACT_TYPE_VALUES:
        return _build_response(
            400,
            {"error": "Invalid artifact_type.", "allowed": sorted(list(ARTIFACT_TYPE_VALUES))},
        )

    if not artifact_id or not ARTIFACT_ID_PATTERN.match(artifact_id):
        return _build_response(
            400,
            {"error": "Invalid artifact id. Must match ^[a-zA-Z0-9\\-]+$."},
        )

    table = dynamodb.Table(TABLE_NAME)

    # Build the pk used in your table, e.g. "MODEL#01JBM6A4F9..."
    pk_value = f"{artifact_type.upper()}#{artifact_id}"

    try:
        # 1) Query all rows for this pk (in case there are multiple sk's)
        query_resp = table.query(
            KeyConditionExpression=Key("pk").eq(pk_value)
        )
        items = query_resp.get("Items", [])

        if not items:
            # Nothing with that pk → 404
            print(f"[delete.lambda] Artifact not found for pk={pk_value}", flush=True)
            return _build_response(
                404,
                {"error": "Artifact does not exist.", "artifact_type": artifact_type, "id": artifact_id},
            )

        # 2) Delete all rows with this pk
        deleted = 0
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={"pk": item["pk"], "sk": item["sk"]}
                )
                deleted += 1

        print(f"[delete.lambda] Deleting pk={pk_value} items={len(items)}", flush=True)
        return _build_response(
            200,
            {
                "message": "Artifact is deleted.",
                "artifact_type": artifact_type,
                "id": artifact_id,
                "deleted_items": deleted,
            },
        )

    except ClientError as e:
        print(f"[delete.lambda] DynamoDB error: {e}", flush=True)
        return _build_response(
            500,
            {"error": "Internal server error while deleting artifact.",
             "details": str(e)},
        )
