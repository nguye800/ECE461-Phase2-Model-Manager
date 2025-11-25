import os
import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.environ.get("ARTIFACTS_TABLE_NAME", "model-metadata")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")  # same shared secret as other endpoints


def _build_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(body),
    }


def _get_header(headers: Dict[str, str], key: str) -> str | None:
    if not headers:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    return lower.get(key.lower())


def _clear_table(table_name: str) -> int:
    table = dynamodb.Table(table_name)
    deleted = 0

    scan_kwargs = {
        "ProjectionExpression": "#pk, #sk",
        "ExpressionAttributeNames": {
            "#pk": "pk",
            "#sk": "sk",
        },
    }

    done = False
    start_key = None

    while not done:
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key

        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        if not items:
            break

        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={"pk": item["pk"], "sk": item["sk"]}
                )
                deleted += 1

        start_key = response.get("LastEvaluatedKey")
        done = start_key is None

    return deleted

def handler(event, context):
    """
    Lambda handler for DELETE /reset

    OpenAPI:
      operationId: RegistryReset
      summary: Reset the registry. (BASELINE)
      description: Reset the registry to a system default state.
      responses:
        200: Registry is reset.
        401: Not used here; we’ll fold into 403.
        403: Authentication failed.
    """
    token = _get_header(event.get("headers", {}) or {}, "X-Authorization")
    if not token or (AUTH_TOKEN is not None and token != AUTH_TOKEN):
        return _build_response(
            403,
            {"error": "Authentication failed due to invalid or missing AuthenticationToken."},
        )

    try:
        deleted_count = _clear_table(TABLE_NAME)

        # OPTIONAL: re-seed default artifacts here if your project spec requires it
        # e.g. read a JSON from S3 specified by SEED_S3_URI and write items into the table.

    except ClientError as e:
        return _build_response(
            500,
            {"error": "Internal server error while resetting registry.",
             "details": str(e)},
        )

    return _build_response(
        200,
        {
            "message": "Registry is reset.",
            "deleted_items": deleted_count,
        },
    )
