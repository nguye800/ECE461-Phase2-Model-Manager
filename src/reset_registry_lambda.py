import os
import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError, BotoCoreError

dynamodb = boto3.resource("dynamodb")
s3_client = boto3.client("s3")
TABLE_NAME = os.environ.get("ARTIFACTS_TABLE_NAME", "model-metadata")
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME", "modelzip-logs-artifacts")


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


def _clear_bucket(bucket_name: str) -> int:
    deleted = 0
    continuation_token = None

    while True:
        list_kwargs = {"Bucket": bucket_name, "MaxKeys": 1000}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)
        contents = response.get("Contents", [])
        if not contents:
            break

        delete_payload = {
            "Objects": [{"Key": obj["Key"]} for obj in contents],
            "Quiet": True,
        }
        s3_client.delete_objects(Bucket=bucket_name, Delete=delete_payload)
        deleted += len(contents)

        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

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
        500: Internal errors during reset.
    """
    method = event.get("httpMethod")
    path = event.get("path") or event.get("rawPath")
    print(
        f"[reset.lambda] Received {method or 'UNKNOWN'} {path or '/'} body={event.get('body')}",
        flush=True,
    )

    try:
        deleted_count = _clear_table(TABLE_NAME)
        print(f"[reset.lambda] Cleared table entries={deleted_count}", flush=True)
    except ClientError as e:
        print(f"[reset.lambda] DynamoDB error resetting registry: {e}", flush=True)
        return _build_response(
            500,
            {"error": "Internal server error while resetting registry.",
             "details": str(e)},
        )

    s3_deleted = None
    if MODEL_BUCKET_NAME:
        try:
            s3_deleted = _clear_bucket(MODEL_BUCKET_NAME)
            print(
                f"[reset.lambda] Cleared {s3_deleted} objects from bucket {MODEL_BUCKET_NAME}",
                flush=True,
            )
        except (BotoCoreError, ClientError) as e:
            print(f"[reset.lambda] S3 error while clearing bucket: {e}", flush=True)
            return _build_response(
                500,
                {"error": "Internal server error while clearing S3 artifacts.",
                 "details": str(e)},
            )
    else:
        print("[reset.lambda] MODEL_BUCKET_NAME not set; skipping S3 cleanup", flush=True)

    print("[reset.lambda] Reset completed successfully", flush=True)
    return _build_response(
        200,
        {
            "message": "Registry is reset.",
            "deleted_items": deleted_count,
            "s3_objects_deleted": s3_deleted,
        },
    )
