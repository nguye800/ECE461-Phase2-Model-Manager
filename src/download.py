"""
Lambda function implementation for retrieving model metadata and artifacts.

This handler reads model records stored in an Aurora Serverless database via
the RDS Data API and returns structured metadata alongside optional presigned
S3 URLs for any related artifacts. The functionality complements the upload
handler by exposing model registry entries to downstream consumers.

Environment variables:
    MODEL_REGISTRY_CLUSTER_ARN      (required)
    MODEL_REGISTRY_SECRET_ARN       (required)
    MODEL_REGISTRY_DATABASE         (required)
    MODEL_REGISTRY_TABLE            (default: model_registry)
    DOWNLOAD_PRESIGN_EXPIRATION     (default: 3600 seconds)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from boto3.session import Session
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_S3_CLIENT: Optional[boto3.client] = None
_RDS_CLIENT: Optional[boto3.client] = None


class DownloadError(Exception):
    """Raised for validation and logical errors during the download process."""


@dataclass
class DownloadRequest:
    """Represents the inbound payload for a download request."""

    model_id: str
    include_logs: bool = True
    generate_presigned_urls: bool = True
    presign_ttl_seconds: int = 3600
    delivery_mode: str = "presigned"
    include_size_details: bool = True

    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> "DownloadRequest":
        """
        Parse the incoming event which may come from API Gateway proxy or direct invocation.
        """
        if "body" in event:
            body = event["body"]
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body)
            payload = json.loads(body)
        else:
            payload = event

        LOGGER.debug("Download event payload: %s", payload)

        try:
            model_id = payload["model_id"]
        except KeyError as exc:
            raise DownloadError("Request must include 'model_id'.") from exc

        include_logs = bool(payload.get("include_logs", True))
        generate_presigned = bool(payload.get("presign", True))
        ttl_default = int(_require_env("DOWNLOAD_PRESIGN_EXPIRATION", optional=True, default="3600"))
        delivery_mode = payload.get("delivery_mode", "presigned")
        if delivery_mode not in {"presigned", "inline"}:
            raise DownloadError("delivery_mode must be either 'presigned' or 'inline'.")

        include_size_details = bool(payload.get("include_size", True))

        if delivery_mode == "inline" and "presign" not in payload:
            # Inline delivery implies the API will return the object bytes directly,
            # so presigned URLs are disabled unless explicitly requested.
            generate_presigned = False

        if payload.get("presign_ttl") is not None:
            try:
                ttl = int(payload["presign_ttl"])
            except ValueError as exc:
                raise DownloadError("'presign_ttl' must be an integer number of seconds.") from exc
        else:
            ttl = ttl_default

        if ttl <= 0:
            raise DownloadError("Presign TTL must be positive.")

        return cls(
            model_id=model_id,
            include_logs=include_logs,
            generate_presigned_urls=generate_presigned,
            presign_ttl_seconds=ttl,
            delivery_mode=delivery_mode,
            include_size_details=include_size_details,
        )


@dataclass
class RDSConfig:
    cluster_arn: str
    secret_arn: str
    database: str
    table_name: str = "model_registry"


def _require_env(name: str, *, optional: bool = False, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if not value and not optional:
        raise DownloadError(f"Environment variable '{name}' is required.")
    return value  # type: ignore[return-value]


def _get_s3_client() -> boto3.client:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        session: Session = boto3.session.Session()
        _S3_CLIENT = session.client("s3")
    return _S3_CLIENT


def _get_rds_client() -> boto3.client:
    global _RDS_CLIENT
    if _RDS_CLIENT is None:
        session: Session = boto3.session.Session()
        _RDS_CLIENT = session.client("rds-data")
    return _RDS_CLIENT


def _load_rds_config() -> RDSConfig:
    return RDSConfig(
        cluster_arn=_require_env("MODEL_REGISTRY_CLUSTER_ARN"),
        secret_arn=_require_env("MODEL_REGISTRY_SECRET_ARN"),
        database=_require_env("MODEL_REGISTRY_DATABASE"),
        table_name=_require_env("MODEL_REGISTRY_TABLE", optional=True, default="model_registry"),
    )


def _decode_field(value: Dict[str, Any]) -> Any:
    """Translate an RDS Data API field result into a native Python value."""
    if value.get("isNull"):
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "longValue" in value:
        return value["longValue"]
    if "booleanValue" in value:
        return value["booleanValue"]
    if "blobValue" in value:
        return base64.b64decode(value["blobValue"])
    return None


def _fetch_model_record(rds_config: RDSConfig, model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a model registry entry by ID."""
    rds_client = _get_rds_client()
    sql = (
        f"SELECT * FROM {rds_config.table_name} "
        "WHERE repo_id = :model_id LIMIT 1"
    )
    response = rds_client.execute_statement(
        resourceArn=rds_config.cluster_arn,
        secretArn=rds_config.secret_arn,
        database=rds_config.database,
        sql=sql,
        parameters=[{"name": "model_id", "value": {"stringValue": model_id}}],
        includeResultMetadata=True,
    )

    records = response.get("records", [])
    metadata = response.get("columnMetadata", [])
    if not records:
        return None

    row = records[0]
    record: Dict[str, Any] = {}
    for column, field in zip(metadata, row):
        name = column.get("name")
        if name:
            record[name] = _decode_field(field)
    return record


def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """Split an S3 URL (s3://bucket/key) into bucket and key components."""
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise DownloadError(f"Invalid S3 URL: '{s3_url}'")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise DownloadError(f"Invalid S3 URL missing key: '{s3_url}'")
    return bucket, key


def _build_artifact_entries(
    record: Dict[str, Any],
    request: DownloadRequest,
    s3_client: Optional[boto3.client],
) -> List[Dict[str, Any]]:
    """Construct artifact descriptors from the database record."""
    artifacts: List[Dict[str, Any]] = []
    artifact_fields = [
        ("model", record.get("s3_location_of_model_zip")),
        ("cloudwatch_log", record.get("s3_location_of_cloudwatch_log_for_database_entry")),
    ]

    for artifact_type, url in artifact_fields:
        if url is None:
            continue
        if artifact_type == "cloudwatch_log" and not request.include_logs:
            continue

        try:
            bucket, key = _parse_s3_url(url)
        except DownloadError:
            LOGGER.warning("Skipping artifact with invalid URL: %s", url)
            continue

        presigned_url = None
        size_bytes = None
        inline_data_b64 = None

        if request.include_size_details and s3_client is not None:
            try:
                head = s3_client.head_object(Bucket=bucket, Key=key)
                size_bytes = head.get("ContentLength")
            except (BotoCoreError, ClientError) as err:
                LOGGER.error("Failed to retrieve size for %s/%s: %s", bucket, key, err)

        if request.delivery_mode == "inline" and s3_client is not None:
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                data_bytes = obj["Body"].read()
                inline_data_b64 = base64.b64encode(data_bytes).decode("utf-8")
                # If size wasn't found from head_object, derive from payload
                if size_bytes is None:
                    size_bytes = len(data_bytes)
            except (BotoCoreError, ClientError) as err:
                LOGGER.error("Failed to download object %s/%s: %s", bucket, key, err)

        if request.generate_presigned_urls and s3_client is not None:
            try:
                presigned_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=request.presign_ttl_seconds,
                )
            except (BotoCoreError, ClientError) as err:
                LOGGER.error("Failed to generate presigned URL for %s/%s: %s", bucket, key, err)

        artifact_entry = {
            "type": artifact_type,
            "bucket": bucket,
            "key": key,
            "s3_url": url,
            "presigned_url": presigned_url,
        }
        if size_bytes is not None:
            artifact_entry["size_bytes"] = size_bytes
        if inline_data_b64 is not None:
            artifact_entry["data"] = inline_data_b64

        artifacts.append(artifact_entry)
    return artifacts


def _format_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handle_download(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler that retrieves model metadata and returns presigned S3 URLs.
    """
    try:
        request = DownloadRequest.from_event(event)
        rds_config = _load_rds_config()

        record = _fetch_model_record(rds_config, request.model_id)
        if record is None:
            LOGGER.info("Model %s not found in registry", request.model_id)
            return _format_response(
                404,
                {
                    "status": "error",
                    "message": f"Model '{request.model_id}' not found.",
                },
            )

        needs_s3_client = (
            request.generate_presigned_urls
            or request.include_size_details
            or request.delivery_mode == "inline"
        )
        s3_client = _get_s3_client() if needs_s3_client else None

        artifacts = _build_artifact_entries(record, request, s3_client)
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {
                "s3_location_of_model_zip",
                "s3_location_of_cloudwatch_log_for_database_entry",
            }
        }

        total_size = sum(
            artifact.get("size_bytes", 0) for artifact in artifacts if artifact.get("size_bytes") is not None
        )

        response_body = {
            "status": "success",
            "model_id": request.model_id,
            "metadata": metadata,
            "artifacts": artifacts,
        }
        if request.include_size_details:
            response_body["total_artifact_size_bytes"] = total_size

        return _format_response(200, response_body)

    except DownloadError as err:
        LOGGER.error("Download error: %s", err, exc_info=True)
        return _format_response(400, {"status": "error", "message": str(err)})

    except (BotoCoreError, ClientError) as err:
        LOGGER.error("AWS service error during download: %s", err, exc_info=True)
        return _format_response(502, {"status": "error", "message": "AWS service failure"})

    except Exception as err:  # pragma: no cover - defensive catch-all
        LOGGER.exception("Unexpected error during download")
        return _format_response(500, {"status": "error", "message": str(err)})


lambda_handler = handle_download
