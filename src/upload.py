"""
Lambda function implementation for handling model uploads.

This module converts the previous pseudo-code into executable logic. The handler
accepts an event describing the model metadata and artifacts, uploads the
artifacts to S3, and records/updates an entry in the model registry stored in
RDS via the Data API.

The implementation intentionally avoids tight coupling to a single RDS dialect.
It first checks whether a record already exists and then issues either an
INSERT or UPDATE statement accordingly. The SQL statements mirror the schema
defined in ``src/dummydb.py``; missing values default to ``NULL``.

Environment variables used:
    MODEL_BUCKET_NAME: S3 bucket where artifacts will be stored (required)
    MODEL_REGISTRY_CLUSTER_ARN: ARN of the Aurora Serverless cluster (required)
    MODEL_REGISTRY_SECRET_ARN: ARN of the Secrets Manager secret for the DB (required)
    MODEL_REGISTRY_DATABASE: Database name inside the cluster (required)
    MODEL_REGISTRY_TABLE: Optional override for the registry table name
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
import zipfile

import boto3
from boto3.session import Session
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_S3_CLIENT: Optional[boto3.client] = None
_RDS_CLIENT: Optional[boto3.client] = None


class UploadError(Exception):
    """Custom error for failures within the upload pipeline."""


@dataclass
class ArtifactDescriptor:
    """Describes a single artifact to be uploaded."""

    name: str
    body: Optional[str] = None  # base64 encoded string
    content_type: str = "application/octet-stream"
    source_bucket: Optional[str] = None
    source_key: Optional[str] = None
    target_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_valid(self) -> None:
        """Verify that enough information exists to upload or copy the object."""
        if self.body is not None:
            return

        if self.source_bucket and self.source_key:
            return

        raise UploadError(
            f"Artifact '{self.name}' must include either 'body' or "
            "'source_bucket'/'source_key'."
        )

    def resolve_target_key(self, prefix: str) -> str:
        """Determine the S3 key to use for the artifact."""
        key = self.target_key or f"{prefix}/{self.name}"
        # Remove leading slash to avoid accidental absolute keys
        return key.lstrip("/")


@dataclass
class UploadRequest:
    """Structured representation of the inbound upload payload."""

    model_id: str
    model_url: str
    metadata: Dict[str, Any]
    artifacts: List[ArtifactDescriptor]
    dataset_link: Optional[str] = None
    dataset_name: Optional[str] = None
    code_repository: Optional[str] = None

    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> "UploadRequest":
        """
        Parse the Lambda event (supports direct invocation or API Gateway proxy).
        """
        LOGGER.debug("Raw event received: %s", event)

        if "body" in event:
            body = event["body"]
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body)
            payload = json.loads(body)
        else:
            payload = event

        try:
            model_id = payload["model_id"]
            model_url = payload["model_url"]
        except KeyError as exc:
            raise UploadError(f"Missing required field '{exc.args[0]}' in payload") from exc

        metadata = payload.get("metadata") or {}
        artifacts_payload = payload.get("artifacts")
        if not artifacts_payload:
            raise UploadError("Payload must include at least one artifact")

        artifacts: List[ArtifactDescriptor] = []
        for artifact in artifacts_payload:
            descriptor = ArtifactDescriptor(
                name=artifact["name"],
                body=artifact.get("body"),
                content_type=artifact.get("content_type", "application/octet-stream"),
                source_bucket=artifact.get("source_bucket"),
                source_key=artifact.get("source_key"),
                target_key=artifact.get("target_key"),
                metadata=artifact.get("metadata") or {},
            )
            descriptor.ensure_valid()
            artifacts.append(descriptor)

        return cls(
            model_id=model_id,
            model_url=model_url,
            metadata=metadata,
            artifacts=artifacts,
            dataset_link=payload.get("dataset_link"),
            dataset_name=payload.get("dataset_name"),
            code_repository=payload.get("code_repository"),
        )


@dataclass
class RDSConfig:
    """Configuration for accessing the model registry within RDS."""

    cluster_arn: str
    secret_arn: str
    database: str
    table_name: str = "model_registry"


def _require_env(name: str, *, optional: bool = False, default: Optional[str] = None) -> str:
    """Fetch an environment variable or raise if missing."""
    value = os.getenv(name, default)
    if not value and not optional:
        raise UploadError(f"Environment variable '{name}' is required for the upload Lambda.")
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


def _upload_artifacts(request: UploadRequest, bucket: str) -> List[Dict[str, Any]]:
    """
    Upload or copy artifacts into S3, returning metadata for each uploaded object.
    """
    s3_client = _get_s3_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"models/{request.model_id}/{timestamp}"
    results: List[Dict[str, Any]] = []

    for artifact in request.artifacts:
        target_key = artifact.resolve_target_key(prefix)
        extracted_metadata: Dict[str, Any] = {}

        try:
            if artifact.body is not None:
                LOGGER.info("Uploading artifact '%s' to s3://%s/%s", artifact.name, bucket, target_key)
                binary_body = base64.b64decode(artifact.body)
                extracted_metadata = _extract_zip_metadata(artifact, binary_body)
                response = s3_client.put_object(
                    Bucket=bucket,
                    Key=target_key,
                    Body=binary_body,
                    ContentType=artifact.content_type,
                    Metadata={
                        **{k: str(v) for k, v in artifact.metadata.items()},
                        "model_id": request.model_id,
                    },
                )
                version_id = response.get("VersionId")
            else:
                copy_source = {"Bucket": artifact.source_bucket, "Key": artifact.source_key}
                LOGGER.info(
                    "Copying artifact '%s' from s3://%s/%s to s3://%s/%s",
                    artifact.name,
                    artifact.source_bucket,
                    artifact.source_key,
                    bucket,
                    target_key,
                )
                response = s3_client.copy_object(
                    Bucket=bucket,
                    Key=target_key,
                    CopySource=copy_source,
                    MetadataDirective="REPLACE",
                    ContentType=artifact.content_type,
                    Metadata={
                        **{k: str(v) for k, v in artifact.metadata.items()},
                        "model_id": request.model_id,
                    },
                )
                version_id = response.get("VersionId")

            results.append(
                {
                    "artifact_name": artifact.name,
                    "bucket": bucket,
                    "key": target_key,
                    "version_id": version_id,
                    "size_bytes": len(binary_body) if artifact.body is not None else None,
                }
            )
            _apply_extracted_metadata(request, extracted_metadata)
        except (BotoCoreError, ClientError) as err:
            raise UploadError(f"Failed to upload artifact '{artifact.name}': {err}") from err
        except Exception as err:
            LOGGER.warning(
                "Unexpected issue while processing artifact '%s': %s", artifact.name, err
            )

    return results


def _extract_first_url(text: str) -> Optional[str]:
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else None


def _extract_zip_metadata(artifact: ArtifactDescriptor, data: bytes) -> Dict[str, Any]:
    """
    Attempt to gather metadata from a ZIP archive containing the model payload.
    """
    info: Dict[str, Any] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            info["zip_file_count"] = len(members)
            info["total_uncompressed_bytes"] = sum(member.file_size for member in members)
            info["zip_sample_listing"] = [member.filename for member in members[:20]]

            # Attempt to read config.json for model metadata
            for candidate in ("config.json", "model_index.json"):
                if candidate in archive.namelist():
                    with archive.open(candidate) as config_file:
                        try:
                            config_data = json.load(config_file)
                            info["config_model_name"] = (
                                config_data.get("model_name")
                                or config_data.get("_name_or_path")
                                or config_data.get("architectures", [None])[0]
                            )
                            base_model = (
                                config_data.get("base_model")
                                or config_data.get("parent_model")
                                or config_data.get("model_type")
                            )
                            if base_model:
                                info["base_model"] = base_model
                            param_count = (
                                config_data.get("num_parameters")
                                or config_data.get("model_size")
                            )
                            if param_count:
                                info["parameter_number"] = param_count
                        except json.JSONDecodeError:
                            LOGGER.debug("Config file in %s is not valid JSON", artifact.name)
                    break

            # Examine README for dataset references
            readme_name = next(
                (name for name in archive.namelist() if name.lower().startswith("readme")),
                None,
            )
            if readme_name:
                with archive.open(readme_name) as readme_file:
                    readme_text = readme_file.read().decode("utf-8", errors="ignore")
                    dataset_link = _extract_first_url(readme_text)
                    if dataset_link:
                        info["dataset_link"] = dataset_link
                        info["dataset_name"] = dataset_link.rstrip("/").split("/")[-1]

    except zipfile.BadZipFile:
        LOGGER.debug("Artifact '%s' is not a ZIP archive or is corrupted.", artifact.name)
    except Exception as err:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to extract metadata from artifact '%s': %s", artifact.name, err)

    return info


def _apply_extracted_metadata(request: UploadRequest, extracted: Dict[str, Any]) -> None:
    """
    Merge automatically derived metadata into the request metadata so it can be
    persisted in the registry.
    """
    if not extracted:
        return

    metadata = request.metadata

    total_bytes = extracted.get("total_uncompressed_bytes")
    if total_bytes and "gb_size_of_model" not in metadata:
        metadata["gb_size_of_model"] = round(total_bytes / (1024**3), 4)

    if extracted.get("parameter_number") and "parameter_number" not in metadata:
        metadata["parameter_number"] = extracted["parameter_number"]

    if extracted.get("base_model") and "base_models_modelID" not in metadata:
        metadata["base_models_modelID"] = extracted["base_model"]

    if not request.dataset_link and extracted.get("dataset_link"):
        request.dataset_link = extracted["dataset_link"]

    if not request.dataset_name and extracted.get("dataset_name"):
        request.dataset_name = extracted["dataset_name"]

    if extracted.get("config_model_name") and "parsed_model_name" not in metadata:
        metadata["parsed_model_name"] = extracted["config_model_name"]

    if extracted.get("zip_sample_listing") and "zip_sample_listing" not in metadata:
        metadata["zip_sample_listing"] = extracted["zip_sample_listing"]


def _record_exists(rds_config: RDSConfig, model_id: str) -> bool:
    """Check whether a model record already exists."""
    rds_client = _get_rds_client()
    sql = (
        f"SELECT repo_id FROM {rds_config.table_name} "
        "WHERE repo_id = :model_id LIMIT 1"
    )
    response = rds_client.execute_statement(
        resourceArn=rds_config.cluster_arn,
        secretArn=rds_config.secret_arn,
        database=rds_config.database,
        sql=sql,
        parameters=[{"name": "model_id", "value": {"stringValue": model_id}}],
    )
    return bool(response.get("records"))


def _build_db_parameters(
    request: UploadRequest,
    uploaded_artifacts: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Map the upload request and S3 results to DB column values."""
    metadata = request.metadata or {}
    artifacts = list(uploaded_artifacts)
    model_artifact = artifacts[0] if artifacts else None
    log_artifact = next(
        (artifact for artifact in artifacts if "log" in artifact["artifact_name"].lower()),
        None,
    )

    if metadata.get("gb_size_of_model") is None and model_artifact and model_artifact.get("size_bytes"):
        metadata["gb_size_of_model"] = round(model_artifact["size_bytes"] / (1024**3), 4)

    return {
        "repo_id": request.model_id,
        "model_url": request.model_url,
        "date_time_entered_to_db": datetime.now(timezone.utc).isoformat(),
        "likes": metadata.get("likes"),
        "downloads": metadata.get("downloads"),
        "license": metadata.get("license"),
        "github_link": request.code_repository or metadata.get("github_link"),
        "github_numContributors": metadata.get("github_numContributors"),
        "base_models_modelID": metadata.get("base_models_modelID"),
        "base_model_urls": metadata.get("base_model_urls"),
        "parameter_number": metadata.get("parameter_number"),
        "gb_size_of_model": metadata.get("gb_size_of_model"),
        "dataset_link": request.dataset_link or metadata.get("dataset_link"),
        "dataset_name": request.dataset_name or metadata.get("dataset_name"),
        "s3_location_of_model_zip": (
            f"s3://{model_artifact['bucket']}/{model_artifact['key']}" if model_artifact else None
        ),
        "s3_location_of_cloudwatch_log_for_database_entry": (
            f"s3://{log_artifact['bucket']}/{log_artifact['key']}" if log_artifact else None
        ),
    }


def _insert_model_record(rds_config: RDSConfig, params: Dict[str, Any]) -> None:
    """Insert a new model record into the registry."""
    rds_client = _get_rds_client()
    columns = ", ".join(params.keys())
    placeholders = ", ".join(f":{name}" for name in params.keys())
    sql = f"INSERT INTO {rds_config.table_name} ({columns}) VALUES ({placeholders})"

    rds_client.execute_statement(
        resourceArn=rds_config.cluster_arn,
        secretArn=rds_config.secret_arn,
        database=rds_config.database,
        sql=sql,
        parameters=[
            {"name": key, "value": _to_rds_value(value)}
            for key, value in params.items()
        ],
    )


def _update_model_record(rds_config: RDSConfig, params: Dict[str, Any]) -> None:
    """Update an existing model record in the registry."""
    rds_client = _get_rds_client()
    assignments = ", ".join(f"{key} = :{key}" for key in params.keys() if key != "repo_id")
    sql = (
        f"UPDATE {rds_config.table_name} SET {assignments} "
        "WHERE repo_id = :repo_id"
    )

    rds_client.execute_statement(
        resourceArn=rds_config.cluster_arn,
        secretArn=rds_config.secret_arn,
        database=rds_config.database,
        sql=sql,
        parameters=[
            {"name": key, "value": _to_rds_value(value)}
            for key, value in params.items()
        ],
    )


def _to_rds_value(value: Any) -> Dict[str, Any]:
    """Convert a Python value into an RDS Data API field representation."""
    if value is None:
        return {"isNull": True}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"longValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _format_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standard Lambda proxy response."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handle_upload(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Entry point for the Lambda function.

    The function returns an API Gateway compatible response. Errors are surfaced
    with a 400-level status code (client issues) or 500-level (server issues).
    """
    try:
        request = UploadRequest.from_event(event)

        bucket_name = _require_env("MODEL_BUCKET_NAME")
        rds_config = _load_rds_config()

        uploaded_artifacts = _upload_artifacts(request, bucket_name)
        db_params = _build_db_parameters(request, uploaded_artifacts)

        if _record_exists(rds_config, request.model_id):
            LOGGER.info("Model %s already exists; updating record.", request.model_id)
            _update_model_record(rds_config, db_params)
            action = "updated"
        else:
            LOGGER.info("Model %s is new; inserting record.", request.model_id)
            _insert_model_record(rds_config, db_params)
            action = "created"

        response_body = {
            "status": "success",
            "action": action,
            "model_id": request.model_id,
            "artifacts": uploaded_artifacts,
        }
        return _format_response(200, response_body)

    except UploadError as err:
        LOGGER.error("Upload error: %s", err, exc_info=True)
        return _format_response(400, {"status": "error", "message": str(err)})

    except (BotoCoreError, ClientError) as err:
        LOGGER.error("AWS service error: %s", err, exc_info=True)
        return _format_response(502, {"status": "error", "message": "AWS service failure"})

    except Exception as err:  # pragma: no cover - defensive catch-all
        LOGGER.exception("Unexpected error during upload")
        return _format_response(500, {"status": "error", "message": str(err)})


# Alias compatible with AWS Lambda's expected handler name
lambda_handler = handle_upload
