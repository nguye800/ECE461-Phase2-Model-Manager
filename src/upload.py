"""
Lambda function implementation for handling model uploads.

This module converts the previous pseudo-code into executable logic. The handler
accepts an event describing the model metadata and artifacts, uploads the
artifacts to S3, and records/updates an entry in the model registry stored in
DynamoDB. Each record mirrors the structure consumed by the metadata and rating
Lambdas so downstream services can read the same data model.

Environment variables used:
    MODEL_BUCKET_NAME: S3 bucket where artifacts will be stored (required)
    ARTIFACTS_DDB_TABLE: DynamoDB table that stores artifact metadata (required)
    ARTIFACTS_DDB_REGION: Optional override for the DynamoDB region
    SCORING_QUEUE_URL: URL of the SQS queue used to trigger rating jobs (optional)
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
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
import zipfile

import boto3
from boto3.session import Session
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_SCORING_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/965919626283/enqueue-rate"

_S3_CLIENT: Optional[boto3.client] = None
_SQS_CLIENT: Optional[boto3.client] = None
_ARTIFACTS_TABLE = None


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


def _get_artifacts_table():
    global _ARTIFACTS_TABLE
    if _ARTIFACTS_TABLE is not None:
        return _ARTIFACTS_TABLE

    table_name = _require_env("ARTIFACTS_DDB_TABLE")
    region = os.getenv("ARTIFACTS_DDB_REGION")
    session: Session = boto3.session.Session()
    dynamo = session.resource("dynamodb", region_name=region) if region else session.resource("dynamodb")
    _ARTIFACTS_TABLE = dynamo.Table(table_name)
    return _ARTIFACTS_TABLE


def _get_sqs_client() -> boto3.client:
    global _SQS_CLIENT
    if _SQS_CLIENT is None:
        session: Session = boto3.session.Session()
        _SQS_CLIENT = session.client("sqs")
    return _SQS_CLIENT


def _resolve_codebase_url(request: UploadRequest) -> Optional[str]:
    metadata = request.metadata or {}
    return (
        request.code_repository
        or metadata.get("code_repository")
        or metadata.get("codebase_url")
        or metadata.get("github_link")
    )


def _resolve_dataset_url(request: UploadRequest) -> Optional[str]:
    metadata = request.metadata or {}
    return request.dataset_link or metadata.get("dataset_link") or metadata.get("dataset_url")


def _is_valid_url(value: Optional[str]) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _model_ready_for_scoring(
    model_url: Optional[str], codebase_url: Optional[str], dataset_url: Optional[str]
) -> bool:
    """A model is ready for scoring when URLs for model, code, and data exist."""
    return all(_is_valid_url(url) for url in (model_url, codebase_url, dataset_url))


def _collect_scoring_urls(item: Dict[str, Any], request: UploadRequest) -> Optional[Dict[str, str]]:
    dataset = item.get("dataset") or {}
    codebase = item.get("codebase") or {}
    model_url = item.get("model_url") or request.model_url
    codebase_url = codebase.get("url") or _resolve_codebase_url(request)
    dataset_url = dataset.get("url") or _resolve_dataset_url(request)

    if _model_ready_for_scoring(model_url, codebase_url, dataset_url):
        return {
            "model_url": model_url,
            "codebase_url": codebase_url,
            "dataset_url": dataset_url,
        }
    return None


def _enqueue_scoring_job(model_id: str, urls: Dict[str, str]) -> bool:
    """
    Enqueue a scoring job on SQS. Returns True when a message was queued.
    """
    queue_url = os.getenv("SCORING_QUEUE_URL", DEFAULT_SCORING_QUEUE_URL)
    if not queue_url:
        LOGGER.debug("SCORING_QUEUE_URL not set; skipping scoring enqueue.")
        return False

    message_body = json.dumps(
        {
            "model_id": model_id,
            "model_url": urls["model_url"],
            "codebase_url": urls["codebase_url"],
            "dataset_url": urls["dataset_url"],
            "queued_at": datetime.utcnow().isoformat(),
        }
    )

    try:
        sqs_client = _get_sqs_client()
        sqs_client.send_message(QueueUrl=queue_url, MessageBody=message_body)
        LOGGER.info("Queued scoring job for model %s", model_id)
        return True
    except (BotoCoreError, ClientError) as exc:
        LOGGER.warning(
            "Failed to enqueue scoring job for model %s: %s",
            model_id,
            exc,
        )
        return False


def _extract_artifact_type(event: Dict[str, Any]) -> str:
    path = event.get("rawPath") or event.get("path", "")
    segments = [segment for segment in path.strip("/").split("/") if segment]
    if len(segments) >= 2 and segments[0] == "artifact":
        return segments[1].lower()
    path_params = event.get("pathParameters") or {}
    return (path_params.get("artifact_type") or "model").lower()


def _build_dynamo_key(artifact_type: str, model_id: str) -> Dict[str, str]:
    pk_field = os.getenv("RATE_PK_FIELD", "pk")
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    pk_prefix = os.getenv("RATE_PK_PREFIX") or f"{artifact_type.upper()}#"
    if not pk_prefix.endswith("#"):
        pk_prefix = f"{pk_prefix}#"
    sk_value = os.getenv("RATE_META_SK", "META")
    return {
        pk_field: f"{pk_prefix}{model_id}",
        sk_field: sk_value,
    }


_METRIC_BREAKDOWN_KEYS = [
    "ramp_up_time",
    "bus_factor",
    "performance_claims",
    "license",
    "size_score",
    "dataset_and_code",
    "dataset_quality",
    "code_quality",
    "reproducibility",
    "reviewedness",
    "tree_score",
]


def _default_metrics() -> Dict[str, Any]:
    breakdown = {
        name: {"value": Decimal("0"), "available": False}
        for name in _METRIC_BREAKDOWN_KEYS
    }
    return {
        "average": Decimal("0"),
        "evidence_coverage": Decimal("0"),
        "breakdown": breakdown,
    }


def _default_scoring() -> Dict[str, Any]:
    return {
        "status": "INITIAL",
        "score_version": "v1",
    }


def _default_eligibility() -> Dict[str, Any]:
    return {"minimum_evidence_met": False, "reason": "Pending initial scoring"}


def _default_approval() -> Dict[str, Any]:
    return {"visible": False, "status": "PENDING"}


def _prepare_for_dynamo(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        prepared: Dict[str, Any] = {}
        for key, entry in value.items():
            if entry is None:
                continue
            prepared[key] = _prepare_for_dynamo(entry)
        return prepared
    if isinstance(value, list):
        return [_prepare_for_dynamo(entry) for entry in value if entry is not None]
    return value


def _merge_dataset_info(request: UploadRequest, metadata: Dict[str, Any], existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dataset = metadata.get("dataset") or existing.get("dataset") or {}
    link = request.dataset_link or metadata.get("dataset_link")
    name = request.dataset_name or metadata.get("dataset_name")
    if link:
        dataset.setdefault("url", link)
    if name:
        dataset.setdefault("name", name)
    return dataset or None


def _merge_codebase_info(request: UploadRequest, metadata: Dict[str, Any], existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    codebase = metadata.get("codebase") or existing.get("codebase") or {}
    repo = _resolve_codebase_url(request)
    if repo:
        codebase.setdefault("url", repo)
    return codebase or None


def _build_ddb_item(
    request: UploadRequest,
    uploaded_artifacts: Iterable[Dict[str, Any]],
    artifact_type: str,
    existing: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = request.metadata or {}
    now = datetime.now(timezone.utc).isoformat()
    artifact_type_upper = artifact_type.upper()

    item: Dict[str, Any] = existing.copy() if existing else {}
    item.update(_build_dynamo_key(artifact_type, request.model_id))
    item["type"] = artifact_type_upper
    item["model_id"] = request.model_id
    item["model_url"] = request.model_url
    if metadata.get("name"):
        item["name"] = metadata["name"]
    elif "name" not in item:
        item["name"] = request.model_id

    if metadata.get("base_models"):
        item["base_models"] = metadata["base_models"]

    dataset_info = _merge_dataset_info(request, metadata, item)
    if dataset_info:
        item["dataset"] = dataset_info

    codebase_info = _merge_codebase_info(request, metadata, item)
    if codebase_info:
        item["codebase"] = codebase_info

    assets = metadata.get("assets") or item.get("assets") or {}
    artifacts = list(uploaded_artifacts)
    if artifacts:
        first = artifacts[0]
        assets.setdefault("model_zip_key", first["key"])
        assets.setdefault("bucket", first["bucket"])
        if first.get("size_bytes") is not None:
            assets.setdefault("model_zip_size_bytes", first["size_bytes"])
    if assets:
        item["assets"] = assets

    if metadata.get("metrics"):
        item["metrics"] = metadata["metrics"]
    else:
        item["metrics"] = item.get("metrics") or _default_metrics()

    item["scoring"] = item.get("scoring") or _default_scoring()
    item["eligibility"] = item.get("eligibility") or _default_eligibility()
    item["approval"] = item.get("approval") or _default_approval()

    if not existing:
        item["created_at"] = now
    item["updated_at"] = now

    return _prepare_for_dynamo(item)


def _fetch_existing_item(table, key: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
        response = table.get_item(Key=key)
    except ClientError:
        LOGGER.exception("Failed to read existing artifact record from DynamoDB")
        raise UploadError("Unable to access artifact metadata store")
    return response.get("Item")


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
        artifact_type = _extract_artifact_type(event)
        table = _get_artifacts_table()

        uploaded_artifacts = _upload_artifacts(request, bucket_name)
        key = _build_dynamo_key(artifact_type, request.model_id)
        existing = _fetch_existing_item(table, key)

        item = _build_ddb_item(request, uploaded_artifacts, artifact_type, existing)
        scoring_urls = _collect_scoring_urls(item, request)
        try:
            table.put_item(Item=item)
        except ClientError as exc:
            LOGGER.error("Failed to write artifact metadata to DynamoDB: %s", exc, exc_info=True)
            return _format_response(
                502, {"status": "error", "message": "Failed to store artifact metadata"}
            )

        action = "updated" if existing else "created"

        queued_scoring_job = False
        if scoring_urls:
            queued_scoring_job = _enqueue_scoring_job(request.model_id, scoring_urls)

        response_body = {
            "status": "success",
            "action": action,
            "model_id": request.model_id,
            "artifacts": uploaded_artifacts,
            "scoring_job_enqueued": queued_scoring_job,
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
