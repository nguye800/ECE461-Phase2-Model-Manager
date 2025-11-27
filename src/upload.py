"""
Lambda function implementation for handling model uploads.

This module converts the previous pseudo-code into executable logic. The handler
accepts an event describing the model metadata and artifacts, uploads the
artifacts to S3, and records/updates an entry in the model registry stored in
DynamoDB. Each record mirrors the structure consumed by the metadata and rating
Lambdas so downstream services can read the same data model.

Environment variables used:
    ARTIFACTS_DDB_TABLE: DynamoDB table that stores artifact metadata (required)
    ARTIFACTS_DDB_REGION: Optional override for the DynamoDB region
    SCORING_QUEUE_URL: URL of the SQS queue used to trigger rating jobs (optional)
Artifacts are stored in the S3 bucket 'modelzip-logs-artifacts'.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import logging
import os
import re
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse

import boto3
from boto3.dynamodb.types import TypeSerializer
from boto3.session import Session
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_SCORING_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/965919626283/enqueue-rate"
SCORING_QUEUE_ARN = "arn:aws:sqs:us-east-1:965919626283:enqueue-rate"
MODEL_BUCKET_NAME = "modelzip-logs-artifacts"
STREAM_CHUNK_SIZE = 8 * 1024 * 1024
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
README_ANALYSIS_PROMPT = (
    "You are analyzing the README for a Hugging Face model. "
    "Identify if it references a dataset (typically a Hugging Face dataset) and/or a GitHub code repository. "
    "Respond with JSON using this schema: "
    '{"dataset_name": "<name or null>", "dataset_url": "<url or null>", '
    '"code_name": "<name or null>", "code_url": "<github url or null>"}.\n'
    "If multiple candidates exist, pick the most prominent. Use null when unsure."
)
MAX_README_CHARS = 6000
HUGGINGFACE_REPO_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?huggingface\.co/([^/]+)/([^/\s]+)"
)
HUGGINGFACE_DATASET_PATTERN = re.compile(
    r"(https?://huggingface\.co/datasets/[^/\s]+/[^/\s)]+)"
)
GITHUB_REPO_PATTERN = re.compile(
    r"(https?://github\.com/[\w\-.]+/[\w\-.]+)"
)

_S3_CLIENT: Optional[boto3.client] = None
_SQS_CLIENT: Optional[boto3.client] = None
_ARTIFACTS_TABLE = None
_DYNAMODB_SERIALIZER = TypeSerializer()
try:  # pragma: no cover - requires AWS creds
    _BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
except Exception as exc:  # pragma: no cover
    _BEDROCK_CLIENT = None
    LOGGER.debug("Bedrock client unavailable: %s", exc)

_ALLOWED_ARTIFACT_TYPES = {"model", "dataset", "code"}


class UploadError(Exception):
    """Custom error for failures within the upload pipeline."""


class ArtifactNotFound(UploadError):
    """Raised when a requested artifact is missing from the registry."""


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


def _extract_http_method(event: Dict[str, Any]) -> str:
    request_context = event.get("requestContext") or {}
    method = (
        request_context.get("http", {}).get("method")
        or event.get("httpMethod")
        or event.get("method")
    )
    return method.upper() if isinstance(method, str) else ""


def _extract_path(event: Dict[str, Any]) -> str:
    request_context = event.get("requestContext") or {}
    return (
        event.get("rawPath")
        or event.get("path")
        or request_context.get("http", {}).get("path")
        or ""
    )


def _parse_path_segments(path: str) -> List[str]:
    return [segment for segment in path.strip("/").split("/") if segment]


def _identify_spec_operation(event: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Determine whether the event targets one of the new artifact endpoints.
    Returns (operation, artifact_type, artifact_id) when matched.
    """
    method = _extract_http_method(event)
    if not method:
        return None

    path_params = event.get("pathParameters") or {}
    path = _extract_path(event)
    segments = _parse_path_segments(path)

    if method == "POST":
        artifact_type = path_params.get("artifact_type")
        if not artifact_type and len(segments) >= 2 and segments[0] == "artifact":
            artifact_type = segments[1]
        if artifact_type:
            return ("create", artifact_type, None)

    if method == "PUT":
        artifact_type = path_params.get("artifact_type")
        artifact_id = path_params.get("id") or path_params.get("artifact_id")
        if (not artifact_type or not artifact_id) and len(segments) >= 3 and segments[0] == "artifacts":
            artifact_type = artifact_type or segments[1]
            artifact_id = artifact_id or segments[2]
        if artifact_type and artifact_id:
            return ("update", artifact_type, artifact_id)

    return None


def _normalize_artifact_type(value: Optional[str]) -> str:
    if not value:
        raise UploadError("Artifact type is required in the request path.")
    normalized = value.lower()
    if normalized not in _ALLOWED_ARTIFACT_TYPES:
        raise UploadError(f"Unsupported artifact type '{value}'.")
    return normalized


def _load_json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    if "body" not in event or event["body"] is None:
        raise UploadError("Request body is required.")

    body = event["body"]
    if isinstance(body, (bytes, bytearray)):
        raw_body = bytes(body)
    elif isinstance(body, str):
        raw_body = body.encode("utf-8")
    else:
        raw_body = json.dumps(body).encode("utf-8")

    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise UploadError("Request body must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise UploadError("Request body must be a JSON object.")
    return payload


def _generate_artifact_id() -> str:
    return uuid.uuid4().hex


def _infer_name_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    candidate = parsed.path.strip("/").split("/")[-1]
    return candidate or None


def _sanitize_key_component(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "-", value.strip("/"))
    return sanitized or "artifact"


def _build_artifact_s3_key(artifact_type: str, artifact_id: str, source_url: str) -> str:
    filename = _infer_name_from_url(source_url) or f"{artifact_id}.bin"
    filename = _sanitize_key_component(filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"artifacts/{artifact_type}/{artifact_id}/{timestamp}-{filename}"


def _build_download_url(bucket: str, key: str) -> str:
    encoded_key = quote(key.lstrip("/"), safe="/")
    return f"https://{bucket}.s3.amazonaws.com/{encoded_key}"


def _download_artifact_from_url(url: str) -> Tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "ModelRegistryUploader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            if not body:
                raise UploadError("Artifact download returned no data.")
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            if ";" in content_type:
                content_type = content_type.split(";", 1)[0]
            return body, content_type or "application/octet-stream"
    except HTTPError as exc:
        raise UploadError(f"Failed to download artifact (HTTP {exc.code}).") from exc
    except URLError as exc:
        raise UploadError(f"Failed to download artifact: {exc.reason}") from exc


def _stream_download_to_s3(
    url: str,
    bucket: str,
    key: str,
    *,
    metadata: Dict[str, Any],
) -> Tuple[int, str, str]:
    """
    Stream a remote artifact directly into S3 using multipart upload when necessary.
    Returns (size_bytes, sha256_hex, content_type).
    """
    request = urllib.request.Request(url, headers={"User-Agent": "ModelRegistryUploader/1.0"})
    s3_client = _get_s3_client()
    metadata_str = {k: str(v) for k, v in metadata.items()}
    upload_id: Optional[str] = None

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            chunk = response.read(STREAM_CHUNK_SIZE)
            if not chunk:
                raise UploadError("Artifact download returned no data.")

            next_chunk = response.read(STREAM_CHUNK_SIZE)
            hasher = hashlib.sha256()

            if not next_chunk:
                hasher.update(chunk)
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=chunk,
                    ContentType=content_type,
                    Metadata=metadata_str,
                )
                return len(chunk), hasher.hexdigest(), content_type

            create_resp = s3_client.create_multipart_upload(
                Bucket=bucket,
                Key=key,
                ContentType=content_type,
                Metadata=metadata_str,
            )
            upload_id = create_resp["UploadId"]
            parts: List[Dict[str, Any]] = []
            total_bytes = 0
            part_number = 1

            def _upload_part(data: bytes) -> None:
                nonlocal part_number, total_bytes
                hasher.update(data)
                total_bytes += len(data)
                result = s3_client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data,
                )
                parts.append({"PartNumber": part_number, "ETag": result["ETag"]})
                part_number += 1

            _upload_part(chunk)
            _upload_part(next_chunk)

            while True:
                chunk = response.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                _upload_part(chunk)

            s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            return total_bytes, hasher.hexdigest(), content_type

    except Exception as exc:
        if upload_id:
            try:
                s3_client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                LOGGER.warning("Failed to abort multipart upload for %s: %s", key, exc)
        raise UploadError(f"Failed to stream artifact '{url}': {exc}") from exc


def _build_spec_registry_item(
    artifact_type: str,
    artifact_id: str,
    name: str,
    source_url: str,
    bucket_name: str,
    object_key: str,
    download_url: str,
    content_type: str,
    size_bytes: int,
    checksum: str,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = copy.deepcopy(existing) if existing else {}
    base.update(_build_dynamo_key(artifact_type, artifact_id))
    base["type"] = artifact_type.upper()
    base["artifact_type"] = artifact_type
    base["model_id"] = artifact_id
    base["name"] = name
    base["name_lc"] = name.lower()
    base["model_url"] = source_url
    base["metadata"] = {"name": name, "id": artifact_id, "type": artifact_type}
    base["data"] = {"url": source_url, "download_url": download_url}

    base_models = base.get("base_models")
    base["base_models"] = copy.deepcopy(base_models) if isinstance(base_models, list) else []
    base["dataset"] = _merge_with_defaults(base.get("dataset"), _SPEC_DATASET_DEFAULTS)
    base["codebase"] = _merge_with_defaults(base.get("codebase"), _SPEC_CODEBASE_DEFAULTS)

    assets = _merge_with_defaults(base.get("assets"), _SPEC_ASSET_DEFAULTS)
    assets.update(
        {
            "key": object_key,
            "model_zip_key": object_key,
            "model_zip_sha256": checksum,
            "bucket": bucket_name,
            "download_url": download_url,
            "content_type": content_type,
            "size_bytes": size_bytes,
        }
    )
    base["assets"] = assets

    if not base.get("metrics"):
        base["metrics"] = _default_metrics()
    if not base.get("scoring"):
        base["scoring"] = _default_scoring()
    else:
        base["scoring"] = _merge_with_defaults(base["scoring"], _default_scoring())
    if not base.get("eligibility"):
        base["eligibility"] = _default_eligibility()
    else:
        base["eligibility"] = _merge_with_defaults(base["eligibility"], _default_eligibility())
    if not base.get("approval"):
        base["approval"] = _default_approval()
    else:
        base["approval"] = _merge_with_defaults(base["approval"], _default_approval())
    if "audits" not in base:
        base["audits"] = []
    return base


def _handle_spec_artifact_create(event: Dict[str, Any], artifact_type_raw: str) -> Dict[str, Any]:
    artifact_type = _normalize_artifact_type(artifact_type_raw)
    payload = _load_json_body(event)
    source_url = payload.get("url")
    if not _is_valid_url(source_url):
        raise UploadError("Request body must include a valid 'url'.")

    requested_name = payload.get("name")
    if not requested_name:
        metadata_section = payload.get("metadata")
        if isinstance(metadata_section, dict):
            requested_name = metadata_section.get("name")
    if requested_name is not None:
        if not isinstance(requested_name, str):
            raise UploadError("Artifact name must be a string when provided.")
        requested_name = requested_name.strip() or None
    artifact_id = _generate_artifact_id()
    name = requested_name or _infer_name_from_url(source_url) or f"{artifact_type}-{artifact_id}"

    bucket_name = MODEL_BUCKET_NAME
    table = _get_artifacts_table()
    object_key = _build_artifact_s3_key(artifact_type, artifact_id, source_url)
    LOGGER.info(
        "Creating artifact %s (%s) from %s into s3://%s/%s",
        name,
        artifact_id,
        source_url,
        bucket_name,
        object_key,
    )
    size_bytes, checksum, content_type = _stream_download_to_s3(
        source_url,
        bucket_name,
        object_key,
        metadata={
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "source_url": source_url,
        },
    )

    now = datetime.now(timezone.utc).isoformat()
    download_url = _build_download_url(bucket_name, object_key)

    metadata_block = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    request = None
    if artifact_type == "model":
        # Reuse the same discovery flow as the legacy uploader so spec uploads
        # can infer datasets/codebases and queue scoring immediately when possible.
        request = UploadRequest(
            model_id=artifact_id,
            model_url=source_url,
            metadata=metadata_block.copy(),
            artifacts=[],
        )
        try:
            _discover_related_resources_for_model(request)
        except Exception as exc:
            LOGGER.warning("Spec create autodiscovery failed for %s: %s", artifact_id, exc)
        metadata_block = request.metadata

    item = _build_spec_registry_item(
        artifact_type,
        artifact_id,
        name,
        source_url,
        bucket_name,
        object_key,
        download_url,
        content_type,
        size_bytes,
        checksum,
    )
    if metadata_block:
        item["metadata"].update(metadata_block)
    if artifact_type == "model":
        if metadata_block.get("dataset"):
            item["dataset"] = metadata_block["dataset"]
        if metadata_block.get("codebase"):
            item["codebase"] = metadata_block["codebase"]
        if metadata_block.get("pending_dependencies"):
            item["pending_dependencies"] = metadata_block["pending_dependencies"]
    item["created_at"] = now
    item["updated_at"] = now
    item.setdefault("audits", [])
    _append_audit_entry(
        item,
        action="CREATE",
        artifact_type=artifact_type,
        user=_resolve_request_user(payload),
    )

    _put_item(table, item)
    _process_dependency_resolution(artifact_type, item)
    scoring_urls = _collect_scoring_urls(item)
    if scoring_urls:
        LOGGER.info(
            "Enqueuing scoring job for new %s artifact %s (all URLs present)", artifact_type, artifact_id
        )
        _enqueue_scoring_job(artifact_id, scoring_urls)
    print(
        f"[upload.spec.create] Completed creation for {artifact_type}/{artifact_id}",
        flush=True,
    )
    return _format_response(201, {"metadata": item["metadata"], "data": item["data"]})


def _handle_spec_artifact_update(
    event: Dict[str, Any], artifact_type_raw: str, artifact_id: str
) -> Dict[str, Any]:
    artifact_type = _normalize_artifact_type(artifact_type_raw)
    payload = _load_json_body(event)
    metadata = payload.get("metadata")
    data = payload.get("data")
    if not isinstance(metadata, dict) or not isinstance(data, dict):
        raise UploadError("Request body must include 'metadata' and 'data' objects.")

    name = metadata.get("name")
    metadata_id = metadata.get("id")
    metadata_type = metadata.get("type") or metadata.get("artifact_type")
    if not isinstance(name, str) or not name.strip() or not metadata_id:
        raise UploadError("Metadata must include 'name' and 'id'.")
    name = name.strip()
    if not isinstance(metadata_id, str):
        metadata_id = str(metadata_id)
    if metadata_type is not None and not isinstance(metadata_type, str):
        metadata_type = str(metadata_type)
    if metadata_id != artifact_id:
        raise UploadError("Metadata artifact id must match request path.")
    if metadata_type and _normalize_artifact_type(metadata_type) != artifact_type:
        raise UploadError("Metadata artifact type must match request path.")

    source_url = data.get("url")
    if not _is_valid_url(source_url):
        raise UploadError("Artifact data must include a valid 'url'.")

    codebase_override = data.get("codebase")
    if codebase_override is not None and not isinstance(codebase_override, dict):
        raise UploadError("Field 'codebase' must be an object when provided.")
    dataset_override = data.get("dataset")
    if dataset_override is not None and not isinstance(dataset_override, dict):
        raise UploadError("Field 'dataset' must be an object when provided.")

    bucket_name = MODEL_BUCKET_NAME
    table = _get_artifacts_table()
    key = _build_dynamo_key(artifact_type, artifact_id)
    existing = _fetch_existing_item(table, key)
    if not existing:
        raise ArtifactNotFound(f"Artifact '{artifact_id}' was not found.")

    existing_name = (existing.get("metadata") or {}).get("name") or existing.get("name")
    if existing_name and existing_name != name:
        raise UploadError("Metadata name does not match the stored artifact.")

    s3_client = _get_s3_client()
    existing_assets = existing.get("assets") or {}
    old_bucket = existing_assets.get("bucket")
    old_key = existing_assets.get("key") or existing_assets.get("model_zip_key")
    if old_bucket and old_key:
        try:
            s3_client.delete_object(Bucket=old_bucket, Key=old_key)
        except (BotoCoreError, ClientError) as exc:
            LOGGER.warning(
                "Failed to delete previous artifact object %s/%s: %s",
                old_bucket,
                old_key,
                exc,
            )

    mutable_existing = copy.deepcopy(existing)
    if codebase_override:
        existing_codebase = _merge_with_defaults(mutable_existing.get("codebase"), _SPEC_CODEBASE_DEFAULTS)
        existing_codebase.update({k: v for k, v in codebase_override.items() if v is not None})
        mutable_existing["codebase"] = existing_codebase
    if dataset_override:
        existing_dataset = _merge_with_defaults(mutable_existing.get("dataset"), _SPEC_DATASET_DEFAULTS)
        existing_dataset.update({k: v for k, v in dataset_override.items() if v is not None})
        mutable_existing["dataset"] = existing_dataset

    object_key = _build_artifact_s3_key(artifact_type, artifact_id, source_url)
    LOGGER.info(
        "Updating artifact %s (%s) from %s into s3://%s/%s",
        name,
        artifact_id,
        source_url,
        bucket_name,
        object_key,
    )
    size_bytes, checksum, content_type = _stream_download_to_s3(
        source_url,
        bucket_name,
        object_key,
        metadata={
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "source_url": source_url,
        },
    )

    now = datetime.now(timezone.utc).isoformat()
    download_url = _build_download_url(bucket_name, object_key)
    updated_item = _build_spec_registry_item(
        artifact_type,
        artifact_id,
        name,
        source_url,
        bucket_name,
        object_key,
        download_url,
        content_type,
        size_bytes,
        checksum,
        existing=mutable_existing,
    )
    updated_item.setdefault("created_at", existing.get("created_at") if existing else now)
    updated_item["updated_at"] = now

    _append_audit_entry(
        updated_item,
        action="UPDATE",
        artifact_type=artifact_type,
        user=_resolve_request_user(payload),
    )
    _put_item(table, updated_item)
    scoring_urls = _collect_scoring_urls(updated_item)
    if scoring_urls:
        _enqueue_scoring_job(artifact_id, scoring_urls)
    print(
        f"[upload.spec.update] Completed update for {artifact_type}/{artifact_id}",
        flush=True,
    )
    return _format_response(200, {"metadata": updated_item["metadata"], "data": updated_item["data"]})



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

    table_name = _require_env("ARTIFACTS_DDB_TABLE", default="model-metadata")
    region = os.getenv("ARTIFACTS_DDB_REGION", "us-east-1")
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


def _collect_scoring_urls(item: Dict[str, Any], request: Optional[UploadRequest] = None) -> Optional[Dict[str, str]]:
    dataset = item.get("dataset") or {}
    codebase = item.get("codebase") or {}
    model_url = item.get("model_url") or (request.model_url if request else None)
    codebase_url = codebase.get("url")
    dataset_url = dataset.get("url")
    if request:
        codebase_url = codebase_url or _resolve_codebase_url(request)
        dataset_url = dataset_url or _resolve_dataset_url(request)

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
        LOGGER.info(
            "Queued scoring job for model %s on %s (%s)",
            model_id,
            queue_url,
            SCORING_QUEUE_ARN,
        )
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


def _build_dynamo_key(artifact_type: str, artifact_id: str) -> Dict[str, str]:
    pk_field = os.getenv("RATE_PK_FIELD", "pk")
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    pk_prefix = os.getenv("RATE_PK_PREFIX") or f"{artifact_type.upper()}#"
    if not pk_prefix.endswith("#"):
        pk_prefix = f"{pk_prefix}#"
    sk_value = os.getenv("RATE_META_SK", "META")
    return {
        pk_field: f"{pk_prefix}{artifact_id}",
        sk_field: sk_value,
    }


_SPEC_DATASET_DEFAULTS = {
    "url": None,
    "version": None,
    "updated_at": None,
}

_SPEC_CODEBASE_DEFAULTS = {
    "url": None,
    "version": None,
    "updated_at": None,
}

_SPEC_ASSET_DEFAULTS = {
    "key": None,
    "model_zip_key": None,
    "model_zip_sha256": None,
    "bucket": None,
    "download_url": None,
    "content_type": None,
    "size_bytes": None,
}


def _merge_with_defaults(
    existing: Optional[Dict[str, Any]], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if existing:
        merged.update(existing)
    return merged


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
        "last_scored_at": None,
        "scorer_build": None,
    }


def _default_eligibility() -> Dict[str, Any]:
    return {"minimum_evidence_met": False, "reason": "Pending initial scoring"}


def _default_approval() -> Dict[str, Any]:
    return {"visible": False, "status": "PENDING", "reason": None}


def _prepare_for_dynamo(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _prepare_for_dynamo(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_prepare_for_dynamo(entry) for entry in value]
    return value


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_nulls(entry) for key, entry in value.items() if entry is not None}
    if isinstance(value, list):
        return [_strip_nulls(entry) for entry in value if entry is not None]
    return value


def _serialize_item_for_dynamo(item: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _prepare_for_dynamo(item)
    serialized = _DYNAMODB_SERIALIZER.serialize(prepared)
    if "M" not in serialized:
        raise UploadError("Serialized Dynamo item must be a map representation.")
    return serialized["M"]


def _put_item(table, item: Dict[str, Any]) -> None:
    serialized = _serialize_item_for_dynamo(item)
    try:
        table.meta.client.put_item(TableName=table.name, Item=serialized)
    except ClientError as exc:
        message = exc.response.get("Error", {}).get("Message", "")
        if "Type mismatch for key" not in message:
            raise
        LOGGER.warning(
            "Dynamo serialization mismatch detected (%s). Falling back to default serialization.",
            message,
        )
        fallback_item = _strip_nulls(item)
        table.put_item(Item=_prepare_for_dynamo(fallback_item))


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

    if item.get("name"):
        item["name_lc"] = str(item["name"]).lower()

    if metadata.get("base_models"):
        item["base_models"] = metadata["base_models"]

    dataset_info = _merge_dataset_info(request, metadata, item)
    if dataset_info:
        item["dataset"] = dataset_info

    codebase_info = _merge_codebase_info(request, metadata, item)
    if codebase_info:
        item["codebase"] = codebase_info

    pending_meta = metadata.get("pending_dependencies")
    if pending_meta:
        cleaned_pending = {
            key: value
            for key, value in pending_meta.items()
            if value and (value.get("name") or value.get("url"))
        }
        if cleaned_pending:
            item["pending_dependencies"] = cleaned_pending
    elif existing and existing.get("pending_dependencies"):
        item["pending_dependencies"] = existing["pending_dependencies"]

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
        item.setdefault("audits", [])
    item["updated_at"] = now

    return item


def _resolve_request_user(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("user") if isinstance(payload, dict) else None
    name = "system"
    is_admin = False
    if isinstance(metadata, dict):
        name = metadata.get("name") or name
        is_admin = bool(metadata.get("is_admin", False))
    return {"name": name, "is_admin": is_admin}


def _append_audit_entry(
    item: Dict[str, Any],
    *,
    action: str,
    artifact_type: str,
    user: Optional[Dict[str, Any]] = None,
) -> None:
    entry = {
        "user": user or {"name": "system", "is_admin": False},
        "date": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "name": item.get("name"),
            "id": item.get("model_id"),
            "type": artifact_type,
        },
        "action": action.upper(),
    }
    audits = item.setdefault("audits", [])
    audits.append(entry)


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


def _discover_related_resources_for_model(request: UploadRequest) -> None:
    readme_text = _fetch_hf_model_readme(request.model_url)
    if not readme_text:
        LOGGER.debug("No README detected for %s; skipping autodiscovery.", request.model_id)
        return

    # Ask Bedrock/heuristics which dataset/code the README references.
    findings = _analyze_readme_entities(readme_text)
    pending = request.metadata.setdefault("pending_dependencies", {})
    dataset_entry = _lookup_artifact_entry(
        "dataset",
        names=[findings.get("dataset_name")],
        urls=[findings.get("dataset_url")],
    )
    if dataset_entry:
        dataset_url = dataset_entry.get("model_url") or dataset_entry.get("dataset", {}).get("url")
        dataset_name = dataset_entry.get("name") or dataset_entry.get("metadata", {}).get("name")
        if dataset_url:
            request.dataset_link = dataset_url
            request.metadata.setdefault("dataset", {}).setdefault("url", dataset_url)
        if dataset_name:
            request.dataset_name = dataset_name
            request.metadata.setdefault("dataset", {}).setdefault("name", dataset_name)
        LOGGER.info(
            "Autodiscovered dataset %s for model %s",
            dataset_name or dataset_url,
            request.model_id,
        )
    elif findings.get("dataset_name") or findings.get("dataset_url"):
        # Record what we saw so we can attach the dataset later when it exists.
        pending["dataset"] = {
            "name": findings.get("dataset_name"),
            "url": findings.get("dataset_url"),
        }

    code_names = [
        findings.get("code_name"),
        request.metadata.get("name"),
        request.model_id,
    ]
    code_entry = _lookup_artifact_entry(
        "code",
        names=code_names,
        urls=[findings.get("code_url")],
    )
    if code_entry:
        code_url = code_entry.get("model_url") or code_entry.get("codebase", {}).get("url")
        code_name = code_entry.get("name") or code_entry.get("metadata", {}).get("name")
        if code_url:
            request.code_repository = code_url
            request.metadata.setdefault("codebase", {}).setdefault("url", code_url)
        if code_name:
            request.metadata.setdefault("codebase", {}).setdefault("name", code_name)
        LOGGER.info(
            "Autodiscovered codebase %s for model %s",
            code_name or code_url,
            request.model_id,
        )
    else:
        # Same deferred flow for code when nothing exists yet; seed with the
        # model name so we can match later if the repo uses a shorter alias.
        pending["code"] = {
            "name": findings.get("code_name") or request.metadata.get("name") or request.model_id,
            "url": findings.get("code_url"),
        }


def _fetch_hf_model_readme(model_url: Optional[str]) -> Optional[str]:
    if not model_url:
        return None
    owner, repo = _extract_hf_repo_segments(model_url)
    if not owner or not repo:
        return None
    for candidate in _candidate_model_readme_urls(owner, repo):
        text = _safe_http_get(candidate)
        if text:
            return text
    return None


def _extract_hf_repo_segments(url: str) -> Tuple[Optional[str], Optional[str]]:
    match = HUGGINGFACE_REPO_PATTERN.search(url)
    if match:
        return match.group(1), match.group(2)
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


def _candidate_model_readme_urls(owner: str, repo: str) -> List[str]:
    base = f"https://huggingface.co/{owner}/{repo}"
    return [
        f"{base}/raw/main/README.md",
        f"{base}/raw/master/README.md",
        f"{base}/resolve/main/README.md",
        f"{base}/resolve/master/README.md",
    ]


def _safe_http_get(url: str) -> Optional[str]:
    if not url:
        return None
    request = urllib.request.Request(url, headers={"User-Agent": "model-manager/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _analyze_readme_entities(readme_text: str) -> Dict[str, Optional[str]]:
    excerpt = (readme_text or "").strip()
    if not excerpt:
        return {}
    if len(excerpt) > MAX_README_CHARS:
        excerpt = excerpt[:MAX_README_CHARS]

    prompt = f"{README_ANALYSIS_PROMPT}\n\nREADME Content:\n```\n{excerpt}\n```"
    parsed = _invoke_bedrock_analysis(prompt)
    if not isinstance(parsed, dict):
        parsed = _heuristic_related_entities(excerpt)
    return {
        "dataset_name": parsed.get("dataset_name"),
        "dataset_url": parsed.get("dataset_url"),
        "code_name": parsed.get("code_name"),
        "code_url": parsed.get("code_url"),
    }


def _invoke_bedrock_analysis(prompt: str) -> Optional[Dict[str, Optional[str]]]:
    client = _BEDROCK_CLIENT
    if client is None:
        return None

    body = {
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_gen_len": 400,
        "temperature": 0.0,
        "top_p": 0.9,
    }

    try:  # pragma: no cover - requires AWS access
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        raw = response.get("body")
        payload = (
            json.loads(raw.read().decode("utf-8"))
            if hasattr(raw, "read")
            else json.loads(raw)
        )
        text = payload.get("generation") or ""
        if not text and "generations" in payload:
            generations = payload.get("generations") or []
            if generations:
                text = generations[0].get("text", "")
        text = text.strip()
        if not text:
            return None
        return json.loads(text)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Bedrock README analysis failed: %s", exc)
        return None


def _heuristic_related_entities(text: str) -> Dict[str, Optional[str]]:
    dataset_url = None
    dataset_match = HUGGINGFACE_DATASET_PATTERN.search(text)
    if dataset_match:
        dataset_url = dataset_match.group(1)
    code_url_match = GITHUB_REPO_PATTERN.search(text)
    code_url = code_url_match.group(1) if code_url_match else None
    return {
        "dataset_url": dataset_url,
        "dataset_name": _infer_name_from_url(dataset_url) if dataset_url else None,
        "code_url": code_url,
        "code_name": _infer_name_from_url(code_url) if code_url else None,
    }


def _lookup_artifact_entry(
    artifact_type: str,
    *,
    names: Optional[Iterable[Optional[str]]] = None,
    urls: Optional[Iterable[Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    table = _get_artifacts_table()
    normalized_urls = [url for url in (urls or []) if _is_valid_url(url)]
    for url in normalized_urls:
        entry = _scan_artifact_by_url(table, artifact_type, url)
        if entry:
            return entry
    for name in names or []:
        if not name:
            continue
        entry = _query_artifact_by_name(table, artifact_type, name)
        if entry:
            return entry
    return None


def _scan_artifact_by_url(table, artifact_type: str, url: str) -> Optional[Dict[str, Any]]:
    try:
        response = table.scan(
            FilterExpression=Attr("type").eq(artifact_type.upper()) & Attr("model_url").eq(url),
            Limit=1,
        )
    except ClientError as exc:
        LOGGER.debug("Failed to scan for %s url=%s: %s", artifact_type, url, exc)
        return None
    items = response.get("Items") or []
    return items[0] if items else None


def _query_artifact_by_name(table, artifact_type: str, name: str) -> Optional[Dict[str, Any]]:
    name_lc = name.strip().lower()
    if not name_lc:
        return None
    try:
        response = table.query(
            IndexName="GSI_ALPHABET_LISTING",
            KeyConditionExpression=Key("type").eq(artifact_type.upper())
            & Key("name_lc").eq(name_lc),
            Limit=1,
        )
    except ClientError as exc:
        LOGGER.debug("Failed to query %s by name=%s: %s", artifact_type, name, exc)
        return None
    items = response.get("Items") or []
    return items[0] if items else None


def _process_dependency_resolution(artifact_type: str, artifact_record: Dict[str, Any]) -> None:
    if artifact_type not in {"dataset", "code"}:
        return
    dependency_key = "dataset" if artifact_type == "dataset" else "code"
    dependency_url = artifact_record.get("model_url")
    dependency_name = (
        artifact_record.get("name")
        or artifact_record.get("metadata", {}).get("name")
        or artifact_record.get("model_id")
    )
    table = _get_artifacts_table()
    pending_models = _find_models_waiting_on_dependency(
        table, dependency_key, dependency_name, dependency_url
    )
    for model_item in pending_models:
        updated = _apply_dependency_to_model(
            model_item, dependency_key, artifact_record, dependency_name, dependency_url
        )
        if not updated:
            continue
        try:
            _put_item(table, updated)
            LOGGER.info(
                "Resolved pending %s dependency for model %s",
                dependency_key,
                updated.get("model_id"),
            )
        except ClientError as exc:
            LOGGER.warning(
                "Failed to update model %s while resolving %s dependency: %s",
                updated.get("model_id"),
                dependency_key,
                exc,
            )
            continue
        scoring_urls = _collect_scoring_urls(updated)
        if scoring_urls:
            _enqueue_scoring_job(updated["model_id"], scoring_urls)


def _find_models_waiting_on_dependency(
    table,
    dependency_key: str,
    dependency_name: Optional[str],
    dependency_url: Optional[str],
) -> List[Dict[str, Any]]:
    filter_expression = Attr("type").eq("MODEL") & Attr("pending_dependencies").exists()
    items: List[Dict[str, Any]] = []
    scan_kwargs: Dict[str, Any] = {"FilterExpression": filter_expression}
    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            pending = (item.get("pending_dependencies") or {}).get(dependency_key)
            if not pending:
                continue
            if _dependency_matches(pending, dependency_name, dependency_url):
                items.append(item)
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key
    return items


def _dependency_matches(
    pending_entry: Dict[str, Any],
    dependency_name: Optional[str],
    dependency_url: Optional[str],
) -> bool:
    pending_url = pending_entry.get("url")
    pending_name = pending_entry.get("name")
    if pending_url and dependency_url and _normalize_str(pending_url) == _normalize_str(dependency_url):
        return True
    if pending_name and dependency_name and _normalize_str(pending_name) == _normalize_str(dependency_name):
        return True
    return False


def _normalize_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().lower()


def _apply_dependency_to_model(
    model_item: Dict[str, Any],
    dependency_key: str,
    dependency_record: Dict[str, Any],
    dependency_name: Optional[str],
    dependency_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    pending = model_item.get("pending_dependencies") or {}
    pending_entry = pending.get(dependency_key)
    if not pending_entry:
        return None
    if not _dependency_matches(pending_entry, dependency_name, dependency_url):
        return None

    if dependency_key == "dataset":
        target_block = model_item.get("dataset") or {}
    else:
        target_block = model_item.get("codebase") or {}

    if dependency_name:
        target_block.setdefault("name", dependency_name)
    if dependency_url:
        target_block["url"] = dependency_url
    target_block["artifact_id"] = dependency_record.get("model_id")
    target_block["updated_at"] = datetime.now(timezone.utc).isoformat()

    if dependency_key == "dataset":
        model_item["dataset"] = target_block
    else:
        model_item["codebase"] = target_block

    pending.pop(dependency_key, None)
    if pending:
        model_item["pending_dependencies"] = pending
    else:
        model_item.pop("pending_dependencies", None)
    model_item["updated_at"] = datetime.now(timezone.utc).isoformat()
    return model_item


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
    print(f"[upload.lambda] Responding with status {status_code}", flush=True)
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
    method = _extract_http_method(event)
    path = _extract_path(event)
    print(
        f"[upload.lambda] Received {method or 'UNKNOWN'} {path or '/'} body={event.get('body')}",
        flush=True,
    )
    try:
        spec_operation = _identify_spec_operation(event)
        if spec_operation:
            operation, artifact_type, artifact_id = spec_operation
            if operation == "create":
                print(f"[upload.lambda] Routing to spec create {artifact_type}", flush=True)
                return _handle_spec_artifact_create(event, artifact_type)
            if operation == "update":
                if not artifact_id:
                    raise UploadError("Artifact id is required for update requests.")
                print(
                    f"[upload.lambda] Routing to spec update {artifact_type}/{artifact_id}",
                    flush=True,
                )
                return _handle_spec_artifact_update(event, artifact_type, artifact_id)

        request = UploadRequest.from_event(event)
        artifact_type = _extract_artifact_type(event)
        if artifact_type == "model":
            try:
                _discover_related_resources_for_model(request)
            except Exception as exc:
                LOGGER.warning(
                    "Automatic related-resource discovery failed for %s: %s",
                    request.model_id,
                    exc,
                )

        bucket_name = MODEL_BUCKET_NAME
        table = _get_artifacts_table()

        uploaded_artifacts = _upload_artifacts(request, bucket_name)
        key = _build_dynamo_key(artifact_type, request.model_id)
        existing = _fetch_existing_item(table, key)

        item = _build_ddb_item(request, uploaded_artifacts, artifact_type, existing)
        scoring_urls = _collect_scoring_urls(item, request)
        try:
            _put_item(table, item)
        except ClientError as exc:
            LOGGER.error("Failed to write artifact metadata to DynamoDB: %s", exc, exc_info=True)
            return _format_response(
                502, {"status": "error", "message": "Failed to store artifact metadata"}
            )
        _process_dependency_resolution(artifact_type, item)

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
        print(
            f"[upload.lambda] Completed generic upload action={action} model_id={request.model_id}",
            flush=True,
        )
        return _format_response(200, response_body)

    except ArtifactNotFound as err:
        LOGGER.error("Artifact not found error: %s", err, exc_info=True)
        print(f"[upload.lambda] Artifact not found error: {err}", flush=True)
        return _format_response(404, {"status": "error", "message": str(err)})

    except UploadError as err:
        LOGGER.error("Upload error: %s", err, exc_info=True)
        print(f"[upload.lambda] Upload error: {err}", flush=True)
        return _format_response(400, {"status": "error", "message": str(err)})

    except (BotoCoreError, ClientError) as err:
        LOGGER.error("AWS service error: %s", err, exc_info=True)
        print(f"[upload.lambda] AWS service error: {err}", flush=True)
        return _format_response(502, {"status": "error", "message": "AWS service failure"})

    except Exception as err:  # pragma: no cover - defensive catch-all
        LOGGER.exception("Unexpected error during upload")
        print(f"[upload.lambda] Unexpected error: {err}", flush=True)
        return _format_response(500, {"status": "error", "message": str(err)})


# Alias compatible with AWS Lambda's expected handler name
lambda_handler = handle_upload
