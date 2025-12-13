import base64
import importlib.machinery
import io
import json
import os
import sys
import types
import unittest
import zipfile
from unittest.mock import MagicMock, patch

# Always stub boto3/botocore to avoid importing hefty dependencies in tests.
boto3_module = types.ModuleType("boto3")
session_module = types.ModuleType("boto3.session")
boto3_module.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)  # type: ignore[attr-defined]
session_module.__spec__ = importlib.machinery.ModuleSpec("boto3.session", loader=None)  # type: ignore[attr-defined]


class _FakeSession:
    def client(self, service_name):
        return MagicMock(name=f"{service_name}_client")


session_module.Session = _FakeSession
boto3_module.session = session_module
sys.modules["boto3"] = boto3_module
sys.modules["boto3.session"] = session_module

botocore_module = types.ModuleType("botocore")
exceptions_module = types.ModuleType("botocore.exceptions")
botocore_module.__spec__ = importlib.machinery.ModuleSpec("botocore", loader=None)  # type: ignore[attr-defined]
exceptions_module.__spec__ = importlib.machinery.ModuleSpec("botocore.exceptions", loader=None)  # type: ignore[attr-defined]


class _FakeBotoCoreError(Exception):
    pass


class _FakeClientError(Exception):
    def __init__(self, error_response, operation_name):
        super().__init__(error_response, operation_name)
        self.response = error_response
        self.operation_name = operation_name


exceptions_module.BotoCoreError = _FakeBotoCoreError
exceptions_module.ClientError = _FakeClientError
botocore_module.exceptions = exceptions_module
sys.modules["botocore"] = botocore_module
sys.modules["botocore.exceptions"] = exceptions_module

import src.upload as upload

ClientError = upload.ClientError


def _sample_event() -> dict:
    """Build a minimal valid upload event."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "config.json",
            json.dumps(
                {
                    "model_name": "TestModel",
                    "num_parameters": 123456,
                    "base_model": "SomeBaseModel",
                }
            ),
        )
        archive.writestr(
            "README.md",
            "This model uses https://huggingface.co/datasets/example/dataset\n",
        )
        archive.writestr("weights.bin", "fake-bytes")

    artifact_body = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "model_id": "model-123",
        "model_url": "https://huggingface.co/example/model-123",
        "metadata": {
            "license": "Apache-2.0",
            "likes": 10,
            "downloads": 50,
            "gb_size_of_model": 1.5,
        },
        "code_repository": "https://github.com/example/model-123",
        "artifacts": [
            {
                "name": "model.bin",
                "body": artifact_body,
                "content_type": "application/octet-stream",
            }
        ],
    }


def _sample_event_without_metadata() -> dict:
    event = _sample_event()
    event["metadata"] = {}
    event.pop("dataset_link", None)
    event.pop("dataset_name", None)
    return event


class UploadHandlerTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "MODEL_BUCKET_NAME": "test-bucket",
                "ARTIFACTS_DDB_TABLE": "model-metadata",
                "ARTIFACTS_DDB_REGION": "us-east-1",
            },
            clear=True,
        )
        self.env_patch.start()

        # Ensure cached clients do not leak across tests
        upload._S3_CLIENT = None
        upload._ARTIFACTS_TABLE = None

    def tearDown(self):
        self.env_patch.stop()
        upload._S3_CLIENT = None
        upload._ARTIFACTS_TABLE = None

    def test_handle_upload_creates_new_record(self):
        event = _sample_event()
        uploaded_artifacts = [
            {
                "artifact_name": "model.bin",
                "bucket": "test-bucket",
                "key": "models/model-123/model.bin",
                "version_id": "1",
            }
        ]

        with patch.object(upload, "_upload_artifacts", return_value=uploaded_artifacts) as mock_upload, \
                patch.object(upload, "_get_artifacts_table") as mock_table_factory, \
                patch.object(upload, "_fetch_existing_item", return_value=None) as mock_fetch, \
                patch.object(upload, "_build_ddb_item", return_value={"pk": "MODEL#model-123"}) as mock_build:

            table_mock = MagicMock()
            mock_table_factory.return_value = table_mock
            response = upload.handle_upload(event, MagicMock())

        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["action"], "created")
        self.assertEqual(body["model_id"], "model-123")
        self.assertEqual(body["artifacts"], uploaded_artifacts)

        self.assertTrue(mock_upload.called)
        request_arg, bucket_arg = mock_upload.call_args.args
        self.assertIsInstance(request_arg, upload.UploadRequest)
        self.assertEqual(bucket_arg, "test-bucket")

        mock_fetch.assert_called_once()
        mock_build.assert_called_once()
        table_mock.put_item.assert_called_once_with(Item={"pk": "MODEL#model-123"})

    def test_handle_upload_updates_existing_record(self):
        event = _sample_event()
        uploaded_artifacts = [
            {
                "artifact_name": "model.bin",
                "bucket": "test-bucket",
                "key": "models/model-123/model.bin",
                "version_id": "2",
            }
        ]

        with patch.object(upload, "_upload_artifacts", return_value=uploaded_artifacts), \
                patch.object(upload, "_get_artifacts_table") as mock_table_factory, \
                patch.object(upload, "_fetch_existing_item", return_value={"pk": "MODEL#model-123"}) as mock_fetch, \
                patch.object(upload, "_build_ddb_item", return_value={"pk": "MODEL#model-123"}) as mock_build:

            table_mock = MagicMock()
            mock_table_factory.return_value = table_mock
            response = upload.handle_upload(event, MagicMock())

        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["action"], "updated")
        mock_fetch.assert_called_once()
        mock_build.assert_called_once()
        table_mock.put_item.assert_called_once()

    def test_handle_upload_missing_required_field(self):
        event = {
            "model_url": "https://huggingface.co/example/model-123",
            "artifacts": [],
        }

        response = upload.handle_upload(event, MagicMock())
        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertEqual(body["status"], "error")
        self.assertIn("Missing required field", body["message"])

    def test_handle_upload_propagates_upload_error(self):
        event = _sample_event()

        with patch.object(upload, "_upload_artifacts", side_effect=upload.UploadError("failure during upload")):
            response = upload.handle_upload(event, MagicMock())

        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertIn("failure during upload", body["message"])

    def test_handle_upload_handles_aws_client_error(self):
        event = _sample_event()
        client_error = ClientError(
            error_response={"Error": {"Code": "InternalFailure", "Message": "upstream error"}},
            operation_name="PutObject",
        )

        with patch.object(upload, "_upload_artifacts", side_effect=client_error):
            response = upload.handle_upload(event, MagicMock())

        self.assertEqual(response["statusCode"], 502)
        body = json.loads(response["body"])
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "AWS service failure")

    def test_zip_metadata_enriched_when_missing(self):
        event = _sample_event_without_metadata()
        request = upload.UploadRequest.from_event(event)

        s3_mock = MagicMock()
        s3_mock.put_object.return_value = {"VersionId": "1"}

        with patch.object(upload, "_get_s3_client", return_value=s3_mock):
            artifacts = upload._upload_artifacts(request, "test-bucket")

        self.assertTrue(any(a["size_bytes"] is not None for a in artifacts))
        self.assertIsNotNone(request.metadata.get("gb_size_of_model"))
        self.assertEqual(request.dataset_link, "https://huggingface.co/datasets/example/dataset")
        self.assertEqual(request.metadata.get("base_models_modelID"), "SomeBaseModel")

    def test_enqueue_scoring_job_sends_message_when_ready(self):
        event = _sample_event()
        event["dataset_link"] = "https://datasets.example.com/model-123"
        request = upload.UploadRequest.from_event(event)

        with patch.dict(
            os.environ, {"SCORING_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123/queue"}, clear=False
        ):
            sqs_mock = MagicMock()
            with patch.object(upload, "_get_sqs_client", return_value=sqs_mock), \
                    patch.object(upload, "_model_ready_for_scoring", return_value=True):
                queued = upload._enqueue_scoring_job(request)

        self.assertTrue(queued)
        sqs_mock.send_message.assert_called_once()

    def test_enqueue_scoring_job_skips_when_urls_missing(self):
        event = _sample_event()
        event.pop("dataset_link", None)
        request = upload.UploadRequest.from_event(event)

        with patch.dict(
            os.environ, {"SCORING_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123/queue"}, clear=False
        ):
            sqs_mock = MagicMock()
            with patch.object(upload, "_get_sqs_client", return_value=sqs_mock), \
                    patch.object(upload, "_model_ready_for_scoring", return_value=False):
                queued = upload._enqueue_scoring_job(request)

        self.assertFalse(queued)
        sqs_mock.send_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
