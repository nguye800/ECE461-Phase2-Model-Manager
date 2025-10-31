import base64
import importlib.machinery
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Always stub boto3/botocore to avoid heavy imports during tests.
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

import src.download as download

ClientError = download.ClientError


def _base_event(payload: dict) -> dict:
    return payload


def _build_rds_response(records, column_names):
    """Utility to construct an RDS Data API shaped response."""
    column_metadata = [{"name": name} for name in column_names]
    converted_records = []
    for record in records:
        row = []
        for value in record:
            if value is None:
                row.append({"isNull": True})
            elif isinstance(value, str):
                row.append({"stringValue": value})
            elif isinstance(value, bool):
                row.append({"booleanValue": value})
            elif isinstance(value, float):
                row.append({"doubleValue": value})
            elif isinstance(value, int):
                row.append({"longValue": value})
            elif isinstance(value, bytes):
                row.append({"blobValue": base64.b64encode(value).decode("utf-8")})
            else:
                row.append({"stringValue": str(value)})
        converted_records.append(row)
    return {"records": converted_records, "columnMetadata": column_metadata}


class DownloadHandlerTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "MODEL_REGISTRY_CLUSTER_ARN": "arn:aws:rds:region:acct:cluster:example",
                "MODEL_REGISTRY_SECRET_ARN": "arn:aws:secretsmanager:region:acct:secret:example",
                "MODEL_REGISTRY_DATABASE": "model_registry",
                "MODEL_REGISTRY_TABLE": "model_registry",
                "DOWNLOAD_PRESIGN_EXPIRATION": "3600",
            },
            clear=True,
        )
        self.env_patch.start()
        download._S3_CLIENT = None
        download._RDS_CLIENT = None

    def tearDown(self):
        self.env_patch.stop()
        download._S3_CLIENT = None
        download._RDS_CLIENT = None

    def test_handle_download_success(self):
        event = _base_event({"model_id": "model-123"})
        column_names = [
            "repo_id",
            "model_url",
            "date_time_entered_to_db",
            "likes",
            "downloads",
            "license",
            "github_link",
            "github_numContributors",
            "base_models_modelID",
            "base_model_urls",
            "parameter_number",
            "gb_size_of_model",
            "dataset_link",
            "dataset_name",
            "s3_location_of_model_zip",
            "s3_location_of_cloudwatch_log_for_database_entry",
        ]
        record_values = [[
            "model-123",
            "https://huggingface.co/example/model-123",
            "2024-01-01T00:00:00Z",
            10,
            100,
            "Apache-2.0",
            "https://github.com/example/model-123",
            5,
            "base-model",
            "https://huggingface.co/base/model",
            1000000,
            1.23,
            "https://huggingface.co/datasets/example/dataset",
            "example/dataset",
            "s3://model-bucket/models/model-123/model.bin",
            "s3://model-bucket/logs/model-123-log.json",
        ]]
        rds_response = _build_rds_response(record_values, column_names)

        mock_rds = MagicMock()
        mock_rds.execute_statement.return_value = rds_response
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.side_effect = [
            "https://presigned/model",
            "https://presigned/log",
        ]
        mock_s3.head_object.side_effect = [
            {"ContentLength": 2048},
            {"ContentLength": 512},
        ]

        with patch.object(download, "_get_rds_client", return_value=mock_rds), \
                patch.object(download, "_get_s3_client", return_value=mock_s3):

            response = download.handle_download(event, MagicMock())

        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["model_id"], "model-123")
        self.assertEqual(len(body["artifacts"]), 2)
        types_returned = {artifact["type"] for artifact in body["artifacts"]}
        self.assertEqual(types_returned, {"model", "cloudwatch_log"})
        self.assertTrue(any(artifact["presigned_url"] == "https://presigned/model" for artifact in body["artifacts"]))
        self.assertTrue(any(artifact["presigned_url"] == "https://presigned/log" for artifact in body["artifacts"]))
        self.assertEqual(body["total_artifact_size_bytes"], 2560)
        self.assertTrue(all("size_bytes" in artifact for artifact in body["artifacts"]))
        # Ensure metadata excludes S3 columns
        self.assertNotIn("s3_location_of_model_zip", body["metadata"])
        self.assertNotIn("s3_location_of_cloudwatch_log_for_database_entry", body["metadata"])
        mock_s3.generate_presigned_url.assert_any_call(
            "get_object",
            Params={"Bucket": "model-bucket", "Key": "models/model-123/model.bin"},
            ExpiresIn=3600,
        )
        mock_s3.get_object.assert_not_called()

    def test_handle_download_not_found(self):
        event = _base_event({"model_id": "missing-model"})
        mock_rds = MagicMock()
        mock_rds.execute_statement.return_value = {"records": [], "columnMetadata": []}

        with patch.object(download, "_get_rds_client", return_value=mock_rds):
            response = download.handle_download(event, MagicMock())

        self.assertEqual(response["statusCode"], 404)
        body = json.loads(response["body"])
        self.assertIn("not found", body["message"])

    def test_handle_download_missing_model_id(self):
        response = download.handle_download({}, MagicMock())
        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertIn("model_id", body["message"])

    def test_handle_download_skips_invalid_s3_url(self):
        event = _base_event({"model_id": "model-123"})
        column_names = ["repo_id", "s3_location_of_model_zip"]
        rds_response = _build_rds_response(
            [["model-123", "invalid-url"]],
            column_names,
        )

        mock_rds = MagicMock()
        mock_rds.execute_statement.return_value = rds_response
        mock_s3 = MagicMock()

        with patch.object(download, "_get_rds_client", return_value=mock_rds), \
                patch.object(download, "_get_s3_client", return_value=mock_s3):

            response = download.handle_download(event, MagicMock())

        body = json.loads(response["body"])
        self.assertEqual(body["artifacts"], [])

    def test_handle_download_without_presign(self):
        event = _base_event({"model_id": "model-123", "presign": False})
        column_names = ["repo_id", "s3_location_of_model_zip"]
        rds_response = _build_rds_response(
            [["model-123", "s3://bucket/key"]],
            column_names,
        )

        mock_rds = MagicMock()
        mock_rds.execute_statement.return_value = rds_response
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 4096}

        with patch.object(download, "_get_rds_client", return_value=mock_rds), \
                patch.object(download, "_get_s3_client", return_value=mock_s3):

            response = download.handle_download(event, MagicMock())

        mock_s3.generate_presigned_url.assert_not_called()
        body = json.loads(response["body"])
        artifact = body["artifacts"][0]
        self.assertIsNone(artifact["presigned_url"])
        self.assertEqual(artifact["size_bytes"], 4096)

    def test_handle_download_invalid_ttl(self):
        event = _base_event({"model_id": "model-123", "presign_ttl": "abc"})
        response = download.handle_download(event, MagicMock())
        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertIn("presign_ttl", body["message"])

    def test_handle_download_rds_client_error(self):
        event = _base_event({"model_id": "model-123"})
        mock_rds = MagicMock()
        mock_rds.execute_statement.side_effect = ClientError(
            error_response={"Error": {"Code": "500", "Message": "boom"}},
            operation_name="ExecuteStatement",
        )

        with patch.object(download, "_get_rds_client", return_value=mock_rds):
            response = download.handle_download(event, MagicMock())

        self.assertEqual(response["statusCode"], 502)
        body = json.loads(response["body"])
        self.assertEqual(body["message"], "AWS service failure")

    def test_handle_download_inline_mode_returns_data(self):
        event = _base_event({"model_id": "model-123", "delivery_mode": "inline"})
        column_names = ["repo_id", "s3_location_of_model_zip"]
        rds_response = _build_rds_response(
            [["model-123", "s3://bucket/key"]],
            column_names,
        )

        mock_rds = MagicMock()
        mock_rds.execute_statement.return_value = rds_response
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 128}
        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"zip-bytes")}

        with patch.object(download, "_get_rds_client", return_value=mock_rds), \
                patch.object(download, "_get_s3_client", return_value=mock_s3):

            response = download.handle_download(event, MagicMock())

        body = json.loads(response["body"])
        artifact = body["artifacts"][0]
        self.assertIn("data", artifact)
        self.assertEqual(base64.b64decode(artifact["data"]), b"zip-bytes")
        self.assertEqual(artifact["size_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
