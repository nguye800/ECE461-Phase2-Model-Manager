import io
import json
import os
import sys
import types
import unittest
from unittest.mock import patch
import importlib.machinery

if "boto3" not in sys.modules:  # pragma: no cover
    boto3_module = types.ModuleType("boto3")
    session_module = types.ModuleType("boto3.session")

    class _FakeSession:
        def client(self, name):
            raise RuntimeError("boto3 client requested in tests")

        def resource(self, name):
            raise RuntimeError("boto3 resource requested in tests")

    session_module.Session = _FakeSession
    boto3_module.session = session_module
    boto3_module.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)
    session_module.__spec__ = importlib.machinery.ModuleSpec(
        "boto3.session", loader=None
    )
    sys.modules["boto3"] = boto3_module
    sys.modules["boto3.session"] = session_module

if "botocore" not in sys.modules:  # pragma: no cover
    botocore_module = types.ModuleType("botocore")
    exceptions_module = types.ModuleType("botocore.exceptions")

    class _FakeClientError(Exception):
        pass

    class _FakeBotoCoreError(Exception):
        pass

    exceptions_module.ClientError = _FakeClientError
    exceptions_module.BotoCoreError = _FakeBotoCoreError
    botocore_module.exceptions = exceptions_module
    botocore_module.__spec__ = importlib.machinery.ModuleSpec("botocore", loader=None)
    exceptions_module.__spec__ = importlib.machinery.ModuleSpec(
        "botocore.exceptions", loader=None
    )
    sys.modules["botocore"] = botocore_module
    sys.modules["botocore.exceptions"] = exceptions_module

if "boto3.dynamodb" not in sys.modules:  # pragma: no cover
    dynamodb_module = types.ModuleType("boto3.dynamodb")
    conditions_module = types.ModuleType("boto3.dynamodb.conditions")

    class _FakeKey:
        def __init__(self, _name):
            self._name = _name

        def eq(self, _value):
            return self

        def begins_with(self, _value):
            return self

        def __and__(self, _other):
            return self

    conditions_module.Key = _FakeKey
    dynamodb_module.conditions = conditions_module
    sys.modules["boto3.dynamodb"] = dynamodb_module
    sys.modules["boto3.dynamodb.conditions"] = conditions_module

import src.metadata as metadata


class StubTable:
    def __init__(self):
        self.item = {
            "pk": "MODEL#demo-model",
            "sk": "META",
            "model_id": "demo-model",
            "artifact_type": "model",
            "base_models": [{"model_id": "base-1"}],
            "gb_size_of_model": 2.5,
            "license": "Apache-2.0",
            "assets": {
                "bucket": "demo-bucket",
                "model_zip_key": "demo/model/model.zip",
                "log_key": "demo/model/log.json",
            },
        }
        self.audit_items = [
            {
                "pk": "MODEL#demo-model",
                "sk": "AUDIT#2025-01-01T00:00:00Z#0001",
                "entity": "AUDIT",
                "action": "DOWNLOAD",
            }
        ]

    def get_item(self, Key):
        if Key["pk"] == "MODEL#demo-model":
            return {"Item": self.item}
        return {}

    def query(self, **kwargs):
        return {"Items": self.audit_items}


class MetadataLambdaTests(unittest.TestCase):
    def setUp(self):
        self.table = StubTable()
        self.patch = patch("src.metadata._get_table", return_value=self.table)
        self.patch.start()
        self.s3_patch = patch("src.metadata._s3_client", return_value=FakeS3())
        self.fake_s3 = self.s3_patch.start()
        os.environ["ARTIFACTS_DDB_TABLE"] = "model-metadata"

    def tearDown(self):
        self.patch.stop()
        self.s3_patch.stop()

    def invoke(self, method: str, path: str, query=None):
        event = {
            "requestContext": {"http": {"method": method}},
            "rawPath": path,
            "queryStringParameters": query,
        }
        return metadata.lambda_handler(event, None)

    def test_get_artifact_download_metadata(self):
        resp = self.invoke("GET", "/artifacts/model/demo-model")
        body = json.loads(resp["body"])
        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["metadata"]["model_id"], "demo-model")
        self.assertEqual(body["artifacts"][0]["bucket"], "demo-bucket")

    def test_get_lineage(self):
        resp = self.invoke("GET", "/artifact/model/demo-model/lineage")
        body = json.loads(resp["body"])
        self.assertEqual(body["base_models"][0]["model_id"], "base-1")

    def test_get_cost(self):
        resp = self.invoke("GET", "/artifact/model/demo-model/cost")
        body = json.loads(resp["body"])
        self.assertAlmostEqual(body["estimated_cost_usd"], 0.3)

    def test_get_audit_entries(self):
        resp = self.invoke("GET", "/artifact/model/demo-model/audit")
        body = json.loads(resp["body"])
        self.assertEqual(len(body["entries"]), 1)

    def test_post_license_check(self):
        resp = metadata.lambda_handler(
            {
                "requestContext": {"http": {"method": "POST"}},
                "rawPath": "/artifact/model/demo-model/license-check",
            },
            None,
        )
        body = json.loads(resp["body"])
        self.assertTrue(body["result"]["fine_tune_allowed"])

    def test_not_found(self):
        resp = self.invoke("GET", "/artifacts/model/unknown")
        self.assertEqual(resp["statusCode"], 404)

    def test_inline_download(self):
        resp = self.invoke(
            "GET",
            "/artifacts/model/demo-model",
            query={"inline": "true"},
        )
        body = json.loads(resp["body"])
        self.assertIn("data", body["artifacts"][0])


class FakeS3:
    def __init__(self):
        self.data = {
            ("demo-bucket", "demo/model/model.zip"): b"binary-data",
            ("demo-bucket", "demo/model/log.json"): b"log-data",
        }

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.data.get((Bucket, Key), b""))}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://mock/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.data.get((Bucket, Key), b""))}


if __name__ == "__main__":
    unittest.main()
