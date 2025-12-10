
import copy
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", os.environ["AWS_DEFAULT_REGION"])

import src.delete_artifact_lambda as delete_lambda
import src.enqueue_rate as enqueue_rate
import src.health as health
import src.metadata as metadata
import src.reset_registry_lambda as reset_lambda
import src.search as search
import src.upload as upload


SAMPLE_METADATA_ITEM = {
    "pk": "MODEL#demo-model",
    "sk": "META",
    "model_id": "demo-model",
    "artifact_type": "model",
    "name": "Demo Model",
    "metadata": {
        "id": "demo-model",
        "name": "Demo Model",
        "type": "model",
        "model_url": "https://huggingface.co/demo/demo-model",
    },
    "model_url": "https://huggingface.co/demo/demo-model",
    "data": {
        "url": "https://huggingface.co/demo/demo-model",
        "download_url": "https://example.com/demo-model.zip",
    },
    "assets": {
        "bucket": "demo-bucket",
        "model_zip_key": "artifacts/demo-model/model.zip",
    },
    "license": "Apache-2.0",
    "lineage": {
        "nodes": [
            {"artifact_id": "demo-model", "name": "Demo Model", "source": "registry"},
            {"artifact_id": "parent-model", "name": "Parent", "source": "base_model"},
        ],
        "edges": [
            {
                "from_node_artifact_id": "parent-model",
                "to_node_artifact_id": "demo-model",
                "relationship": "base_model",
            }
        ],
    },
    "audits": [
        {
            "user": {"name": "tester", "is_admin": True},
            "date": "2025-01-01T00:00:00Z",
            "artifact": {"name": "Demo Model", "id": "demo-model", "type": "model"},
            "action": "UPLOAD",
        }
    ],
}

SAMPLE_SEARCH_RECORD = {
    "name": "Demo Model",
    "id": "demo-model",
    "artifact_type": "MODEL",
    "metadata": {"name": "Demo Model", "id": "demo-model", "type": "model"},
}


class MetadataTableStub:
    def __init__(self, item):
        self.item = item

    def get_item(self, Key):
        if Key == {"pk": "MODEL#demo-model", "sk": "META"}:
            return {"Item": copy.deepcopy(self.item)}
        return {}

    def query(self, **kwargs):
        return {"Items": copy.deepcopy(self.item.get("audits", []))}


class FakeSearchRepository:
    def __init__(self, records):
        self.records = records

    def search(self, queries, total_needed, start_key):
        return copy.deepcopy(self.records), None

    def fetch_by_name(self, name):
        if any(record["name"] == name or record.get("metadata", {}).get("name") == name for record in self.records):
            return copy.deepcopy(self.records)
        return []

    def regex_search(self, compiled, total_needed, start_key):
        return copy.deepcopy(self.records), None


class FakeDeleteBatchWriter:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def delete_item(self, Key):
        self.table.deleted.append(Key)


class FakeDeleteTable:
    def __init__(self):
        self.items = [{"pk": "MODEL#demo-model", "sk": "META"}]
        self.deleted = []

    def query(self, **kwargs):
        return {"Items": copy.deepcopy(self.items)}

    def batch_writer(self):
        return FakeDeleteBatchWriter(self)


class FakeDynamoResource:
    def __init__(self, table):
        self._table = table

    def Table(self, name):  # pragma: no cover - simple pass-through
        return self._table


class HealthIntegrationTests(unittest.TestCase):
    def test_health_route_reports_ok(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/health",
        }
        dynamo_status = {"ok": True, "latency_ms": 1, "table_status": "ACTIVE", "item_count": 1}
        s3_status = {"ok": True, "latency_ms": 1}
        with patch("src.health._probe_dynamodb", return_value=dynamo_status), patch(
            "src.health._probe_s3", return_value=s3_status
        ):
            response = health.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["dependencies"]["dynamodb"]["table"], health.DYNAMODB_TABLE_NAME)

    def test_health_components_route_lists_services(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/health/components",
            "queryStringParameters": None,
        }
        dynamo_status = {"ok": True, "latency_ms": 1, "table_status": "ACTIVE", "item_count": 1}
        s3_status = {"ok": True, "latency_ms": 1}
        lambda_status = {
            "ok": True,
            "latency_ms": 1,
            "runtime": "python3.11",
            "code_size": 1,
            "handler": "handler",
            "version": "$LATEST",
        }
        with patch("src.health._probe_dynamodb", return_value=dynamo_status), patch(
            "src.health._probe_s3", return_value=s3_status
        ), patch("src.health._probe_lambda", return_value=lambda_status):
            response = health.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        components = json.loads(response["body"])["components"]
        self.assertGreaterEqual(len(components), 3)

    def test_tracks_endpoint_returns_planned_tracks(self):
        event = {"requestContext": {"http": {"method": "GET"}}, "rawPath": "/tracks"}
        response = health.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertIn("plannedTracks", body)


class SearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.repo = FakeSearchRepository([copy.deepcopy(SAMPLE_SEARCH_RECORD)])

    def test_post_artifacts_route_returns_results(self):
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/artifacts",
            "body": json.dumps([{"name": "Demo Model"}]),
        }
        with patch("src.search._get_repository", return_value=self.repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 200)
        results = json.loads(response["body"])
        self.assertEqual(results[0]["name"], "Demo Model")

    def test_get_artifact_by_name_route(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifact/byName/Demo%20Model",
            "pathParameters": {"name": "Demo Model"},
        }
        with patch("src.search._get_repository", return_value=self.repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(len(payload), 1)

    def test_post_regex_route(self):
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/artifact/byRegEx",
            "body": json.dumps({"regex": "Demo"}),
        }
        with patch("src.search._get_repository", return_value=self.repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(payload[0]["id"], "demo-model")


class DeletionIntegrationTests(unittest.TestCase):
    def test_reset_endpoint_invokes_cleanup(self):
        event = {"httpMethod": "DELETE", "rawPath": "/reset"}
        with patch("src.reset_registry_lambda._clear_table", return_value=5) as clear_table, patch(
            "src.reset_registry_lambda._clear_bucket", return_value=3
        ) as clear_bucket:
            response = reset_lambda.handler(event, None)
        clear_table.assert_called_once()
        clear_bucket.assert_called_once()
        self.assertEqual(response["statusCode"], 200)

    def test_delete_artifact_endpoint(self):
        event = {
            "httpMethod": "DELETE",
            "rawPath": "/artifacts/model/demo-model",
            "pathParameters": {"artifact_type": "model", "id": "demo-model"},
        }
        fake_table = FakeDeleteTable()
        fake_dynamo = FakeDynamoResource(fake_table)
        with patch.object(delete_lambda, "dynamodb", fake_dynamo):
            response = delete_lambda.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(len(fake_table.deleted), 1)


class MetadataIntegrationTests(unittest.TestCase):
    def _table_patch(self):
        table = MetadataTableStub(copy.deepcopy(SAMPLE_METADATA_ITEM))
        return patch("src.metadata._get_table", return_value=table)

    def test_get_artifact_envelope(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifacts/model/demo-model",
        }
        with self._table_patch():
            response = metadata.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(payload["metadata"]["id"], "demo-model")

    def test_license_check_endpoint(self):
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/artifact/model/demo-model/license-check",
            "body": json.dumps({"github_url": "https://github.com/demo/repo"}),
        }
        with self._table_patch(), patch("src.metadata._fetch_github_license_identifier", return_value="apache-2.0"):
            response = metadata.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        result = json.loads(response["body"])
        self.assertTrue(result)

    def test_lineage_endpoint_returns_graph(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifact/model/demo-model/lineage",
        }
        with self._table_patch():
            response = metadata.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(payload["nodes"][0]["artifact_id"], "demo-model")

    def test_cost_endpoint_uses_object_size(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifact/model/demo-model/cost",
        }
        with self._table_patch(), patch("src.metadata._object_size_mb", return_value=1.5):
            response = metadata.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(payload["demo-model"]["total_cost"], 1.5)

    def test_audit_endpoint_returns_entries(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifact/model/demo-model/audit",
        }
        with self._table_patch():
            response = metadata.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertEqual(len(payload), 1)


class UploadIntegrationTests(unittest.TestCase):
    def test_spec_create_route_invokes_helper(self):
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/artifact/model",
            "body": json.dumps({"url": "https://example.com/model.bin"}),
        }
        expected = {"statusCode": 201, "body": json.dumps({"status": "created"})}
        with patch("src.upload._handle_spec_artifact_create", return_value=expected) as create_mock:
            response = upload.handle_upload(event, None)
        create_mock.assert_called_once()
        self.assertEqual(response, expected)

    def test_spec_update_route_invokes_helper(self):
        event = {
            "requestContext": {"http": {"method": "PUT"}},
            "rawPath": "/artifacts/model/demo-model",
            "pathParameters": {"artifact_type": "model", "id": "demo-model"},
            "body": json.dumps({"url": "https://example.com/model.bin"}),
        }
        expected = {"statusCode": 202, "body": json.dumps({"status": "updated"})}
        with patch("src.upload._handle_spec_artifact_update", return_value=expected) as update_mock:
            response = upload.handle_upload(event, None)
        update_mock.assert_called_once()
        self.assertEqual(response, expected)


class RateIntegrationTests(unittest.TestCase):
    def test_rate_endpoint_returns_metrics(self):
        breakdown = {
            metric: {"value": 0.9, "available": True, "latency_ms": 5}
            for metric in enqueue_rate.OPENAPI_METRIC_FIELDS
        }
        sample_item = {
            "model_id": "demo-model",
            "metrics": {"average": 0.9, "breakdown": breakdown},
            "scoring": {"status": "COMPLETED", "net_score_latency": 15},
            "eligibility": {"minimum_evidence_met": True},
        }
        fake_table = FakeRateTable(sample_item)
        fake_dynamo = FakeDynamoResource(fake_table)
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/artifact/model/demo-model/rate",
            "pathParameters": {"model_id": "demo-model"},
        }
        with patch.dict(os.environ, {"MODELS_TABLE": "models-table"}, clear=False), patch.object(
            enqueue_rate, "dynamo", fake_dynamo
        ):
            response = enqueue_rate.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertIn("net_score", payload)


class FakeRateTable:
    def __init__(self, item):
        self.item = item

    def get_item(self, Key):
        return {"Item": copy.deepcopy(self.item)}


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
