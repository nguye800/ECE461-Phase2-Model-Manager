import base64
import json
import os
import unittest
from decimal import Decimal

import boto3
from moto import mock_aws
from unittest import mock

from src import search


class TestSearchLambdaDynamo(unittest.TestCase):
    TABLE_NAME = "model-registry"

    def setUp(self):
        self.mock = mock_aws()
        self.mock.start()
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        os.environ["MODEL_REGISTRY_TABLE"] = self.TABLE_NAME
        self.dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        self._create_table()
        self._seed_data()
        search._REPOSITORY = None  # ensure repository reinitialises per test

    def tearDown(self):
        search._REPOSITORY = None
        self.mock.stop()

    # ------------------------------------------------------------------ helpers

    def _create_table(self):
        table = self.dynamodb.create_table(
            TableName=self.TABLE_NAME,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
                {"AttributeName": "type", "AttributeType": "S"},
                {"AttributeName": "name", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI_ALPHABET_LISTING",
                    "KeySchema": [
                        {"AttributeName": "type", "KeyType": "HASH"},
                        {"AttributeName": "name", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )
        table.wait_until_exists()
        self.table = table

    def _seed_data(self):
        items = [
            self._model_item(
                model_id="model-alpha",
                name="AlphaModel",
                license="apache-2.0",
                avg=Decimal("0.55"),
                eligibility=False,
                base_models=[
                    {"model_id": "base-A", "relation": "derived_from"},
                ],
                readme="Alpha README text",
            ),
            self._model_item(
                model_id="model-beta",
                name="BetaModel",
                license="mit",
                avg=Decimal("0.85"),
                eligibility=True,
                base_models=[
                    {"model_id": "base-B", "relation": "fine_tuned_from"},
                ],
                readme="Beta README mentions quantum effects.",
            ),
            self._model_item(
                model_id="model-gamma",
                name="GammaModel",
                license="apache-2.0",
                avg=Decimal("0.93"),
                eligibility=True,
                base_models=[
                    {"model_id": "base-A", "relation": "fine_tuned_from"},
                ],
                readme="Gamma README covers advanced math topics.",
            ),
        ]
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

    def _model_item(
        self,
        *,
        model_id: str,
        name: str,
        license: str,
        avg: Decimal,
        eligibility: bool,
        base_models: list[dict],
        readme: str,
    ) -> dict:
        return {
            "pk": f"MODEL#{model_id}",
            "sk": "META",
            "type": "MODEL",
            "model_id": model_id,
            "name": name,
            "name_lc": name.lower(),
            "license": license,
            "model_url": f"https://example.com/{model_id}",
            "created_at": "2025-01-01T00:00:00Z",
            "base_models": base_models,
            "metrics": {
                "average": avg,
                "breakdown": {
                    "ramp_up_time": {"value": Decimal("0.0"), "available": False},
                },
            },
            "eligibility": {"minimum_evidence_met": eligibility},
            "readme_text": readme,
        }

    def _event(
        self,
        method: str,
        path: str,
        *,
        body=None,
        headers=None,
        query=None,
        path_params=None,
        raw_path=None,
        base64_encode=False,
    ):
        event = {
            "httpMethod": method,
            "path": path,
            "headers": headers or {},
            "queryStringParameters": query,
            "multiValueQueryStringParameters": None,
            "pathParameters": path_params,
            "requestContext": {},
        }
        if raw_path:
            event["rawPath"] = raw_path
        if body is not None:
            raw_body = json.dumps(body) if isinstance(body, (dict, list)) else body
            if base64_encode:
                encoded = base64.b64encode(raw_body.encode("utf-8")).decode("utf-8")
                event["body"] = encoded
                event["isBase64Encoded"] = True
            else:
                event["body"] = raw_body
        return event

    def _parse_body(self, response):
        return json.loads(response["body"])

    def _decode_token(self, token: str) -> dict:
        padding = "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8")
        return json.loads(decoded)

    # ------------------------------------------------------------------ tests

    def test_post_artifacts_basic_pagination(self):
        event = self._event(
            "POST",
            "/artifacts",
            headers={"x-limit": "1"},
            body={"artifact_queries": [{"name": "*"}]},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(body["page"]["limit"], 1)
        self.assertEqual(len(body["artifacts"]), 1)
        self.assertIn("x-next-offset", response["headers"])

        token = response["headers"]["x-next-offset"]
        next_event = self._event(
            "POST",
            "/artifacts",
            headers={"x-limit": "1", "x-offset": token},
            body={"artifact_queries": [{"name": "*"}]},
        )
        next_response = search.handle_search(next_event, None)
        next_body = self._parse_body(next_response)

        self.assertEqual(next_response["statusCode"], 200)
        self.assertNotEqual(
            body["artifacts"][0]["model_id"],
            next_body["artifacts"][0]["model_id"],
        )

    def test_post_artifacts_invalid_pagination_token(self):
        event = self._event(
            "POST",
            "/artifacts",
            headers={"x-offset": "%%%bad%%%", "x-limit": "2"},
            body={"artifact_queries": [{"name": "*"}]},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(body["page"]["offset"], 0)

    def test_post_artifacts_numeric_offset_header(self):
        event = self._event(
            "POST",
            "/artifacts",
            headers={"x-offset": "2", "x-limit": "2"},
            body={"artifact_queries": [{"name": "*"}]},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(body["page"]["offset"], 2)

    def test_post_artifacts_license_filter(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={"artifact_queries": [{"filters": {"license": "mit"}}]},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(len(body["artifacts"]), 1)
        self.assertEqual(body["artifacts"][0]["name"], "BetaModel")

    def test_post_artifacts_metrics_average_range(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"metrics.average": {"min": 0.9}}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = [a["name"] for a in body["artifacts"]]

        self.assertIn("GammaModel", names)
        self.assertNotIn("AlphaModel", names)

    def test_post_artifacts_metrics_average_between_range(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"metrics.average": {"min": 0.6, "max": 0.9}}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = {a["name"] for a in body["artifacts"]}

        self.assertEqual(names, {"BetaModel"})

    def test_post_artifacts_metrics_average_exact_string(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={"artifact_queries": [{"filters": {"metrics.average": "0.85"}}]},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(len(body["artifacts"]), 1)
        self.assertEqual(body["artifacts"][0]["name"], "BetaModel")

    def test_post_artifacts_base_model_filter(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"base_models[].model_id": "base-A"}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = {a["name"] for a in body["artifacts"]}

        self.assertEqual(names, {"AlphaModel", "GammaModel"})

    def test_post_artifacts_base_model_filter_list(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"base_models[].model_id": ["base-B", "base-Z"]}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = {a["name"] for a in body["artifacts"]}

        self.assertEqual(names, {"BetaModel"})

    def test_post_artifacts_base_model_filter_no_match(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"base_models[].model_id": "unknown-id"}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(body["artifacts"], [])

    def test_post_artifacts_eligibility_filter(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"eligibility.minimum_evidence_met": True}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = {a["name"] for a in body["artifacts"]}

        self.assertTrue({"BetaModel", "GammaModel"}.issubset(names))

    def test_post_artifacts_eligibility_false_filter(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={
                "artifact_queries": [
                    {"filters": {"eligibility.minimum_evidence_met": False}}
                ]
            },
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)
        names = {a["name"] for a in body["artifacts"]}

        self.assertEqual(names, {"AlphaModel"})

    def test_post_artifacts_rejects_non_list_queries(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={"artifact_queries": "not-a-list"},
        )
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_post_artifacts_non_object_body(self):
        event = self._event(
            "POST",
            "/artifacts",
            body='["bad"]',
        )
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_post_artifacts_invalid_json(self):
        event = self._event(
            "POST",
            "/artifacts",
            body="{invalid-json",
        )
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_post_artifacts_base64_body(self):
        payload = {"artifact_queries": [{"name": "beta"}]}
        event = self._event(
            "POST",
            "/artifacts",
            body=payload,
            base64_encode=True,
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(len(body["artifacts"]), 1)
        self.assertEqual(body["artifacts"][0]["name"], "BetaModel")

    def test_post_artifacts_repository_error(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={"artifact_queries": [{"name": "*"}]},
        )
        fake_repo = mock.Mock()
        fake_repo.search.side_effect = search.RepositoryError("boom")
        with mock.patch("src.search._get_repository", return_value=fake_repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 500)

    def test_get_artifact_by_name_path_param(self):
        event = self._event(
            "GET",
            "/artifact/byName",
            path_params={"name": "GammaModel"},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(body["artifacts"][0]["name"], "GammaModel")

    def test_get_artifact_by_name_suffix(self):
        event = self._event("GET", "/artifact/byName/BetaModel")
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(body["artifacts"][0]["model_id"], "model-beta")

    def test_get_artifact_by_name_not_found(self):
        event = self._event("GET", "/artifact/byName/Unknown")
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 404)

    def test_get_artifact_by_name_case_mismatch_not_found(self):
        event = self._event(
            "GET",
            "/artifact/byName",
            path_params={"name": "gammamodel"},
        )
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 404)

    def test_get_artifact_by_name_missing(self):
        event = self._event("GET", "/artifact/byName")
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_get_artifact_by_name_repository_error(self):
        event = self._event("GET", "/artifact/byName/BetaModel")
        fake_repo = mock.Mock()
        fake_repo.fetch_by_name.side_effect = search.RepositoryError("boom")
        with mock.patch("src.search._get_repository", return_value=fake_repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 500)

    def test_regex_search_matches_readme(self):
        event = self._event(
            "POST",
            "/artifact/byRegEx",
            body={"pattern": "quantum"},
        )
        response = search.handle_search(event, None)
        body = self._parse_body(response)

        self.assertEqual(len(body["artifacts"]), 1)
        self.assertEqual(body["artifacts"][0]["name"], "BetaModel")

    def test_regex_search_with_pagination_token(self):
        first = self._event(
            "POST",
            "/artifact/byRegEx",
            headers={"x-limit": "1"},
            body={"pattern": "model"},
        )
        response = search.handle_search(first, None)
        body = self._parse_body(response)
        token = response["headers"]["x-next-offset"]

        second = self._event(
            "POST",
            "/artifact/byRegEx",
            headers={"x-limit": "1", "x-offset": token},
            body={"pattern": "model"},
        )
        next_response = search.handle_search(second, None)
        next_body = self._parse_body(next_response)

        self.assertNotEqual(
            body["artifacts"][0]["model_id"],
            next_body["artifacts"][0]["model_id"],
        )

    def test_regex_search_repository_error(self):
        event = self._event(
            "POST",
            "/artifact/byRegEx",
            body={"pattern": "model"},
        )
        fake_repo = mock.Mock()
        fake_repo.regex_search.side_effect = search.RepositoryError("oops")
        with mock.patch("src.search._get_repository", return_value=fake_repo):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 500)

    def test_regex_search_missing_pattern(self):
        event = self._event("POST", "/artifact/byRegEx", body={})
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_regex_search_invalid_pattern(self):
        event = self._event(
            "POST",
            "/artifact/byRegEx",
            body={"pattern": "("},
        )
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 400)

    def test_regex_search_without_pagination_header(self):
        event = self._event(
            "POST",
            "/artifact/byRegEx",
            headers={"x-limit": "5"},
            body={"pattern": "Alpha"},
        )
        response = search.handle_search(event, None)
        self.assertNotIn("x-next-offset", response["headers"])

    def test_unknown_route_returns_404(self):
        event = self._event("DELETE", "/unknown")
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 404)

    def test_normalize_path_removes_stage_prefix(self):
        event = {
            "rawPath": "/prod/artifact/byName/Alpha",
            "requestContext": {"stage": "prod"},
        }
        normalized = search._normalize_path(event)
        self.assertEqual(normalized, "/artifact/byName/Alpha")

    def test_handle_search_uses_request_context_method(self):
        event = self._event("GET", "/artifact/byName/AlphaModel")
        event.pop("httpMethod")
        event["requestContext"] = {"http": {"method": "get"}}
        response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 200)

    def test_repository_error_bubbles_to_500(self):
        event = self._event(
            "POST",
            "/artifacts",
            body={"artifact_queries": [{"name": "*"}]},
        )

        class ExplodingRepo:
            def search(self, *_args, **_kwargs):
                raise search.RepositoryError("nope")

        with mock.patch("src.search._get_repository", return_value=ExplodingRepo()):
            response = search.handle_search(event, None)
        self.assertEqual(response["statusCode"], 500)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
