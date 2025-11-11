import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.config import ModelPaths, ModelURLs
from src.metrics.ramp_up_time import RampUpMetric


class TestRampUpMetric(unittest.TestCase):
    def setUp(self):
        self.metric = RampUpMetric()
        self.metric.set_url(
            ModelURLs(
                model="https://huggingface.co/acme/demo-model",
                codebase="https://github.com/acme/demo-repo",
            )
        )
        self._requests_patch = patch(
            "src.metrics.ramp_up_time.requests.get",
            side_effect=AssertionError("RampUpMetric tests must not make HTTP calls."),
        )
        self._requests_patch.start()
        self.addCleanup(self._requests_patch.stop)

    @patch.object(
        RampUpMetric,
        "_load_model_card_text",
        return_value="DemoModel is a classifier. To integrate, call pipeline().",
    )
    @patch.object(
        RampUpMetric,
        "_load_codebase_readme",
        return_value="Install deps with pip and call DemoModel.predict().",
    )
    @patch.object(RampUpMetric, "_invoke_bedrock", side_effect=[0.9, 0.5])
    def test_average_bedrock_scores(self, mock_bedrock, mock_code, mock_model):
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertAlmostEqual(score, 0.7)
        self.assertEqual(mock_bedrock.call_count, 2)

    @patch.object(
        RampUpMetric,
        "_load_model_card_text",
        return_value="Step-by-step installation guide " * 40,
    )
    @patch.object(
        RampUpMetric,
        "_load_codebase_readme",
        return_value="Detailed instructions with code samples " * 40,
    )
    @patch.object(
        RampUpMetric,
        "_invoke_bedrock",
        side_effect=RuntimeError("Bedrock unavailable"),
    )
    def test_fallback_without_bedrock(self, mock_bedrock, mock_code, mock_model):
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertGreater(score, 0.0)
        self.assertEqual(mock_bedrock.call_count, 2)

    @patch.object(RampUpMetric, "_load_model_card_text", return_value="")
    @patch.object(RampUpMetric, "_load_codebase_readme", return_value="")
    @patch.object(RampUpMetric, "_invoke_bedrock")
    def test_no_docs_returns_zero(self, mock_bedrock, mock_code, mock_model):
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertEqual(score, 0.0)
        mock_bedrock.assert_not_called()


class TestRampUpMetricInternals(unittest.TestCase):
    def setUp(self):
        self.metric = RampUpMetric()

    def test_parse_score_from_json(self):
        self.assertEqual(self.metric._parse_score('{"score": 0.42}'), 0.42)

    def test_parse_score_from_plain_text(self):
        self.assertEqual(self.metric._parse_score("score: 0.37/1.0"), 0.37)

    def test_parse_score_empty_text(self):
        self.assertIsNone(self.metric._parse_score(""))

    def test_heuristic_score_structure_bonus(self):
        text = "# Heading\n```code sample```"
        score = self.metric._heuristic_score(text)
        self.assertGreater(score, self.metric._heuristic_score("plain text"))

    def test_invoke_bedrock_with_stream_body(self):
        payload = {"content": [{"type": "text", "text": '{"score":0.9}'}]}
        body = io.BytesIO(json.dumps(payload).encode("utf-8"))
        stub_client = MagicMock()
        stub_client.invoke_model.return_value = {"body": body}
        self.metric.bedrock_client = stub_client
        score = self.metric._invoke_bedrock("prompt")
        self.assertEqual(score, 0.9)

    def test_invoke_bedrock_with_string_body(self):
        payload = {"content": [{"type": "text", "text": "0.5"}]}
        stub_client = MagicMock()
        stub_client.invoke_model.return_value = {"body": json.dumps(payload)}
        self.metric.bedrock_client = stub_client
        score = self.metric._invoke_bedrock("prompt")
        self.assertEqual(score, 0.5)

    def test_invoke_bedrock_reads_output_field(self):
        payload = {"output": [{"content": [{"type": "text", "text": "0.3"}]}]}
        stub_client = MagicMock()
        stub_client.invoke_model.return_value = {"body": io.BytesIO(json.dumps(payload).encode())}
        self.metric.bedrock_client = stub_client
        score = self.metric._invoke_bedrock("prompt")
        self.assertEqual(score, 0.3)

    def test_score_text_falls_back_to_heuristic(self):
        text = "instructions " * 500
        with patch.object(self.metric, "_invoke_bedrock", side_effect=RuntimeError("boom")):
            score = self.metric._score_text(text, "model card")
        self.assertGreater(score, 0.0)

    def test_heuristic_score_zero_for_empty(self):
        self.assertEqual(self.metric._heuristic_score(""), 0.0)

    def test_read_local_code_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            (code_dir / "README.md").write_text("hello", encoding="utf-8")
            self.metric.local_directory = ModelPaths(codebase=code_dir)
            content = self.metric._read_local_code_readme()
            self.assertEqual(content, "hello")

    def test_read_local_code_readme_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            (code_dir / "README_extra.txt").write_text("extra", encoding="utf-8")
            self.metric.local_directory = ModelPaths(codebase=code_dir)
            self.assertEqual(self.metric._read_local_code_readme(), "extra")

    def test_load_codebase_readme_prefers_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            (code_dir / "README.md").write_text("local", encoding="utf-8")
            self.metric.local_directory = ModelPaths(codebase=code_dir)
            self.assertEqual(self.metric._load_codebase_readme(), "local")

    def test_load_codebase_readme_remote(self):
        self.metric.local_directory = ModelPaths(codebase=None)
        self.metric.url = ModelURLs(codebase="https://github.com/foo/bar")
        with patch.object(self.metric, "_fetch_remote_code_readme", return_value="remote"):
            self.assertEqual(self.metric._load_codebase_readme(), "remote")

    def test_fetch_remote_code_readme_prefers_github(self):
        self.metric.url = ModelURLs(codebase="https://github.com/owner/repo")
        with patch.object(self.metric, "_safe_http_get", return_value="remote readme") as mock_get:
            content = self.metric._fetch_remote_code_readme()
        self.assertEqual(content, "remote readme")
        self.assertTrue(mock_get.called)

    def test_fetch_remote_code_readme_direct(self):
        self.metric.url = ModelURLs(codebase="https://example.com/readme")
        with patch.object(self.metric, "_safe_http_get", return_value="direct"):
            self.assertEqual(self.metric._fetch_remote_code_readme(), "direct")

    @patch("src.metrics.ramp_up_time.requests.get")
    def test_safe_http_get_handles_errors(self, mock_get):
        mock_get.side_effect = Exception("fail")
        self.assertIsNone(self.metric._safe_http_get("https://example.com"))

    @patch("src.metrics.ramp_up_time.requests.get")
    def test_safe_http_get_success(self, mock_get):
        resp = MagicMock()
        resp.text = "body"
        mock_get.return_value = resp
        self.assertEqual(self.metric._safe_http_get("https://example.com"), "body")

    def test_candidate_model_card_urls(self):
        urls = self.metric._candidate_model_card_urls("owner", "repo")
        self.assertIn("https://huggingface.co/owner/repo/raw/main/README.md", urls)

    def test_extract_hf_repo_variants(self):
        owner, repo = self.metric._extract_hf_repo("https://huggingface.co/foo/bar")
        self.assertEqual((owner, repo), ("foo", "bar"))
        owner, repo = self.metric._extract_hf_repo("foo/bar")
        self.assertEqual((owner, repo), ("foo", "bar"))

    def test_extract_github_repo(self):
        self.assertEqual(
            self.metric._extract_github_repo("https://github.com/user/repo"),
            ("user", "repo"),
        )

    def test_load_model_card_text_candidate_cycle(self):
        self.metric.url = ModelURLs(model="https://huggingface.co/owner/repo")
        with patch.object(self.metric, "_safe_http_get", side_effect=[None, "card"]):
            self.assertEqual(self.metric._load_model_card_text(), "card")

    def test_load_model_card_text_direct_fetch(self):
        self.metric.url = ModelURLs(model="https://example.com/model")
        with patch.object(self.metric, "_safe_http_get", return_value="direct"):
            self.assertEqual(self.metric._load_model_card_text(), "direct")
