import unittest
from pathlib import Path
import tempfile
import json

from src.metrics.performance_claims import PerformanceClaimsMetric
from src.config import ModelPaths


class _StubBedrockClient:
    def __init__(self, payload):
        self._payload = payload

    def invoke_model(self, **kwargs):
        return {"body": json.dumps(self._payload)}


class TestPerformanceClaimsMetric(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmpdir.name) / "modelA"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metric = PerformanceClaimsMetric()
        dirs = ModelPaths()
        dirs.model = self.model_dir
        self.metric.set_local_directory(dirs)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_setup_resources_sets_paths(self):
        self.metric.setup_resources()
        self.assertEqual(self.metric.model_dir, self.model_dir)
        self.assertEqual(self.metric.readme_file, self.model_dir / "README.md")

    def test_calculate_score_no_readme_returns_zero(self):
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_parses_json_from_bedrock(self):
        (self.model_dir / "README.md").write_text(
            "Model achieves 93.5% accuracy on CIFAR-10 and references arXiv:1234.5678.",
            encoding="utf-8",
        )
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_client = perf_mod.BEDROCK_CLIENT
        perf_mod.BEDROCK_CLIENT = _StubBedrockClient(
            {
                "content": [
                    {
                        "type": "text",
                        "text": '{"score": 0.73, "explanation": "well documented"}',
                    }
                ]
            }
        )
        try:
            score = self.metric.calculate_score()
            self.assertAlmostEqual(score, 0.73, places=6)
        finally:
            perf_mod.BEDROCK_CLIENT = original_client

    def test_calculate_score_handles_non_numeric_response(self):
        (self.model_dir / "README.md").write_text("Some content", encoding="utf-8")
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_client = perf_mod.BEDROCK_CLIENT
        perf_mod.BEDROCK_CLIENT = _StubBedrockClient(
            {"content": [{"type": "text", "text": "no numeric score present"}]}
        )
        try:
            score = self.metric.calculate_score()
            self.assertEqual(score, 0.0)
        finally:
            perf_mod.BEDROCK_CLIENT = original_client

    def test_calculate_score_uses_last_float_when_multiple_present(self):
        (self.model_dir / "README.md").write_text("Content", encoding="utf-8")
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_client = perf_mod.BEDROCK_CLIENT
        perf_mod.BEDROCK_CLIENT = _StubBedrockClient(
            {"content": [{"type": "text", "text": "intermediate 0.12 final 0.56"}]}
        )
        try:
            score = self.metric.calculate_score()
            self.assertAlmostEqual(score, 0.56, places=6)
        finally:
            perf_mod.BEDROCK_CLIENT = original_client

    def test_heuristic_fallback_used_when_no_client(self):
        (self.model_dir / "README.md").write_text(
            "Benchmark accuracy 93.5% compared to baseline; see arxiv 1234.5678.",
            encoding="utf-8",
        )
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_client = perf_mod.BEDROCK_CLIENT
        perf_mod.BEDROCK_CLIENT = None
        try:
            score = self.metric.calculate_score()
            self.assertGreater(score, 0.0)
        finally:
            perf_mod.BEDROCK_CLIENT = original_client


if __name__ == "__main__":
    unittest.main()
