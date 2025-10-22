import unittest
from pathlib import Path
import tempfile
import json
import types

from src.metrics.performance_claims import PerformanceClaimsMetric
from src.config import ModelPaths

class _StubResponse:
    def __init__(self, payload):
        self._payload = payload
    def json(self):
        return self._payload

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

    def test_calculate_score_parses_float_from_response(self):
        # Create README with some plausible content
        (self.model_dir / "README.md").write_text(
            "This model achieves 93.5% accuracy on CIFAR-10 and references arXiv:1234.5678.",
            encoding="utf-8",
        )
        self.metric.setup_resources()

        # Stub requests.post to return a valid-looking LLM response
        import src.metrics.performance_claims as perf_mod
        original_post = perf_mod.requests.post
        try:
            payload = {
                "choices": [
                    {"message": {"content": "Final score: 0.73\n"}}
                ]
            }
            perf_mod.requests.post = lambda *a, **k: _StubResponse(payload)
            score = self.metric.calculate_score()
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            # Should match parsed float
            self.assertAlmostEqual(score, 0.73, places=6)
        finally:
            perf_mod.requests.post = original_post

    def test_calculate_score_handles_non_numeric_response(self):
        (self.model_dir / "README.md").write_text("Some content", encoding="utf-8")
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_post = perf_mod.requests.post
        try:
            payload = {
                "choices": [
                    {"message": {"content": "no numeric score present"}}
                ]
            }
            perf_mod.requests.post = lambda *a, **k: _StubResponse(payload)
            score = self.metric.calculate_score()
            self.assertEqual(score, 0.0)
        finally:
            perf_mod.requests.post = original_post

    def test_calculate_score_uses_last_float_when_multiple_present(self):
        (self.model_dir / "README.md").write_text("Content", encoding="utf-8")
        self.metric.setup_resources()

        import src.metrics.performance_claims as perf_mod
        original_post = perf_mod.requests.post
        try:
            payload = {
                "choices": [
                    {"message": {"content": "intermediate: 0.12 final: 0.56"}}
                ]
            }
            perf_mod.requests.post = lambda *a, **k: _StubResponse(payload)
            score = self.metric.calculate_score()
            self.assertAlmostEqual(score, 0.56, places=6)
        finally:
            perf_mod.requests.post = original_post


if __name__ == "__main__":
    unittest.main()