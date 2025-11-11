import unittest
from unittest.mock import patch

from src.config import ModelURLs
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
