import io
import json
import unittest
from unittest.mock import patch

from src.config import ModelURLs
from src.metrics.ramp_up_time import RampUpMetric


class BedrockStub:
    def __init__(self, scores):
        self._scores = list(scores)

    def invoke_model(self, **kwargs):
        if not self._scores:
            raise RuntimeError("No further scores available")
        score = self._scores.pop(0)
        payload = json.dumps(
            {"content": [{"type": "text", "text": json.dumps({"score": score})}]}
        ).encode("utf-8")
        return {"body": io.BytesIO(payload)}


class TestRampUpMetric(unittest.TestCase):
    def setUp(self):
        self.metric = RampUpMetric()
        self.metric.set_url(
            ModelURLs(
                model="https://huggingface.co/acme/demo-model",
                codebase="https://github.com/acme/demo-repo",
            )
        )

    @patch.object(
        RampUpMetric,
        "_load_codebase_readme",
        return_value="Install deps with pip and call DemoModel.predict().",
    )
    @patch.object(
        RampUpMetric,
        "_load_model_card_text",
        return_value="DemoModel is a classifier. To integrate, call pipeline().",
    )
    def test_average_bedrock_scores(self, mock_model, mock_code):
        self.metric.bedrock_client = BedrockStub([0.9, 0.5])
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertAlmostEqual(score, 0.7)

    @patch.object(
        RampUpMetric,
        "_load_codebase_readme",
        return_value="Detailed instructions with code samples " * 40,
    )
    @patch.object(
        RampUpMetric,
        "_load_model_card_text",
        return_value="Step-by-step installation guide " * 40,
    )
    def test_fallback_without_bedrock(self, mock_model, mock_code):
        self.metric.bedrock_client = None
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertGreater(score, 0.0)

    @patch.object(RampUpMetric, "_load_codebase_readme", return_value="")
    @patch.object(RampUpMetric, "_load_model_card_text", return_value="")
    def test_no_docs_returns_zero(self, mock_model, mock_code):
        self.metric.setup_resources()
        score = self.metric.calculate_score()
        self.assertEqual(score, 0.0)
