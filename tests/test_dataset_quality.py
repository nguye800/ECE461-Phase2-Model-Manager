import unittest
from src.metrics.dataset_quality import *
from src.metric import ModelURLs


class TestDatasetQuality(unittest.TestCase):

    def test_good_dataset(self):
        metric_instance = DatasetQualityMetric()
        metric_instance.set_url(
            ModelURLs(
                model="google-bert/bert-base-uncased",
                dataset="https://huggingface.co/datasets/bookcorpus/bookcorpus",
            )
        )
        metric_instance.run()
        self.assertIsInstance(metric_instance.score, float)
        if isinstance(metric_instance.score, dict):
            return

        # should be high-ish
        self.assertGreaterEqual(metric_instance.score, 0.8)

    def test_niche_dataset(self):
        metric_instance = DatasetQualityMetric()
        metric_instance.set_url(
            ModelURLs(
                model="google-bert/bert-base-uncased",
                dataset="https://huggingface.co/datasets/arjunpatel/best-selling-video-games",
            )
        )

        metric_instance.run()
        self.assertIsInstance(metric_instance.score, float)
        if isinstance(metric_instance.score, dict):
            return
        # should be pretty low
        self.assertLessEqual(metric_instance.score, 0.2)

    def test_zero_likes(self):
        metric_instance = DatasetQualityMetric()
        self.assertAlmostEqual(
            metric_instance.scale_logarithmically(0, 0, 10 * 10 ^ 3), 0.0
        )

    def test_no_model_url(self):
        metric_instance = DatasetQualityMetric()
        metric_instance.set_url(
            ModelURLs(model="https://huggingface.co/openai/whisper-tiny")
        )
        self.assertAlmostEqual(metric_instance.calculate_score(), 0.0)
