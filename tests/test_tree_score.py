import os
import unittest

from src.metrics.tree_score import TreeScoreMetric, _normalize_model_id
from src.config import extract_model_repo_id
from src.metric import ModelURLs


ADAPTER_MODEL_URL = "https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb"


class TestTreeScore(unittest.TestCase):
    def test_normalize_accepts_single_segment_ids(self):
        self.assertEqual(_normalize_model_id("bert-base-uncased"), "bert-base-uncased")
        self.assertEqual(
            _normalize_model_id("https://huggingface.co/bert-base-uncased"),
            "bert-base-uncased",
        )

    def test_parents_detected_for_adapterhub_model(self):
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(ADAPTER_MODEL_URL)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        # Expect at least one parent (AdapterHub cards typically reference base model)
        self.assertIsInstance(parents, set)
        self.assertGreaterEqual(len(parents), 1)

    def test_treescore_runs_and_returns_score(self):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model=ADAPTER_MODEL_URL))
        score = metric.run().score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_no_parents_detected_for_base_model(self):
        base_model = "https://huggingface.co/bert-base-uncased"
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(base_model)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        # Expect zero parents for a base model repo
        self.assertIsInstance(parents, set)
        self.assertEqual(len(parents), 0)

    def test_score_neutral_when_no_parents(self):
        base_model = "https://huggingface.co/bert-base-uncased"
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model=base_model))
        score = metric.run().score
        # Neutral score (no penalty) when no parents are found
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
