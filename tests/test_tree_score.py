import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.metrics.tree_score import TreeScoreMetric, _normalize_model_id
from src.config import ModelPaths, ModelURLs, extract_model_repo_id


ADAPTER_MODEL_URL = "https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb"


def _hf_info(card=None, config=None):
    return SimpleNamespace(cardData=card or {}, config=config or {})


class TestTreeScore(unittest.TestCase):
    def setUp(self):
        # TreeScoreMetric relies heavily on network calls; stub them out by default.
        self.hf_download_patch = patch(
            "src.metrics.tree_score.hf_hub_download",
            side_effect=AssertionError("TreeScore tests must not hit hf_hub_download."),
        )
        self.hf_download_patch.start()
        self.addCleanup(self.hf_download_patch.stop)

    def test_normalize_accepts_single_segment_ids(self):
        self.assertEqual(_normalize_model_id("bert-base-uncased"), "bert-base-uncased")
        self.assertEqual(
            _normalize_model_id("https://huggingface.co/bert-base-uncased"),
            "bert-base-uncased",
        )

    @patch("src.metrics.tree_score.model_info")
    def test_parents_detected_for_adapterhub_model(self, mock_model_info):
        mock_model_info.return_value = _hf_info(card={"base_model": "bert-base-uncased"})
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(ADAPTER_MODEL_URL)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        self.assertIn("bert-base-uncased", parents)

    @patch("src.metrics.tree_score.model_info")
    def test_no_parents_detected_for_base_model(self, mock_model_info):
        mock_model_info.return_value = _hf_info()
        base_model = "https://huggingface.co/bert-base-uncased"
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(base_model)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        self.assertEqual(len(parents), 0)

    @patch("src.metrics.tree_score.run_workflow")
    @patch.object(TreeScoreMetric, "_stage_base_metrics")
    @patch("src.metrics.tree_score.DownloadManager")
    @patch("src.metrics.tree_score.generate_model_paths")
    @patch.object(TreeScoreMetric, "_infer_parent_urls", return_value={})
    @patch.object(TreeScoreMetric, "_extract_parents_from_card", return_value={"adapter/base"})
    def test_treescore_runs_and_returns_score(
        self,
        mock_extract,
        mock_infer,
        mock_generate_paths,
        mock_download_manager,
        mock_stage_metrics,
        mock_run_workflow,
    ):
        mock_generate_paths.return_value = ModelPaths()
        mock_download_manager.return_value.download_model_resources.return_value = (
            None,
            None,
            None,
        )
        stub_metric = MagicMock(metric_name="size_score", score=0.8)
        mock_stage_metrics.return_value.metrics = [stub_metric]
        mock_run_workflow.return_value = SimpleNamespace(
            metrics=[stub_metric], score=0.8
        )

        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model=ADAPTER_MODEL_URL))
        score = metric.calculate_score()

        self.assertAlmostEqual(score, 0.8)
        mock_run_workflow.assert_called_once()

    @patch.object(TreeScoreMetric, "_extract_parents_from_card", return_value=set())
    def test_score_neutral_when_no_parents(self, _mock_extract):
        base_model = "https://huggingface.co/bert-base-uncased"
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model=base_model))
        score = metric.calculate_score()
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
