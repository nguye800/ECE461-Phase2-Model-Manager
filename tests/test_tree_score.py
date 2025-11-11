import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.metrics.tree_score import TreeScoreMetric, _normalize_model_id
from src.config import ConfigContract, ModelPaths, ModelURLs, extract_model_repo_id


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

    def test_normalize_rejects_invalid_ids(self):
        self.assertIsNone(_normalize_model_id("bad id"))

    def test_normalize_handles_empty_string(self):
        self.assertIsNone(_normalize_model_id(""))

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

    @patch("src.metrics.tree_score.model_info")
    def test_parents_detected_from_list(self, mock_model_info):
        mock_model_info.return_value = _hf_info(card={"base_models": ["foo/bar", "baz/qux"]})
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(ADAPTER_MODEL_URL)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        self.assertIn("foo/bar", parents)

    @patch("src.metrics.tree_score.model_info")
    def test_parents_detected_from_config(self, mock_model_info):
        mock_model_info.return_value = _hf_info(config={"base_model": "foo/base"})
        metric = TreeScoreMetric()
        repo_id = extract_model_repo_id(ADAPTER_MODEL_URL)
        parents = metric._extract_parents_from_card(repo_id)  # type: ignore[attr-defined]
        self.assertIn("foo/base", parents)

    @patch("src.metrics.tree_score.model_info", side_effect=Exception("boom"))
    def test_extract_parents_handles_errors(self, mock_info):
        metric = TreeScoreMetric()
        parents = metric._extract_parents_from_card("owner/model")  # type: ignore[attr-defined]
        self.assertEqual(parents, set())

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

    def test_calculate_score_without_model_url(self):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model=None))
        self.assertEqual(metric.calculate_score(), 0.0)


class TestTreeScoreUtilities(unittest.TestCase):
    def setUp(self):
        self.metric = TreeScoreMetric()

    @patch("src.metrics.tree_score.model_info")
    @patch.object(TreeScoreMetric, "_fetch_readme_text", return_value="See https://github.com/foo/bar")
    def test_infer_parent_urls(self, mock_readme, mock_info):
        mock_info.return_value = _hf_info(card={"datasets": "datasets/user/set"})
        inferred = self.metric._infer_parent_urls("owner/repo")
        self.assertEqual(inferred["dataset"], "https://huggingface.co/datasets/user/set")
        self.assertEqual(inferred["codebase"], "https://github.com/foo/bar")

    @patch("src.metrics.tree_score.model_info", return_value=_hf_info(card={}))
    @patch.object(
        TreeScoreMetric,
        "_fetch_readme_text",
        return_value="Dataset docs at https://huggingface.co/datasets/readme/entry",
    )
    def test_infer_parent_urls_readme_absolute_dataset(self, mock_readme, mock_info):
        inferred = self.metric._infer_parent_urls("owner/repo")
        self.assertEqual(
            inferred["dataset"], "https://huggingface.co/datasets/readme/entry"
        )

    def test_fetch_readme_text_uses_download(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("text")
        with patch("src.metrics.tree_score.hf_hub_download", return_value=tmp.name):
            text = self.metric._fetch_readme_text("owner/repo")
        self.assertEqual(text, "text")

    @patch("src.metrics.tree_score.hf_hub_download", side_effect=Exception("missing"))
    def test_fetch_readme_text_handles_missing(self, mock_download):
        self.assertIsNone(self.metric._fetch_readme_text("owner/repo"))

    @patch("src.metrics.tree_score.model_info", return_value=_hf_info())
    def test_adapter_config_detection(self, mock_info):
        config = {
            "base_model": "foo/bar",
            "results": [{"task": {"source_model": "nested/base"}}],
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(config, tmp)
        with patch("src.metrics.tree_score.hf_hub_download", return_value=tmp.name):
            parents = self.metric._extract_parents_from_card("adapter/repo")
        self.assertIn("foo/bar", parents)
        self.assertIn("nested/base", parents)

    @patch("src.metrics.tree_score.model_info", side_effect=Exception("boom"))
    @patch.object(TreeScoreMetric, "_fetch_readme_text", return_value="")
    def test_infer_parent_urls_handles_model_info_error(self, mock_readme, mock_info):
        self.assertEqual(self.metric._infer_parent_urls("owner/repo"), {})

    @patch("src.metrics.tree_score.model_info")
    @patch.object(TreeScoreMetric, "_fetch_readme_text", return_value="")
    def test_infer_parent_urls_dataset_list(self, mock_readme, mock_info):
        mock_info.return_value = _hf_info(card={"datasets": ["datasets/u/v"]})
        inferred = self.metric._infer_parent_urls("owner/repo")
        self.assertEqual(
            inferred["dataset"], "https://huggingface.co/datasets/u/v"
        )

    @patch("src.metrics.tree_score.model_info", return_value=_hf_info(card={}, config={}))
    @patch.object(TreeScoreMetric, "_fetch_readme_text", return_value="(./datasets/foo/bar)")
    def test_infer_parent_urls_relative_dataset(self, mock_readme, mock_info):
        inferred = self.metric._infer_parent_urls("owner/repo")
        self.assertEqual(
            inferred["dataset"], "https://huggingface.co/datasets/foo/bar"
        )

    @patch("src.metrics.tree_score.MetricStager")
    def test_stage_base_metrics_respects_flags(self, mock_stager):
        with tempfile.TemporaryDirectory() as tmp:
            config = ConfigContract(
                num_processes=1,
                run_multi=False,
                priority_function="PFReciprocal",
                target_platform="pc",
                local_storage_directory=tmp,
                model_path_name="models",
                code_path_name="code",
                dataset_path_name="datasets",
            )
            instance = mock_stager.return_value
            instance.attach_metric.return_value = instance
            metric = TreeScoreMetric()
            metric._stage_base_metrics(
                config,
                include_bus_factor=True,
                include_dataset_quality=True,
                include_dataset_and_code=True,
                include_size=True,
                include_license=True,
                include_code_quality=True,
                include_performance_claims=True,
            )
            self.assertGreater(instance.attach_metric.call_count, 0)

    @patch("src.metrics.tree_score.MetricStager")
    def test_stage_base_metrics_handles_attach_errors(self, mock_stager):
        with tempfile.TemporaryDirectory() as tmp:
            config = ConfigContract(
                num_processes=1,
                run_multi=False,
                priority_function="PFReciprocal",
                target_platform="pc",
                local_storage_directory=tmp,
                model_path_name="models",
                code_path_name="code",
                dataset_path_name="datasets",
            )
            instance = mock_stager.return_value
            instance.attach_metric.side_effect = [
                Exception("size"),
                Exception("bus"),
                Exception("dataset_code"),
                Exception("dataset_quality"),
                Exception("license"),
                Exception("code_quality"),
                Exception("perf"),
            ]
            metric = TreeScoreMetric()
            metric._stage_base_metrics(
                config,
                include_bus_factor=True,
                include_dataset_quality=True,
                include_dataset_and_code=True,
                include_size=True,
                include_license=True,
                include_code_quality=True,
                include_performance_claims=True,
            )

    @patch("builtins.print")
    @patch("src.metrics.tree_score.hf_hub_download", side_effect=Exception("missing"))
    @patch("src.metrics.tree_score.run_workflow")
    @patch.object(TreeScoreMetric, "_stage_base_metrics")
    @patch("src.metrics.tree_score.DownloadManager")
    @patch("src.metrics.tree_score.generate_model_paths")
    @patch.object(TreeScoreMetric, "_infer_parent_urls")
    @patch.object(TreeScoreMetric, "_extract_parents_from_card")
    def test_calculate_score_handles_download_errors(
        self,
        mock_extract,
        mock_infer,
        mock_generate,
        mock_dm,
        mock_stage,
        mock_run,
        mock_hf,
        mock_print,
    ):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/child/model"))
        mock_extract.return_value = {"parent/one", "parent/two"}
        mock_infer.side_effect = [
            {"dataset": "https://huggingface.co/datasets/ds", "codebase": "https://github.com/code/repo"},
            {},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            mock_generate.return_value = ModelPaths(model=Path(tmp) / "model")
        dm_instance = MagicMock()
        mock_dm.return_value = dm_instance
        dm_instance.download_model_resources.side_effect = [
            (None, Path("."), None),
            Exception("boom"),
        ]
        stage_obj = MagicMock()
        stage_obj.metrics = [MagicMock(metric_name="size_score", score=0.4)]
        mock_stage.return_value = stage_obj
        class BrokenMetric:
            metric_name = "broken"

            @property
            def score(self):
                raise RuntimeError("bad score")

        mock_run.return_value = SimpleNamespace(
            metrics=[BrokenMetric()],
            score=0.4,
        )

        with patch("pathlib.Path.mkdir", side_effect=Exception("mkdir boom")):
            score = metric.calculate_score()
        self.assertAlmostEqual(score, 0.4)

    @patch("builtins.print")
    @patch("src.metrics.tree_score.hf_hub_download")
    @patch("src.metrics.tree_score.run_workflow")
    @patch.object(TreeScoreMetric, "_stage_base_metrics")
    @patch("src.metrics.tree_score.DownloadManager")
    @patch("src.metrics.tree_score.generate_model_paths")
    @patch.object(TreeScoreMetric, "_infer_parent_urls", return_value={})
    @patch.object(TreeScoreMetric, "_extract_parents_from_card", return_value={"parent/license"})
    def test_calculate_score_sets_license_ready(
        self,
        mock_parents,
        mock_infer,
        mock_generate,
        mock_dm,
        mock_stage,
        mock_run,
        mock_hf,
        mock_print,
    ):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/child"))
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model"
            mock_generate.return_value = ModelPaths(model=model_path)
            temp_file = Path(tmp) / "README.md"
            temp_file.write_text("contents", encoding="utf-8")
            mock_hf.return_value = str(temp_file)
            dm_instance = MagicMock()
            mock_dm.return_value = dm_instance
            dm_instance.download_model_resources.return_value = (None, None, None)
            stage_obj = MagicMock()
            stage_obj.metrics = [MagicMock(metric_name="size_score", score=0.9)]
            mock_stage.return_value = stage_obj
            mock_run.return_value = SimpleNamespace(metrics=stage_obj.metrics, score=0.9)
            score = metric.calculate_score()
        self.assertAlmostEqual(score, 0.9)

    @patch("src.metrics.tree_score.extract_model_repo_id", side_effect=Exception("bad"))
    def test_calculate_score_invalid_repo(self, mock_extract):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model="bad"))
        self.assertEqual(metric.calculate_score(), 0.0)

    @patch("builtins.print")
    @patch("src.metrics.tree_score.run_workflow")
    @patch("src.metrics.tree_score.DownloadManager")
    @patch("src.metrics.tree_score.generate_model_paths", return_value=ModelPaths())
    @patch.object(TreeScoreMetric, "_infer_parent_urls", return_value={})
    @patch.object(TreeScoreMetric, "_stage_base_metrics", side_effect=RuntimeError("stage boom"))
    @patch.object(TreeScoreMetric, "_extract_parents_from_card", return_value={"parent/only"})
    def test_calculate_score_handles_stage_failure(
        self,
        mock_parents,
        mock_stage,
        mock_infer,
        mock_gen,
        mock_dm,
        mock_run,
        mock_print,
    ):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/child"))
        score = metric.calculate_score()
        self.assertEqual(score, 0.0)

    @patch("builtins.print")
    @patch("src.metrics.tree_score.hf_hub_download", side_effect=Exception("missing"))
    @patch("src.metrics.tree_score.run_workflow")
    @patch.object(TreeScoreMetric, "_stage_base_metrics")
    @patch("src.metrics.tree_score.DownloadManager")
    @patch("src.metrics.tree_score.generate_model_paths")
    @patch.object(TreeScoreMetric, "_infer_parent_urls", return_value={})
    @patch.object(TreeScoreMetric, "_extract_parents_from_card", return_value={"parent/three"})
    def test_calculate_score_readme_download_failures(
        self,
        mock_parents,
        mock_infer,
        mock_generate,
        mock_dm,
        mock_stage,
        mock_run,
        mock_hf,
        mock_print,
    ):
        metric = TreeScoreMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/child"))
        with tempfile.TemporaryDirectory() as tmp:
            mock_generate.return_value = ModelPaths(model=Path(tmp) / "model")
            dm_instance = MagicMock()
            mock_dm.return_value = dm_instance
            dm_instance.download_model_resources.return_value = (None, None, None)
            stage_obj = MagicMock()
            stage_obj.metrics = [MagicMock(metric_name="size_score", score=0.3)]
            mock_stage.return_value = stage_obj
            mock_run.return_value = SimpleNamespace(metrics=stage_obj.metrics, score=0.3)
            metric.calculate_score()


if __name__ == "__main__":
    unittest.main()
