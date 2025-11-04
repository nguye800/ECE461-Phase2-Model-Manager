import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import os
from huggingface_hub import snapshot_download
from src.download_manager import DownloadManager
from src.metric import ModelURLs


class TestDownloadManager(unittest.TestCase):
    """Test DownloadManager in isolation using mocks"""

    def setUp(self):
        """Set up test fixtures before each test"""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.test_dir) / "models"
        self.codebases_dir = Path(self.test_dir) / "codebases"
        self.datasets_dir = Path(self.test_dir) / "datasets"

        self.dm = DownloadManager(
            models_dir=str(self.models_dir),
            codebases_dir=str(self.codebases_dir),
            datasets_dir=str(self.datasets_dir),
        )

    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.test_dir)

    def test_init_creates_directories(self):
        """Test that __init__ creates model, codebase, and dataset directories"""
        self.assertTrue(self.models_dir.exists())
        self.assertTrue(self.codebases_dir.exists())
        self.assertTrue(self.datasets_dir.exists())

    def test_extract_repo_id_standard_url(self):
        """Test extracting repo ID from standard HuggingFace URL"""
        url = "https://huggingface.co/bert-base-uncased"
        result = self.dm._extract_repo_id(url)
        self.assertEqual(result, "bert-base-uncased")

    def test_extract_repo_id_with_tree_main(self):
        """Test extracting repo ID from URL with /tree/main"""
        url = "https://huggingface.co/username/model-name/tree/main"
        result = self.dm._extract_repo_id(url)
        self.assertEqual(result, "username/model-name")

    def test_extract_repo_id_with_blob_main(self):
        """Test extracting repo ID from URL with /blob/main"""
        url = "https://huggingface.co/username/model-name/blob/main/README.md"
        result = self.dm._extract_repo_id(url)
        self.assertEqual(result, "username/model-name")

    def test_extract_repo_id_dataset_url(self):
        """Test extracting repo ID from dataset URL"""
        url = "https://huggingface.co/datasets/username/dataset-name"
        result = self.dm._extract_repo_id(url)
        self.assertEqual(result, "username/dataset-name")

    def test_extract_repo_id_with_trailing_slash(self):
        """Test extracting repo ID with trailing slash"""
        url = "https://huggingface.co/username/model/"
        result = self.dm._extract_repo_id(url)
        self.assertEqual(result, "username/model")

    def test_extract_repo_name_standard_github_url(self):
        """Test extracting repo name from GitHub URL"""
        url = "https://github.com/username/repo-name"
        result = self.dm._extract_repo_name(url)
        self.assertEqual(result, "repo-name")

    def test_extract_repo_name_with_git_extension(self):
        """Test extracting repo name with .git extension"""
        url = "https://github.com/username/repo-name.git"
        result = self.dm._extract_repo_name(url)
        self.assertEqual(result, "repo-name")

    def test_extract_repo_name_with_trailing_slash(self):
        """Test extracting repo name with trailing slash"""
        url = "https://github.com/username/repo-name/"
        result = self.dm._extract_repo_name(url)
        self.assertEqual(result, "repo-name")

    @patch("src.download_manager.snapshot_download")
    def test_download_model_success(self, mock_snapshot):
        """Test successful model download"""
        model_url = "https://huggingface.co/bert-base-uncased"

        result = self.dm.download_model(model_url)

        # Check snapshot_download was called correctly
        mock_snapshot.assert_called_once_with(
            repo_id="bert-base-uncased",
            local_dir=str(self.models_dir / "bert-base-uncased"),
            revision="main",
            force_download=True,
            force_download=False,
        )

        # Check return value
        self.assertEqual(result, self.models_dir / "bert-base-uncased")

    @patch("src.download_manager.snapshot_download")
    def test_download_model_already_exists_updates(self, mock_snapshot):
        """Test that existing model is updated (not skipped)"""
        model_url = "https://huggingface.co/bert-base-uncased"
        local_path = self.models_dir / "bert-base-uncased"
        local_path.mkdir(parents=True)

        result = self.dm.download_model(model_url)

        # SHOULD call snapshot_download to update
        mock_snapshot.assert_called_once_with(
            repo_id="bert-base-uncased",
            local_dir=str(local_path),
            revision="main",
            force_download=True,
            force_download=False,
        )

        # Should return path
        self.assertEqual(result, local_path)

    @patch("src.download_manager.shutil.rmtree")
    @patch("src.download_manager.snapshot_download")
    def test_download_model_update_failure_redownloads(
        self, mock_snapshot, mock_rmtree
    ):
        """Test model re-download on update failure"""
        model_url = "https://huggingface.co/bert-base-uncased"
        local_path = self.models_dir / "bert-base-uncased"
        local_path.mkdir(parents=True)

        # First call fails, second succeeds
        mock_snapshot.side_effect = [Exception("Update failed"), None]

        result = self.dm.download_model(model_url)

        # Should remove directory and retry
        mock_rmtree.assert_called_once_with(local_path)

        # Should call snapshot_download twice
        self.assertEqual(mock_snapshot.call_count, 2)

        # Check return value
        self.assertEqual(result, local_path)

    @patch("src.download_manager.snapshot_download")
    def test_download_model_failure(self, mock_snapshot):
        """Test model download failure handling"""
        model_url = "https://huggingface.co/nonexistent/model"
        mock_snapshot.side_effect = Exception("Download failed")

        with self.assertRaises(Exception) as context:
            self.dm.download_model(model_url)
            self.assertIn("Download failed", str(context.exception))

    @patch("src.download_manager.snapshot_download")
    def test_download_dataset_success(self, mock_snapshot):
        """Test successful dataset download"""
        dataset_url = "https://huggingface.co/datasets/squad"

        result = self.dm.download_dataset(dataset_url)

        # Check snapshot_download was called correctly with repo_type="dataset"
        mock_snapshot.assert_called_once_with(
            repo_id="squad",
            repo_type="dataset",
            local_dir=str(self.datasets_dir / "squad"),
            revision="main",
            force_download=True,
            force_download=False,
        )

        # Check return value
        self.assertEqual(result, self.datasets_dir / "squad")

    @patch("src.download_manager.snapshot_download")
    def test_download_dataset_already_exists_updates(self, mock_snapshot):
        """Test that existing dataset is updated (not skipped)"""
        dataset_url = "https://huggingface.co/datasets/squad"
        local_path = self.datasets_dir / "squad"
        local_path.mkdir(parents=True)

        result = self.dm.download_dataset(dataset_url)

        # SHOULD call snapshot_download to update
        mock_snapshot.assert_called_once_with(
            repo_id="squad",
            repo_type="dataset",
            local_dir=str(local_path),
            revision="main",
            force_download=True,
            force_download=False,
        )

        # Should return path
        self.assertEqual(result, local_path)

    @patch("src.download_manager.shutil.rmtree")
    @patch("src.download_manager.snapshot_download")
    def test_download_dataset_update_failure_redownloads(
        self, mock_snapshot, mock_rmtree
    ):
        """Test dataset re-download on update failure"""
        dataset_url = "https://huggingface.co/datasets/squad"
        local_path = self.datasets_dir / "squad"
        local_path.mkdir(parents=True)

        # First call fails, second succeeds
        mock_snapshot.side_effect = [Exception("Update failed"), None]

        result = self.dm.download_dataset(dataset_url)

        # Should remove directory and retry
        mock_rmtree.assert_called_once_with(local_path)

        # Should call snapshot_download twice
        self.assertEqual(mock_snapshot.call_count, 2)

        # Check return value
        self.assertEqual(result, local_path)

    @patch("src.download_manager.git.Repo")
    def test_download_codebase_success(self, mock_repo_class):
        """Test successful codebase clone"""
        code_url = "https://github.com/username/repo"

        result = self.dm.download_codebase(code_url)

        # Check clone_from was called correctly
        mock_repo_class.clone_from.assert_called_once_with(
            code_url, self.codebases_dir / "repo"
        )

        # Check return value
        self.assertEqual(result, self.codebases_dir / "repo")

    @patch("src.download_manager.git.Repo")
    def test_download_codebase_already_exists_updates(self, mock_repo_class):
        """Test that existing codebase is updated with hard reset"""
        code_url = "https://github.com/username/repo"
        local_path = self.codebases_dir / "repo"
        local_path.mkdir(parents=True)

        # Mock the repo instance
        mock_repo_instance = MagicMock()
        mock_repo_instance.active_branch.name = "main"
        mock_repo_class.return_value = mock_repo_instance

        result = self.dm.download_codebase(code_url)

        # Should call Repo() to open existing repo
        mock_repo_class.assert_called_once_with(local_path)

        # Should fetch and reset
        mock_repo_instance.remotes.origin.fetch.assert_called_once()
        mock_repo_instance.git.reset.assert_called_once_with("--hard", "origin/main")

        # Should NOT call clone_from
        mock_repo_class.clone_from.assert_not_called()

        self.assertEqual(result, local_path)

    @patch("src.download_manager.shutil.rmtree")
    @patch("src.download_manager.git.Repo")
    def test_download_codebase_update_failure_reclones(
        self, mock_repo_class, mock_rmtree
    ):
        """Test that update failure causes re-clone"""
        code_url = "https://github.com/username/repo"
        local_path = self.codebases_dir / "repo"
        local_path.mkdir(parents=True)

        # Mock failed update
        mock_repo_instance = MagicMock()
        mock_repo_instance.remotes.origin.fetch.side_effect = Exception("Fetch failed")
        mock_repo_class.side_effect = [
            mock_repo_instance,
            None,
        ]  # First call fails, clone_from succeeds

        result = self.dm.download_codebase(code_url)

        # Should remove directory
        mock_rmtree.assert_called_once_with(local_path)

        # Should call clone_from after failure
        mock_repo_class.clone_from.assert_called_once_with(code_url, local_path)

        self.assertEqual(result, local_path)

    @patch("src.download_manager.git.Repo")
    def test_download_codebase_clone_failure(self, mock_repo_class):
        """Test codebase clone failure handling"""
        code_url = "https://github.com/username/repo"
        mock_repo_class.clone_from.side_effect = Exception("Clone failed")

        with self.assertRaises(Exception) as context:
            self.dm.download_codebase(code_url)

        self.assertIn("Clone failed", str(context.exception))

    def test_check_local_model_exists(self):
        """Test checking for existing local model"""
        model_url = "https://huggingface.co/bert-base-uncased"
        local_path = self.models_dir / "bert-base-uncased"
        local_path.mkdir(parents=True)

        result = self.dm.check_local_model(model_url)

        self.assertEqual(result, local_path)

    def test_check_local_model_not_exists(self):
        """Test checking for non-existent local model"""
        model_url = "https://huggingface.co/bert-base-uncased"

        result = self.dm.check_local_model(model_url)

        self.assertIsNone(result)

    def test_check_local_dataset_exists(self):
        """Test checking for existing local dataset"""
        dataset_url = "https://huggingface.co/datasets/squad"
        local_path = self.datasets_dir / "squad"
        local_path.mkdir(parents=True)

        result = self.dm.check_local_dataset(dataset_url)

        self.assertEqual(result, local_path)

    def test_check_local_dataset_not_exists(self):
        """Test checking for non-existent local dataset"""
        dataset_url = "https://huggingface.co/datasets/squad"

        result = self.dm.check_local_dataset(dataset_url)

        self.assertIsNone(result)

    def test_check_local_codebase_exists(self):
        """Test checking for existing local codebase"""
        code_url = "https://github.com/username/repo"
        local_path = self.codebases_dir / "repo"
        local_path.mkdir(parents=True)

        result = self.dm.check_local_codebase(code_url)

        self.assertEqual(result, local_path)

    def test_check_local_codebase_not_exists(self):
        """Test checking for non-existent local codebase"""
        code_url = "https://github.com/username/repo"

        result = self.dm.check_local_codebase(code_url)

        self.assertIsNone(result)

    @patch("src.download_manager.snapshot_download")
    @patch("src.download_manager.git.Repo")
    def test_download_model_resources_all(self, mock_git, mock_snapshot):
        """Test downloading model, codebase, and dataset"""
        model_urls = ModelURLs(
            model="https://huggingface.co/bert-base-uncased",
            codebase="https://github.com/username/repo",
            dataset="https://huggingface.co/datasets/squad",
        )

        model_path, codebase_path, dataset_path = self.dm.download_model_resources(
            model_urls
        )

        # Check all were attempted
        self.assertEqual(mock_snapshot.call_count, 2)  # model and dataset
        mock_git.clone_from.assert_called_once()

        # Check return values
        self.assertEqual(model_path, self.models_dir / "bert-base-uncased")
        self.assertEqual(codebase_path, self.codebases_dir / "repo")
        self.assertEqual(dataset_path, self.datasets_dir / "squad")

    @patch("src.download_manager.snapshot_download")
    def test_download_model_resources_model_only(self, mock_snapshot):
        """Test downloading only model (no codebase or dataset URL)"""
        model_urls = ModelURLs(
            model="https://huggingface.co/bert-base-uncased",
            codebase=None,
            dataset=None,
        )

        model_path, codebase_path, dataset_path = self.dm.download_model_resources(
            model_urls
        )

        # Check model was downloaded
        mock_snapshot.assert_called_once()

        # Check return values
        self.assertEqual(model_path, self.models_dir / "bert-base-uncased")
        self.assertIsNone(codebase_path)
        self.assertIsNone(dataset_path)

    @patch("src.download_manager.git.Repo")
    def test_download_model_resources_codebase_only(self, mock_git):
        """Test downloading only codebase (no model or dataset download)"""
        model_urls = ModelURLs(
            model="https://huggingface.co/bert-base-uncased",
            codebase="https://github.com/username/repo",
            dataset=None,
        )

        model_path, codebase_path, dataset_path = self.dm.download_model_resources(
            model_urls,
            download_model=False,
            download_codebase=True,
            download_dataset=False,
        )

        # Check codebase was downloaded
        mock_git.clone_from.assert_called_once()

        # Check return values
        self.assertIsNone(model_path)
        self.assertEqual(codebase_path, self.codebases_dir / "repo")
        self.assertIsNone(dataset_path)

    @patch("src.download_manager.snapshot_download")
    def test_download_model_resources_dataset_only(self, mock_snapshot):
        """Test downloading only dataset"""
        model_urls = ModelURLs(
            model="shouldn't matter",
            codebase=None,
            dataset="https://huggingface.co/datasets/squad",
        )

        model_path, codebase_path, dataset_path = self.dm.download_model_resources(
            model_urls,
            download_model=False,
            download_codebase=False,
            download_dataset=True,
        )

        # Check dataset was downloaded
        mock_snapshot.assert_called_once_with(
            repo_id="squad",
            repo_type="dataset",
            local_dir=str(self.datasets_dir / "squad"),
            revision="main",
            force_download=True,
            force_download=False,
        )

        # Check return values
        self.assertIsNone(model_path)
        self.assertIsNone(codebase_path)
        self.assertEqual(dataset_path, self.datasets_dir / "squad")

    @patch("src.download_manager.snapshot_download")
    @patch("src.download_manager.git.Repo")
    def test_download_model_resources_always_updates(self, mock_git, mock_snapshot):
        """Test that existing resources are always updated"""
        # Create existing directories
        model_path = self.models_dir / "bert-base-uncased"
        codebase_path = self.codebases_dir / "repo"
        dataset_path = self.datasets_dir / "squad"
        model_path.mkdir(parents=True)
        codebase_path.mkdir(parents=True)
        dataset_path.mkdir(parents=True)

        # Mock git repo for update
        mock_repo_instance = MagicMock()
        mock_repo_instance.active_branch.name = "main"
        mock_git.return_value = mock_repo_instance

        model_urls = ModelURLs(
            model="https://huggingface.co/bert-base-uncased",
            codebase="https://github.com/username/repo",
            dataset="https://huggingface.co/datasets/squad",
        )

        result_model, result_codebase, result_dataset = (
            self.dm.download_model_resources(model_urls)
        )

        # Should call download functions to UPDATE (not skip)
        self.assertEqual(mock_snapshot.call_count, 2)  # model and dataset updates

        # Should call git.Repo to update existing codebase
        mock_git.assert_called_once_with(codebase_path)
        mock_repo_instance.remotes.origin.fetch.assert_called_once()
        mock_repo_instance.git.reset.assert_called_once()

        # Check return values
        self.assertEqual(result_model, model_path)
        self.assertEqual(result_codebase, codebase_path)
        self.assertEqual(result_dataset, dataset_path)

    @patch("src.download_manager.snapshot_download")
    @patch("src.download_manager.git.Repo")
    def test_download_model_resources_selective_download(self, mock_git, mock_snapshot):
        """Test selective downloading with all URLs present but selective flags"""
        model_urls = ModelURLs(
            model="https://huggingface.co/bert-base-uncased",
            codebase="https://github.com/username/repo",
            dataset="https://huggingface.co/datasets/squad",
        )

        # Only download model and dataset, skip codebase
        model_path, codebase_path, dataset_path = self.dm.download_model_resources(
            model_urls,
            download_model=True,
            download_codebase=False,
            download_dataset=True,
        )

        # Check model and dataset were downloaded
        self.assertEqual(mock_snapshot.call_count, 2)

        # Codebase should NOT be downloaded
        mock_git.clone_from.assert_not_called()
        mock_git.assert_not_called()

        # Check return values
        self.assertEqual(model_path, self.models_dir / "bert-base-uncased")
        self.assertIsNone(codebase_path)
        self.assertEqual(dataset_path, self.datasets_dir / "squad")


if __name__ == "__main__":
    unittest.main()
