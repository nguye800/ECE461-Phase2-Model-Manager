import unittest
import os
import tempfile
from pathlib import Path

from src.config import (
    PFExponentialDecay,
    PFReciprocal,
    PRIORITY_FUNCTIONS,
    ModelURLs,
    ModelPaths,
    ConfigContract,
    extract_model_repo_id,
    extract_dataset_repo_id,
    extract_code_repo_name,
    generate_model_paths,
)


class TestPriorityFunctions(unittest.TestCase):
    def test_pf_exponential_decay_basic(self):
        pf = PFExponentialDecay(base_coefficient=2)
        # priority = 1 => 2^-(0) = 1
        self.assertAlmostEqual(pf.calculate_priority_weight(1), 1.0)
        # priority = 2 => 2^-1 = 0.5
        self.assertAlmostEqual(pf.calculate_priority_weight(2), 0.5)
        # priority = 3 => 2^-2 = 0.25
        self.assertAlmostEqual(pf.calculate_priority_weight(3), 0.25)

    def test_pf_exponential_decay_assertion(self):
        with self.assertRaises(AssertionError):
            PFExponentialDecay(base_coefficient=1)

    def test_pf_reciprocal_basic(self):
        pf = PFReciprocal()
        self.assertAlmostEqual(pf.calculate_priority_weight(1), 1.0)
        self.assertAlmostEqual(pf.calculate_priority_weight(2), 0.5)
        self.assertAlmostEqual(pf.calculate_priority_weight(4), 0.25)

    def test_priority_functions_mapping(self):
        self.assertIn("PFExponentialDecay", PRIORITY_FUNCTIONS)
        self.assertIn("PFReciprocal", PRIORITY_FUNCTIONS)
        self.assertIsInstance(PRIORITY_FUNCTIONS["PFReciprocal"], PFReciprocal)
        self.assertIsInstance(
            PRIORITY_FUNCTIONS["PFExponentialDecay"], PFExponentialDecay
        )


class TestUrlExtraction(unittest.TestCase):
    def test_extract_model_repo_id_standard(self):
        url = "https://huggingface.co/bert-base-uncased"
        self.assertEqual(extract_model_repo_id(url), "bert-base-uncased")

    def test_extract_model_repo_id_with_user_and_tree(self):
        url = "https://huggingface.co/user/model-name/tree/main"
        self.assertEqual(extract_model_repo_id(url), "user/model-name")

    def test_extract_model_repo_id_trailing_slash(self):
        url = "https://huggingface.co/user/model-name/"
        self.assertEqual(extract_model_repo_id(url), "user/model-name")

    def test_extract_dataset_repo_id_standard(self):
        url = "https://huggingface.co/datasets/user/dataset-name"
        self.assertEqual(extract_dataset_repo_id(url), "user/dataset-name")

    def test_extract_dataset_repo_id_with_tree(self):
        url = "https://huggingface.co/datasets/user/dataset-name/tree/main"
        self.assertEqual(extract_dataset_repo_id(url), "user/dataset-name")

    def test_extract_dataset_repo_id_trailing_slash(self):
        url = "https://huggingface.co/datasets/user/dataset-name/"
        self.assertEqual(extract_dataset_repo_id(url), "user/dataset-name")

    def test_extract_code_repo_name_standard(self):
        url = "https://github.com/username/repo-name"
        self.assertEqual(extract_code_repo_name(url), "repo-name")

    def test_extract_code_repo_name_git_suffix(self):
        url = "https://github.com/username/repo-name.git"
        self.assertEqual(extract_code_repo_name(url), "repo-name")

    def test_extract_code_repo_name_trailing_slash(self):
        url = "https://github.com/username/repo-name/"
        self.assertEqual(extract_code_repo_name(url), "repo-name")


class TestConfigContractValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_valid_config_contract(self):
        cfg = ConfigContract(
            local_storage_directory=self.tmp_path,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )
        self.assertEqual(cfg.local_storage_directory, self.tmp_path)
        self.assertEqual(cfg.model_path_name, "models")
        self.assertEqual(cfg.code_path_name, "code")
        self.assertEqual(cfg.dataset_path_name, "datasets")

    def test_invalid_local_storage_directory_not_readable(self):
        # create a directory and remove read permission
        unreadable_dir = Path(self.tmp_path) / "unreadable"
        unreadable_dir.mkdir()
        # Remove read for owner
        current_mode = unreadable_dir.stat().st_mode
        try:
            unreadable_dir.chmod(0o000)
            with self.assertRaises(IOError):
                ConfigContract(
                    local_storage_directory=str(unreadable_dir),
                    model_path_name="models",
                    code_path_name="code",
                    dataset_path_name="datasets",
                )
        finally:
            # Restore so temp cleanup works on all OS
            unreadable_dir.chmod(current_mode)

    def test_invalid_path_names_with_slash(self):
        with self.assertRaises(NameError):
            ConfigContract(
                local_storage_directory=self.tmp_path,
                model_path_name="bad/name",
                code_path_name="code",
                dataset_path_name="datasets",
            )
        with self.assertRaises(NameError):
            ConfigContract(
                local_storage_directory=self.tmp_path,
                model_path_name="models",
                code_path_name="bad/name",
                dataset_path_name="datasets",
            )
        with self.assertRaises(NameError):
            ConfigContract(
                local_storage_directory=self.tmp_path,
                model_path_name="models",
                code_path_name="code",
                dataset_path_name="bad/name",
            )


class TestGenerateModelPaths(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmpdir.name
        self.cfg = ConfigContract(
            local_storage_directory=self.tmp_path,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generate_paths_model_only_simple(self):
        urls = ModelURLs(model="https://huggingface.co/bert-base-uncased")
        paths = generate_model_paths(self.cfg, urls)
        expected = Path(self.tmp_path) / "models" / "bert-base-uncased"
        self.assertEqual(paths.model, expected)
        self.assertIsNone(paths.codebase)
        self.assertIsNone(paths.dataset)

    def test_generate_paths_model_with_user_replaces_slash(self):
        urls = ModelURLs(model="https://huggingface.co/user/model-name")
        paths = generate_model_paths(self.cfg, urls)
        # user/model-name -> user_model-name
        expected = Path(self.tmp_path) / "models" / "user_model-name"
        self.assertEqual(paths.model, expected)

    def test_generate_paths_all(self):
        urls = ModelURLs(
            model="https://huggingface.co/user/model-name",
            codebase="https://github.com/user/repo",
            dataset="https://huggingface.co/datasets/user/ds",
        )
        paths = generate_model_paths(self.cfg, urls)
        self.assertEqual(
            paths.model, Path(self.tmp_path) / "models" / "user_model-name"
        )
        self.assertEqual(paths.codebase, Path(self.tmp_path) / "code" / "repo")
        self.assertEqual(paths.dataset, Path(self.tmp_path) / "datasets" / "user_ds")


if __name__ == "__main__":
    unittest.main()
