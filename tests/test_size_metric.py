import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.config import ModelURLs, ModelPaths  # adjust if these live elsewhere
from src.metrics.size_metric import DEVICE_SPECS, SizeMetric


class TestSizeMetric(unittest.TestCase):
    def setUp(self):
        self.metric = SizeMetric()

        # Provide a dummy URL to trigger fallback logic (no network dependency)
        self.metric.set_url(ModelURLs(model="invalid/repo", dataset=None, codebase=None))

    def test_setup_resources_populates_fields(self):
        # Should not raise; should populate internal fields using fallback
        self.metric.setup_resources()
        self.assertIsNotNone(self.metric.model_info)
        self.assertIn("config", self.metric.model_info)
        self.assertIsInstance(self.metric.storage_size_mb, (int, float))
        self.assertIsInstance(self.metric.memory_size_mb, (int, float))
        self.assertGreater(self.metric.storage_size_mb, 0)
        self.assertGreater(self.metric.memory_size_mb, 0)

    def test_calculate_score_returns_dict_in_range(self):
        result = self.metric.run()
        self.assertIsInstance(result.score, dict)
        self.assertTrue(result.score)  # not empty
        for k, v in result.score.items():
            self.assertIn(k, DEVICE_SPECS)
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_calculate_device_score_storage_fail(self):
        # Force huge storage size to ensure storage check returns 0.0
        self.metric.storage_size_mb = 10**9  # 1,000,000 MB
        self.metric.param_source = None
        score = self.metric.calculate_device_score(DEVICE_SPECS["raspberry_pi"])
        self.assertEqual(score, 0.0)

    def test_calculate_device_score_fallback_zero(self):
        # When param_source is 'fallback', device score should be 0.0
        self.metric.storage_size_mb = 1.0
        self.metric.param_source = "fallback"
        score = self.metric.calculate_device_score(DEVICE_SPECS["desktop_pc"])
        self.assertEqual(score, 0.0)

    def test_calculate_device_score_memory_brackets(self):
        # Test thresholds: >1.2 => 0.01, between 0.8 and 1.2 => in [0.05, 0.8), <0.8 => near 1.0
        spec = DEVICE_SPECS["desktop_pc"]
        available = spec["memory_mb"] * spec["available_memory_ratio"]

        self.metric.param_source = None
        self.metric.storage_size_mb = 1.0

        # > 1.2x available memory
        self.metric.memory_size_mb = available * 1.21
        self.assertEqual(self.metric.calculate_device_score(spec), 0.01)

        # In the 0.8..1.2 range
        self.metric.memory_size_mb = available * 0.9
        mid = self.metric.calculate_device_score(spec)
        self.assertGreaterEqual(mid, 0.05)
        self.assertLess(mid, 0.8)

        # < 0.8
        self.metric.memory_size_mb = available * 0.5
        high = self.metric.calculate_device_score(spec)
        self.assertGreater(high, 0.8)
        self.assertLessEqual(high, 1.0)

    def test_get_size_details_valid_device(self):
        # Prepare plausible state
        self.metric.setup_resources()
        details = self.metric.get_size_details("desktop_pc")
        self.assertIn("storage_size_mb", details)
        self.assertIn("memory_size_mb", details)
        self.assertIn("device_storage_mb", details)
        self.assertIn("device_memory_mb", details)
        self.assertIn("available_memory_mb", details)
        self.assertIn("storage_fits", details)
        self.assertIn("memory_fits", details)

    def test_get_size_details_invalid_device_raises(self):
        with self.assertRaises(ValueError):
            self.metric.get_size_details("nonexistent_device")

    def test_setup_resources_wraps_fetch_errors(self):
        metric = SizeMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/owner/model"))
        with patch.object(SizeMetric, "_fetch_model_info", side_effect=RuntimeError("boom")):
            with self.assertRaises(IOError):
                metric.setup_resources()

    def test_extract_repo_id_requires_url(self):
        metric = SizeMetric()
        with self.assertRaises(ValueError):
            metric._extract_repo_id_from_url()
        metric.set_url(ModelURLs(model="owner/model"))
        self.assertEqual(metric._extract_repo_id_from_url(), "owner/model")

    def test_get_parameter_count_prefers_known_fields(self):
        for key in ["n_params", "num_parameters", "parameter_count", "total_params", "nparams"]:
            self.metric.model_info = {"config": {key: 7}}
            self.assertEqual(self.metric._get_parameter_count(), 7)

    def test_estimate_parameters_additional_terms(self):
        cfg = {
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "vocab_size": 10,
            "intermediate_size": 16,
            "max_position_embeddings": 8,
            "type_vocab_size": 1,
            "add_pooling_layer": True,
            "tie_word_embeddings": False,
        }
        params = self.metric._estimate_parameters_from_config(cfg)
        self.assertGreater(params, 0)

    def test_get_tensor_type_default(self):
        self.metric.model_info = {}
        self.assertEqual(self.metric._get_tensor_type(), "float32")
    def test_extract_repo_id_variants(self):
        self.metric.set_url(ModelURLs(model="https://huggingface.co/owner/model/tree/main"))
        self.assertEqual(self.metric._extract_repo_id_from_url(), "owner/model")
        self.metric.set_url(ModelURLs(model="owner/model"))
        self.assertEqual(self.metric._extract_repo_id_from_url(), "owner/model")

    def test_get_fallback_model_info_defaults(self):
        info = self.metric._get_fallback_model_info()
        self.assertEqual(info["param_source"], "fallback")
        self.assertIn("config", info)

    def test_calculate_storage_size_uses_siblings(self):
        self.metric.model_info = {"siblings": [{"size": 1024}, {"size": 2048}]}
        size_mb = self.metric._calculate_storage_size()
        self.assertAlmostEqual(size_mb, (1024 + 2048) / (1024 * 1024))

    def test_calculate_memory_size_with_kv_cache(self):
        self.metric.model_info = {
            "config": {
                "model_type": "gptx",
                "max_position_embeddings": 128,
                "hidden_size": 64,
                "num_hidden_layers": 4,
            }
        }
        with patch.object(self.metric, "_get_parameter_count", return_value=1000), patch.object(
            self.metric, "_get_tensor_type", return_value="float32"
        ):
            mem_mb = self.metric._calculate_memory_size()
        self.assertGreater(mem_mb, 0)

    def test_get_parameter_count_prefers_config_fields(self):
        self.metric.model_info = {"config": {"num_parameters": 123}}
        self.assertEqual(self.metric._get_parameter_count(), 123)

    def test_estimate_parameters_requires_config(self):
        with self.assertRaises(ValueError):
            self.metric._estimate_parameters_from_config(
                {"num_hidden_layers": None, "vocab_size": None}
            )

    def test_get_tensor_type_strips_prefix(self):
        self.metric.model_info = {"config": {"torch_dtype": "torch.float16"}}
        self.assertEqual(self.metric._get_tensor_type(), "float16")

    @patch("src.metrics.size_metric.requests.get")
    def test_fetch_model_info_falls_back_when_request_fails(self, mock_get):
        mock_get.side_effect = requests.RequestException("boom")
        self.metric.set_url(ModelURLs(model="https://huggingface.co/owner/model"))
        info = self.metric._fetch_model_info()
        self.assertEqual(info["param_source"], "fallback")

    @patch("src.metrics.size_metric.requests.get")
    def test_fetch_model_info_success_path(self, mock_get):
        resp_config = MagicMock()
        resp_config.raise_for_status = MagicMock()
        resp_config.json.return_value = {"siblings": []}
        resp_extra = MagicMock()
        resp_extra.status_code = 200
        resp_extra.json.return_value = {"hidden_size": 16}
        mock_get.side_effect = [resp_config, resp_extra]
        self.metric.set_url(ModelURLs(model="https://huggingface.co/owner/model"))
        info = self.metric._fetch_model_info()
        self.assertIn("config", info)
        self.assertEqual(info["config"]["hidden_size"], 16)
