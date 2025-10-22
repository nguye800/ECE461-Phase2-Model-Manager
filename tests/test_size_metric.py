import unittest
from pathlib import Path
import types

from metrics.size_metric import SizeMetric, DEVICE_SPECS
from metric import BaseMetric
from config import ModelURLs, ModelPaths  # adjust if these live elsewhere


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