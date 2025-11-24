import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.metrics.dataset_and_code import *
from src.metric import ModelPaths, ModelURLs


class MockPath:
    """Simple mock for a README file."""

    def __init__(self, content="", exists=True):
        self._content = content
        self.exists = exists

    def read_text(self, *args, **kwargs):
        return self._content


class TestDatasetAndCodeScoreMetric(unittest.TestCase):
    metric: DatasetAndCodeScoreMetric

    def setUp(self):
        self.metric = DatasetAndCodeScoreMetric()

    def test_score_calculation_no_resources(self):
        urls = ModelURLs(model="nonexistent")
        urls.dataset = None
        urls.codebase = None
        self.metric.set_url(urls)

        self.metric.readme_file = MockPath("")

        score = self.metric.calculate_score()
        self.assertEqual(score, 0.0)

    def test_score_calculation_full_documentation(self):

        readme = """
        Model Documentation

        This model uses a comprehensive dataset with detailed data description.

        Usage
        Here's how to use this model effectively.

        Example
        Sample usage and code examples.

        Requirements
        Installation and dependency requirements.

        Limitations
        Known limitations and constraints.
        """.lower()

        self.metric.readme_file = MockPath(readme)
        score = self.metric.calculate_score()
        expected_score = 0.2
        self.assertAlmostEqual(score, expected_score, places=2)

    def test_score_calculation_working_urls(self):
        metric = DatasetAndCodeScoreMetric()
        urls = ModelURLs(model="shouldn't matter", dataset="https://datasets", codebase="https://code")
        metric.set_url(urls)
        metric.readme_file = MockPath("", exists=False)
        with patch("src.metrics.dataset_and_code.requests.head") as mock_head, patch(
            "src.metrics.dataset_and_code.requests.get"
        ) as mock_get:
            mock_head.side_effect = [
                MagicMock(status_code=200),  # dataset head
                MagicMock(status_code=200),  # code head
            ]
            mock_get.return_value = MagicMock(text="Dataset description available here.")
            score = metric.calculate_score()
        self.assertAlmostEqual(score, 0.8, places=2)

    def test_documentation_scoring_logic(self):

        urls = ModelURLs(model="nonexistent")
        urls.dataset = None
        urls.codebase = None
        self.metric.set_url(urls)

        # Test cases with different documentation levels
        test_cases = [
            ("", 0.0),
            ("dataset", 0.04),
            ("dataset usage", 0.08),
            ("dataset data description usage how to use", 0.16),
            ("dataset usage example requirements limitations", 0.2),
        ]

        for readme_content, expected_doc_score in test_cases:
            self.metric.readme_file = MockPath(readme_content)
            score = self.metric.calculate_score()
            self.assertAlmostEqual(score, expected_doc_score, places=2)

    def test_setup_resources_requires_local_directory(self):
        metric = DatasetAndCodeScoreMetric()
        with self.assertRaises(ValueError):
            metric.setup_resources()

    @patch("src.metrics.dataset_and_code.requests.head")
    def test_calculate_score_handles_head_exception(self, mock_head):
        mock_head.side_effect = requests.RequestException("boom")
        metric = DatasetAndCodeScoreMetric()
        urls = ModelURLs(model="x", dataset="https://datasets")
        metric.set_url(urls)
        metric.readme_file = MockPath("")
        score = metric.calculate_score()
        self.assertEqual(score, 0.0)

    @patch("src.metrics.dataset_and_code.requests.head")
    def test_code_head_exception(self, mock_head):
        mock_head.side_effect = requests.RequestException("boom")
        metric = DatasetAndCodeScoreMetric()
        urls = ModelURLs(model="x", codebase="https://github.com/test/repo")
        metric.set_url(urls)
        metric.readme_file = MockPath("")
        score = metric.calculate_score()
        self.assertEqual(score, 0.0)

    @patch("src.metrics.dataset_and_code.requests.get")
    def test_calculate_score_handles_get_exception(self, mock_get):
        mock_get.side_effect = requests.RequestException("boom")
        metric = DatasetAndCodeScoreMetric()
        urls = ModelURLs(model="x", dataset="https://datasets")
        metric.set_url(urls)
        metric.readme_file = MockPath("dataset usage example requirements limitations")
        score = metric.calculate_score()
        self.assertAlmostEqual(score, 0.2, places=2)

    def test_setup_resources_sets_paths(self):
        metric = DatasetAndCodeScoreMetric()
        with tempfile.TemporaryDirectory() as tmp:
            paths = ModelPaths(model=tmp)
            metric.set_local_directory(paths)
            metric.setup_resources()
            self.assertEqual(metric.readme_file, Path(tmp) / "README.md")
