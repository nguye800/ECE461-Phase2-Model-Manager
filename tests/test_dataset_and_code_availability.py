import unittest
from src.metrics.dataset_and_code import *
from unittest.mock import MagicMock
from src.metric import ModelURLs


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
        test_cases = [
            ("https://example.com", 0.3),
            ("https://example.com", 0.3),
        ]
        for url, expected_score in test_cases:
            metric = DatasetAndCodeScoreMetric()
            urls = ModelURLs(model="shouldn't matter")
            urls.dataset = url
            urls.codebase = None
            metric.set_url(urls)
            metric.readme_file = MockPath(exists=False)
            score = metric.calculate_score()
            self.assertAlmostEqual(score, expected_score, places=2)

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
