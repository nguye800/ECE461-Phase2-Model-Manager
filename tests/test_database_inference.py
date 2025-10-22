from ..src.infer_dataset import *
from unittest import TestCase
from src.database import SQLiteAccessor
from src.database import ModelStats


class MockPath:
    """Simple mock for a README file."""

    def __init__(self, content="", exists=True):
        self._content = content
        self.exists = exists

    def read_text(self, *args, **kwargs):
        return self._content


class TestDatabaseInference(TestCase):
    schema1: list = [
        FloatMetric("size", 0.3, 10),
        FloatMetric("setup", 0.2, 30),
        DictMetric("compatibility", {"windows": 0.5, "mac": 0.2, "linux": 0.8}, 29),
    ]

    def test_one_url(self):
        db = SQLiteAccessor(db_location=None, metric_schema=self.schema1)
        db.add_to_db(
            model_stats=ModelStats(
                "https://huggingface.co/google-bert/bert-case-uncased",
                "google-bert/bert-case-uncased",
                "example.com/cool-dataset",
                "",
                0.39,
                39,
                self.schema1,
            )
        )
        readme_contents = """
# New Model
We're going to blow google out of the water with this one

# Dataset
We used [The Same Dataset](example.com/cool-dataset) as that google one
"""
        readme_file = MockPath(content=readme_contents)
        metrics = get_linked_dataset_metrics(
            readme_file, db, [FloatMetric("setup", 0.0, 0)]  # pyright: ignore
        )
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics[1][0].data, float)  # pyright: ignore
        self.assertAlmostEqual(metrics[1][0].data, 0.2)  # type: ignore

    def test_two_urls(self):
        db = SQLiteAccessor(db_location=None, metric_schema=self.schema1)
        db.add_to_db(
            model_stats=ModelStats(
                "https://huggingface.co/google-bert/bert-case-uncased",
                "google-bert/bert-case-uncased",
                "example.com/cool-dataset",
                "",
                0.39,
                39,
                self.schema1,
            )
        )
        readme_contents = """
# Newer Model
We're going to blow google out of the water with this one. Forget about the [last time](example.com/awful-model). Please

# Dataset
We used [The Same Dataset](example.com/cool-dataset) as that google one
"""
        readme_file = MockPath(content=readme_contents)
        metrics = get_linked_dataset_metrics(
            readme_file, db, [FloatMetric("setup", 0.0, 0)]  # pyright: ignore
        )
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics[1][0].data, float)  # pyright: ignore
        self.assertAlmostEqual(metrics[1][0].data, 0.2)  # type: ignore

    def test_zero_urls(self):
        db = SQLiteAccessor(db_location=None, metric_schema=self.schema1)
        db.add_to_db(
            model_stats=ModelStats(
                "https://huggingface.co/google-bert/bert-case-uncased",
                "google-bert/bert-case-uncased",
                "example.com/cool-dataset",
                "",
                0.39,
                39,
                self.schema1,
            )
        )
        readme_contents = """
# Newest Model
We actually know what we're doing now. Model is linked on huggingface, no need to repeat it here.
"""
        readme_file = MockPath(content=readme_contents)
        metrics = get_linked_dataset_metrics(
            readme_file, db, [FloatMetric("setup", 0.0, 0)]  # pyright: ignore
        )
        self.assertIsNone(metrics)
