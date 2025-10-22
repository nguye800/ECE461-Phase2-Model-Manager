import unittest
from metrics.code_quality import *
from metric import ModelPaths
from pathlib import Path


class TestCodeQuality(unittest.TestCase):
    def test_score_calculation(self):
        metric_inst: CodeQualityMetric = CodeQualityMetric()
        directories = ModelPaths(codebase=Path(os.path.dirname(os.path.abspath(__file__))))
        metric_inst.set_local_directory(directories)

        metric_inst.setup_resources()
        metric_score: float = metric_inst.calculate_score()

        self.assertGreater(metric_score, 0.5)

    def test_no_python(self):
        metric_inst: CodeQualityMetric = CodeQualityMetric()
        directories = ModelPaths(
            codebase=Path(os.path.dirname(os.path.abspath(__file__))) / "/dummy_bad_dir"
        )
        metric_inst.set_local_directory(directories)
        metric_inst.setup_resources()
        metric_score: float = metric_inst.calculate_score()

        self.assertAlmostEqual(metric_score, 0.0)

    def test_score_calculation_full(self):
        metric_inst: CodeQualityMetric = CodeQualityMetric()
        directories = ModelPaths(codebase=Path(os.path.dirname(os.path.abspath(__file__))))
        metric_inst.set_local_directory(directories)
        metric_inst.setup_resources()
        metric_inst: CodeQualityMetric = metric_inst.run()
        self.assertIsInstance(metric_inst.score, float)

        metric_score: float = metric_inst.score

        self.assertGreater(metric_score, 0.5)

    def test_no_python_full(self):
        metric_inst: CodeQualityMetric = CodeQualityMetric()
        directories = ModelPaths(
            codebase=Path(os.path.dirname(os.path.abspath(__file__))) / "/dummy_bad_dir"
        )
        metric_inst.set_local_directory(directories)
        metric_inst.setup_resources()
        metric_inst: CodeQualityMetric = metric_inst.run()
        self.assertIsInstance(metric_inst.score, float)

        metric_score: float = metric_inst.score

        self.assertAlmostEqual(metric_score, 0.0)