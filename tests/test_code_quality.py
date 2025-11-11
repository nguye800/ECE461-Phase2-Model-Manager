import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.metrics.code_quality import *
from src.metric import ModelPaths


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

    def test_setup_resources_collects_python_files(self):
        metric_inst = CodeQualityMetric()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            code_dir = Path(tmp) / "code"
            model_dir.mkdir()
            code_dir.mkdir()
            (model_dir / "a.py").write_text("print('a')")
            (code_dir / "b.py").write_text("print('b')")
            metric_inst.set_local_directory(ModelPaths(model=model_dir, codebase=code_dir))
            metric_inst.setup_resources()
            self.assertIn(str(model_dir / "a.py"), metric_inst.file_list)
            self.assertIn(str(code_dir / "b.py"), metric_inst.file_list)

    @patch("src.metrics.code_quality.pylinter.MANAGER")
    @patch("src.metrics.code_quality.Run")
    def test_calculate_score_parses_rating(self, mock_run, mock_manager):
        metric_inst = CodeQualityMetric()
        metric_inst.file_list = ["fake.py"]

        def fake_run(args, reporter, exit):
            reporter.out.write("Your code has been rated at 8.50/10")

        mock_run.side_effect = fake_run
        score = metric_inst.calculate_score()
        mock_manager.clear_cache.assert_called_once()
        self.assertAlmostEqual(score, 0.85)

    @patch("src.metrics.code_quality.pylinter.MANAGER")
    @patch("src.metrics.code_quality.Run")
    def test_calculate_score_handles_missing_rating(self, mock_run, mock_manager):
        metric_inst = CodeQualityMetric()
        metric_inst.file_list = ["fake.py"]

        def fake_run(args, reporter, exit):
            reporter.out.write("no rating here")

        mock_run.side_effect = fake_run
        score = metric_inst.calculate_score()
        self.assertEqual(score, 0.0)
