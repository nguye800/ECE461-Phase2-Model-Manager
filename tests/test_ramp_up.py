import unittest
import tempfile
from pathlib import Path
from src.metric import ModelPaths

from huggingface_hub import snapshot_download
from src.metrics.ramp_up_time import RampUpMetric


class TestRampUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a shared temporary workspace for this test class
        cls._class_tmp = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls._class_tmp.name)

        # Create a codebase dir where we’ll place the downloaded model (acts as "repo")
        cls.model_dir = cls.workspace / "model"
        cls.model_dir.mkdir(parents=True, exist_ok=True)

        # Download a very small Hugging Face model into the codebase directory
        # Using a tiny config-only repo to minimize size and time
        # You can swap this with any small public model if desired.
        cls.model_local_dir = cls.model_dir / "sshleifer_tiny-distilroberta-base"
        if not cls.model_local_dir.exists():
            snapshot_download(
                repo_id="sshleifer/tiny-distilroberta-base",
                local_dir=str(cls.model_local_dir),
                revision="main",
            )

        # Add minimal ancillary files that typical code quality checks might expect
        (cls.model_dir / "README.md").write_text(
            "# Tiny Model Repo\n\nAuto-downloaded for testing.\n"
        )
        (cls.model_dir / "pyproject.toml").write_text("[tool]\nname='tiny'\n")

    @classmethod
    def tearDownClass(cls):
        cls._class_tmp.cleanup()

    def setUp(self):
        # Fresh metric instance per test; preserve existing test style
        self.metric = RampUpMetric(0.1, "cpu")

    def tearDown(self):
        pass

    def test_url(self):
        metric: RampUpMetric = RampUpMetric(0.25, "cpu")
        metric.set_local_directory(ModelPaths(model=self.model_local_dir))
        metric.setup_resources()

        self.assertGreater(metric.run().score, 0.0)

    def test_with_url(self):
        metric: RampUpMetric = RampUpMetric(0.25, "cpu")
        metric.set_local_directory(ModelPaths(model="shagooba"))
        metric.setup_resources()

        self.assertAlmostEqual(metric.run().score, 0.0)
