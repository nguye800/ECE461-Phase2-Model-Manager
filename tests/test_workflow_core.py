import tempfile
import unittest

from src.config import ConfigContract, ModelPaths, ModelURLs
from src.metric import BaseMetric
from src.workflow import MetricRunner, MetricStager, run_metric, run_workflow


class SimpleMetric(BaseMetric):
    metric_name = "simple"

    def __init__(self):
        super().__init__()
        self.setup_called = False

    def setup_resources(self):
        self.setup_called = True

    def calculate_score(self):
        return 0.5


class WorkflowCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ConfigContract(
            num_processes=1,
            run_multi=False,
            priority_function="PFReciprocal",
            target_platform="desktop_pc",
            local_storage_directory=self.tmp.name,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_metric_invokes_metric(self):
        metric = SimpleMetric()
        result = run_metric(metric)
        self.assertTrue(result.setup_called)
        self.assertEqual(result.score, 0.5)

    def test_metric_runner_requires_pool_when_multi(self):
        runner = MetricRunner([SimpleMetric()])
        with self.assertRaises(Exception):
            runner.run()

    def test_metric_runner_single_thread(self):
        runner = MetricRunner([SimpleMetric()])
        runner.run_multi = False
        results = runner.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.5)

    def test_run_workflow_single_thread(self):
        stager = MetricStager(self.config)
        stager.attach_metric(SimpleMetric(), 1)
        output = run_workflow(
            stager,
            ModelURLs(model="https://huggingface.co/owner/model"),
            ModelPaths(),
            self.config,
        )
        self.assertIn("simple", output.individual_scores)
