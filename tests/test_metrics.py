import unittest
import tempfile
from src.workflow import *  # pyright: ignore[reportWildcardImportFromLibrary, reportMissingTypeStubs]
from src.metric import *  # pyright: ignore[reportWildcardImportFromLibrary, reportMissingTypeStubs]
from src.config import *


class DummyMetric1(BaseMetric):
    @typing_extensions.override
    def calculate_score(self):
        return 0.5

    @typing_extensions.override
    def setup_resources(self):
        pass


class DummyMetric2(BaseMetric):
    @typing_extensions.override
    def calculate_score(self):
        return 0.7

    @typing_extensions.override
    def setup_resources(self):
        pass


class DummyMetric4(BaseMetric):
    @typing_extensions.override
    def calculate_score(self):
        return 0.0

    @typing_extensions.override
    def setup_resources(self):
        pass


class DummyMetric3(BaseMetric):
    @typing_extensions.override
    def calculate_score(self):
        return 1.0

    @typing_extensions.override
    def setup_resources(self):
        pass


class DummyMetric5(BaseMetric):
    @typing_extensions.override
    def calculate_score(self):
        return {"a": 1.0}

    @typing_extensions.override
    def setup_resources(self):
        pass


MODEL_URLS_1: ModelURLs = ModelURLs(model="https://github.com/user/repo")
MODEL_URLS_2: ModelURLs = ModelURLs(
    model="https://github.com/user/repo", dataset="https://github.com/user/repo"
)
MODEL_URLS_3: ModelURLs = ModelURLs(
    model="https://github.com/user/repo",
    dataset="https://github.com/user/repo",
    codebase="https://github.com/user/repo",
)


class BaseMetricTestCase(unittest.TestCase):
    def setUp(self):
        # Create a valid temporary directory structure that satisfies validators
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

        # Subdirectories for config path names
        self.models_dir_name = "models"
        self.code_dir_name = "code"
        self.datasets_dir_name = "datasets"

        # Create subdirectories (not strictly required by validator but useful for realism)
        (self.root / self.models_dir_name).mkdir(exist_ok=True)
        (self.root / self.code_dir_name).mkdir(exist_ok=True)
        (self.root / self.datasets_dir_name).mkdir(exist_ok=True)

        # Valid ConfigContract shared by tests
        self.CONFIG_1 = ConfigContract(
            num_processes=1,
            priority_function="PFReciprocal",
            target_platform="pc",
            local_storage_directory=self.tmpdir.name,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )
        self.CONFIG_2 = ConfigContract(
            num_processes=2,
            priority_function="PFReciprocal",
            target_platform="pc",
            local_storage_directory=self.tmpdir.name,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )
        self.CONFIG_3 = ConfigContract(
            num_processes=2,
            priority_function="PFExponentialDecay",
            target_platform="pc",
            local_storage_directory=self.tmpdir.name,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="datasets",
        )

    def tearDown(self):
        self.tmpdir.cleanup()


class TestMetricStaging(BaseMetricTestCase):
    def test_staging_valid(self):
        stager: MetricStager = MetricStager(self.CONFIG_1)
        stager.attach_metric(DummyMetric1(), 1)
        stager.attach_metric(DummyMetric1(), 2)
        stager.attach_metric(DummyMetric2(), 2)
        stager.attach_metric(DummyMetric3(), 3)

        self.assertEqual(len(stager.metrics), 4)

        directories = generate_model_paths(self.CONFIG_1, MODEL_URLS_1)
        runner = stager.attach_model_sources(MODEL_URLS_1, directories)
        self.assertEqual(len(runner.metrics), 4)

    def test_staging_invalid(self):
        stager: MetricStager = MetricStager(self.CONFIG_1)
        with self.assertRaises(AssertionError):
            stager.attach_metric(DummyMetric1(), 0)


class TestMetricRunner(BaseMetricTestCase):
    def test_run_single_threaded(self):
        runner: MetricRunner = MetricRunner(
            [DummyMetric1(), DummyMetric2(), DummyMetric3()]
        )
        runner.set_num_processes(3)
        results: list[BaseMetric] = runner.run()

        result_scores = [metric.score for metric in results]

        self.assertEqual(len(results), 3)
        self.assertEqual(result_scores, [0.5, 0.7, 1.0])


class TestNetScoreCalculator(BaseMetricTestCase):
    def test_sum_scores(self):
        scores: list[float] = [0.5, 0.7, 1.0]
        sum_scores: float = NetScoreCalculator(PFReciprocal()).sum_scores(scores)
        self.assertAlmostEqual(sum_scores, 0.7333333333333)

    def test_compress_priorities(self):
        priority_organized_scores: SortedDict = SortedDict({1: [0.5, 0.7], 2: [1.0]})
        compressed_scores: list[list[float]] = NetScoreCalculator(
            PFReciprocal
        ).compress_priorities(priority_organized_scores)
        self.assertEqual(compressed_scores, [[0.5, 0.7], [1.0]])

        priority_organized_scores: SortedDict = SortedDict({1: [0.5, 0.7], 3: [1.0]})
        compressed_scores: list[list[float]] = NetScoreCalculator(
            PFReciprocal()
        ).compress_priorities(priority_organized_scores)
        self.assertEqual(compressed_scores, [[0.5, 0.7], [1.0]])

    def test_generate_scores_priority_dict(self):
        metrics: list[BaseMetric] = [
            DummyMetric1().set_params(1, "").run(),
            DummyMetric2().set_params(2, "").run(),
            DummyMetric3().set_params(3, "").run(),
        ]

        priority_organized_scores: SortedDict = NetScoreCalculator(
            PFReciprocal()
        ).generate_scores_priority_dict(metrics)
        self.assertEqual(priority_organized_scores, {1: [0.5], 2: [0.7], 3: [1.0]})

    def test_get_priority_weights(self):
        compressed_scores: list[list[float]] = [[0.5, 0.7], [1.0]]
        total_size: int = 2
        priority_weights: list[float] = NetScoreCalculator(
            PFReciprocal()
        ).get_priority_weights(compressed_scores, total_size)
        for m, n in zip(priority_weights, [12 / 15, 3 / 15]):
            self.assertAlmostEqual(m, n)

    def test_net_score_calculation_1(self):
        metrics: list[BaseMetric] = [
            DummyMetric1().set_params(1, "").run(),
            DummyMetric2().set_params(1, "").run(),
            DummyMetric3().set_params(3, "").run(),
        ]

        net_score: float = NetScoreCalculator(PFReciprocal()).calculate_net_score(
            metrics
        )
        self.assertAlmostEqual(net_score, 0.68)

    def test_net_score_calculation_2(self):
        metrics: list[BaseMetric] = [
            DummyMetric3().set_params(1, "").run(),
            DummyMetric3().set_params(2, "").run(),
            DummyMetric3().set_params(3, "").run(),
        ]

        net_score: float = NetScoreCalculator(PFReciprocal()).calculate_net_score(
            metrics
        )
        self.assertAlmostEqual(net_score, 1.0)

    def test_net_score_calculation_3(self):
        metrics: list[BaseMetric] = [
            DummyMetric4().set_params(1, "").run(),
            DummyMetric4().set_params(2, "").run(),
            DummyMetric4().set_params(3, "").run(),
        ]

        net_score: float = NetScoreCalculator(PFReciprocal()).calculate_net_score(
            metrics
        )
        self.assertAlmostEqual(net_score, 0.0)

    def test_net_score_calculation_3(self):
        metrics: list[BaseMetric] = [
            DummyMetric5().set_params(1, "").run(),
        ]

        net_score: float = NetScoreCalculator(PFReciprocal()).calculate_net_score(
            metrics
        )
        self.assertAlmostEqual(net_score, 1.0)
