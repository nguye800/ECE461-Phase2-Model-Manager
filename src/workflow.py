from multiprocessing import Pool
from typing import Optional
from pydantic import BaseModel, field_validator, ValidationError
import typing
import typing_extensions
from metric import BaseMetric, AnalyzerOutput, PRIORITY_FUNCTIONS
from config import (
    ConfigContract,
    ModelPaths,
    ModelURLs,
    PriorityFunction,
    PRIORITY_FUNCTIONS,
)


def run_metric(metric: BaseMetric) -> BaseMetric:
    """
    Runs a single metric by calling its run method.
    Args:
        metric (BaseMetric): The metric instance to run.
    Returns:
        BaseMetric: The metric instance after running.
    """
    return metric.run()


class MetricRunner:
    def __init__(self, metrics: list[BaseMetric]):  # single threaded by default
        self.metrics: list[BaseMetric] = metrics
        self.multiprocessing_pool: Optional[Pool] = None
        self.run_multi = True  # for debugging purposes

    def run(self) -> list[BaseMetric]:
        """
        Runs all metrics using the configured multiprocessing pool.
        Returns:
            list[BaseMetric]: List of processed metric instances.
        Raises:
            Exception: If no multiprocessing pool has been created.
        """
        if self.run_multi:
            if self.multiprocessing_pool:
                with self.multiprocessing_pool as pool:
                    results: list[BaseMetric] = pool.map(run_metric, self.metrics)
            else:
                raise Exception("No multiprocessing pool has been created")
        else:
            results = [run_metric(metric) for metric in self.metrics]

        return results

    def set_num_processes(self, num_processes: int) -> typing_extensions.Self:
        """
        Sets the number of processes for multiprocessing.
        Args:
            num_processes (int): Number of processes to use.
        Returns:
            Self: The MetricRunner instance with updated pool.
        """
        self.multiprocessing_pool = Pool(num_processes)
        return self


class MetricStager:
    """
    Stages metrics by grouping them and attaching configuration and URLs.
    """

    def __init__(self, config: ConfigContract):
        """
        Initializes the MetricStager.
        Args:
            config (ConfigContract): Configuration for staging metrics.
        """
        self.metrics: list[BaseMetric] = []
        self.config: ConfigContract = config

    def attach_metric(self, metric: BaseMetric, priority: int) -> typing_extensions.Self:
        """
        Attaches a metric to a group with a given priority and platform.
        Args:
            group (str): The group to attach the metric to ('dataset', 'codebase', or 'model').
            metric (BaseMetric): The metric instance to attach.
            priority (int): The priority value for the metric.
        Returns:
            Self: The MetricStager instance with the metric attached.
        Raises:
            KeyError: If the group is invalid.
        """
        assert priority >= 1
        metric.set_params(priority, self.config.target_platform)
        self.metrics.append(metric)

        return self

    def attach_model_sources(
        self, model_urls: ModelURLs, model_paths: ModelPaths
    ) -> MetricRunner:
        """
        Attaches URLs from model metadata to the corresponding metrics.
        Args:
            model_metadata (ModelURLs): The model metadata containing URLs.
        Returns:
            MetricRunner: A MetricRunner instance with staged metrics.
        """

        for metric in self.metrics:
            metric.set_local_directory(model_paths)
            metric.set_url(model_urls)

        return MetricRunner(self.metrics)


def run_workflow(
    metric_stager: MetricStager,
    input_urls: ModelURLs,
    input_paths: ModelPaths,
    config: ConfigContract,
) -> AnalyzerOutput:
    """
    Runs the complete workflow: attaches URLs, sets up multiprocessing, runs metrics,
    and returns the analysis output.
    Args
        metric_stager (MetricStager): The metric stager with staged metrics.
        input_urls (ModelURLs): The input model URLs.
        config (ConfigContract): The workflow configuration.
    Returns:
        AnalyzerOutput: The output of the analysis.
    """

    # HERE it should check if the inputted models are already stored locally
    metric_runner: MetricRunner = metric_stager.attach_model_sources(
        input_urls, input_paths
    )
    if config.run_multi:
        metric_runner.set_num_processes(config.num_processes)
    else:
        metric_runner.multiprocessing_pool = None
    metric_runner.run_multi = config.run_multi
    processed_metrics: list[BaseMetric] = metric_runner.run()

    priority_fn: PriorityFunction = PRIORITY_FUNCTIONS[config.priority_function]

    return AnalyzerOutput(priority_fn, processed_metrics, input_urls)
