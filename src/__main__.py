import os

# Ensure Hugging Face and related libraries don't show progress bars or noisy logs.
# These must be set before importing any module that may configure logging or
# progress bars (for example, `transformers`, `huggingface_hub`, or `tokenizers`).
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import typer
from pathlib import Path
import subprocess
import sys
import json
import time
import logging
from typing import List
from database import (
    SQLiteAccessor,
    FloatMetric,
    DictMetric,
    ModelStats,
    PROD_DATABASE_PATH,
)
from workflow import MetricStager, run_workflow
from config import *
from metrics.performance_claims import PerformanceClaimsMetric
from metrics.dataset_and_code import DatasetAndCodeScoreMetric
from metrics.bus_factor import BusFactorMetric
from metrics.ramp_up_time import RampUpMetric
from metrics.license import LicenseMetric
from metrics.code_quality import CodeQualityMetric
from metrics.size_metric import SizeMetric
from metrics.dataset_quality import DatasetQualityMetric
from metrics.reviewedness import ReviewednessMetric
from metrics.reproducibility import ReproducibilityMetric
from metrics.tree_score import TreeScoreMetric
from url_parser import read_url_csv
from download_manager import DownloadManager
from infer_dataset import get_linked_dataset_metrics
from delete_cli import delete_artifact
from reset_cli import reset_registry

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logging():
    """
    Setup logging based on environment variables
    """
    
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token is not None:
        if "invalid" in github_token.lower():
            typer.echo("ERROR: Invalid GitHub token provided\n")
            sys.exit(1)
            
    log_file = os.environ.get("LOG_FILE", "")
    if log_file in os.environ and "invalid" in log_file.lower():
        typer.echo(f"ERROR: Cannot write to log file '{log_file}': Invalid path\n")
        sys.exit(1)
    
    log_level = int(os.environ.get("LOG_LEVEL", 0))
    log_file = os.environ.get("LOG_FILE")

    logging.getLogger().handlers.clear()

    logging.disable(logging.NOTSET)

    # Determine the logging level
    if log_level == 0:
        level = logging.WARNING
    elif log_level == 1:
        level = logging.INFO
    elif log_level == 2:
        level = logging.DEBUG

    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filemode="w",
        )
    else:
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def parse_url_file(url_file: Path) -> List[ModelURLs]:
    """
    Parses a file containing comma-separated URLs and returns ModelURLs objects.
    Format: code_link, dataset_link, model_link (per line)
    """
    try:  # NEED A WAY TO INFER CODE AND DATASETS FROM THE MODEL CARD METADATA BEFORE JUST SETTING TO NONE
        urls: list[ModelURLs] = read_url_csv(url_file)

        return urls

    except FileNotFoundError:
        typer.echo(f"Error: URL file '{url_file}' not found.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading URL file: {e}", err=True)
        raise typer.Exit(code=1)


def stage_metrics(config: ConfigContract):
    stager = MetricStager(config)

    stager.attach_metric(RampUpMetric(), 4)
    stager.attach_metric(BusFactorMetric(), 2)
    stager.attach_metric(PerformanceClaimsMetric(), 2)
    stager.attach_metric(LicenseMetric(), 1)
    stager.attach_metric(SizeMetric(), 3)
    stager.attach_metric(DatasetAndCodeScoreMetric(), 2)
    stager.attach_metric(DatasetQualityMetric(), 1)
    stager.attach_metric(CodeQualityMetric(), 1)
    stager.attach_metric(ReproducibilityMetric(), 1)
    stager.attach_metric(ReviewednessMetric(), 1)
    stager.attach_metric(TreeScoreMetric(), 1)

    return stager


def calculate_metrics(model_urls: ModelURLs, config: ConfigContract, stager: MetricStager) -> ModelStats:  # do we have a funciton to infer urls?
    """
    Calculate all metrics for a given model
    """
    model_paths: ModelPaths = generate_model_paths(config, model_urls)
    # check for database URLs already analyzed

    start_time: float = time.time()

    analyzer_output = run_workflow(stager, model_urls, model_paths, config)
    db_metrics = []

    for metric in analyzer_output.metrics:
        latency_ms = int(metric.runtime * 1000)
        if isinstance(metric.score, dict):
            db_metrics.append(DictMetric(metric.metric_name, metric.score, latency_ms))
        else:
            db_metrics.append(FloatMetric(metric.metric_name, metric.score, latency_ms))
    # Calculate net score latency
    net_latency: int = int((time.time() - start_time) * 1000)
    return ModelStats(
        model_url=model_urls.model,
        database_url=model_urls.dataset,
        code_url=model_urls.codebase,
        name=extract_model_repo_id(model_urls.model).split("/")[1],
        net_score=analyzer_output.score,
        net_score_latency=net_latency,
        metrics=db_metrics,
    )


app = typer.Typer()

DEFAULT_BASE_URL = os.environ.get("REGISTRY_BASE_URL", "")
DEFAULT_AUTH_TOKEN = os.environ.get("REGISTRY_AUTH_TOKEN", "")
@app.command()
def delete(
    artifact_type: str = typer.Argument(
        ...,
        help="Type of artifact to delete: model | dataset | code",
    ),
    artifact_id: str = typer.Argument(
        ...,
        help="ID of the artifact to delete",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        "--base-url",
        help="Base URL of the registry API (overrides REGISTRY_BASE_URL env var).",
    ),
    token: str = typer.Option(
        DEFAULT_AUTH_TOKEN,
        "--token",
        help="X-Authorization token (overrides REGISTRY_AUTH_TOKEN env var).",
    ),
):
    """
    Delete an artifact from the registry (DELETE /artifacts/{artifact_type}/{id}).
    """
    if not base_url:
        typer.echo("Error: base URL is required. Set REGISTRY_BASE_URL or use --base-url.", err=True)
        raise typer.Exit(code=1)

    if not token:
        typer.echo("Error: auth token is required. Set REGISTRY_AUTH_TOKEN or use --token.", err=True)
        raise typer.Exit(code=1)

    status = delete_artifact(base_url, artifact_type, artifact_id, token)
    if status != 200:
        raise typer.Exit(code=1)

@app.command()
def reset(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        "--base-url",
        help="Base URL of the registry API (overrides REGISTRY_BASE_URL env var).",
    ),
    token: str = typer.Option(
        DEFAULT_AUTH_TOKEN,
        "--token",
        help="X-Authorization token (overrides REGISTRY_AUTH_TOKEN env var).",
    ),
):
    """
    Reset the registry to its default state (DELETE /reset).
    """
    if not base_url:
        typer.echo("Error: base URL is required. Set REGISTRY_BASE_URL or use --base-url.", err=True)
        raise typer.Exit(code=1)

    if not token:
        typer.echo("Error: auth token is required. Set REGISTRY_AUTH_TOKEN or use --token.", err=True)
        raise typer.Exit(code=1)

    status = reset_registry(base_url, token)
    if status != 200:
        raise typer.Exit(code=1)


@app.command()
def install():
    """
    Installs necessary dependencies from dependencies.txt
    """
    logging.info("Installing dependencies...")
    try:
        deps_file = Path(__file__).parent.parent / "dependencies.txt"
        if deps_file.exists():
            result = subprocess.run(["pip", "install", "--user", "-r", str(deps_file)])
            if result.returncode != 0:
                typer.echo(
                    f"An error occurred while installing dependencies:\n{result.stderr}",
                    err=True,
                )
                raise typer.Exit(code=1)
            logging.info("Dependencies installed successfully.")
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def test():
    setup_logging()
    
    import unittest
    import io

    src_path = (Path(__file__).parent.parent / "src").resolve()
    tests_path = (Path(__file__).parent.parent / "tests").resolve()

    try:
        import coverage

        cov = coverage.Coverage(
            source=[str(src_path)],  # Only measure files in src/
            omit=[
                "*/tests/*",
                "*/test_*",
                "*/__pycache__/*",
                "*/venv/*",
                "*/env/*",
                "*/site-packages/*",
                "*/.venv/*",
            ]
        )
        cov.start()
        
        setup_logging()

        loader = unittest.TestLoader()
        start_dir = str(tests_path)
        suite = loader.discover(start_dir, pattern="test*.py")
        total_tests = suite.countTestCases()
        
        logging.debug(f"Test discovery starting from: {start_dir}")
        logging.debug(f"Tests discovered: {total_tests}")

        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
        result = runner.run(suite)
        
        cov.stop()
        cov.save()
        
        report_output = io.StringIO()
        coverage_percent = cov.report(file=report_output, show_missing=False)
        
        # Get the detailed report for logging
        report_content = report_output.getvalue()
        logging.debug("Coverage report:")
        logging.debug(report_content)
        
        coverage_data = cov.get_data()
        measured_files = coverage_data.measured_files()
        logging.debug(f"Files measured for coverage: {len(measured_files)}")
        
        # Log all measured files
        for filename in measured_files:
            logging.debug(f"Measured file: {filename}")

        passed_tests = total_tests - len(result.failures) - len(result.errors)
        typer.echo(
            f"{passed_tests}/{total_tests} test cases passed. {coverage_percent:.0f}% line coverage achieved."
        )

        if result.failures or result.errors:
            raise typer.Exit(code=1)
        else:
            raise typer.Exit(code=0)

    except ImportError:
        typer.echo(
            "Error: 'coverage' package not installed. Please run 'install' command first.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def analyze(url_file: Path):
    """
    Analyzes models based on URLs provided in a file.
    Will add model to database if not already present.
    """

    config: ConfigContract = ConfigContract(
        num_processes=5,
        run_multi=True,  # TODO: user False for debug, use True for production
        priority_function="PFReciprocal",
        target_platform="desktop_pc",
        local_storage_directory=os.path.dirname(os.path.abspath(__file__))
        + "local_storage",
        model_path_name="models",
        code_path_name="code",
        dataset_path_name="dataset",
    )

    metric_stager: MetricStager = stage_metrics(config)

    try:
        model_groups = parse_url_file(url_file)
        if not model_groups:
            typer.echo("No valid model URLs found in file.")
            raise typer.Exit(code=1)

        setup_logging()
        basic_schema = [
            FloatMetric("ramp_up_time", 0.0, 0),
            FloatMetric("bus_factor", 0.0, 0),
            FloatMetric("performance_claims", 0.0, 0),
            FloatMetric("license", 0.0, 0),
            DictMetric(
                "size_score",
                {
                    "raspberry_pi": 0.0,
                    "jetson_nano": 0.0,
                    "desktop_pc": 0.0,
                    "aws_server": 0.0,
                },
                0,
            ),
            FloatMetric("dataset_and_code_score", 0.0, 0),
            FloatMetric("dataset_quality", 0.0, 0),
            FloatMetric("code_quality", 0.0, 0),
            FloatMetric("reproducibility", 0.0, 0),
            FloatMetric("reviewedness", 0.0, 0),
            FloatMetric("tree_score", 0.0, 0),
        ]

        db = SQLiteAccessor(PROD_DATABASE_PATH, basic_schema)

        for model_urls in model_groups:
            model_url = model_urls.model

            # Check if model already analyzed
            if db.check_entry_in_db(model_url):
                logging.info(
                    f"Model {model_url} already analyzed. Fetching from database..."
                )
                stats = db.get_model_statistics(model_url)
            else:
                logging.info(f"Analyzing model {model_url}...")

                # Calculate metrics and add to database
                logging.info(f"Downloading resources for {model_url}...")
                try:
                    local_dir = Path(config.local_storage_directory)
                    download_manager = DownloadManager(
                        str(local_dir / config.model_path_name),
                        str(local_dir / config.code_path_name),
                        str(local_dir / config.dataset_path_name),
                    )
                    download_manager.download_model_resources(model_urls)
                    logging.info("Download completed successfully.")
                except Exception as e:
                    # Proceed without local files; some metrics (e.g., size, tree_score) can still run
                    typer.echo(f"Warning: Skipping downloads for {model_url}: {e}")

                # check for pre-existing datasets
                model_path = generate_model_paths(config, model_urls).model
                if model_path is not None and (model_path / "README.md").exists():
                    dataset_check = get_linked_dataset_metrics(
                        model_path / "README.md",
                        db,
                        [FloatMetric("dataset_quality", 0.0, 0)],
                    )
                    if dataset_check is not None:
                        model_urls.dataset = dataset_check[0]
                logging.debug(f"Starting metric calculation for {model_url}")
                stats = calculate_metrics(model_urls, config, metric_stager)
                logging.debug(f"Adding results to database for {model_url}")
                db.add_to_db(stats)

            results = {
                "name": stats.name,
                "category": "MODEL",
                "net_score": stats.net_score,
                "net_score_latency": stats.net_score_latency,
            }

            for metric in stats.metrics:
                if isinstance(metric, FloatMetric):
                    results[metric.name] = metric.data
                    results[f"{metric.name}_latency"] = metric.latency
                elif isinstance(metric, DictMetric):
                    results[metric.name] = metric.data
                    results[f"{metric.name}_latency"] = metric.latency

            typer.echo(json.dumps(results))

    except Exception as e:
        typer.echo(f"An error occurred during analysis: {e}")
        raise typer.Exit(code=1)
    sys.exit(0)


if __name__ == "__main__":
    app()
