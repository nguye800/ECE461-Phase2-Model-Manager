from metric import BaseMetric
from pathlib import Path
from typing_extensions import override
import requests
import re
import os
import io
import json
import time
import tarfile
import tempfile
import boto3
from botocore.exceptions import ClientError
from config import extract_model_repo_id


class PerformanceClaimsMetric(BaseMetric):
    metric_name: str = "performance_claims"
    model_dir: Path
    readme_file: Path

    def __init__(self):
        super().__init__()

    @override
    def setup_resources(self):
        if self.local_directory is None or self.local_directory.model is None:
            raise ValueError("Local model directory not specified")
        self.model_dir = Path(self.local_directory.model)
        self.readme_file = self.model_dir / "README.md"

    @override
    def calculate_score(self) -> float:
        """
        Evaluate performance claims by:
        - Scraping claimed benchmarks from the HuggingFace model card (README.md).
        - Launching an AWS SageMaker Processing job (container image in ECR) to run the
          provided model weights and code on a demo dataset.
        - Comparing measured metrics from the job against the claimed benchmarks.

        Required inputs (expected via config/environment):
        - self.url.model: HuggingFace model URL
        - self.url.dataset: Downloadable dataset URL (HTTP/HTTPS)
        - self.local_directory.model: local path containing a model zip (weights)
        - self.local_directory.codebase: local path to code that loads weights in PyTorch

        Environment variables expected:
        - ECR_IMAGE: ECR image URI for the evaluation container
        - SAGEMAKER_ROLE_ARN: IAM role for SageMaker
        - S3_BUCKET: S3 bucket for job inputs/outputs
        - AWS_REGION: AWS region (default us-east-1)
        - SAGEMAKER_INSTANCE_TYPE: instance type (default ml.m5.large)
        """

        # Basic validation of required fields
        if self.url is None or self.url.model is None:
            return 0.0

        ecr_image = os.getenv("ECR_IMAGE")
        role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
        s3_bucket = os.getenv("S3_BUCKET")
        region = os.getenv("AWS_REGION", "us-east-1")
        instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")

        if not all([ecr_image, role_arn, s3_bucket]):
            # Missing critical infra configuration
            return 0.0

        sagemaker = boto3.client("sagemaker", region_name=region)
        s3 = boto3.client("s3", region_name=region)

        # 1) Scrape model card from HuggingFace to extract claimed benchmarks
        def fetch_model_card(md_url: str) -> str | None:
            try:
                repo = extract_model_repo_id(md_url)
                url = f"https://huggingface.co/{repo}/raw/main/README.md"
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                return r.text
            except Exception:
                return None

        def parse_claimed_metrics(markdown: str) -> dict:
            """
            Very simple parser for common benchmark tables/mentions.
            Looks for markdown table rows or inline patterns with metrics.
            Extracts: accuracy, f1, bleu, rouge, precision, recall, mAP.
            """
            claimed: dict[str, float] = {}
            if not markdown:
                return claimed

            text = markdown.lower()
            # Inline patterns e.g., "accuracy: 0.93", "f1 = 84.1%"
            patterns = {
                "accuracy": r"accuracy\s*[:=]\s*(\d+\.?\d*)%?",
                "f1": r"f1\s*[:=]\s*(\d+\.?\d*)%?",
                "bleu": r"bleu\s*[:=]\s*(\d+\.?\d*)%?",
                "rouge": r"rouge(?:-l)?\s*[:=]\s*(\d+\.?\d*)%?",
                "precision": r"precision\s*[:=]\s*(\d+\.?\d*)%?",
                "recall": r"recall\s*[:=]\s*(\d+\.?\d*)%?",
                "map": r"m\s*ap\s*[:=]\s*(\d+\.?\d*)%?",
            }
            for name, pat in patterns.items():
                try:
                    m = re.search(pat, text)
                    if m:
                        val = float(m.group(1))
                        # Convert percentage if likely (values > 1.0)
                        claimed[name] = val / 100.0 if val > 1.0 else val
                except Exception:
                    pass

            # Parse markdown tables lines with '|' and numeric cells
            for line in text.splitlines():
                if "|" in line and any(k in line for k in patterns.keys()):
                    # e.g., | Metric | Accuracy | Value | 93.2 |
                    nums = re.findall(r"(\d+\.?\d*)%?", line)
                    if nums:
                        # Heuristic: associate the first known metric keyword with last number
                        for k in patterns.keys():
                            if k in line and k not in claimed:
                                val = float(nums[-1])
                                claimed[k] = val / 100.0 if val > 1.0 else val
                                break

            return claimed

        model_card = fetch_model_card(self.url.model)
        claimed_metrics = parse_claimed_metrics(model_card or "")
        if not claimed_metrics:
            # No claims found; cannot evaluate
            return 0.0

        # 2) Prepare inputs in S3: model zip and evaluation code archive
        job_name = f"perf-claims-{int(time.time())}"
        s3_prefix = f"performance_claims/{job_name}"
        output_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/output"

        def upload_file(fp: Path, key: str) -> str | None:
            try:
                s3.upload_file(str(fp), s3_bucket, key)
                return f"s3://{s3_bucket}/{key}"
            except Exception:
                return None

        # Find model zip in local_directory.model
        model_zip_s3 = None
        if self.local_directory and self.local_directory.model and self.local_directory.model.exists():
            model_dir = Path(self.local_directory.model)
            zips = list(model_dir.glob("*.zip"))
            if zips:
                model_zip_s3 = upload_file(zips[0], f"{s3_prefix}/inputs/{zips[0].name}")

        # Tar.gz the codebase directory if provided
        code_s3 = None
        if self.local_directory and self.local_directory.codebase and Path(self.local_directory.codebase).exists():
            code_dir = Path(self.local_directory.codebase)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
                tar_path = Path(tf.name)
            try:
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(code_dir, arcname=code_dir.name)
                code_s3 = upload_file(tar_path, f"{s3_prefix}/inputs/{code_dir.name}.tar.gz")
            finally:
                try:
                    os.unlink(tar_path)
                except Exception:
                    pass

        # 3) Launch SageMaker Processing Job using the provided ECR image
        processing_inputs = []
        if model_zip_s3:
            processing_inputs.append({
                "InputName": "model_zip",
                "S3Input": {
                    "S3Uri": model_zip_s3,
                    "LocalPath": "/opt/ml/processing/input/model.zip",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            })
        if code_s3:
            processing_inputs.append({
                "InputName": "code_archive",
                "S3Input": {
                    "S3Uri": code_s3,
                    "LocalPath": "/opt/ml/processing/input/code.tar.gz",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            })

        environment = {
            "DATASET_HTTP_URL": self.url.dataset or "",
            "HUGGINGFACE_MODEL_URL": self.url.model,
            "CLAIMED_METRICS_JSON": json.dumps(claimed_metrics),
            "OUTPUT_DIR": "/opt/ml/processing/output"
        }

        try:
            sagemaker.create_processing_job(
                ProcessingJobName=job_name,
                RoleArn=role_arn,
                AppSpecification={
                    "ImageUri": ecr_image,
                },
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": instance_type,
                        "VolumeSizeInGB": 30,
                    }
                },
                ProcessingInputs=processing_inputs,
                ProcessingOutputConfig={
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": output_s3_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                Environment=environment,
            )
        except ClientError:
            return 0.0

        # 4) Poll for job completion (basic)
        status = "InProgress"
        start = time.time()
        timeout_s = 60 * 60  # 60 minutes
        while status in ("InProgress", "Stopping"):
            if time.time() - start > timeout_s:
                return 0.0
            time.sleep(15)
            try:
                desc = sagemaker.describe_processing_job(ProcessingJobName=job_name)
                status = desc.get("ProcessingJobStatus", "Failed")
            except ClientError:
                return 0.0

        if status != "Completed":
            return 0.0

        # 5) Download evaluation metrics from S3 and compare
        # Expect metrics.json under the output prefix
        key = f"{s3_prefix}/output/metrics.json"
        try:
            buf = io.BytesIO()
            s3.download_fileobj(s3_bucket, key, buf)
            buf.seek(0)
            measured = json.loads(buf.read().decode("utf-8"))
        except Exception:
            return 0.0

        measured_metrics = measured.get("metrics", {}) if isinstance(measured, dict) else {}
        if not measured_metrics:
            return 0.0

        # Compare with tolerance; score is fraction of claims met/exceeded
        tolerance = 0.0  # strict compare; adjust if desired
        names = set(k for k in claimed_metrics.keys() if k in measured_metrics)
        if not names:
            return 0.0

        met = 0
        for name in names:
            claimed_v = float(claimed_metrics[name])
            measured_v = float(measured_metrics[name])
            if measured_v + tolerance >= claimed_v:
                met += 1

        return met / len(names)
