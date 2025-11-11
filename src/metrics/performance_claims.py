from metric import BaseMetric
from pathlib import Path
from typing_extensions import override
from typing import Optional
import os
import re
import json
import logging
import boto3

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

try:  # pragma: no cover - requires AWS creds
    BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
except Exception as exc:  # pragma: no cover
    BEDROCK_CLIENT = None
    logger.debug("PerformanceClaimsMetric Bedrock client unavailable: %s", exc)


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
        if not self.readme_file.exists():
            return 0.0

        readme_content = self.readme_file.read_text(encoding="utf-8").lower()

        score = self._score_with_bedrock(readme_content)
        if score is not None:
            return score

        return self._heuristic_score(readme_content)

    def _score_with_bedrock(self, readme: str) -> Optional[float]:
        client = BEDROCK_CLIENT
        if client is None:
            return None

        excerpt = readme[:6000]
        prompt = (
            "You are auditing a machine learning model card. Evaluate the reliability of its "
            "performance claims. Consider whether the claims cite specific benchmarks/tasks, "
            "provide numerical results, reference external evidence, and align with the described "
            "dataset/task. Respond ONLY with JSON in the form "
            '{"score": <number between 0 and 1>, "explanation": "<short reason>"}.\n'
            f"README excerpt:\n```\n{excerpt}\n```"
        )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }

        try:
            response = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("PerformanceClaimsMetric Bedrock call failed: %s", exc)
            return None

        payload = response.get("body")
        payload_json = (
            json.loads(payload.read().decode("utf-8"))
            if hasattr(payload, "read")
            else json.loads(payload)
        )
        text_segments = [
            part.get("text", "")
            for part in payload_json.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        combined = "\n".join(text_segments).strip()
        if not combined:
            return None

        try:
            data = json.loads(combined)
            score = float(data.get("score"))
            return max(0.0, min(1.0, score))
        except (ValueError, json.JSONDecodeError, TypeError):
            matches = re.findall(r"0?\.\d+", combined)
            if matches:
                score = float(matches[-1])
                return max(0.0, min(1.0, score))
        return None

    def _heuristic_score(self, readme: str) -> float:
        # Fallback heuristic: count evidence signals
        score = 0.0
        if "benchmark" in readme or "leaderboard" in readme:
            score += 0.4
        if re.search(r"\d+\.\d+\s*%", readme):
            score += 0.3
        if "paper" in readme or "arxiv" in readme or "reference" in readme:
            score += 0.2
        if "compared to" in readme or "vs." in readme:
            score += 0.1
        return max(0.0, min(1.0, score))
