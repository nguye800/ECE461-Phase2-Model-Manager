import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import requests
from typing_extensions import override
from dotenv import load_dotenv

from config import ModelURLs
from metric import BaseMetric

logger = logging.getLogger(__name__)

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0",
)

try:  # pragma: no cover - exercised only when AWS creds are present
    BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
except Exception as exc:  # pragma: no cover
    BEDROCK_CLIENT = None
    logger.debug("Bedrock client unavailable: %s", exc)

README_CANDIDATES = (
    "README.md",
    "Readme.md",
    "readme.md",
    "README",
    "readme",
)

HUGGINGFACE_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?huggingface\.co/([^/]+)/([^/\s]+)"
)
GITHUB_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/\s]+)")

MAX_DOC_CHARS = 6000

DOC_PROMPT = (
    "You are reviewing documentation to estimate how quickly a developer can integrate "
    "a Hugging Face model. Consider clarity, setup steps, dependency info, API usage, "
    "code examples, and troubleshooting advice. Respond with JSON shaped exactly as "
    '{"score": <number between 0 and 1>, "explanation": "<short reason>"}.'
)


class RampUpMetric(BaseMetric):
    metric_name: str = "ramp_up_time"

    def __init__(self) -> None:
        super().__init__()
        self.model_card_text: str = ""
        self.code_readme_text: str = ""
        self.bedrock_client = BEDROCK_CLIENT

    @override
    def setup_resources(self) -> None:
        self.model_card_text = self._load_model_card_text()
        self.code_readme_text = self._load_codebase_readme()

    @override
    def calculate_score(self) -> float:
        scores = []
        model_score = self._score_text(self.model_card_text, "model card")
        if model_score is not None:
            scores.append(model_score)
        code_score = self._score_text(self.code_readme_text, "code repository")
        if code_score is not None:
            scores.append(code_score)

        if not scores:
            return 0.0

        composite = sum(scores) / len(scores)
        return self._clamp(composite)

    def _score_text(self, text: Optional[str], source_name: str) -> Optional[float]:
        if not text or not text.strip():
            return None

        excerpt = text.strip()
        if len(excerpt) > MAX_DOC_CHARS:
            excerpt = excerpt[:MAX_DOC_CHARS]

        prompt = f"{DOC_PROMPT}\n\nDocumentation source: {source_name}\n```\n{excerpt}\n```"

        try:
            score = self._invoke_bedrock(prompt)
            if score is not None:
                return self._clamp(score)
        except Exception as exc:
            logger.warning(
                "RampUpMetric Bedrock scoring failed for %s: %s", source_name, exc
            )

        return self._heuristic_score(excerpt)

    def _invoke_bedrock(self, prompt: str) -> Optional[float]:
        client = self.bedrock_client
        if client is None:
            raise RuntimeError("Bedrock client is not configured")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
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

        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        raw = response.get("body")
        payload = (
            json.loads(raw.read().decode("utf-8"))
            if hasattr(raw, "read")
            else json.loads(raw)
        )

        content = payload.get("content", [])
        text_parts = [
            piece.get("text", "")
            for piece in content
            if isinstance(piece, dict) and piece.get("type") == "text"
        ]
        if not text_parts and isinstance(payload.get("output"), list):
            for message in payload["output"]:
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

        combined = "\n".join(filter(None, text_parts)).strip()
        return self._parse_score(combined)

    def _parse_score(self, raw_text: str) -> Optional[float]:
        if not raw_text:
            return None

        cleaned = raw_text.strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "score" in data:
                return float(data["score"])
        except json.JSONDecodeError:
            pass

        match = re.search(r"([01](?:\.\d+)?)", cleaned)
        if match:
            return float(match.group(1))
        return None

    def _heuristic_score(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = max(1, len(text.split()))
        coverage = min(1.0, tokens / 1200)
        structure_bonus = 0.1 if ("```" in text or "#" in text) else 0.0
        return self._clamp(min(1.0, coverage + structure_bonus))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _load_model_card_text(self) -> str:
        if not self.url or not self.url.model:
            return ""

        owner, repo = self._extract_hf_repo(self.url.model)
        if not owner:
            return self._safe_http_get(self.url.model) or ""

        for candidate in self._candidate_model_card_urls(owner, repo):
            text = self._safe_http_get(candidate)
            if text:
                return text
        return ""

    def _candidate_model_card_urls(self, owner: str, repo: str) -> List[str]:
        base = f"https://huggingface.co/{owner}/{repo}"
        return [
            f"{base}/raw/main/README.md",
            f"{base}/raw/master/README.md",
            f"{base}/resolve/main/README.md",
            f"{base}/resolve/master/README.md",
        ]

    def _extract_hf_repo(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        match = HUGGINGFACE_PATTERN.search(url)
        if match:
            return match.group(1), match.group(2)
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return None, None

    def _load_codebase_readme(self) -> str:
        local = self._read_local_code_readme()
        if local:
            return local
        return self._fetch_remote_code_readme()

    def _read_local_code_readme(self) -> Optional[str]:
        if not self.local_directory or not self.local_directory.codebase:
            return None

        base_path = Path(self.local_directory.codebase)
        if not base_path.exists():
            return None

        for candidate in README_CANDIDATES:
            file_path = base_path / candidate
            if file_path.exists():
                try:
                    return file_path.read_text(encoding="utf-8")
                except Exception:
                    continue

        for file_path in base_path.glob("README*"):
            if file_path.is_file():
                try:
                    return file_path.read_text(encoding="utf-8")
                except Exception:
                    continue
        return None

    def _fetch_remote_code_readme(self) -> str:
        if not self.url or not self.url.codebase:
            return ""

        owner_repo = self._extract_github_repo(self.url.codebase)
        if owner_repo:
            owner, repo = owner_repo
            repo = repo.replace(".git", "")
            for branch in ("main", "master"):
                raw_url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
                )
                text = self._safe_http_get(raw_url)
                if text:
                    return text

        fetched = self._safe_http_get(self.url.codebase)
        return fetched or ""

    def _extract_github_repo(self, url: str) -> Optional[Tuple[str, str]]:
        match = GITHUB_PATTERN.search(url)
        if match:
            return match.group(1), match.group(2)
        return None

    def _safe_http_get(self, url: str) -> Optional[str]:
        if not url:
            return None
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "model-manager/1.0"},
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.debug("RampUpMetric failed to fetch %s: %s", url, exc)
            return None


if __name__ == "__main__":  # pragma: no cover
    load_dotenv()

    model_url = "https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb"
    code_url = "https://github.com/adapter-hub/adapters"

    metric = RampUpMetric()
    metric.set_url(ModelURLs(model=model_url, codebase=code_url))

    print("Running RampUpMetric smoke test...")
    metric.setup_resources()
    score = metric.calculate_score()

    print("\n" + "=" * 60)
    print("Ramp-Up Metric Results")
    print("=" * 60)
    print(f"Model URL   : {model_url}")
    print(f"Code URL    : {code_url}")
    print(f"Score       : {score:.3f}")
    if metric.model_card_text:
        print(f"Model card  : {len(metric.model_card_text.split())} words fetched")
    else:
        print("Model card  : not available")
    if metric.code_readme_text:
        print(f"Code README : {len(metric.code_readme_text.split())} words fetched")
    else:
        print("Code README : not available")
