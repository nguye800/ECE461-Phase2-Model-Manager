from metric import BaseMetric
from typing_extensions import override
from pathlib import Path
import requests


class DatasetAndCodeScoreMetric(BaseMetric):
    # UPDATE ME TO USE LOCAL COPIES OF REPOS
    metric_name: str = "dataset_and_code_score"
    model_dir: Path
    readme_file: Path

    def __init__(self):
        super().__init__()

    @override
    def setup_resources(self):
        if self.local_directory is None or self.local_directory.model is None:
            raise ValueError("Local directory not specified")
        self.model_dir = Path(self.local_directory.model)
        self.readme_file = self.model_dir / "README.md"

    @override
    def calculate_score(self) -> float:
        score = 0.0
        signals: list[str] = []

        # Check dataset availability
        try:
            if self.url and self.url.dataset:
                response = requests.head(self.url.dataset)
                if response.status_code == 200:
                    score += 0.3
                    signals.append("dataset_url reachable")
        except requests.RequestException:
            pass

        # Check code availability
        try:
            if self.url and self.url.codebase:
                response = requests.head(self.url.codebase)
                if response.status_code == 200:
                    score += 0.3
                    signals.append("code_url reachable")
        except requests.RequestException:
            pass

        # Check online documentation
        try:
            if self.url and self.url.dataset:
                response = requests.get(self.url.dataset)
                if "dataset description" in response.text.lower():
                    score += 0.2
                    signals.append("dataset page has description")
        except requests.RequestException:
            pass

        # Check README for dataset and code info
        doc_score = 0.0
        if self.readme_file.exists:
            readme_content = self.readme_file.read_text(encoding="utf-8").lower()
            documentation_markers = {
                "dataset": ["dataset", "data description", "training data"],
                "usage": ["usage", "how to use", "getting started"],
                "examples": ["example", "sample usage"],
                "requirements": ["requirements", "dependencies", "installation"],
                "limitations": ["limitations", "constraints", "known issues"],
            }
            for section in documentation_markers.values():
                for marker in section:
                    if marker in readme_content:
                        doc_score += 0.2 / len(documentation_markers)
            doc_score = min(0.2, doc_score)
            if doc_score > 0:
                signals.append("README mentions dataset/code markers")

        final_score = min(1, score + doc_score)
        details = ", ".join(signals) if signals else "no signals detected"
        self._set_debug_details(
            f"{details} -> base={score:.3f} doc_bonus={doc_score:.3f}"
        )
        return final_score
