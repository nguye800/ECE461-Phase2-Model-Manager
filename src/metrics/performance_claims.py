from metric import BaseMetric
from pathlib import Path
from typing_extensions import override
import requests
import re

API_KEY = "sk-c089cffb672740b4b99d38dad7c97677"
LLM_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"


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

        prompt_base = """Use the following README content to score the model's performance claims on a scale of 0 to 1. Look for 
        benchmark keywords, numerical results, academic references, and performance comparisons. The benchmark score has a weight 
        of 0.4, numerical results have a weight of 0.3, academic references have a weight of 0.2, and performance comparisions have a 
        weight of 0.1. Output the calculated score as a floating point and nothing else."""

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama3.1:latest",
            "messages": [
                {"role": "user", "content": readme_content},
                {"role": "system", "content": prompt_base},
            ],
        }

        final_score = requests.post(LLM_API_URL, headers=headers, json=data)
        # remove every non-number character
        response: str = final_score.json()["choices"][0]["message"]["content"]
        try:
            final_number = re.findall(r"0\.\d+", response)[-1]
            final_score = float(final_number)
        except:
            return 0.0
        return final_score
