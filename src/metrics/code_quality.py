import os
import re
from io import StringIO
from typing_extensions import override
from pylint.lint import pylinter, Run
from pylint.reporters.text import TextReporter

from metric import BaseMetric


class CodeQualityMetric(BaseMetric):
    metric_name: str = "code_quality"

    def __init__(self):
        super().__init__()
        self.file_list: list[str] = []

    @override
    def setup_resources(self):
        if self.local_directory is None:
            return
        if self.local_directory.codebase is not None:
            for root, _, files in os.walk(self.local_directory.codebase):
                for file in files:
                    if file.endswith(".py"):
                        self.file_list.append(os.path.join(root, file))
        if self.local_directory.model is not None:
            for root, _, files in os.walk(self.local_directory.model):
                for file in files:
                    if file.endswith(".py"):
                        self.file_list.append(os.path.join(root, file))

    @override
    def calculate_score(self) -> float:
        if not self.file_list:
            self._set_debug_details("no python files detected for linting")
            return 0.0

        pylinter.MANAGER.clear_cache()

        output_stream: StringIO = StringIO()
        reporter: TextReporter = TextReporter(output_stream)

        Run(
            ["--disable=line-too-long", "--disable=bad-indentation", "--disable=import-error"] + self.file_list, reporter=reporter, exit=False
        )
        match = re.search(r"rated at ([0-9]+\.[0-9]+)/10", output_stream.getvalue())

        if match is None:
            self._set_debug_details("pylint did not return an aggregate score")
            return 0.0

        raw_score = float(match.group(1)) / 10
        self._set_debug_details(
            f"analyzed_files={len(self.file_list)} pylint_score={float(match.group(1)):.2f}/10"
        )
        return raw_score
