import typing
import time
from io import StringIO
from math import exp, log
from typing_extensions import override, Literal
import logging

import contextlib
import torch
from transformers import AutoTokenizer, AutoModel

from metric import BaseMetric


class RampUpMetric(BaseMetric):
    metric_name: str = "ramp_up_time"

    def __init__(
        self,
        half_score_time_minutes: float,
        device_type: Literal["cpu", "mps", "cuda", "cuda:0"],
    ):
        super().__init__()
        assert half_score_time_minutes > 0.0
        self.device_type: Literal["cpu", "mps", "cuda", "cuda:0"] = device_type
        self.exponential_coefficient: float = -log(0.5) / half_score_time_minutes

    @override
    def setup_resources(self):
        pass

    def installation_spin_up_score(self):
        start_load_time: float = time.time()
        buf: StringIO = StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tokenizer: typing.Any = AutoTokenizer.from_pretrained(
                self.local_directory.model.resolve()
            )
            model: typing.Any = AutoModel.from_pretrained(
                self.local_directory.model.resolve()
            ).to(self.device_type)
            inputs = tokenizer("Hello world", return_tensors="pt").to(self.device_type)
            with torch.no_grad():
                _ = model(**inputs)

            total_time: float = time.time() - start_load_time

        return exp(-self.exponential_coefficient * (total_time / 60))

    @override
    def calculate_score(self) -> float:
        try:
            ramp_up: float = self.installation_spin_up_score()
            return ramp_up
        except Exception:
            logging.debug("No local model access")
            return 0.0
