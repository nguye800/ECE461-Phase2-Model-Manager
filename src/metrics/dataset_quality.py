from typing_extensions import override

from metric import BaseMetric
from huggingface_hub import dataset_info
from config import extract_dataset_repo_id
import math


class DatasetQualityMetric(BaseMetric):
    metric_name: str = "dataset_quality"

    def __init__(self):
        return super().__init__()

    def scale_logarithmically(self, value: float, zero: float, one: float) -> float:
        if value == 0:
            return 0
        if zero == 0:
            zero = 1
        raw = math.log10(value / zero) / math.log10(one)
        return min(max(0.0, raw), 1.0)

    @override
    def setup_resources(self): ...

    @override
    def calculate_score(self) -> float:
        if self.url is None or self.url.dataset is None:
            return 0.0
        dataset_stats = dataset_info(extract_dataset_repo_id(self.url.dataset))
        dataset_card = dataset_stats.card_data
        if dataset_card is None:
            row_score = 0.0
        else:
            total_rows = sum(
                [
                    split["num_examples"]
                    for split in dataset_card["dataset_info"]["splits"]
                ]
            )
            row_score = self.scale_logarithmically(total_rows, 10**2, 10**6)

        downloads = dataset_stats.downloads_all_time
        if downloads is None:
            downloads = dataset_stats.downloads
        if downloads is None:
            download_score = 0.0
        else:
            download_score = self.scale_logarithmically(downloads, 10, 10**4)

        likes = dataset_stats.likes
        if likes is None:
            like_score = 0.0
        else:
            like_score = self.scale_logarithmically(likes, 0, 10 * 10**2)

        return row_score * 0.5 + download_score * 0.3 + like_score * 0.2
