import math
import re
from typing import Any, Iterable, Sequence

from typing_extensions import override

from config import extract_dataset_repo_id
from huggingface_hub import dataset_info
from metric import BaseMetric
from metrics.license import license_score


SIZE_CATEGORY_ESTIMATES: dict[str, int] = {
    "n<1k": 500,
    "1k<n<10k": 5_500,
    "10k<n<100k": 55_000,
    "100k<n<1m": 550_000,
    "1m<n<10m": 5_500_000,
    "10m<n<100m": 55_000_000,
    "100m<n<1b": 550_000_000,
}
SIZE_TAG_PREFIX = "size_categories:"
UNIT_FACTORS = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def _safe_get(obj: Any, path: Sequence[str]) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_numeric_token(token: str) -> float | None:
    match = re.match(r"(?P<num>\d+(?:\.\d+)?)(?P<unit>[kKmMbB]?)", token)
    if not match:
        return None
    num = float(match.group("num"))
    unit = match.group("unit").lower()
    return num * UNIT_FACTORS.get(unit, 1)


def _estimate_from_size_hint(hint: str) -> float | None:
    normalized = hint.strip().lower().replace("–", "-")
    normalized = normalized.replace(" ", "")
    if not normalized:
        return None
    if "<n<" in normalized:
        low, high = normalized.split("<n<", 1)
    elif "-" in normalized:
        low, high = normalized.split("-", 1)
    else:
        return _parse_numeric_token(normalized.strip("<>"))
    low_val = _parse_numeric_token(low.strip("<>="))
    high_val = _parse_numeric_token(high.strip("<>="))
    if low_val and high_val:
        return (low_val + high_val) / 2.0
    return low_val or high_val


def _extract_first_numeric(candidate: Any) -> float | None:
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, str):
        numeric = re.findall(r"\d+(?:\.\d+)?", candidate.replace(",", ""))
        if numeric:
            try:
                return float(numeric[0])
            except ValueError:
                return None
    return None


class DatasetQualityMetric(BaseMetric):
    metric_name: str = "dataset_quality"

    def __init__(self):
        return super().__init__()

    def scale_logarithmically(self, value: float | None, zero: float, one: float) -> float:
        if value is None or value <= 0:
            return 0.0
        if zero == 0:
            zero = 1
        raw = math.log10(value / zero) / math.log10(one)
        return min(max(0.0, raw), 1.0)

    @override
    def setup_resources(self): ...

    def _estimate_row_count(self, dataset_stats, dataset_card: dict[str, Any]) -> float | None:
        dataset_info_block = dataset_card.get("dataset_info") or {}
        splits = dataset_info_block.get("splits") or dataset_card.get("splits")
        if not splits:
            splits = getattr(dataset_stats, "splits", None)
        if isinstance(splits, list):
            total_rows = 0
            for split in splits:
                if isinstance(split, dict):
                    value = split.get("num_examples") or split.get("num_rows")
                    if isinstance(value, (int, float)):
                        total_rows += int(value)
            if total_rows > 0:
                return float(total_rows)

        size_hints: list[str] = []
        size_categories = dataset_card.get("size_categories")
        if isinstance(size_categories, str):
            size_hints.append(size_categories)
        elif isinstance(size_categories, Iterable):
            size_hints.extend([str(item) for item in size_categories if item])

        tags = dataset_card.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            for tag in tags:
                if isinstance(tag, str) and tag.lower().startswith(SIZE_TAG_PREFIX):
                    size_hints.append(tag.split(":", 1)[1])

        if "size" in dataset_card and isinstance(dataset_card["size"], str):
            size_hints.append(dataset_card["size"])

        for hint in size_hints:
            normalized = hint.strip().lower()
            if normalized in SIZE_CATEGORY_ESTIMATES:
                return float(SIZE_CATEGORY_ESTIMATES[normalized])
            approx = _estimate_from_size_hint(normalized)
            if approx:
                return approx

        return None

    def _extract_number(
        self,
        dataset_stats,
        dataset_card: dict[str, Any],
        attr_name: str,
        candidate_paths: Sequence[Sequence[str]],
    ) -> float | None:
        direct = getattr(dataset_stats, attr_name, None)
        numeric = _extract_first_numeric(direct)
        if numeric is not None:
            return numeric
        for path in candidate_paths:
            candidate = _safe_get(dataset_card, path)
            numeric = _extract_first_numeric(candidate)
            if numeric is not None:
                return numeric
        return None

    def _normalize_license(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower() or None
        if isinstance(value, dict):
            for key in ("id", "identifier", "value", "name"):
                normalized = self._normalize_license(value.get(key))
                if normalized:
                    return normalized
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for entry in value:
                normalized = self._normalize_license(entry)
                if normalized:
                    return normalized
        return None

    def _extract_license_id(self, dataset_stats, dataset_card: dict[str, Any]) -> str | None:
        candidates: list[Any] = [
            getattr(dataset_stats, "license", None),
            dataset_card.get("license"),
            _safe_get(dataset_card, ("dataset_info", "license")),
            dataset_card.get("licenses"),
            _safe_get(dataset_card, ("dataset_info", "licenses")),
        ]

        tags = dataset_card.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            for tag in tags:
                if isinstance(tag, str) and tag.lower().startswith("license:"):
                    candidates.append(tag.split(":", 1)[1])

        for candidate in candidates:
            normalized = self._normalize_license(candidate)
            if normalized:
                return normalized
        return None

    @override
    def calculate_score(self) -> float:
        if self.url is None or self.url.dataset is None:
            return 0.0

        dataset_stats = dataset_info(extract_dataset_repo_id(self.url.dataset))
        dataset_card = dataset_stats.card_data or {}

        row_count = self._estimate_row_count(dataset_stats, dataset_card)
        row_score = self.scale_logarithmically(row_count, 10**2, 10**6)

        downloads_value = self._extract_number(
                dataset_stats,
                dataset_card,
                "downloads_all_time",
                [
                    ("downloads",),
                    ("dataset_info", "downloads"),
                    ("stats", "downloads"),
                ],
            )
        if downloads_value is None:
            downloads_value = self._extract_number(
                dataset_stats,
                dataset_card,
                "downloads",
                [
                    ("statistics", "downloads"),
                    ("community_stats", "downloads"),
                ],
            )
        download_score = self.scale_logarithmically(
            downloads_value,
            10,
            10**4,
        )

        likes_value = self._extract_number(
                dataset_stats,
                dataset_card,
                "likes",
                [
                    ("likes",),
                    ("statistics", "likes"),
                    ("community_stats", "likes"),
                ],
            )
        likes_score = self.scale_logarithmically(
            likes_value,
            1,
            10 * 10**2,
        )

        license_id = self._extract_license_id(dataset_stats, dataset_card)
        license_component = license_score.get(license_id, 0.0) if license_id else 0.0

        final_score = (
            row_score * 0.45
            + download_score * 0.25
            + likes_score * 0.2
            + license_component * 0.1
        )
        self._set_debug_details(
            "rows="
            f"{row_count if row_count is not None else 'unknown'}, "
            f"downloads={downloads_value if downloads_value is not None else 'unknown'}, "
            f"likes={likes_value if likes_value is not None else 'unknown'}, "
            f"license={license_id or 'unknown'}"
        )
        return final_score
