from typing_extensions import override
from typing import List, Optional, Set
import os
from pathlib import Path

from metric import BaseMetric
from config import (
    ConfigContract,
    ModelURLs,
    ModelPaths,
    extract_model_repo_id,
    generate_model_paths,
)
from workflow import MetricStager, run_workflow

from huggingface_hub import model_info, hf_hub_download
from download_manager import DownloadManager


def _normalize_model_id(candidate: str) -> Optional[str]:
    """
    Normalize a candidate string into a Hugging Face id.
    Accepts both namespaced ("owner/name") and single-segment ids (e.g., "bert-base-uncased").
    Returns None if it cannot be normalized.
    """
    if not candidate:
        return None
    s = candidate.strip()
    if s.startswith("http://") or s.startswith("https://"):
        s = s.replace("https://huggingface.co/", "").replace(
            "http://huggingface.co/", ""
        )
    s = s.replace("/tree/main", "").strip("/")
    # Accept owner/name or single token ids (e.g., bert-base-uncased, imdb datasets)
    if "/" in s:
        return s
    # Single-segment HF ids are valid; basic sanity check on characters
    import re
    return s if re.fullmatch(r"[A-Za-z0-9_.\-]+", s) else None


class TreeScoreMetric(BaseMetric):
    metric_name: str = "tree_score"

    def __init__(self):
        super().__init__()

    @override
    def setup_resources(self):
        # No local resources required; data fetched via Hugging Face Hub API
        ...

    def _extract_parents_from_card(self, repo_id: str) -> Set[str]:
        parents: Set[str] = set()
        try:
            info = model_info(repo_id)
        except Exception:
            return parents

        # 1) Card data hints
        card = getattr(info, "cardData", None) or {}
        for key in [
            "base_model",
            "base_models",
            "base_model_repo",
            "base_model_name_or_path",
            "source_model",
        ]:
            value = card.get(key)
            if isinstance(value, str):
                normalized = _normalize_model_id(value)
                if normalized:
                    parents.add(normalized)
            elif isinstance(value, list):
                for item in value:
                    normalized = _normalize_model_id(str(item))
                    if normalized:
                        parents.add(normalized)

        # 2) Config hints
        cfg = getattr(info, "config", None) or {}
        for key in ["base_model", "base_model_name_or_path"]:
            if key in cfg:
                normalized = _normalize_model_id(str(cfg[key]))
                if normalized:
                    parents.add(normalized)

        # 3) Adapter configs (PEFT)
        for fname in [
            "adapter_config.json",
            "lora_config.json",
            "model_index.json",
        ]:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=None)
                import json

                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                for key in [
                    "base_model",
                    "base_model_name_or_path",
                    "base_model_repo",
                    "model_name",
                ]:
                    if key in obj:
                        normalized = _normalize_model_id(str(obj[key]))
                        if normalized:
                            parents.add(normalized)
                # Some model_index.json store nested metadata
                if isinstance(obj, dict) and "results" in obj:
                    for res in obj.get("results", []):
                        if isinstance(res, dict):
                            src = (
                                res.get("task", {}).get("source_model")
                                or res.get("source_model")
                            )
                            normalized = _normalize_model_id(str(src)) if src else None
                            if normalized:
                                parents.add(normalized)
            except Exception:
                # If file not found or network issues, skip silently
                continue

        return parents

    def _stage_base_metrics(
        self,
        config: ConfigContract,
        include_bus_factor: bool,
        include_dataset_quality: bool,
        include_dataset_and_code: bool,
        include_size: bool = True,
        include_license: bool = False,
        include_code_quality: bool = False,
        include_performance_claims: bool = False,
    ) -> MetricStager:
        """
        Build a MetricStager dynamically based on available resources.

        - Always prefers lightweight metrics.
        - Skips metrics that require unavailable URLs/local files.
        """
        stager = MetricStager(config)

        # Size metric: uses HF API only
        if include_size:
            try:
                from metrics.size_metric import SizeMetric  # type: ignore
                stager.attach_metric(SizeMetric(), 3)
            except Exception:
                pass

        # Bus factor: can compute from HF model commits even without GitHub
        if include_bus_factor:
            try:
                from metrics.bus_factor import BusFactorMetric  # type: ignore
                stager.attach_metric(BusFactorMetric(), 2)
            except Exception:
                pass

        # Dataset-related metrics only if dataset URL present
        if include_dataset_and_code:
            try:
                from metrics.dataset_and_code import DatasetAndCodeScoreMetric  # type: ignore
                stager.attach_metric(DatasetAndCodeScoreMetric(), 2)
            except Exception:
                pass
        if include_dataset_quality:
            try:
                from metrics.dataset_quality import DatasetQualityMetric  # type: ignore
                stager.attach_metric(DatasetQualityMetric(), 1)
            except Exception:
                pass

        # File-based metrics (only if local resources are present)
        if include_license:
            try:
                from metrics.license import LicenseMetric  # type: ignore
                stager.attach_metric(LicenseMetric(), 1)
            except Exception:
                pass

        if include_code_quality:
            try:
                from metrics.code_quality import CodeQualityMetric  # type: ignore
                stager.attach_metric(CodeQualityMetric(), 1)
            except Exception:
                pass

        if include_performance_claims:
            try:
                from metrics.performance_claims import PerformanceClaimsMetric  # type: ignore
                stager.attach_metric(PerformanceClaimsMetric(), 2)
            except Exception:
                pass

        return stager

    def _fetch_readme_text(self, repo_id: str) -> Optional[str]:
        for fname in ["README.md", "README.MD", "Readme.md"]:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=None)
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                continue
        return None

    def _infer_parent_urls(self, repo_id: str) -> dict:
        """
        Attempt to infer dataset and codebase URLs from card, config, and README.
        Returns a dict with optional keys: 'dataset', 'codebase'.
        """
        inferred: dict = {}
        try:
            info = model_info(repo_id)
        except Exception:
            info = None

        # Datasets from card data
        if info is not None:
            card = getattr(info, "cardData", None) or {}
            for key in ["datasets", "dataset"]:
                val = card.get(key)
                if isinstance(val, str):
                    ds = _normalize_model_id(val.replace("datasets/", ""))
                    if ds:
                        inferred["dataset"] = f"https://huggingface.co/datasets/{ds}"
                        break
                elif isinstance(val, list) and val:
                    ds = _normalize_model_id(str(val[0]).replace("datasets/", ""))
                    if ds:
                        inferred["dataset"] = f"https://huggingface.co/datasets/{ds}"
                        break

        # README-based inference
        readme = self._fetch_readme_text(repo_id)
        if readme:
            import re

            # GitHub repo link
            m = re.search(r"https?://github\.com/([\w\-\.]+)/([\w\-\.]+)(?:/|\b)", readme)
            if m and "codebase" not in inferred:
                inferred["codebase"] = f"https://github.com/{m.group(1)}/{m.group(2)}"

            # Hugging Face dataset link (absolute or relative)
            md = re.search(r"https?://huggingface\.co/datasets/([\w\-\.]+)/([\w\-\.]+)", readme)
            if md and "dataset" not in inferred:
                inferred["dataset"] = f"https://huggingface.co/datasets/{md.group(1)}/{md.group(2)}"
            if "dataset" not in inferred:
                md2 = re.search(r"\((?:\./)?datasets/([\w\-\.]+)/([\w\-\.]+)\)", readme)
                if md2:
                    inferred["dataset"] = f"https://huggingface.co/datasets/{md2.group(1)}/{md2.group(2)}"

        return inferred

    @override
    def calculate_score(self) -> float:
        # Must have a model URL
        if self.url is None or self.url.model is None:
            return 0.0

        try:
            repo_id = extract_model_repo_id(self.url.model)
        except Exception as e:
            return 0.0

        parents = list(self._extract_parents_from_card(repo_id))
        print(f"[TreeScore] Model repo_id: {repo_id}")
        print(f"[TreeScore] Discovered parents: {parents if parents else 'None'}")
        if not parents:
            return 0.0

        # Build a lightweight config for inner evaluations
        local_storage = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_storage")
        config = ConfigContract(
            num_processes=1,
            run_multi=False,
            priority_function="PFReciprocal",
            target_platform=self.target_platform or "desktop_pc",
            local_storage_directory=local_storage,
            model_path_name="models",
            code_path_name="code",
            dataset_path_name="dataset",
        )

        print("[TreeScore] Note: TreeScore only uses metrics that have required resources for parents.")

        scores: List[float] = []
        for parent_id in parents:
            try:
                parent_url = f"https://huggingface.co/{parent_id}"
                print(f"[TreeScore] Scoring parent: {parent_url}")
                parent_urls = ModelURLs(model=parent_url)
                inferred = self._infer_parent_urls(parent_id)
                if inferred.get("dataset"):
                    parent_urls.dataset = inferred["dataset"]
                if inferred.get("codebase"):
                    parent_urls.codebase = inferred["codebase"]
                print(f"  inferred URLs: dataset={parent_urls.dataset or 'None'}, codebase={parent_urls.codebase or 'None'}")
                parent_paths: ModelPaths = generate_model_paths(config, parent_urls)
                # Optionally download resources so file-based metrics can run
                downloaded_model = downloaded_code = downloaded_dataset = False
                license_ready = False
                try:
                    dm = DownloadManager(
                        str(os.path.join(config.local_storage_directory, config.model_path_name)),
                        str(os.path.join(config.local_storage_directory, config.code_path_name)),
                        str(os.path.join(config.local_storage_directory, config.dataset_path_name)),
                    )
                    mp, cp, dp = dm.download_model_resources(
                        parent_urls,
                        # Skip downloading heavy model files
                        download_model=False,
                        download_codebase=bool(parent_urls.codebase),
                        download_dataset=bool(parent_urls.dataset),
                    )
                    # We intentionally did not download models
                    downloaded_model = False
                    downloaded_code = cp is not None
                    downloaded_dataset = dp is not None
                    # Fetch only small model files needed for LicenseMetric (README/LICEN[CS]E)
                    try:
                        if parent_paths.model is not None:
                            Path(parent_paths.model).mkdir(parents=True, exist_ok=True)
                            for fname in [
                                "README.md",
                                "README.MD",
                                "Readme.md",
                                "LICENSE",
                                "LICENSE.md",
                                "LICENSE.txt",
                            ]:
                                try:
                                    hf_hub_download(
                                        repo_id=parent_id,
                                        filename=fname,
                                        local_dir=str(parent_paths.model),
                                    )
                                    license_ready = True
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    print(
                        f"  downloaded: model={downloaded_model}, code={downloaded_code}, dataset={downloaded_dataset}, license_files={license_ready}"
                    )
                except Exception as de:
                    print(f"  download error: {de}")
                # Build stager based on what we have for this parent
                stager = self._stage_base_metrics(
                    config,
                    include_bus_factor=True,  # can use HF commit history
                    include_dataset_quality=bool(parent_urls.dataset),
                    include_dataset_and_code=bool(parent_urls.dataset or parent_urls.codebase),
                    include_size=True,
                    # Enable license scoring if we fetched README/LICENSE only
                    include_license=license_ready,
                    include_code_quality=downloaded_code,
                    include_performance_claims=False,
                )
                print(f"  metrics enabled: {[m.metric_name for m in stager.metrics]}")
                output = run_workflow(stager, parent_urls, parent_paths, config)
                # Print per-metric scores for this parent
                for m in output.metrics:
                    try:
                        print(f"  - metric={m.metric_name} score={m.score}")
                    except Exception as m_err:
                        print(f"  - metric={getattr(m,'metric_name','?')} error printing score: {m_err}")
                print(f"  > parent_net_score={output.score}")
                scores.append(float(output.score))
            except Exception as e:
                print(f"[TreeScore] Failed to score parent {parent_id}: {e}")
                continue

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        print(f"[TreeScore] Aggregated average across {len(scores)} parents: {avg}")
        return avg


if __name__ == "__main__":
    # Lightweight CLI to test TreeScoreMetric
    import sys
    from types import SimpleNamespace

    # Example smaller model with known base (AdapterHub usually lists base_model)
    # You can override by passing a URL arg.
    default_model = "https://huggingface.co/AdapterHub/bert-base-uncased-pf-imdb"
    test_url = sys.argv[1] if len(sys.argv) > 1 else default_model

    metric = TreeScoreMetric()
    metric.set_url(SimpleNamespace(model=test_url))

    print("Testing TreeScoreMetric on:", test_url)
    try:
        score = metric.run().score
        print(f"TreeScore: {score:.4f}")
    except Exception as e:
        print("Error computing TreeScore:", e)
