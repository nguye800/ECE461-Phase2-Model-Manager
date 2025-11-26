import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from huggingface_hub import snapshot_download
import git
from metric import ModelURLs


class DownloadManager:
    """
    Manages downloading and caching of models, datasets, and codebases.
    Always updates existing resources when downloading.
    """

    def __init__(
        self,
        models_dir: str = "models",
        codebases_dir: str = "codebases",
        datasets_dir: str = "datasets",
    ):
        """
        Args:
            models_dir: Directory to store downloaded models
            codebases_dir: Directory to store downloaded codebases
            datasets_dir: Directory to store downloaded datasets
        """
        self.models_dir = Path(models_dir)
        self.codebases_dir = Path(codebases_dir)
        self.datasets_dir = Path(datasets_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.codebases_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)

    def _extract_repo_id(self, url: str) -> str:
        """
        Extract repository ID from HuggingFace URL or repo identifier.
        """
        if not url:
            raise ValueError("Repository URL is required")

        candidate = url.strip()
        parsed = urlparse(candidate)

        if parsed.scheme and parsed.netloc:
            if "huggingface.co" not in parsed.netloc:
                # Not a Hugging Face URL; return last path component
                path_part = parsed.path.strip("/") or parsed.netloc
                return path_part.split("/")[-1]

            path_segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
            cleaned_segments = []
            skip_prefixes = {"datasets", "models", "spaces"}
            for segment in path_segments:
                if not cleaned_segments and segment in skip_prefixes:
                    continue
                cleaned_segments.append(segment)

            if not cleaned_segments:
                raise ValueError(f"Unable to extract repo id from URL: {url}")

            repo_id = "/".join(cleaned_segments[:2])
        else:
            repo_id = candidate.strip("/")

        repo_id = repo_id.replace("/tree/main", "")
        repo_id = repo_id.replace("/blob/main", "")
        repo_id = repo_id.replace("/resolve/main", "")
        repo_id = repo_id.rstrip("/")

        parts = repo_id.split("/")
        if len(parts) > 2:
            repo_id = "/".join(parts[:2])

        if not repo_id or repo_id == "https:":
            raise ValueError(f"Unable to extract repo id from URL: {url}")

        return repo_id

    def _extract_repo_name(self, code_url: str) -> str:
        """
        Extract repository name from git URL.

        Args:
            code_url: Git repository URL

        Returns:
            Repository name
        """
        repo_name = code_url.rstrip("/").split("/")[-1]
        repo_name = repo_name.replace(".git", "")
        return repo_name

    def download_model(self, model_url: str) -> Path:
        """
        Download or update a HuggingFace model to local storage.
        Always updates if model exists locally.

        Args:
            model_url: HuggingFace model URL

        Returns:
            Path to the downloaded model directory
        """
        repo_id = self._extract_repo_id(model_url)
        local_path = self.models_dir / repo_id.replace("/", "_")
        local_path.mkdir(parents=True, exist_ok=True)

        #if local_path.exists():
            #logging.info(f"Updating existing model at {local_path}...")
        #else:
            #logging.info(f"Downloading model from {model_url}...")

        try:
            # snapshot_download efficiently handles updates - only downloads changed files
            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local_path),            # tests expect str
                "revision": "main",
                "resume_download": local_path.exists(),  # True if already there
                "force_download": False,                 # test expects False
            }
            snapshot_download(**kwargs)
            #logging.info(f"Model ready at {local_path}")
            return local_path
        except Exception as e:
            # If update fails, clear and re-download
            if local_path.exists():
                #logging.warning(f"Update failed, clearing and re-downloading: {e}")
                shutil.rmtree(local_path)
                local_path.mkdir(parents=True, exist_ok=True)
                try:
                    kwargs = {
                        "repo_id": repo_id,
                        "local_dir": str(local_path),
                        "revision": "main",
                        "force_download": False,
                    }
                    snapshot_download(**kwargs)
                    return local_path
                except Exception as retry_error:
                    #logging.error(f"Failed to re-download model: {retry_error}")
                    raise
            else:
                #logging.error(f"Failed to download model from {model_url}: {e}")
                raise

    def download_dataset(self, dataset_url: str) -> Path:
        """
        Download or update a HuggingFace dataset to local storage.
        Always updates if dataset exists locally.

        Args:
            dataset_url: HuggingFace dataset URL

        Returns:
            Path to the downloaded dataset directory
        """
        dataset_id = self._extract_repo_id(dataset_url)
        local_path = self.datasets_dir / dataset_id.replace("/", "_")
        local_path.mkdir(parents=True, exist_ok=True)

        #if local_path.exists():
            #logging.info(f"Updating existing dataset at {local_path}...")
        #else:
            #logging.info(f"Downloading dataset from {dataset_url}...")

        try:
            # Use snapshot_download with repo_type="dataset"
            kwargs = {
                "repo_id": dataset_id,
                "repo_type": "dataset",
                "local_dir": str(local_path),
                "revision": "main",
                "resume_download": local_path.exists(),  # True if already there
                "force_download": False,                 # test expects False
            }
            snapshot_download(**kwargs)
            #logging.info(f"Dataset ready at {local_path}")
            return local_path
        except Exception as e:
            # If update fails, clear and re-download
            if local_path.exists():
                #logging.warning(f"Update failed, clearing and re-downloading: {e}")
                shutil.rmtree(local_path)
                try:
                    local_path.mkdir(parents=True, exist_ok=True)

                    kwargs = {
                        "repo_id": dataset_id,
                        "repo_type": "dataset",
                        "local_dir": str(local_path),
                        "revision": "main",
                        "force_download": False,
                    }
                    snapshot_download(**kwargs)
                    #logging.info(f"Dataset re-downloaded to {local_path}")
                    return local_path
                except Exception as retry_error:
                    #logging.error(f"Failed to re-download dataset: {retry_error}")
                    raise
            else:
                #logging.error(f"Failed to download dataset from {dataset_url}: {e}")
                raise

    def download_codebase(self, code_url: str) -> Path:
        """
        Download or update a git repository to local storage.
        Always pulls latest changes if repository exists.

        Args:
            code_url: Git repository URL

        Returns:
            Path to the downloaded codebase directory
        """
        repo_name = self._extract_repo_name(code_url)
        local_path = self.codebases_dir / repo_name

        if local_path.exists():
            #logging.info(f"Updating existing codebase at {local_path}...")
            try:
                repo = git.Repo(local_path)
                origin = repo.remotes.origin

                # Fetch latest changes
                # Suppress git progress output by passing progress=None
                fetch_kwargs = {}
                origin.fetch(**fetch_kwargs)

                # Get current branch
                current_branch = repo.active_branch.name

                # Reset to origin to ensure clean update
                repo.git.reset("--hard", f"origin/{current_branch}")

                #logging.info(f"Codebase updated at {local_path}")
                return local_path
            except Exception as e:
                # If any git operation fails, remove and re-clone
                #logging.warning(f"Git update failed, re-cloning: {e}")
                shutil.rmtree(local_path)

        # Clone repository (either doesn't exist or update failed)
        #logging.info(f"Cloning codebase from {code_url}...")
        try:
            clone_kwargs = {}
            # Suppress clone progress output
            git.Repo.clone_from(code_url, local_path, **clone_kwargs)
            #logging.info(f"Codebase cloned to {local_path}")
            return local_path
        except Exception as e:
            #logging.error(f"Failed to clone codebase from {code_url}: {e}")
            raise

    def check_local_model(self, model_url: str) -> Optional[Path]:
        """
        Args:
            model_url: HuggingFace model URL

        Returns:
            Path to model if exists, None otherwise
        """
        repo_id = self._extract_repo_id(model_url)
        local_path = self.models_dir / repo_id.replace("/", "_")

        if local_path.exists():
            return local_path
        return None

    def check_local_dataset(self, dataset_url: str) -> Optional[Path]:
        """
        Args:
            dataset_url: HuggingFace dataset URL

        Returns:
            Path to dataset if exists, None otherwise
        """
        dataset_id = self._extract_repo_id(dataset_url)
        local_path = self.datasets_dir / dataset_id.replace("/", "_")

        if local_path.exists():
            return local_path
        return None

    def check_local_codebase(self, code_url: str) -> Optional[Path]:
        """
        Args:
            code_url: Git repository URL

        Returns:
            Path to codebase if exists, None otherwise
        """
        repo_name = self._extract_repo_name(code_url)
        local_path = self.codebases_dir / repo_name

        if local_path.exists():
            return local_path
        return None

    def download_model_resources(
        self,
        model_urls: ModelURLs,
        download_model: bool = True,
        download_codebase: bool = True,
        download_dataset: bool = True,
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Args:
            model_urls: ModelURLs object containing URLs
            download_model: Whether to download the model
            download_codebase: Whether to download the codebase
            download_dataset: Whether to download the dataset

        Returns:
            Tuple of (model_path, codebase_path, dataset_path)
        """
        model_path = None
        codebase_path = None
        dataset_path = None

        # Download/update model if requested and URL exists
        if download_model and model_urls.model:
            model_path = self.download_model(model_urls.model)

        # Download/update codebase if requested and URL exists
        if download_codebase and model_urls.codebase:
            codebase_path = self.download_codebase(model_urls.codebase)

        # Download/update dataset if requested and URL exists
        if download_dataset and model_urls.dataset:
            dataset_path = self.download_dataset(model_urls.dataset)

        return model_path, codebase_path, dataset_path
