import os
import pydantic
from pydantic import BaseModel, field_validator
pydantic.main.BaseModel.model_config = {"protected_namespaces": ()}
from typing import Literal, Optional
from typing_extensions import override
from abc import ABC, abstractmethod
from pathlib import Path
import logging


DATASET = "dataset"
CODEBASE = "codebase"
MODEL = "model"


class PriorityFunction(ABC):
    """
    Abstract base class for priority weighting functions.
    """

    @abstractmethod
    def calculate_priority_weight(self, priority: int) -> float:
        """
        Calculates the weight for a given priority.
        Args:
            priority (int): The priority value.
        Returns:
            float: The calculated weight.
        """
        pass


class PFExponentialDecay(PriorityFunction):
    """
    Priority function using exponential decay.
    """

    def __init__(self, base_coefficient: int):
        """
        Initializes the exponential decay function.
        Args:
            base_coefficient (int): The base coefficient (> 1).
        """
        assert base_coefficient > 1
        self.base_coefficient: int = base_coefficient

    @override
    def calculate_priority_weight(self, priority: int) -> float:
        """
        Calculates the weight using exponential decay.
        Args:
            priority (int): The priority value.
        Returns:
            float: The calculated weight.
        """
        return self.base_coefficient ** -(priority - 1)


class PFReciprocal(PriorityFunction):
    """
    Priority function using reciprocal weighting.
    """

    @override
    def calculate_priority_weight(self, priority: int) -> float:
        """
        Calculates the weight as the reciprocal of the priority.
        Args:
            priority (int): The priority value.
        Returns:
            float: The calculated weight.
        """
        return 1 / priority


PRIORITY_FUNCTIONS: dict[str, PriorityFunction] = {
    "PFExponentialDecay": PFExponentialDecay(2),
    "PFReciprocal": PFReciprocal(),
}


class ModelURLs(BaseModel):
    """
    Stores URLs related to a model, including model, codebase, and dataset URLs.
    """

    model: Optional[str] = None
    codebase: Optional[str] = None
    dataset: Optional[str] = None

    def __eq__(self, other):
        return (
            self.model == other.model
            and self.codebase == other.codebase
            and self.dataset == other.dataset
        )

    @field_validator("codebase", "dataset", mode="after")
    @classmethod
    def check_empty_url(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            return None
        return value

    @field_validator("model", mode="after")
    @classmethod
    def check_empty_url_model(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            raise ValueError("Must have a model url")
        return value


class ModelPaths(BaseModel):
    model: Optional[Path] = None
    codebase: Optional[Path] = None
    dataset: Optional[Path] = None

    # @field_validator('model_local_path', 'code_local_path', 'dataset_local_path', mode='before')
    # @classmethod
    # def check_path_or_create(cls, directory):


class ConfigContract(BaseModel):
    """
    Configuration contract for the workflow, specifying number of processes,
    priority function, and target platform.
    """

    num_processes: int = 1
    run_multi: bool = True
    priority_function: Literal["PFReciprocal", "PFExponentialDecay"] = "PFReciprocal"
    target_platform: str = ""
    local_storage_directory: str
    model_path_name: str
    code_path_name: str
    dataset_path_name: str

    @field_validator("local_storage_directory", mode="before")
    @classmethod
    def validate_local_storage_directory(cls, directory: str) -> str:
        if not os.path.isdir(directory):
            #logging.debug("The provided local directory is invalid. Creating")
            Path(directory).mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.R_OK):
            raise IOError("The provided local directory is not readable")

        return directory

    @field_validator(
        "model_path_name", "code_path_name", "dataset_path_name", mode="before"
    )
    @classmethod
    def validate_path_names(cls, name: str) -> str:
        if "/" in name:
            raise NameError("cannot put / in path name")
        return name


def extract_model_repo_id(model_url: str) -> str:
    """
    Args:
        model_url: HuggingFace model URL

    Returns:
        Repository ID (e.g., "username/model-name")
    """
    # Remove common URL patterns
    repo_id = model_url.replace("https://huggingface.co/", "")
    repo_id = repo_id.replace("/tree/main", "")
    repo_id = repo_id.rstrip("/")
    return repo_id


def extract_dataset_repo_id(dataset_url: str) -> str:
    """
    Args:
        dataset_url: HuggingFace dataset URL

    Returns:
        Dataset repository ID (e.g., "username/dataset-name")
    """
    # Remove common URL patterns for datasets
    repo_id = dataset_url.replace("https://huggingface.co/datasets/", "")
    repo_id = repo_id.replace("/tree/main", "")
    repo_id = repo_id.rstrip("/")
    return repo_id


def extract_code_repo_name(code_url: str) -> str:
    """
    Args:
        code_url: Git repository URL

    Returns:
        Repository name
    """
    repo_name = code_url.rstrip("/").split("/")[-1]
    repo_name = repo_name.replace(".git", "")
    return repo_name


def generate_model_paths(contract: ConfigContract, urls: ModelURLs) -> ModelPaths:
    parent_path: Path = Path(contract.local_storage_directory)
    directories: ModelPaths = ModelPaths()

    if urls.model:
        directories.model = (
            parent_path / contract.model_path_name
        ) / extract_model_repo_id(urls.model).replace("/", "_")

    if urls.codebase:
        directories.codebase = (
            parent_path / contract.code_path_name
        ) / extract_code_repo_name(urls.codebase)

    if urls.dataset:
        directories.dataset = (
            parent_path / contract.dataset_path_name
        ) / extract_dataset_repo_id(urls.dataset).replace("/", "_")

    return directories
