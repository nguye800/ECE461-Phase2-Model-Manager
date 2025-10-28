from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
import requests
from dotenv import load_dotenv
import os, re, json
import heapq
from huggingface_hub import list_repo_commits
from config import extract_model_repo_id
from datetime import datetime

class ReproducibilityMetric(BaseMetric):
    metric_name: str = "reproducibility"
    graphql_query = # demo code from model card

    def __init__(self):
        self.response = None
        super().__init__()