from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
import requests
from dotenv import load_dotenv
import os, re, json
import heapq
from huggingface_hub import list_repo_commits
from config import extract_model_repo_id
from datetime import datetime

class ReviewednessMetric(BaseMetric):
    metric_name: str = "reviewedness"
    """
    How many people reviewed or audited the model
    How much community activity the repo has
    Whether anyone besides the original author contributed
    Whether the model has undergone external evaluation"""
    graphql_query = # api: # of commits, # of prs, # discussion threads; commit history: # contributors, frequency of updates, # model card revisions

    def __init__(self):
        self.response = None
        super().__init__()