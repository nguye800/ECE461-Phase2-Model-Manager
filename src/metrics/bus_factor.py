from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
import requests
from dotenv import load_dotenv
import os, re, json
import heapq
from huggingface_hub import list_repo_commits
from config import extract_model_repo_id
from datetime import datetime

github_pattern = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/[^/]+)?/?$"
)

# Bus factor metric
# Assumes that the url for this metric points to a github codebase
class BusFactorMetric(BaseMetric):
    metric_name: str = "bus_factor"
    # get most recent 30 commits on (most) branches since 2020
    graphql_query = """
{
repository(name:"%s", owner:"%s"){
    refs(refPrefix:"refs/heads/", first:30){
      edges{
        node{
          target{
        ...on Commit{
          history(first:30, since:"2020-01-01T00:00:00.000Z") {
            edges {
              node {
                author{
                  email
                }
              }
            }
          }
        }
      }
        }
      }
    }
  }
  }"""

    def __init__(self):
        self.response = None
        super().__init__()

    def get_response(self, url: str):
        """
        Queries the GitHub GraphQL API for commit data using the repository URL.

        Returns:
            requests.Response object containing the GraphQL JSON payload.
        Raises:
            ValueError if the URL is invalid or if no GitHub token is found.
        """
        load_dotenv()  # ensure .env variables are loaded

        # Validate and extract owner/repo from URL
        matches = github_pattern.match(url.strip())
        if matches is None:
            raise ValueError(f"Invalid GitHub URL: {url}")

        owner, name = matches.groups()
        if not owner or not name:
            raise ValueError(f"Invalid GitHub URL: {url}")

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing GitHub API token in environment (GITHUB_TOKEN)")

        # Construct GraphQL query payload
        graphql_url = "https://api.github.com/graphql"
        query = self.graphql_query % (name, owner)
        payload = {"query": query}
        headers = {"Authorization": f"Bearer {token}"}

        # Perform the request
        try:
            response = requests.post(graphql_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()  # HTTP-level errors
        except requests.exceptions.RequestException as e:
            raise ValueError(f"GitHub API request failed: {e}")

        self.response = response

    # separated into functions for testing

    # parse the given response
    # Returns: total number of commits and dictionary of authors and commit counts
    def parse_response(self) -> tuple[int, dict[str, int]]:
        # create dictionary of commit counts
        if self.response is None:
            return (0, {})
        try:
            obj = json.loads(self.response.text)
        except Exception:
            raise ValueError("Repository is not public or does not exist")
        
        # GraphQL errors (common for invalid/private repos)
        if obj.get("errors"):
            # surface message or keep generic
            msg = obj["errors"][0].get("message", "Repository is not public or does not exist")
            raise ValueError(msg)

        data = obj.get("data")
        if not isinstance(data, dict):
            raise ValueError("Repository is not public or does not exist")

        repo = data.get("repository")
        if repo is None:
            raise ValueError("Repository is not public or does not exist")

        refs = repo.get("refs") or {}
        edges = refs.get("edges") or []

        commit_score: dict[str, int] = {}
        total_commits = 0

        for branch in edges:
            history_edges = (
                ((branch.get("node") or {}).get("target") or {}).get("history") or {}
            ).get("edges", [])
            for commit in history_edges:
                author = ((commit.get("node") or {}).get("author") or {}).get("email")
                if not author:
                    continue
                commit_score[author] = commit_score.get(author, 0) + 1
                total_commits += 1

        return total_commits, commit_score

    def calculate_bus_factor(
        self, total_commits: int, commit_score: dict[str, int]
    ) -> float:
        if total_commits < 1:
            return 0.0
        pqueue = [
            (total_commits - commits, commits)
            for _, commits in list(commit_score.items())
        ]
        heapq.heapify(pqueue)
        num_contributors = len(pqueue)

        # start taking away authors
        bus_numerator = 0
        remaining_commits = total_commits
        while remaining_commits / total_commits > 0.5:
            bussed_author_commits = heapq.heappop(pqueue)[1]
            remaining_commits -= bussed_author_commits
            bus_numerator += 1
        if bus_numerator <= 1:
            return 0.0
        bus_factor = 2 * bus_numerator / num_contributors
        return bus_factor if bus_factor < 1.0 else 1.0

    def calc_weighted_sum(
        self,
        codebase_commits: int,
        model_commits: int,
        codebase_factor: float,
        model_factor: float,
    ) -> float:
        if codebase_commits + model_commits == 0:
            return 0.0
        if codebase_commits == 0:
            return model_factor
        if model_commits == 0:
            return codebase_factor
        return (codebase_factor * codebase_commits + model_factor * model_commits) / (
            codebase_commits + model_commits
        )

    def parse_model(self) -> tuple[int, dict[str, int]]:
        if self.url is None:
            return (0, {})
        total_commits = 0
        score_dict: dict[str, int] = {}
        try:
            commits = list_repo_commits(extract_model_repo_id(self.url.model))
        except:
            return (0, {})
        for commit in commits:
            if commit.created_at >= datetime.fromisoformat("2020-01-01T00:00:00.000+00:00"):
                total_commits += 1
                for author in commit.authors:
                    score_dict[author] = score_dict.get(author, 0) + 1
        return (total_commits, score_dict)

    def calculate_score(self) -> float:
        model_commits, model_score = self.parse_model()
        codebase_commits, codebase_score = self.parse_response()
        codebase_factor = self.calculate_bus_factor(codebase_commits, codebase_score)
        model_factor = self.calculate_bus_factor(model_commits, model_score)
        return self.calc_weighted_sum(
            codebase_commits, model_commits, codebase_factor, model_factor
        )

    def setup_resources(self):
        load_dotenv()

        if self.url is not None and self.url.codebase is not None:
            # parse out name and owner
            matches = github_pattern.match(self.url.codebase)
            if matches is None:
                raise ValueError("invalid GitHub URL")

            owner = matches.group(1)
            name = matches.group(2)

            # this should theoretically never run but will cause errors to be
            # raised if the regex parsing is faulty
            if type(owner) is not str or type(name) is not str:
                raise ValueError("invalid GitHub URL")  # pragma: no cover

            url = "https://api.github.com/graphql"
            json = {"query": self.graphql_query % (name, owner)}
            headers = {"Authorization": f"bearer {os.getenv('GITHUB_TOKEN')}"}
            self.response = requests.post(url=url, json=json, headers=headers)
        else:
            self.response = None

        return super().setup_resources()
