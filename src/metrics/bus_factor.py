from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
import requests
from dotenv import load_dotenv
import os, re, json
import heapq
from huggingface_hub import list_repo_commits
from config import extract_model_repo_id
from datetime import datetime

github_pattern = re.compile(r"^(.*)?github.com\/([^\/]+)\/([^\/]+)\/?(.*)$")


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

    # separated into functions for testing

    # parse the given response
    # Returns: total number of commits and dictionary of authors and commit counts
    def parse_response(self) -> tuple[int, dict[str, int]]:
        # create dictionary of commit counts
        if self.response is None:
            return (0, {})
        response_obj = json.loads(self.response.text)
        try:
            response_obj["data"]["repository"]["refs"]["edges"]
        except (TypeError, KeyError):
            raise ValueError("Repository is not public or does not exist")
        commit_score: dict[str, int] = {}
        total_commits = 0
        for branch in response_obj["data"]["repository"]["refs"]["edges"]:
            for commit in branch["node"]["target"]["history"]["edges"]:
                author = commit["node"]["author"]["email"]
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

            owner = matches.group(2)
            name = matches.group(3)

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
