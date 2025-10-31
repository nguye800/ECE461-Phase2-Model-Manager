import json
import os
import unittest
from datetime import datetime
from unittest.mock import patch

from src.metrics.bus_factor import BusFactorMetric
from src.metric import ModelURLs


class _FakeResponse:
    def __init__(self, payload, ok=True, status_code=200):
        self.text = json.dumps(payload)
        self.ok = ok
        self.status_code = status_code

    def raise_for_status(self):
        if not self.ok or self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"Status code {self.status_code}")


def _build_commit_payload(commit_map: dict[str, int]) -> dict:
    edges = []
    for email, count in commit_map.items():
        hist_edges = [
            {"node": {"author": {"email": email}}}
            for _ in range(count)
        ]
        edges.append({"node": {"target": {"history": {"edges": hist_edges}}}})
    return {"data": {"repository": {"refs": {"edges": edges}}}}


class TestBustFactor(unittest.TestCase):
    def setUp(self):
        self.metric_instance = BusFactorMetric()
        os.environ["GITHUB_TOKEN"] = "dummy-token"

        def fake_post(url, json=None, headers=None, timeout=None):
            query = json.get("query", "") if isinstance(json, dict) else ""
            owner = ""
            repo = ""
            if "owner:\"" in query and "repository(name:\"" in query:
                try:
                    owner = query.split("owner:\"")[1].split("\"")[0]
                    repo = query.split("repository(name:\"")[1].split("\"")[0]
                except IndexError:
                    pass
            key = (owner, repo)
            if key == ("silica-dev", "TerrorCTF"):
                payload = _build_commit_payload(
                    {
                        "silicasandwhich@github.com": 18,
                        "marinom@rose-hulman.edu": 6,
                        "rogerscm@rose-hulman.edu": 5,
                        "102613108+CarsonRogers@users.noreply.github.com": 1,
                    }
                )
                return _FakeResponse(payload)
            if key == ("silica-dev", "2nd_to_ft_conversion_script"):
                payload = _build_commit_payload({"owner@example.com": 500})
                return _FakeResponse(payload)
            if key == ("silica-dev", "PLEASEDONTACTUALLYMAKETHIS"):
                return _FakeResponse({"errors": [{"message": "Not Found"}]}, ok=False, status_code=404)
            if key == ("leftwm", "leftwm"):
                return _FakeResponse(_build_commit_payload({}))
            return _FakeResponse(_build_commit_payload({}))

        self.requests_patcher = patch("src.metrics.bus_factor.requests.post", side_effect=fake_post)
        self.requests_patcher.start()

        def fake_list_repo_commits(repo_id):
            class DummyCommit:
                def __init__(self, created_at, authors):
                    self.created_at = created_at
                    self.authors = authors

            if repo_id == "google-bert/bert-base-uncased":
                base_time = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
                commit_map = {
                    "alice": 40,
                    "bob": 30,
                    "charlie": 3,
                    "diana": 3,
                    "eve": 3,
                    "frank": 3,
                    "grace": 3,
                    "heidi": 3,
                    "ivan": 3,
                    "judy": 3,
                }
                commits = []
                for author, count in commit_map.items():
                    commits.extend(
                        DummyCommit(base_time, [author]) for _ in range(count)
                    )
                return commits
            return []

        self.hf_patcher = patch("src.metrics.bus_factor.list_repo_commits", side_effect=fake_list_repo_commits)
        self.hf_patcher.start()

    def tearDown(self):
        self.requests_patcher.stop()
        self.hf_patcher.stop()

    # calculation testing

    def test_lopsided_team(self):
        team = {"hard_carry": 35, "normal_dev": 10, "lazy_user": 1}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def test_equal_team(self):
        team = {"cool_dev": 20, "great_dev": 20, "some_guy": 20}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 1.0
        )

    def test_solo_dev(self):
        team = {"one_guy_from_nebraska": 172}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def test_empty_repo(self):
        team: dict[str, int] = {}
        total_commits = 0
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def test_average_team(self):
        team: dict[str, int] = {
            "lead_programmer": 20,
            "team_member": 10,
            "team_member_2": 10,
            "tryhard": 15,
            "future_successor": 20,
            "intern": 3,
        }
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team),
            2 * 2 / len(team),
        )

    def test_team_repo(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/TerrorCTF"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        parsed_response = self.metric_instance.parse_response()
        commit_score = {
            "silicasandwhich@github.com": 18,
            "marinom@rose-hulman.edu": 6,
            "rogerscm@rose-hulman.edu": 5,
            "102613108+CarsonRogers@users.noreply.github.com": 1,
        }
        self.assertDictEqual(parsed_response[1], commit_score)
        self.assertEqual(parsed_response[0], 30)

    def test_solo_repo(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://www.github.com/silica-dev/2nd_to_ft_conversion_script"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        parsed_response = self.metric_instance.parse_response()
        self.assertGreaterEqual(parsed_response[0], 500)

    def test_nonexistent_repo(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/PLEASEDONTACTUALLYMAKETHIS"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        with self.assertRaises(ValueError):
            self.metric_instance.get_response(urls.codebase)

    def test_invalid_url(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "sdvx.org"
        self.metric_instance.set_url(urls)
        with self.assertRaises(ValueError):
            self.metric_instance.setup_resources()

    def test_no_http(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "github.com/silica-dev/TerrorCTF"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        self.assertIsNotNone(self.metric_instance.response)
        self.assertTrue(self.metric_instance.response.ok)

    def test_specific_branch(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/leftwm/leftwm/tree/flake_update"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        self.assertIsNotNone(self.metric_instance.response)
        self.assertTrue(self.metric_instance.response.ok)

    def test_team_repo_full(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/TerrorCTF"
        self.metric_instance.set_url(urls)
        self.metric_instance.run()
        self.assertIsInstance(self.metric_instance.score, float)

    def test_huggingface_url(self):
        urls = ModelURLs(model="https://huggingface.co/google-bert/bert-base-uncased")
        self.metric_instance.set_url(urls)
        self.metric_instance.run()
        self.assertAlmostEqual(self.metric_instance.score, 0.4)

    def test_weighted_sum(self):
        factor_1 = 0.5
        commits_1 = 100
        factor_2 = 0.25
        commits_2 = 50
        self.assertAlmostEqual(
            self.metric_instance.calc_weighted_sum(
                commits_1, commits_2, factor_1, factor_2
            ),
            0.416666667,
        )


if __name__ == "__main__":
    unittest.main()
