import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.metrics.bus_factor import *  # pyright: ignore[reportWildcardImportFromLibrary]
from src.metric import ModelURLs


class TestBustFactor(unittest.TestCase):
    metric_instance: BusFactorMetric

    def setUp(self):
        self.metric_instance = BusFactorMetric()

    # calculation testing

    def testLopsidedTeam(self):
        team = {"hard_carry": 35, "normal_dev": 10, "lazy_user": 1}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def testEqualTeam(self):
        team = {"cool_dev": 20, "great_dev": 20, "some_guy": 20}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 1.0
        )

    def testSoloDev(self):
        team = {"one_guy_from_nebraska": 172}
        total_commits = sum(team.values())
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def testEmptyRepo(self):
        team: dict[str, int] = {}
        total_commits = 0
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team), 0.0
        )

    def testAverageTeam(self):
        team: dict[str, int] = {
            "lead_programmer": 20,
            "team_member": 10,
            "team_member_2": 10,
            "tryhard": 15,
            "future_successor": 20,
            "intern": 3,
        }
        total_commits = sum(team.values())
        # total commits is 78, 78/2 = 39
        #  remove lead programmer and future successor, remove half the commits.
        self.assertAlmostEqual(
            self.metric_instance.calculate_bus_factor(total_commits, team),
            2 * 2 / len(team),
        )

    # test repo parsing
    def test_team_repo(self):
        # archived project, unlikely to change much
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/TerrorCTF"
        self.metric_instance.set_url(urls)
        commit_score = {
            "silicasandwhich@github.com": 18,
            "marinom@rose-hulman.edu": 6,
            "rogerscm@rose-hulman.edu": 5,
            "102613108+CarsonRogers@users.noreply.github.com": 1,
        }
        total_commits = 30
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        parsed_response = self.metric_instance.parse_response()
        self.assertDictEqual(parsed_response[1], commit_score)
        self.assertEqual(parsed_response[0], total_commits)

    def test_solo_repo(self):
        # you can't remove commits from a repository
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://www.github.com/silica-dev/2nd_to_ft_conversion_script"
        self.metric_instance.set_url(urls)
        total_commits = 1000
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        parsed_response = self.metric_instance.parse_response()
        self.assertGreaterEqual(total_commits, parsed_response[0])
        self.assertGreaterEqual(
            total_commits,
            parsed_response[1]["43558271+Silicasandwhich@users.noreply.github.com"],
        )

    def test_nonexistent_repo(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/PLEASEDONTACTUALLYMAKETHIS"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        with self.assertRaises(ValueError):
            self.metric_instance.parse_response()

    # url parsing
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
        if self.metric_instance.response is None:
            return
        self.assertTrue(self.metric_instance.response.ok)

    def test_specific_branch(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/leftwm/leftwm/tree/flake_update"
        self.metric_instance.set_url(urls)
        self.metric_instance.setup_resources()
        self.metric_instance.get_response(urls.codebase)
        self.assertIsNotNone(self.metric_instance.response)
        if self.metric_instance.response is None:
            return
        self.assertTrue(self.metric_instance.response.ok)

    # full integration

    # test repo parsing
    def test_team_repo_full(self):
        # archived project, unlikely to change much
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/silica-dev/TerrorCTF"
        self.metric_instance.set_url(urls)
        # commit_score = {
        #    "silicasandwhich@github.com": 18,
        #    "marinom@rose-hulman.edu": 6,
        #    "rogerscm@rose-hulman.edu": 5,
        #    "102613108+CarsonRogers@users.noreply.github.com": 1,
        # }
        # remove silicasandwhich@github.com and more than 50% is gone
        self.metric_instance.run()
        self.assertIsInstance(self.metric_instance.score, float)
        if isinstance(self.metric_instance.score, dict):
            return
        self.assertAlmostEqual(self.metric_instance.score, 0.0)

    def test_huggingface_url(self):
        # almost all huggingface models have pretty horrendous bus factors, this one isn't awful
        urls = ModelURLs(model="https://huggingface.co/google-bert/bert-base-uncased")
        self.metric_instance.set_url(urls)
        self.metric_instance.run()
        self.assertIsInstance(self.metric_instance.score, float)
        if isinstance(self.metric_instance.score, dict):
            return
        self.assertAlmostEqual(self.metric_instance.score, 3 / 15 * 2)

    def test_weighted_sum(self):
        # almost all huggingface models have pretty horrendous bus factors
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


class TestBusFactorUnit(unittest.TestCase):
    def setUp(self):
        self.metric = BusFactorMetric()

    def test_get_response_validates_url(self):
        with self.assertRaises(ValueError):
            self.metric.get_response("not a github url")

    def test_get_response_requires_token(self):
        with patch("src.metrics.bus_factor.os.getenv", return_value=None):
            with self.assertRaises(ValueError):
                self.metric.get_response("https://github.com/owner/repo")

    @patch("src.metrics.bus_factor.requests.post")
    def test_get_response_handles_request_errors(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("boom")
        with patch("src.metrics.bus_factor.os.getenv", return_value="token"):
            with self.assertRaises(ValueError):
                self.metric.get_response("https://github.com/owner/repo")

    @patch("src.metrics.bus_factor.requests.post")
    def test_get_response_success(self, mock_post):
        fake_response = MagicMock()
        mock_post.return_value = fake_response
        with patch("src.metrics.bus_factor.os.getenv", return_value="token"):
            self.metric.get_response("https://github.com/owner/repo")
        self.assertIs(self.metric.response, fake_response)
        mock_post.assert_called_once()

    def test_parse_response_without_payload(self):
        total, scores = self.metric.parse_response()
        self.assertEqual(total, 0)
        self.assertEqual(scores, {})

    def test_parse_response_invalid_json(self):
        self.metric.response = SimpleNamespace(text="not-json")
        with self.assertRaises(ValueError):
            self.metric.parse_response()

    def test_parse_response_graphql_errors(self):
        payload = {"errors": [{"message": "nope"}]}
        self.metric.response = SimpleNamespace(text=json.dumps(payload))
        with self.assertRaises(ValueError):
            self.metric.parse_response()

    def test_parse_response_success(self):
        payload = {
            "data": {
                "repository": {
                    "refs": {
                        "edges": [
                            {
                                "node": {
                                    "target": {
                                        "history": {
                                            "edges": [
                                                {"node": {"author": {"email": "a@x"}}},
                                                {"node": {"author": {"email": "b@x"}}},
                                                {"node": {"author": {"email": "a@x"}}},
                                            ]
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        self.metric.response = SimpleNamespace(text=json.dumps(payload))
        total, scores = self.metric.parse_response()
        self.assertEqual(total, 3)
        self.assertEqual(scores["a@x"], 2)
        self.assertEqual(scores["b@x"], 1)

    @patch("src.metrics.bus_factor.list_repo_commits")
    def test_parse_model_handles_errors(self, mock_list_commits):
        mock_list_commits.side_effect = RuntimeError("boom")
        self.metric.set_url(ModelURLs(model="https://huggingface.co/owner/model"))
        total, scores = self.metric.parse_model()
        self.assertEqual(total, 0)
        self.assertEqual(scores, {})

    @patch("src.metrics.bus_factor.list_repo_commits")
    def test_parse_model_counts_recent_commits(self, mock_list_commits):
        class Commit:
            def __init__(self, ts, authors):
                self.created_at = ts
                self.authors = authors

        recent = datetime(2024, 1, 1, tzinfo=timezone.utc)
        older = datetime(2019, 1, 1, tzinfo=timezone.utc)
        mock_list_commits.return_value = [
            Commit(recent, ["a", "b"]),
            Commit(older, ["c"]),
        ]
        self.metric.set_url(ModelURLs(model="https://huggingface.co/owner/model"))
        total, scores = self.metric.parse_model()
        self.assertEqual(total, 1)
        self.assertEqual(scores, {"a": 1, "b": 1})

    def test_calc_weighted_sum_edge_cases(self):
        self.assertEqual(self.metric.calc_weighted_sum(0, 0, 0.5, 0.5), 0.0)
        self.assertEqual(self.metric.calc_weighted_sum(0, 5, 0.5, 0.7), 0.7)
        self.assertEqual(self.metric.calc_weighted_sum(3, 0, 0.6, 0.4), 0.6)

    @patch("src.metrics.bus_factor.requests.post")
    def test_setup_resources_without_codebase(self, mock_post):
        metric = BusFactorMetric()
        metric.set_url(ModelURLs(model="https://huggingface.co/model"))
        metric.setup_resources()
        self.assertIsNone(metric.response)
        mock_post.assert_not_called()

    @patch("src.metrics.bus_factor.requests.post")
    def test_setup_resources_with_codebase(self, mock_post):
        fake_resp = MagicMock()
        mock_post.return_value = fake_resp
        metric = BusFactorMetric()
        urls = ModelURLs(model="https://huggingface.co/model", codebase="https://github.com/owner/repo")
        metric.set_url(urls)
        with patch("src.metrics.bus_factor.os.getenv", return_value="token"):
            metric.setup_resources()
        self.assertIs(metric.response, fake_resp)
