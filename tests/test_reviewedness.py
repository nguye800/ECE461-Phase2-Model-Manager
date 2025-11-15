import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import requests

from src.metric import ModelURLs
from src.metrics.reviewedness import ReviewednessMetric


class TestReviewednessMetric(unittest.TestCase):
    metric_instance: ReviewednessMetric

    def setUp(self):
        self.metric_instance = ReviewednessMetric()

    # URL parsing tests

    def test_invalid_url(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "sdvx.org"
        self.metric_instance.set_url(urls)
        with self.assertRaises(ValueError):
            self.metric_instance._get_owner_repo()

    def test_valid_github_url(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/pytorch/pytorch"
        self.metric_instance.set_url(urls)
        owner, repo = self.metric_instance._get_owner_repo()
        self.assertEqual(owner, "pytorch")
        self.assertEqual(repo, "pytorch")

    def test_github_url_with_tree(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo/tree/branch-name"
        self.metric_instance.set_url(urls)
        owner, repo = self.metric_instance._get_owner_repo()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo, "repo")

    def test_github_url_without_https(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "github.com/owner/repo"
        self.metric_instance.set_url(urls)
        owner, repo = self.metric_instance._get_owner_repo()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo, "repo")

    def test_github_url_with_trailing_slash(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo/"
        self.metric_instance.set_url(urls)
        owner, repo = self.metric_instance._get_owner_repo()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo, "repo")

    # Date manipulation tests

    def test_subtract_months(self):
        date_str = "2024-06-15T12:00:00Z"
        result = self.metric_instance._subtract_months(date_str, months=6)
        # Should be approximately 6 months earlier (180 days)
        original_dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        result_dt = datetime.fromisoformat(result.replace('Z', '+00:00'))
        diff_days = (original_dt - result_dt).days
        self.assertAlmostEqual(diff_days, 180, delta=5)

    def test_subtract_months_single_month(self):
        date_str = "2024-06-15T12:00:00Z"
        result = self.metric_instance._subtract_months(date_str, months=1)
        original_dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        result_dt = datetime.fromisoformat(result.replace('Z', '+00:00'))
        diff_days = (original_dt - result_dt).days
        self.assertAlmostEqual(diff_days, 30, delta=2)

    # Score calculation tests

    def test_calculate_score_no_prs(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "total_additions": 0,
            "reviewed_additions": 0,
            "unreviewed_additions": 0,
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_all_reviewed(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "total_additions": 1000,
            "reviewed_additions": 1000,
            "unreviewed_additions": 0,
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 1.0)

    def test_calculate_score_none_reviewed(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "total_additions": 1000,
            "reviewed_additions": 0,
            "unreviewed_additions": 1000,
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_half_reviewed(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "total_additions": 1000,
            "reviewed_additions": 500,
            "unreviewed_additions": 500,
        }
        score = self.metric_instance.calculate_score()
        self.assertAlmostEqual(score, 0.5)

    def test_calculate_score_partial_reviewed(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "total_additions": 1000,
            "reviewed_additions": 750,
            "unreviewed_additions": 250,
        }
        score = self.metric_instance.calculate_score()
        self.assertAlmostEqual(score, 0.75)

    def test_calculate_score_no_repo(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "invalid-url"
        self.metric_instance.set_url(urls)
        with self.assertRaises(ValueError):
            self.metric_instance.calculate_score()

    # Mock-based integration tests

    @patch('src.metrics.reviewedness.requests.post')
    def test_ghql_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"repository": {"defaultBranchRef": {"name": "main"}}}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        
        with patch.dict('os.environ', {'GITHUB_TOKEN': 'fake_token'}):
            result = self.metric_instance._ghql("query { }", {})
            self.assertIn("repository", result)

    @patch('src.metrics.reviewedness.requests.post')
    def test_ghql_with_errors(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "errors": [{"message": "API rate limit exceeded"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        
        with patch.dict('os.environ', {'GITHUB_TOKEN': 'fake_token'}):
            with self.assertRaises(RuntimeError):
                self.metric_instance._ghql("query { }", {})

    @patch("src.metrics.reviewedness.os.getenv", return_value=None)
    def test_ghql_missing_token(self, mock_env):
        with self.assertRaises(ValueError):
            self.metric_instance._ghql("query { }", {})

    # PR analysis tests (using mocked data)

    def test_analyze_prs_with_reviews(self):
        # Mock PR data with reviews
        mock_pr_data = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 1,
                            "mergedAt": "2024-06-15T12:00:00Z",
                            "additions": 100,
                            "author": {"login": "author1"},
                            "reviews": {
                                "nodes": [
                                    {
                                        "author": {"login": "reviewer1"},
                                        "state": "APPROVED"
                                    }
                                ]
                            }
                        }
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                }
            }
        }
        
        with patch.object(self.metric_instance, '_ghql', return_value=mock_pr_data):
            results = self.metric_instance._analyze_prs_in_window(
                "owner", "repo", "main", "2024-01-01T00:00:00Z", max_prs=10
            )
            
            self.assertEqual(results["total_prs"], 1)
            self.assertEqual(results["prs_with_reviews"], 1)
            self.assertEqual(results["prs_without_reviews"], 0)
            self.assertEqual(results["reviewed_additions"], 100)
            self.assertEqual(results["unreviewed_additions"], 0)

    def test_analyze_prs_without_reviews(self):
        # Mock PR data without reviews
        mock_pr_data = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 1,
                            "mergedAt": "2024-06-15T12:00:00Z",
                            "additions": 200,
                            "author": {"login": "author1"},
                            "reviews": {"nodes": []}
                        }
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                }
            }
        }
        
        with patch.object(self.metric_instance, '_ghql', return_value=mock_pr_data):
            results = self.metric_instance._analyze_prs_in_window(
                "owner", "repo", "main", "2024-01-01T00:00:00Z", max_prs=10
            )
            
            self.assertEqual(results["total_prs"], 1)
            self.assertEqual(results["prs_with_reviews"], 0)
            self.assertEqual(results["prs_without_reviews"], 1)
            self.assertEqual(results["reviewed_additions"], 0)
            self.assertEqual(results["unreviewed_additions"], 200)

    def test_analyze_prs_self_review_excluded(self):
        # Mock PR data where author reviews their own PR
        mock_pr_data = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 1,
                            "mergedAt": "2024-06-15T12:00:00Z",
                            "additions": 150,
                            "author": {"login": "author1"},
                            "reviews": {
                                "nodes": [
                                    {
                                        "author": {"login": "author1"},
                                        "state": "APPROVED"
                                    }
                                ]
                            }
                        }
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                }
            }
        }
        
        with patch.object(self.metric_instance, '_ghql', return_value=mock_pr_data):
            results = self.metric_instance._analyze_prs_in_window(
                "owner", "repo", "main", "2024-01-01T00:00:00Z", max_prs=10
            )
            
            self.assertEqual(results["total_prs"], 1)
            self.assertEqual(results["prs_with_reviews"], 0)
            self.assertEqual(results["prs_without_reviews"], 1)
            self.assertEqual(results["unreviewed_additions"], 150)

    def test_analyze_prs_mixed_reviews(self):
        # Mock PR data with mixed reviewed and unreviewed PRs
        mock_pr_data = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 1,
                            "mergedAt": "2024-06-15T12:00:00Z",
                            "additions": 100,
                            "author": {"login": "author1"},
                            "reviews": {
                                "nodes": [
                                    {
                                        "author": {"login": "reviewer1"},
                                        "state": "APPROVED"
                                    }
                                ]
                            }
                        },
                        {
                            "number": 2,
                            "mergedAt": "2024-06-14T12:00:00Z",
                            "additions": 200,
                            "author": {"login": "author2"},
                            "reviews": {"nodes": []}
                        }
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                }
            }
        }
        
        with patch.object(self.metric_instance, '_ghql', return_value=mock_pr_data):
            results = self.metric_instance._analyze_prs_in_window(
                "owner", "repo", "main", "2024-01-01T00:00:00Z", max_prs=10
            )
            
            self.assertEqual(results["total_prs"], 2)
            self.assertEqual(results["prs_with_reviews"], 1)
            self.assertEqual(results["prs_without_reviews"], 1)
            self.assertEqual(results["reviewed_additions"], 100)
            self.assertEqual(results["unreviewed_additions"], 200)
            self.assertEqual(results["total_additions"], 300)

    @patch("builtins.print")
    @patch.object(ReviewednessMetric, "_ghql")
    def test_get_latest_commit_date_future_debug(self, mock_ghql, mock_print):
        future = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        mock_ghql.return_value = {
            "repository": {"defaultBranchRef": {"target": {"committedDate": future}}}
        }
        latest = self.metric_instance._get_latest_commit_date("owner", "repo", debug=True)
        self.assertLessEqual(latest, future)

    @patch("builtins.print")
    @patch.object(ReviewednessMetric, "_ghql")
    def test_analyze_prs_in_window_early_exit_debug(self, mock_ghql, mock_print):
        mock_ghql.return_value = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 1,
                            "mergedAt": "2023-01-01T00:00:00Z",
                            "additions": 10,
                            "author": {"login": "dev"},
                            "reviews": {"nodes": []},
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        result = self.metric_instance._analyze_prs_in_window(
            "owner", "repo", "main", "2024-01-01T00:00:00Z", debug=True
        )
        self.assertEqual(result["total_prs"], 0)

    @patch("builtins.print")
    @patch.object(ReviewednessMetric, "_ghql")
    def test_analyze_prs_in_window_full_debug(self, mock_ghql, mock_print):
        mock_ghql.return_value = {
            "repository": {
                "pullRequests": {
                    "nodes": [
                        {
                            "number": 3,
                            "mergedAt": "2024-06-10T00:00:00Z",
                            "additions": 25,
                            "author": {"login": "author"},
                            "reviews": {
                                "nodes": [
                                    {"author": {"login": "reviewer"}, "state": "APPROVED"}
                                ]
                            },
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        result = self.metric_instance._analyze_prs_in_window(
            "owner", "repo", "main", "2024-01-01T00:00:00Z", debug=True
        )
        self.assertEqual(result["reviewed_additions"], 25)

    @patch("src.metrics.reviewedness.os.getenv", return_value=None)
    def test_ghql_requires_token(self, mock_env):
        with self.assertRaises(ValueError):
            self.metric_instance._ghql("query", {})

    @patch("src.metrics.reviewedness.requests.post")
    def test_ghql_handles_errors(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("boom")
        with patch("src.metrics.reviewedness.os.getenv", return_value="token"):
            with self.assertRaises(requests.exceptions.RequestException):
                self.metric_instance._ghql("query", {})

    def test_get_default_branch_fallback(self):
        with patch.object(self.metric_instance, "_ghql", return_value={"repository": {"defaultBranchRef": None}}):
            branch = self.metric_instance._get_default_branch("owner", "repo")
        self.assertEqual(branch, "main")

    def test_get_latest_commit_date_future_adjustment(self):
        future = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = {"repository": {"defaultBranchRef": {"target": {"committedDate": future}}}}
        with patch.object(self.metric_instance, "_ghql", return_value=data):
            latest = self.metric_instance._get_latest_commit_date("owner", "repo")
        self.assertLessEqual(latest, future)

    def test_subtract_months(self):
        date = "2024-06-30T00:00:00Z"
        result = self.metric_instance._subtract_months(date, months=2)
        self.assertTrue(result.startswith("2024"))

    def test_analyze_prs_in_window_no_nodes(self):
        empty = {"repository": {"pullRequests": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}
        with patch.object(self.metric_instance, "_ghql", return_value=empty):
            results = self.metric_instance._analyze_prs_in_window("o", "r", "main", "2024-01-01T00:00:00Z")
        self.assertEqual(results["total_prs"], 0)

    def test_setup_resources_without_codebase(self):
        self.metric_instance.set_url(ModelURLs(model="x", codebase=None))
        self.metric_instance.setup_resources()
        self.assertIsNone(self.metric_instance.response)

    def test_calculate_score_handles_zero_total(self):
        urls = ModelURLs(codebase="https://github.com/owner/repo")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {"total_additions": 0, "reviewed_additions": 0}
        self.assertEqual(self.metric_instance.calculate_score(), 0.0)

    @patch.object(ReviewednessMetric, "_analyze_prs_in_window", return_value={
        "reviewed_additions": 10,
        "unreviewed_additions": 5,
        "total_additions": 15,
        "prs_with_reviews": 1,
        "prs_without_reviews": 0,
        "total_prs": 1,
    })
    @patch.object(ReviewednessMetric, "_get_latest_commit_date", return_value="2024-07-01T00:00:00Z")
    @patch.object(ReviewednessMetric, "_get_default_branch", return_value="main")
    @patch.object(ReviewednessMetric, "_get_owner_repo", return_value=("owner", "repo"))
    @patch("builtins.print")
    def test_setup_resources_populates_response(self, mock_print, mock_owner, mock_branch, mock_commit, mock_analyze):
        self.metric_instance.set_url(ModelURLs(codebase="https://github.com/owner/repo"))
        self.metric_instance.setup_resources()
        self.assertEqual(self.metric_instance.response["default_branch"], "main")
        self.assertEqual(self.metric_instance.response["total_additions"], 15)


if __name__ == "__main__":
    unittest.main()
