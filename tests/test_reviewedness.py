import unittest
from unittest.mock import patch, MagicMock
from src.metrics.reviewedness import ReviewednessMetric
from src.metric import ModelURLs
from datetime import datetime, timedelta


class TestReviewednessMetric(unittest.TestCase):
    metric_instance: ReviewednessMetric

    def setUp(self):
        self.metric_instance = ReviewednessMetric()

    # URL parsing tests

    def test_invalid_url(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "sdvx.org"
        self.metric_instance.set_url(urls)
        owner, repo = self.metric_instance._get_owner_repo()
        self.assertIsNone(owner)
        self.assertIsNone(repo)

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
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, -1.0)

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

    def test_ghql_missing_token(self):
        urls = ModelURLs(model="nonexistent")
        urls.codebase = "https://github.com/owner/repo"
        self.metric_instance.set_url(urls)
        
        with patch.dict('os.environ', {}, clear=True):
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


if __name__ == "__main__":
    unittest.main()