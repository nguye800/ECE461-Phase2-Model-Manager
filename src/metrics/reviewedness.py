from metric import BaseMetric
import requests
from dotenv import load_dotenv
import os, re
from types import SimpleNamespace
from datetime import datetime, timedelta
from metrics.graphql_queries import (GET_DEFAULT_BRANCH, GET_LATEST_COMMIT, LAST_N_MERGED_PRS)

github_pattern = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/[^/]+)?/?$")

class ReviewednessMetric(BaseMetric):
    metric_name: str = "reviewedness"

    def __init__(self):
        self.response = None
        self.stats = {}
        super().__init__()

    def _ghql(self, query: str, variables: dict):
        load_dotenv()
        url = "https://api.github.com/graphql"
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing GitHub API token in environment (GITHUB_TOKEN)")
        headers = {"Authorization": f"bearer {token}"}
        r = requests.post(url=url, json={"query": query, "variables": variables}, headers=headers)
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(f"GitHub GraphQL error: {data['errors']}")
        return data["data"]

    def _get_owner_repo(self):
        if self.url is None or self.url.codebase is None:
            return None, None
        m = github_pattern.match(self.url.codebase)
        if not m:
            raise ValueError("invalid GitHub URL")
        return m.group(1), m.group(2)

    def _get_default_branch(self, owner, name):
        data = self._ghql(GET_DEFAULT_BRANCH, {"owner": owner, "name": name})
        ref = data["repository"]["defaultBranchRef"]
        return (ref["name"] if ref and ref.get("name") else "main")

    def _get_latest_commit_date(self, owner, name, debug=False):
        """Get the date of the most recent commit on the default branch."""
        data = self._ghql(GET_LATEST_COMMIT, {"owner": owner, "name": name})
        latest_date = data["repository"]["defaultBranchRef"]["target"]["committedDate"]
        
        # Sanity check: if date is in the future, use current date
        dt = datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
        now = datetime.utcnow().replace(tzinfo=dt.tzinfo)
        
        if dt > now:
            if debug:
                print(f"⚠️  Warning: Latest commit date {latest_date} is in the future! Using current date.")
            latest_date = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if debug:
            print(f"Latest commit date: {latest_date}")
        
        return latest_date

    def _subtract_months(self, date_str, months=6):
        """Subtract N months from an ISO date string."""
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        # Approximate: subtract 30 days per month
        days_to_subtract = months * 30
        new_dt = dt - timedelta(days=days_to_subtract)
        
        return new_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def _analyze_prs_in_window(self, owner, name, base_ref, since_date, max_prs=200, debug=False):
        """
        Analyze merged PRs within a time window.
        Returns stats about reviewed vs unreviewed PRs.
        """
        reviewed_additions = 0
        unreviewed_additions = 0
        prs_with_reviews = 0
        prs_without_reviews = 0
        total_prs = 0
        after = None
        
        if debug:
            print(f"\n=== Fetching merged PRs since {since_date} ===")
        
        while total_prs < max_prs:
            batch_size = min(100, max_prs - total_prs)
            
            data = self._ghql(LAST_N_MERGED_PRS, {
                "owner": owner,
                "name": name,
                "baseRef": base_ref,
                "first": batch_size,
                "afterPR": after
            })
            
            prs = data["repository"]["pullRequests"]
            nodes = prs["nodes"]
            
            if not nodes:
                break
            
            for pr in nodes:
                merged_at = pr.get("mergedAt")
                
                # Stop if we've gone past our time window
                if not merged_at or merged_at < since_date:
                    if debug:
                        print(f"\nReached PRs older than {since_date}, stopping.")
                    return {
                        "reviewed_additions": reviewed_additions,
                        "unreviewed_additions": unreviewed_additions,
                        "total_additions": reviewed_additions + unreviewed_additions,
                        "prs_with_reviews": prs_with_reviews,
                        "prs_without_reviews": prs_without_reviews,
                        "total_prs": total_prs,
                    }
                
                additions = pr["additions"] or 0
                pr_author = pr["author"]["login"] if pr["author"] else None
                
                # Check for third-party reviews
                has_third_party_review = False
                for review in pr["reviews"]["nodes"]:
                    reviewer = review["author"]["login"] if review["author"] else None
                    review_state = review["state"]
                    
                    if (reviewer and reviewer != pr_author and 
                        review_state in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED")):
                        has_third_party_review = True
                        break
                
                if has_third_party_review:
                    reviewed_additions += additions
                    prs_with_reviews += 1
                else:
                    unreviewed_additions += additions
                    prs_without_reviews += 1
                
                if debug and total_prs < 5:
                    print(f"PR #{pr['number']}: {additions} adds, reviewed={has_third_party_review}")
                
                total_prs += 1
                if total_prs >= max_prs:
                    break
            
            if not prs["pageInfo"]["hasNextPage"] or total_prs >= max_prs:
                break
            
            after = prs["pageInfo"]["endCursor"]
        
        if debug:
            print(f"\n=== PR Analysis Complete ===")
            print(f"Total PRs analyzed: {total_prs}")
            print(f"PRs with third-party reviews: {prs_with_reviews}")
            print(f"PRs without third-party reviews: {prs_without_reviews}")
            print(f"Reviewed code additions: {reviewed_additions:,}")
            print(f"Unreviewed code additions: {unreviewed_additions:,}")
        
        return {
            "reviewed_additions": reviewed_additions,
            "unreviewed_additions": unreviewed_additions,
            "total_additions": reviewed_additions + unreviewed_additions,
            "prs_with_reviews": prs_with_reviews,
            "prs_without_reviews": prs_without_reviews,
            "total_prs": total_prs,
        }

    def setup_resources(self):
        load_dotenv()
        owner, name = self._get_owner_repo()
        if owner is None:
            self.response = None
            return super().setup_resources()

        default_branch = self._get_default_branch(owner, name)
        
        # Get the most recent commit date
        latest_commit_date = self._get_latest_commit_date(owner, name, debug=True)
        
        # Calculate the date 6 months before the latest commit
        start_date = self._subtract_months(latest_commit_date, months=6)
        
        print(f"\n📅 Analysis window: {start_date} to {latest_commit_date}")
        print(f"   (Last 6 months of activity before most recent commit)\n")
        
        # Analyze PRs within this 6-month window (max 200 PRs)
        results = self._analyze_prs_in_window(
            owner, name, default_branch,
            since_date=start_date,
            max_prs=200,
            debug=True
        )

        self.response = {
            "default_branch": default_branch,
            "analysis_start_date": start_date,
            "analysis_end_date": latest_commit_date,
            **results
        }
        self.stats.update(self.response)
        return super().setup_resources()

    def calculate_score(self) -> float:
        owner, name = self._get_owner_repo()
        if owner is None or name is None:
            return -1.0  # no linked repo

        if not self.response:
            self.setup_resources()

        total = self.response["total_additions"]
        if total == 0:
            return 0.0
        
        reviewed = self.response["reviewed_additions"]
        return float(reviewed) / float(total)


if __name__ == "__main__":  # pragma: no cover
    load_dotenv()
    
    if os.getenv("GITHUB_TOKEN") is None:
        raise RuntimeError("You must set GITHUB_TOKEN in your environment.")

    test_url = "https://github.com/pytorch/pytorch"
    url_obj = SimpleNamespace(codebase=test_url)

    metric = ReviewednessMetric()
    metric.set_url(url_obj)

    print("Running setup_resources() ...")
    metric.setup_resources()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    score = metric.calculate_score()
    
    if metric.response and metric.response["total_additions"] > 0:
        reviewed = metric.response["reviewed_additions"]
        unreviewed = metric.response["unreviewed_additions"]
        total = metric.response["total_additions"]
        prs_reviewed = metric.response["prs_with_reviews"]
        prs_unreviewed = metric.response["prs_without_reviews"]
        total_prs = metric.response["total_prs"]
        
        print(f"\n📊 Reviewedness Analysis:")
        print(f"   Time Period: {metric.response['analysis_start_date'][:10]} to {metric.response['analysis_end_date'][:10]}")
        print(f"   Total PRs: {total_prs}")
        print(f"\n   PRs with reviews:    {prs_reviewed:3d} ({prs_reviewed/total_prs*100:5.1f}%)")
        print(f"   PRs without reviews: {prs_unreviewed:3d} ({prs_unreviewed/total_prs*100:5.1f}%)")
        print(f"\n   Code Additions:")
        print(f"   ├─ Reviewed:   {reviewed:8,} lines ({reviewed/total*100:5.1f}%)")
        print(f"   ├─ Unreviewed: {unreviewed:8,} lines ({unreviewed/total*100:5.1f}%)")
        print(f"   └─ Total:      {total:8,} lines")
        print(f"\n✅ Reviewedness Score: {score:.4f} ({score*100:.2f}%)")
        print(f"\n⚠️  Note: This only counts code from PRs. Direct commits are not")
        print(f"   included because GitHub's API doesn't reliably link them to PRs")
        print(f"   when repos use squash/rebase merge strategies.")
    else:
        print(f"\n✅ Reviewedness Score: {score:.4f}")
        print("   (No PRs found in analysis window)")
