GET_DEFAULT_BRANCH = """
query GetDefaultBranch($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { defaultBranchRef { name } }
}
"""

# Get the most recent commit date
GET_LATEST_COMMIT = """
query GetLatestCommit($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      target {
        ... on Commit {
          committedDate
        }
      }
    }
  }
}
"""

# Query for last N merged PRs with reviews
LAST_N_MERGED_PRS = """
query LastNMergedPRs($owner: String!, $name: String!, $baseRef: String!, $first: Int!, $afterPR: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: MERGED,
      baseRefName: $baseRef,
      first: $first,
      after: $afterPR,
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        mergedAt
        author { login }
        reviews(first: 50) {
          nodes {
            author { login }
            state
          }
        }
        additions
      }
    }
  }
}
"""