import os
import pytest
import requests

pytestmark = pytest.mark.integration

API_BASE = os.environ.get(
    "API_BASE",
    "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com",
)


def test_health_endpoint_reports_dependencies():
    resp = requests.get(f"{API_BASE}/health", timeout=30)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body.get("status") in ("ok", "critical"), body
    deps = body.get("dependencies", {})
    assert "dynamodb" in deps and "s3" in deps, body
    for name in ("dynamodb", "s3"):
        assert "latency_ms" in deps[name], body


def test_health_components_lists_resources():
    resp = requests.get(
        f"{API_BASE}/health/components",
        params={"includeTimeline": "true"},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    components = body.get("components", [])
    assert components, body

    # Ensure DynamoDB and S3 entries are present
    ids = {c.get("id") for c in components}
    assert any(str(i).startswith("dynamodb:") for i in ids), ids
    assert any(str(i).startswith("s3:") for i in ids), ids

    for component in components:
        assert "status" in component, component
        assert "latency_ms" in component.get("metrics", {}), component
