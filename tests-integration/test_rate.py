# MODEL_ID="5ffd1e7956d94cb9a93cbf2c6e695940" python -m pytest -q -m integration tests-integration/test_rate.py


import json
import os
import pytest
import requests

pytestmark = pytest.mark.integration

API_BASE = os.environ.get("API_BASE", "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com")

def test_rate_endpoint_returns_expected_metrics():
    model_id = os.environ.get("MODEL_ID")
    assert model_id, "Set MODEL_ID to an ingested model's id before running."

    r = requests.get(f"{API_BASE}/artifact/model/{model_id}/rate", timeout=120)
    if r.status_code in (202, 500) and "pending initial scoring" in r.text.lower():
        pytest.skip("Rate job is still pending initial scoring.")
    assert r.status_code == 200, r.text

    body = r.json()

    for field in [
        "name", "category", "net_score", "net_score_latency",
        "ramp_up_time", "bus_factor", "performance_claims", "license",
        "dataset_and_code", "dataset_quality", "code_quality",
        "reproducibility", "reviewedness",
    ]:
        assert field in body, f"Missing {field} in {body}"
