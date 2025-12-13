#artifact lookup: curl -i "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com/artifacts/model/5ffd1e7956d94cb9a93cbf2c6e695940"

# GITHUB_URL="https://github.com/google-research/bert" \
# ARTIFACT_TYPE="model" ARTIFACT_ID="5ffd1e7956d94cb9a93cbf2c6e695940" \
# python -m pytest -q -m integration tests-integration/test_artifacts.py


import os
import uuid
import pytest
import requests

pytestmark = pytest.mark.integration

API_BASE = os.environ.get("API_BASE", "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com")
ARTIFACT_ID = os.environ["ARTIFACT_ID"]
ARTIFACT_TYPE = os.environ.get("ARTIFACT_TYPE", "model")


def test_get_artifact_metadata_200():
    r = requests.get(f"{API_BASE}/artifacts/{ARTIFACT_TYPE}/{ARTIFACT_ID}", timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()

    # Match the observed payload shape:
    assert "metadata" in body, body
    assert body["metadata"].get("id") == ARTIFACT_ID, body
    assert (body["metadata"].get("type") == ARTIFACT_TYPE) or (body["metadata"].get("type") == ARTIFACT_TYPE.lower()), body

    assert "data" in body, body
    # At least one of these should exist based on your output
    assert ("download_url" in body["data"]) or ("url" in body["data"]), body


def test_get_artifact_cost_200():
    r = requests.get(f"{API_BASE}/artifact/{ARTIFACT_TYPE}/{ARTIFACT_ID}/cost", timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()

    # Match the observed payload shape: { "<id>": { "total_cost": 0.0 } }
    assert ARTIFACT_ID in body, body
    assert "total_cost" in body[ARTIFACT_ID], body


def test_get_artifact_audit_200():
    r = requests.get(f"{API_BASE}/artifact/{ARTIFACT_TYPE}/{ARTIFACT_ID}/audit", timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()

    # Match the observed payload shape: a list of audit entries
    assert isinstance(body, list), body
    assert len(body) >= 1, body
    assert "action" in body[0], body[0]


def test_post_license_check_200_for_model():
    if ARTIFACT_TYPE not in ("model", "MODEL"):
        pytest.skip("license-check only supported for model artifacts")

    github_url = os.environ.get("GITHUB_URL")
    assert github_url, "Set GITHUB_URL to a valid GitHub URL (e.g. https://github.com/google-research/bert)"

    r = requests.post(
        f"{API_BASE}/artifact/{ARTIFACT_TYPE}/{ARTIFACT_ID}/license-check",
        json={"github_url": github_url},
        timeout=30,
    )
    assert r.status_code == 200, r.text


def test_put_mismatched_id_returns_400():
    """
    Validate Upload Lambda rejects path/body id mismatches before mutating data.
    """
    mismatch_id = f"{ARTIFACT_ID}-mismatch"
    payload = {
        "metadata": {
            "id": mismatch_id,
            "name": "Integration Mismatch",
            "type": ARTIFACT_TYPE,
        },
        "data": {"dummy": True},
    }

    r = requests.put(
        f"{API_BASE}/artifacts/{ARTIFACT_TYPE}/{ARTIFACT_ID}",
        json=payload,
        timeout=30,
    )
    assert r.status_code in (400, 422), r.text


def test_put_unknown_artifact_returns_404_or_400():
    """
    Using a random id should return a client error without creating new records.
    """
    random_id = uuid.uuid4().hex
    payload = {
        "metadata": {
            "id": random_id,
            "name": "Integration Missing",
            "type": ARTIFACT_TYPE,
        },
        "data": {"dummy": True},
    }

    r = requests.put(
        f"{API_BASE}/artifacts/{ARTIFACT_TYPE}/{random_id}",
        json=payload,
        timeout=30,
    )
    assert r.status_code in (400, 404), r.text
