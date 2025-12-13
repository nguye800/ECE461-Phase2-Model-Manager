import os
import uuid

import pytest
import requests

pytestmark = pytest.mark.integration

API_BASE = os.environ.get(
    "API_BASE",
    "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com",
)
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")


def _auth_headers():
    if not AUTH_TOKEN:
        return {}
    return {"X-Authorization": AUTH_TOKEN}


def test_reset_clears_registry_when_allowed():
    """
    Guarded by ALLOW_RESET to avoid wiping shared environments accidentally.
    """
    if os.environ.get("ALLOW_RESET", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Set ALLOW_RESET=true to run reset against this environment.")
    if not AUTH_TOKEN:
        pytest.skip("Set AUTH_TOKEN for reset authentication.")

    resp = requests.delete(
        f"{API_BASE}/reset",
        headers=_auth_headers(),
        timeout=120,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "deleted_items" in body, body


def test_delete_invalid_id_rejected():
    if not AUTH_TOKEN:
        pytest.skip("Set AUTH_TOKEN for delete authentication.")

    resp = requests.delete(
        f"{API_BASE}/artifacts/model/!!!",
        headers=_auth_headers(),
        timeout=30,
    )
    assert resp.status_code == 400, resp.text


def test_delete_known_artifact():
    """
    Delete a known test artifact (provide DELETE_ARTIFACT_ID/TYPE).
    """
    artifact_id = os.environ.get("DELETE_ARTIFACT_ID")
    artifact_type = os.environ.get("DELETE_ARTIFACT_TYPE", "model")
    if not artifact_id:
        pytest.skip("Set DELETE_ARTIFACT_ID (and optional DELETE_ARTIFACT_TYPE) to run this test.")
    if not AUTH_TOKEN:
        pytest.skip("Set AUTH_TOKEN for delete authentication.")

    delete_resp = requests.delete(
        f"{API_BASE}/artifacts/{artifact_type}/{artifact_id}",
        headers=_auth_headers(),
        timeout=60,
    )
    assert delete_resp.status_code in (200, 404), delete_resp.text

    # Follow-up GET should report not found.
    get_resp = requests.get(
        f"{API_BASE}/artifacts/{artifact_type}/{artifact_id}",
        timeout=30,
    )
    assert get_resp.status_code in (400, 404), get_resp.text


def test_delete_unknown_artifact_is_safe():
    """
    Random ids should return 404/400 without side effects.
    """
    if not AUTH_TOKEN:
        pytest.skip("Set AUTH_TOKEN for delete authentication.")

    random_id = uuid.uuid4().hex
    resp = requests.delete(
        f"{API_BASE}/artifacts/model/{random_id}",
        headers=_auth_headers(),
        timeout=30,
    )
    assert resp.status_code in (400, 404), resp.text
