import os
import urllib.parse
import uuid

import pytest
import requests

pytestmark = pytest.mark.integration

API_BASE = os.environ.get(
    "API_BASE",
    "https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com",
)
ARTIFACT_ID = os.environ.get("ARTIFACT_ID")
ARTIFACT_TYPE = os.environ.get("ARTIFACT_TYPE", "model")


def _require_artifact():
    if not ARTIFACT_ID:
        pytest.skip("Set ARTIFACT_ID to an existing artifact id.")


def _fetch_metadata():
    _require_artifact()
    resp = requests.get(
        f"{API_BASE}/artifacts/{ARTIFACT_TYPE}/{ARTIFACT_ID}",
        timeout=60,
    )
    if resp.status_code != 200:
        pytest.skip(f"Metadata fetch failed: {resp.status_code} {resp.text}")
    return resp.json()


def _artifact_matches(record):
    return (
        record.get("model_id") == ARTIFACT_ID
        or record.get("id") == ARTIFACT_ID
        or record.get("metadata", {}).get("id") == ARTIFACT_ID
    )


def test_search_by_name_returns_artifact():
    meta = _fetch_metadata()
    name = meta.get("metadata", {}).get("name") or ARTIFACT_ID
    encoded = urllib.parse.quote(name, safe="")

    resp = requests.get(f"{API_BASE}/artifact/byName/{encoded}", timeout=60)
    if resp.status_code == 404:
        pytest.skip("Artifact name not indexed yet (returned 404).")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    artifacts = body.get("artifacts") or []
    assert any(_artifact_matches(a) for a in artifacts), body


def test_search_post_with_name_filter():
    meta = _fetch_metadata()
    name = meta.get("metadata", {}).get("name") or ARTIFACT_ID
    payload = {"artifact_queries": [{"name": name}]}

    resp = requests.post(
        f"{API_BASE}/artifacts",
        json=payload,
        headers={"x-limit": "5"},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    artifacts = body.get("artifacts") or []
    assert artifacts, body
    assert any(_artifact_matches(a) for a in artifacts), body


def test_regex_search_finds_name_fragment():
    meta = _fetch_metadata()
    name = meta.get("metadata", {}).get("name") or ARTIFACT_ID
    fragment = name[:6] or name
    resp = requests.post(
        f"{API_BASE}/artifact/byRegEx",
        json={"pattern": fragment},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    artifacts = body.get("artifacts") or []
    assert any(_artifact_matches(a) for a in artifacts), body


def test_regex_search_invalid_pattern_returns_400():
    resp = requests.post(
        f"{API_BASE}/artifact/byRegEx",
        json={"pattern": "("},
        timeout=30,
    )
    assert resp.status_code == 400, resp.text


def test_search_by_name_not_found_returns_empty_or_404():
    random_name = f"missing-{uuid.uuid4().hex}"
    resp = requests.get(f"{API_BASE}/artifact/byName/{random_name}", timeout=30)
    if resp.status_code == 404:
        return
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not body.get("artifacts"), body
