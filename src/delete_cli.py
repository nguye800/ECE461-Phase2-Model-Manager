# src/delete_cli.py
import re
import requests

_VALID_TYPES = {"model", "dataset", "code"}
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-]+$")


def _validate_artifact_type(artifact_type: str) -> str:
    if artifact_type not in _VALID_TYPES:
        raise ValueError(f"artifact_type must be one of {_VALID_TYPES}, got '{artifact_type}'")
    return artifact_type


def _validate_artifact_id(artifact_id: str) -> str:
    if not artifact_id or not _ID_PATTERN.match(artifact_id):
        raise ValueError("artifact_id must match regex ^[a-zA-Z0-9\\-]+$")
    return artifact_id


def delete_artifact(base_url: str, artifact_type: str, artifact_id: str, auth_token: str) -> int:
    """
    Call the backend DELETE /artifacts/{artifact_type}/{id} endpoint.

    Returns the HTTP status code.
    Raises ValueError on bad inputs.
    """
    artifact_type = _validate_artifact_type(artifact_type)
    artifact_id = _validate_artifact_id(artifact_id)

    url = f"{base_url.rstrip('/')}/artifacts/{artifact_type}/{artifact_id}"
    headers = {
        "X-Authorization": auth_token,
        "Accept": "application/json",
    }

    resp = requests.delete(url, headers=headers)

    # Let the Typer command decide what to print / exit with;
    # here we just print a small message and return.
    if resp.status_code == 200:
        print(f"[OK] Artifact deleted: type={artifact_type}, id={artifact_id}")
    elif resp.status_code == 400:
        print(f"[BAD REQUEST] {resp.text}")
    elif resp.status_code == 403:
        print("[FORBIDDEN] Authentication failed. Check your token.")
    elif resp.status_code == 404:
        print("[NOT FOUND] Artifact does not exist.")
    else:
        print(f"[ERROR] Unexpected status {resp.status_code}: {resp.text}")

    return resp.status_code
