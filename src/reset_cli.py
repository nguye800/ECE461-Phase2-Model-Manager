# src/reset_cli.py
import requests


def reset_registry(base_url: str, auth_token: str) -> int:
    """
    Call the backend DELETE /reset endpoint.

    Returns the HTTP status code.
    """
    url = f"{base_url.rstrip('/')}/reset"
    headers = {
        "X-Authorization": auth_token,
        "Accept": "application/json",
    }

    resp = requests.delete(url, headers=headers)

    if resp.status_code == 200:
        print("[OK] Registry reset successfully.")
        try:
            body = resp.json()
            deleted_items = body.get("deleted_items")
            if deleted_items is not None:
                print(f"Deleted items: {deleted_items}")
        except Exception:
            pass
    elif resp.status_code == 403:
        print("[FORBIDDEN] Authentication failed. Check your token.")
    else:
        print(f"[ERROR] Unexpected status {resp.status_code}: {resp.text}")

    return resp.status_code
