#python -m pytest -q -m integration tests-integration/test_rate_invalid.py

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so we can import rate.py from src/

import rate 


def test_rate_invalid_id_returns_400_and_does_not_hit_dynamo():
    event = {"pathParameters": {"id": "   "}, "httpMethod": "GET"}

    with patch("rate.boto3.resource") as mock_resource:
        resp = rate.handler(event, None)

    assert resp["statusCode"] == 400
    mock_resource.assert_not_called()
