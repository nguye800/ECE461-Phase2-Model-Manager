#python -m pytest -q -m integration tests-integration/test_rate_invalid.py

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so we can import rate.py from src/

import rate


def test_rate_invalid_sqs_payload_returns_failure_and_skips_dynamo():
    """
    Rate Lambda is SQS-triggered; invalid/missing model_id should mark the record as failed
    without touching DynamoDB.
    """
    event = {
        "Records": [
            {
                "messageId": "msg-1",
                "body": "{}",
            }
        ]
    }

    with patch("rate.boto3.resource") as mock_resource:
        resp = rate.handler(event, None)

    assert resp.get("batchItemFailures") == [{"itemIdentifier": "msg-1"}]
    mock_resource.assert_not_called()
