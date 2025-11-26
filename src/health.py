# src/health_lambda.py

import json
from datetime import datetime, timezone


def heartbeat_handler(event, context):
    """
    Handler for GET /health
    Baseline liveness check: just return 200 + small JSON.
    """
    body = {
        "status": "ok",
        "service": "model-registry",
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def components_handler(event, context):
    """
    Handler for GET /health/components

    Matches the RegistryHealthComponents spec:

    - Accepts query params:
      * windowMinutes: integer [5, 1440], default 60
      * includeTimeline: boolean, default false
    - Returns a HealthComponentCollection:
      {
        "components": [ HealthComponentDetail, ... ],
        "generated_at": "...",
        "window_minutes": 60
      }
    """

    # HTTP API v2 shape: queryStringParameters is a dict of strings
    qs = event.get("queryStringParameters") or {}

    # windowMinutes: 5–1440, default 60
    window_minutes = 60
    raw_wm = qs.get("windowMinutes")
    if raw_wm is not None:
        try:
            wm = int(raw_wm)
            # clamp to [5, 1440] as spec requires
            window_minutes = max(5, min(1440, wm))
        except (TypeError, ValueError):
            # on bad input, fall back to default 60
            window_minutes = 60

    # includeTimeline: boolean, default false
    raw_it = qs.get("includeTimeline")
    include_timeline = False
    if raw_it is not None:
        include_timeline = str(raw_it).lower() in ("1", "true", "yes", "y")

    now = datetime.now(timezone.utc).isoformat()

    # Minimal HealthComponentDetail satisfying the schema
    component = {
        "id": "registry-api",
        "display_name": "Model Registry API",
        "status": "ok",          # HealthStatus enum
        "observed_at": now,      # required datetime
        "description": "HTTP API + Lambda functions for the model registry.",
        "metrics": {
            # HealthMetricMap: values are HealthMetricValue (int/float/string/bool)
            "window_minutes": window_minutes,
            "dummy_request_rate_rpm": 0,
        },
        "issues": [],    # [] is allowed
        "timeline": [],
        "logs": [],
    }

    if include_timeline:
        # HealthTimelineEntry: requires bucket (datetime) and value (number), unit optional
        component["timeline"] = [
            {
                "bucket": now,
                "value": 0.0,
                "unit": "requests/min",
            }
        ]

    body = {
        "components": [component],
        "generated_at": now,
        "window_minutes": window_minutes,
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
