import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote

import boto3
from botocore.exceptions import BotoCoreError, ClientError


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "model-metadata")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "modelzip-logs-artifacts")

_DEFAULT_LAMBDA_FUNCTIONS = [
    "Upload",
    "Rate",
    "Enqueue-Rate",
    "model-registry-reset-registry",
    "health-components",
    "Metadata",
    "Health",
    "search",
    "model-registry-delete-artifact",
]

_ENV_LAMBDA = [
    fn.strip()
    for fn in os.getenv("LAMBDA_FUNCTION_NAMES", "").split(",")
    if fn.strip()
]
LAMBDA_FUNCTIONS = _ENV_LAMBDA or _DEFAULT_LAMBDA_FUNCTIONS

DEFAULT_LOG_GROUPS = {
    "Upload": "/aws/lambda/Upload",
    "Rate": "/aws/lambda/Rate",
    "Enqueue-Rate": "/aws/lambda/Enqueue-Rate",
    "model-registry-reset-registry": "/aws/lambda/model-registry-reset-registry",
    "health-components": "/aws/lambda/health-components",
    "Metadata": "/aws/lambda/Metadata",
    "Health": "/aws/lambda/Health",
    "search": "/aws/lambda/search",
    "model-registry-delete-artifact": "/aws/lambda/model-registry-delete-artifact",
}


def _parse_log_group_env():
    raw = os.getenv("LAMBDA_LOG_GROUPS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return {k: str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        return {}


LOG_GROUP_MAP = {**DEFAULT_LOG_GROUPS, **_parse_log_group_env()}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start_time):
    return round((time.perf_counter() - start_time) * 1000, 2)


def _extract_http_method(event):
    http_ctx = event.get("requestContext", {}).get("http", {})
    method = http_ctx.get("method")
    if method:
        return method.upper()
    route_key = event.get("routeKey")
    if route_key and " " in route_key:
        return route_key.split(" ", 1)[0].upper()
    return (event.get("httpMethod") or "GET").upper()


def _extract_http_path(event):
    for key in ("rawPath", "path"):
        value = event.get(key)
        if value:
            return value
    http_ctx = event.get("requestContext", {}).get("http", {})
    if http_ctx.get("path"):
        return http_ctx["path"]
    route_key = event.get("routeKey")
    if route_key and " " in route_key:
        return route_key.split(" ", 1)[1]
    return event.get("resource") or ""


def _normalize_path(path_value):
    if not path_value:
        return ""
    normalized = path_value.strip()
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")
    return normalized


@lru_cache(maxsize=None)
def _aws_client(service_name):
    return boto3.client(service_name, region_name=AWS_REGION)


def _build_issue(summary, severity="error", details=None):
    issue = {
        "code": "dependency-check",
        "severity": severity,
        "summary": summary,
    }
    if details:
        issue["details"] = details
    return issue


def _cloudwatch_log_url(log_group):
    norm = log_group if log_group.startswith("/") else f"/{log_group}"
    encoded = quote(norm, safe="")
    return (
        f"https://console.aws.amazon.com/cloudwatch/home"
        f"?region={AWS_REGION}#logsV2:log-groups/log-group/{encoded}"
    )


def _probe_dynamodb():
    client = _aws_client("dynamodb")
    start = time.perf_counter()
    try:
        response = client.describe_table(TableName=DYNAMODB_TABLE_NAME)
        table = response.get("Table", {})
        return {
            "ok": True,
            "latency_ms": _elapsed_ms(start),
            "table_status": table.get("TableStatus", "unknown"),
            "item_count": table.get("ItemCount", 0),
        }
    except (BotoCoreError, ClientError) as exc:
        return {
            "ok": False,
            "latency_ms": _elapsed_ms(start),
            "error": str(exc),
        }


def _probe_s3():
    client = _aws_client("s3")
    start = time.perf_counter()
    try:
        client.head_bucket(Bucket=S3_BUCKET_NAME)
        # head_bucket succeeded, optionally fetch object count hint
        return {
            "ok": True,
            "latency_ms": _elapsed_ms(start),
        }
    except (BotoCoreError, ClientError) as exc:
        return {
            "ok": False,
            "latency_ms": _elapsed_ms(start),
            "error": str(exc),
        }


def _probe_lambda(function_name):
    client = _aws_client("lambda")
    start = time.perf_counter()
    try:
        response = client.get_function(FunctionName=function_name)
        config = response.get("Configuration", {})
        return {
            "ok": True,
            "latency_ms": _elapsed_ms(start),
            "runtime": config.get("Runtime"),
            "last_modified": config.get("LastModified"),
            "code_size": config.get("CodeSize"),
            "handler": config.get("Handler"),
            "version": config.get("Version"),
        }
    except (BotoCoreError, ClientError) as exc:
        return {
            "ok": False,
            "latency_ms": _elapsed_ms(start),
            "error": str(exc),
        }


def _timeline(include_timeline, now, is_ok):
    if not include_timeline:
        return []
    return [
        {
            "bucket": now,
            "value": 1 if is_ok else 0,
            "unit": "pass/fail",
        }
    ]


def heartbeat_handler(event, context):
    """
    Handler for GET /health.
    Performs reachability checks against DynamoDB + S3 and reports aggregate status.
    """

    now = _now_iso()
    dynamodb_status = _probe_dynamodb()
    s3_status = _probe_s3()
    all_ok = dynamodb_status["ok"] and s3_status["ok"]
    status_code = 200 if all_ok else 503
    body = {
        "status": "ok" if all_ok else "critical",
        "service": "model-registry",
        "checked_at": now,
        "region": AWS_REGION,
        "dependencies": {
            "dynamodb": {
                "table": DYNAMODB_TABLE_NAME,
                **dynamodb_status,
            },
            "s3": {
                "bucket": S3_BUCKET_NAME,
                **s3_status,
            },
        },
    }

    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    """
    Unified Lambda entrypoint that dispatches /health and /health/components.
    """

    method = _extract_http_method(event)
    path = _normalize_path(_extract_http_path(event))
    path_lower = path.lower()

    if method == "GET" and path_lower.endswith("/health/components"):
        return components_handler(event, context)

    if method == "GET" and path_lower.endswith("/health"):
        return heartbeat_handler(event, context)

    return {
        "statusCode": 404,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "message": "Not Found",
                "requested_path": path or "unknown",
            }
        ),
    }


def _dynamodb_component(now, window_minutes, include_timeline):
    result = _probe_dynamodb()
    issues = []
    if not result["ok"]:
        issues.append(
            _build_issue(
                "DynamoDB table unreachable",
                details=result.get("error") or "unknown error",
            )
        )
    metrics = {
        "window_minutes": window_minutes,
        "latency_ms": result["latency_ms"],
    }
    if result["ok"]:
        metrics["table_status"] = result.get("table_status")
        metrics["item_count"] = result.get("item_count", 0)
    component = {
        "id": f"dynamodb:{DYNAMODB_TABLE_NAME}",
        "display_name": f"DynamoDB ({DYNAMODB_TABLE_NAME})",
        "status": "ok" if result["ok"] else "critical",
        "observed_at": now,
        "description": "Primary metadata store for registry artifacts.",
        "metrics": metrics,
        "issues": issues,
        "timeline": _timeline(include_timeline, now, result["ok"]),
        "logs": [
            {
                "label": f"DynamoDB Table {DYNAMODB_TABLE_NAME}",
                "url": (
                    f"https://console.aws.amazon.com/dynamodbv2/home"
                    f"?region={AWS_REGION}#table?name={DYNAMODB_TABLE_NAME}"
                ),
                "tail_available": False,
                "last_updated_at": now,
            }
        ],
    }
    return component


def _s3_component(now, window_minutes, include_timeline):
    result = _probe_s3()
    issues = []
    if not result["ok"]:
        issues.append(
            _build_issue(
                "S3 bucket unreachable",
                details=result.get("error") or "unknown error",
            )
        )
    metrics = {
        "window_minutes": window_minutes,
        "latency_ms": result["latency_ms"],
    }
    component = {
        "id": f"s3:{S3_BUCKET_NAME}",
        "display_name": f"S3 ({S3_BUCKET_NAME})",
        "status": "ok" if result["ok"] else "critical",
        "observed_at": now,
        "description": "Artifact bundle storage bucket.",
        "metrics": metrics,
        "issues": issues,
        "timeline": _timeline(include_timeline, now, result["ok"]),
        "logs": [
            {
                "label": f"S3 Bucket {S3_BUCKET_NAME}",
                "url": (
                    f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET_NAME}"
                    f"?region={AWS_REGION}&tab=objects"
                ),
                "tail_available": False,
                "last_updated_at": now,
            }
        ],
    }
    return component


def _lambda_component(function_name, now, window_minutes, include_timeline):
    result = _probe_lambda(function_name)
    issues = []
    if not result["ok"]:
        issues.append(
            _build_issue(
                f"Lambda {function_name} unreachable",
                details=result.get("error") or "unknown error",
            )
        )
    log_group = LOG_GROUP_MAP.get(
        function_name, f"/aws/lambda/{function_name}"
    )
    metrics = {
        "window_minutes": window_minutes,
        "latency_ms": result["latency_ms"],
    }
    if result["ok"]:
        metrics.update(
            {
                "runtime": result.get("runtime"),
                "code_size": result.get("code_size"),
                "handler": result.get("handler"),
                "version": result.get("version"),
            }
        )
    component = {
        "id": f"lambda:{function_name}",
        "display_name": f"Lambda ({function_name})",
        "status": "ok" if result["ok"] else "critical",
        "observed_at": now,
        "description": f"Serverless component `{function_name}` supporting the registry.",
        "metrics": metrics,
        "issues": issues,
        "timeline": _timeline(include_timeline, now, result["ok"]),
        "logs": [
            {
                "label": f"CloudWatch {function_name}",
                "url": _cloudwatch_log_url(log_group),
                "tail_available": True,
                "last_updated_at": now,
            }
        ],
    }
    return component


def components_handler(event, context):
    """
    Handler for GET /health/components

    Performs health diagnostics for DynamoDB, S3, and each Lambda function.
    """

    qs = event.get("queryStringParameters") or {}

    window_minutes = 60
    raw_wm = qs.get("windowMinutes")
    if raw_wm is not None:
        try:
            wm = int(raw_wm)
            window_minutes = max(5, min(1440, wm))
        except (TypeError, ValueError):
            window_minutes = 60

    raw_it = qs.get("includeTimeline")
    include_timeline = False
    if raw_it is not None:
        include_timeline = str(raw_it).lower() in ("1", "true", "yes", "y")

    now = _now_iso()
    components = [
        _dynamodb_component(now, window_minutes, include_timeline),
        _s3_component(now, window_minutes, include_timeline),
    ]

    for fn in LAMBDA_FUNCTIONS:
        components.append(
            _lambda_component(fn, now, window_minutes, include_timeline)
        )

    body = {
        "components": components,
        "generated_at": now,
        "window_minutes": window_minutes,
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
