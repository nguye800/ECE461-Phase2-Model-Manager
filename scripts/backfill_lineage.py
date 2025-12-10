"""
One-time lineage backfill utility.

Scans every MODEL record in the artifacts table, rebuilds its lineage graph using
the same helpers as the upload lambda, and writes the lineage field back.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import boto3
from boto3.dynamodb.conditions import Attr

from src.upload import (
    _build_dynamo_key,
    _build_lineage_graph_payload,
    _get_artifacts_table,
    _put_item,
)


def _convert(value: Any):
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value


def backfill_lineage(table=None, *, force: bool = False) -> Dict[str, int]:
    table = table or _get_artifacts_table()
    sk_field = os.getenv("RATE_SK_FIELD", "sk")
    meta_sk = os.getenv("RATE_META_SK", "META")

    filter_expression = Attr("type").eq("MODEL") & Attr(sk_field).eq(meta_sk)
    scan_kwargs: Dict[str, Any] = {"FilterExpression": filter_expression}

    processed = 0
    updated = 0

    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            processed += 1
            plain = _convert(item)
            graph, _ = _build_lineage_graph_payload(table, plain)
            if not force and plain.get("lineage") == graph:
                continue
            plain["lineage"] = graph
            plain["updated_at"] = datetime.now(timezone.utc).isoformat()
            _put_item(table, plain)
            updated += 1
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    return {"processed": processed, "updated": updated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill lineage graphs in DynamoDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite lineage graphs even if they already exist.",
    )
    args = parser.parse_args()
    stats = backfill_lineage(force=args.force)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
