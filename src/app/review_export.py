"""Export helpers for review items."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from app.review_models import ReviewItem


_CSV_COLUMNS = [
    "change_id",
    "table_name",
    "section",
    "page_t1",
    "page_t2",
    "indicator_name",
    "indicator_type",
    "table_status",
    "review_status",
    "comment",
    "confidence",
]


def export_review_items_csv(items: list[ReviewItem]) -> str:
    """Serialize review items to CSV text with one row per indicator."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for item in items:
        base = item.to_dict()
        indicators = base.get("indicators", [])
        if not indicators:
            writer.writerow({
                "change_id": base.get("change_id", ""),
                "table_name": base.get("table_name", ""),
                "section": base.get("section", ""),
                "page_t1": base.get("page_t1", ""),
                "page_t2": base.get("page_t2", ""),
                "indicator_name": base.get("indicator", ""),
                "indicator_type": base.get("change_type", ""),
                "table_status": base.get("table_status", ""),
                "review_status": base.get("review_status", ""),
                "comment": base.get("comment", ""),
                "confidence": base.get("confidence", ""),
            })
        else:
            for ind in indicators:
                writer.writerow({
                    "change_id": base.get("change_id", ""),
                    "table_name": base.get("table_name", ""),
                    "section": base.get("section", ""),
                    "page_t1": base.get("page_t1", ""),
                    "page_t2": base.get("page_t2", ""),
                    "indicator_name": ind.get("name", ""),
                    "indicator_type": ind.get("type", ""),
                    "table_status": base.get("table_status", ""),
                    "review_status": base.get("review_status", ""),
                    "comment": base.get("comment", ""),
                    "confidence": base.get("confidence", ""),
                })
    return buffer.getvalue()


def export_review_items_json_fr(
    items: list[ReviewItem],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialize review items to JSON (French-oriented schema)."""
    payload = {
        "metadata": metadata or {},
        "total": len(items),
        "items": [item.to_dict() for item in items],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
