"""Debug sidecar writer for extraction metrics. Does not affect pipeline or comparison output."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_extraction_debug(
    bank: str,
    quarter: str,
    year: int,
    tables: list[Any],
) -> Path | None:
    """
    Write extraction debug JSON to outputs/debug_extractions/.
    Only runs if ENABLE_EXTRACTION_DEBUG=1. Otherwise returns None immediately.

    Filename: {bank}_{quarter}_{year}_{timestamp}.json

    Each table record includes:
    page_number, section, table_id, title_raw, title_clean,
    row_count_before_merge, row_count_after_merge, merge_count,
    footnote_row_filtered_count, indicator_count, duplicate_ratio,
    header_like_ratio, first_column_raw, first_column_clean.
    """
    if os.environ.get("ENABLE_EXTRACTION_DEBUG") != "1":
        return None

    try:
        out_dir = Path("outputs") / "debug_extractions"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{bank}_{quarter}_{year}_{ts}.json"
        out_path = out_dir / filename

        records: list[dict[str, Any]] = []
        for t in tables:
            dm = getattr(t, "debug_metrics", None) or {}
            first_raw = list(getattr(t, "first_column_indicators_raw", None) or [])
            first_clean = list(getattr(t, "first_column_indicators", []) or [])

            records.append({
                "page_number": int(getattr(t, "page_number", 0) or 0),
                "section": getattr(t, "section", None),
                "table_id": getattr(t, "table_id", ""),
                "title_raw": getattr(t, "title_raw", None),
                "title_clean": getattr(t, "title_clean", None),
                "row_count_before_merge": dm.get("row_count_before_merge"),
                "row_count_after_merge": dm.get("row_count_after_merge"),
                "merge_count": dm.get("merge_count", 0),
                "footnote_row_filtered_count": dm.get("footnote_row_filtered_count", 0),
                "indicator_count": dm.get("indicator_count", len(first_clean)),
                "duplicate_ratio": dm.get("duplicate_ratio", 0.0),
                "header_like_ratio": dm.get("header_like_ratio", 0.0),
                "first_column_raw": first_raw,
                "first_column_clean": first_clean,
            })

        payload: dict[str, Any] = {
            "bank": bank,
            "quarter": quarter,
            "year": year,
            "timestamp": ts,
            "table_count": len(records),
            "tables": records,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Extraction debug written to %s", out_path)
        return out_path
    except Exception:
        pass
    return None
