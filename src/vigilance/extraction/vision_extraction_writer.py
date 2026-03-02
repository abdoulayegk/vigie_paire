"""Writer for Vision extraction audit: indicators.json and footnotes.json."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.footnotes_utils import (
    count_stringified_dict_suspects,
    footnotes_list_to_dict,
)

logger = logging.getLogger(__name__)


def _table_entry_indicators(
    table: Any,
    source: str,
) -> dict[str, Any]:
    """Build indicators entry for one table."""
    table_id = str(getattr(table, "table_id", "") or "")
    title = (
        getattr(table, "title_clean", None)
        or getattr(table, "title", None)
        or ""
    )
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    indicators = list(getattr(table, "first_column_indicators", []) or [])
    indicators_raw = list(getattr(table, "first_column_indicators_raw", None) or indicators)
    unit_context = getattr(table, "unit_context", None) or ""

    sections: list[dict[str, Any]] = []
    if indicators_raw:
        sections.append({
            "section": title or "Indicateurs",
            "indicators": [str(x).strip() for x in indicators_raw if str(x).strip()],
        })

    return {
        "table_id": table_id,
        "title": str(title),
        "date_reference": str(unit_context),
        "page": page,
        "source": source,
        "sections": sections,
    }


def _table_entry_footnotes(
    table: Any,
    source: str,
) -> dict[str, Any]:
    """Build footnotes entry for one table."""
    table_id = str(getattr(table, "table_id", "") or "")
    title = (
        getattr(table, "title_clean", None)
        or getattr(table, "title", None)
        or ""
    )
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    footnotes_raw = getattr(table, "footnotes", None) or []
    fn_dict = footnotes_list_to_dict(footnotes_raw)
    footnote_markers = list(fn_dict.keys())
    repr_suspects = count_stringified_dict_suspects(footnotes_raw)

    return {
        "table_id": table_id,
        "title": str(title),
        "page": page,
        "source": source,
        "has_footnotes": bool(fn_dict),
        "footnote_markers": footnote_markers,
        "footnotes_content": fn_dict,
        "_repr_suspect_count": int(repr_suspects),
    }


def write_indicators_json(
    tables_t1: list[Any],
    tables_t2: list[Any],
    out_dir: Path,
    bank_code: str,
    run_id: str,
) -> Path:
    """
    Write indicators.json for audit.

    Each entry: table_id, title, date_reference, page, source (t1/t2), sections.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for t in tables_t1:
        entries.append(_table_entry_indicators(t, "t1"))
    for t in tables_t2:
        entries.append(_table_entry_indicators(t, "t2"))

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "tables": entries,
    }

    out_path = out_dir / "indicators.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote indicators.json to %s", out_path)
    return out_path


def write_footnotes_json(
    tables_t1: list[Any],
    tables_t2: list[Any],
    out_dir: Path,
    bank_code: str,
    run_id: str,
) -> Path:
    """
    Write footnotes.json for audit.

    Each entry: table_id, title, page, source (t1/t2), has_footnotes,
    footnote_markers, footnotes_content.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for t in tables_t1:
        entries.append(_table_entry_footnotes(t, "t1"))
    for t in tables_t2:
        entries.append(_table_entry_footnotes(t, "t2"))

    tables_total = len(entries)
    tables_with_footnotes = sum(1 for e in entries if bool(e.get("has_footnotes")))
    footnote_entries_total = sum(len(e.get("footnotes_content", {}) or {}) for e in entries)
    repr_suspect_count = sum(int(e.get("_repr_suspect_count", 0) or 0) for e in entries)

    for entry in entries:
        entry.pop("_repr_suspect_count", None)

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "meta": {
            "tables_total": tables_total,
            "tables_with_footnotes": tables_with_footnotes,
            "footnote_entries_total": footnote_entries_total,
            "repr_suspect_count": repr_suspect_count,
        },
        "tables": entries,
    }
    if repr_suspect_count > 0:
        payload["warnings"] = [
            {
                "code": "repr_suspect_detected",
                "message": "Stringified-dict footnotes detected in source payload.",
                "count": repr_suspect_count,
            }
        ]

    out_path = out_dir / "footnotes.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote footnotes.json to %s", out_path)
    return out_path
