"""JSON export utilities for section ranges and table extraction outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.models.section_models import SectionRangesResult
from vigilance.models.table_models import TableArtifact


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_out_dir(out_dir: str | Path) -> Path:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_section_ranges(out_dir: str | Path, result: SectionRangesResult) -> Path:
    """Write `section_ranges.json` in *out_dir* and return its path."""
    target_dir = _prepare_out_dir(out_dir)
    out_path = target_dir / "section_ranges.json"
    payload: dict[str, Any] = {
        "metadata": {
            "bank_code": result.bank_code,
            "quarter": result.quarter,
            "pdf_path": result.pdf_path,
            "created_at": _utc_now_iso(),
        },
        "ranges": [item.to_dict() for item in result.ranges],
        "skipped_pages": result.skipped_pages,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def write_tables_docling(out_dir: str | Path, tables: list[TableArtifact]) -> Path:
    """Write `tables_docling.json` in *out_dir* and return its path."""
    target_dir = _prepare_out_dir(out_dir)
    out_path = target_dir / "tables_docling.json"
    first = tables[0] if tables else None
    payload: dict[str, Any] = {
        "metadata": {
            "bank_code": first.bank_code if first else "",
            "quarter": first.quarter if first else "",
            "pdf_path": first.pdf_path if first else "",
            "created_at": _utc_now_iso(),
        },
        "tables": [table.to_dict() for table in tables],
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
