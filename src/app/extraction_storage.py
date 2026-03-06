"""Persistance des extractions par banque/annee/trimestre pour decouplage extraction/comparaison."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.models.table_models import TableArtifact

logger = logging.getLogger(__name__)

STORAGE_SCHEMA_VERSION = 2


def _normalize_storage_quarter(quarter: str) -> str:
    """Normalize quarter labels to ``t1``..``t4`` for storage."""
    value = str(quarter or "").strip().lower()
    match = None
    if value:
        import re

        match = re.search(r"([qt])\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        return f"t{match.group(2)}"
    return value or "t1"


def _extraction_dir(base_dir: Path, bank_code: str, year: int, quarter: str) -> Path:
    """Return the directory for one report: {base_dir}/{bank}/{year}/{quarter}/."""
    return base_dir / str(bank_code) / str(year) / str(quarter).lower()


def _footnote_to_canonical(item: dict[str, Any]) -> dict[str, str]:
    """Normalize a footnote item to canonical stored form: {"id": str, "text": str}."""
    fid = (item.get("id") or item.get("marker") or "").strip()
    text = str(item.get("text") or "").strip()
    return {"id": fid, "text": text}


def table_artifact_from_dict(d: dict[str, Any]) -> TableArtifact:
    """Reconstruct TableArtifact from a dict (from to_dict / JSON load)."""
    # Normalize keys and handle optional fields
    bank_code = str(d.get("bank_code", "") or "")
    section = str(d.get("section", "") or "")
    page_pdf = int(d.get("page_pdf", 0) or 0)
    table_id = str(d.get("table_id", "") or "")
    title = d.get("title")
    headers = list(d.get("headers") or [])
    rows = list(d.get("rows") or [])
    for i, row in enumerate(rows):
        rows[i] = list(row) if isinstance(row, (list, tuple)) else [str(row)]
    first_column_indicators = list(d.get("first_column_indicators") or [])
    extraction_method = str(d.get("extraction_method") or "docling")
    title_clean = d.get("title_clean")
    title_raw = d.get("title_raw")
    table_number = d.get("table_number")
    bbox = d.get("bbox")
    quarter = d.get("quarter")
    pdf_path = d.get("pdf_path")
    first_column_indicators_raw = d.get("first_column_indicators_raw")
    footnotes = d.get("footnotes")
    # Backward compat: old stored extractions may have footnotes as dict {marker: text}
    if isinstance(footnotes, dict):
        footnotes = [
            {"id": str(k), "text": str(v)}
            for k, v in footnotes.items()  # preserves insertion order (Python 3.7+)
        ]
    elif footnotes is not None and not isinstance(footnotes, list):
        footnotes = None
    fragmentation_detected = bool(d.get("fragmentation_detected", False))
    debug_metrics = d.get("debug_metrics")
    if debug_metrics is not None and not isinstance(debug_metrics, dict):
        debug_metrics = None

    return TableArtifact(
        bank_code=bank_code,
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=headers,
        rows=rows,
        first_column_indicators=first_column_indicators,
        extraction_method=extraction_method,
        title_clean=title_clean,
        title_raw=title_raw,
        table_number=table_number,
        bbox=bbox,
        quarter=quarter,
        pdf_path=pdf_path,
        first_column_indicators_raw=first_column_indicators_raw,
        footnotes=footnotes,
        fragmentation_detected=fragmentation_detected,
        debug_metrics=debug_metrics,
    )


def save_extraction(
    bank_code: str,
    year: int,
    quarter: str,
    tables: list[TableArtifact],
    meta: dict[str, Any],
    base_dir: Path,
) -> Path:
    """
    Save extraction for one report.

    Writes tables.json and meta.json in outputs/extractions/{bank_code}/{year}/{quarter}/.
    Returns the directory path.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    target_dir.mkdir(parents=True, exist_ok=True)

    tables_path = target_dir / "tables.json"
    tables_dicts = [t.to_dict() for t in tables]
    for d in tables_dicts:
        raw_footnotes = d.get("footnotes") or []
        d["footnotes"] = [_footnote_to_canonical(fn) for fn in raw_footnotes]
    tables_payload = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "tables": tables_dicts,
        "bank_code": bank_code,
        "year": year,
        "quarter": quarter_norm,
    }
    tables_path.write_text(
        json.dumps(tables_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_path = target_dir / "meta.json"
    meta_payload = dict(meta or {})
    meta_payload.setdefault("schema_version", STORAGE_SCHEMA_VERSION)
    meta_payload.setdefault("bank_code", bank_code)
    meta_payload.setdefault("year", year)
    meta_payload.setdefault("quarter", quarter_norm)
    meta_payload.setdefault("extracted_at", datetime.now(timezone.utc).isoformat())
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Extraction enregistree: %s/%s/%s (%d tableaux)",
        bank_code,
        year,
        quarter_norm,
        len(tables),
    )
    return target_dir


def load_extraction(
    bank_code: str,
    year: int,
    quarter: str,
    base_dir: Path,
) -> tuple[list[TableArtifact], dict[str, Any]] | None:
    """
    Load extraction for one report.

    Returns (tables, meta) or None if not found or invalid.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    tables_path = target_dir / "tables.json"
    meta_path = target_dir / "meta.json"

    if not tables_path.exists():
        return None
    try:
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        _ = payload.get("schema_version")  # optional: for future version checks / logging
        tables_data = payload.get("tables", [])
        if not isinstance(tables_data, list):
            return None
        tables = [table_artifact_from_dict(t) for t in tables_data]
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (tables, meta)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug("Load extraction failed %s: %s", target_dir, e)
        return None


def load_stored_extractions(
    bank_code: str,
    year: int,
    base_dir: Path,
) -> (
    tuple[list[TableArtifact], list[TableArtifact], dict[str, Any], dict[str, Any]]
    | None
):
    """
    Load both t1 and t2 extractions if they exist.

    Returns (tables_t1, tables_t2, meta_t1, meta_t2) or None if either is missing.
    """
    t1 = load_extraction(bank_code, year, "t1", base_dir)
    t2 = load_extraction(bank_code, year, "t2", base_dir)
    if t1 is None or t2 is None:
        return None
    tables_t1, meta_t1 = t1
    tables_t2, meta_t2 = t2
    return (tables_t1, tables_t2, meta_t1, meta_t2)
