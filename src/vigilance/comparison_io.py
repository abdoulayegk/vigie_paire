"""I/O helpers, coercion utilities, and table-transform functions for the comparison pipeline.

Extracted from compare_gpt.py. compare_gpt.py re-exports all names from this module
so that all existing imports remain valid.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction-status buckets
# ---------------------------------------------------------------------------

_BUSINESS_EXTRACTION_STATUSES: frozenset[str] = frozenset(
    {"ok", "rescued", "suspect_unresolved"}
)
_ARTIFACT_EXTRACTION_STATUSES: frozenset[str] = frozenset({"confirmed_no_table"})
_SUSPECT_EXTRACTION_STATUSES: frozenset[str] = (
    frozenset()
)  # No longer excluding any tables from matching


# ---------------------------------------------------------------------------
# TypedDict interfaces (annotation-only, no runtime cost)
# ---------------------------------------------------------------------------


class TableCard(TypedDict):
    table_id: str
    section: str
    title: str
    table_summary: str
    page: int | None
    row_count: int
    first_indicator: str
    footnote_count: int
    headers: list[str]
    indicators: list[str]
    footnotes: list[dict[str, str]]


class TableSnapshot(TypedDict):
    table_id: str
    title: str
    table_summary: str
    extraction_status: str
    page: int | None
    section: str
    bbox: object
    row_count: int
    headers: list[str]
    indicators: list[str]
    footnotes: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Ghost-table detection
# ---------------------------------------------------------------------------


def _is_ghost_table(entry: dict[str, Any]) -> bool:
    """Return True if the table has 0 real indicators (not a real table)."""
    from vigilance.extraction.vision_full_extractor import _count_real_indicators

    indicators = list(entry.get("indicators", []) or [])
    headers = list(entry.get("headers", []) or [])
    return _count_real_indicators(indicators) == 0 and len(headers) == 0


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_pathlike(value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Chemin requis manquant: {field}")
    try:
        return Path(text)
    except TypeError as exc:
        raise ValueError(f"Chemin invalide pour {field}: {value!r}") from exc


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _load_tables_payload(extraction_dir: Path) -> dict[str, Any]:
    tables_path = extraction_dir / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Missing required file: {tables_path}")
    payload = _load_json(tables_path)
    schema_version = _coerce_int(payload.get("schema_version"))
    if schema_version != 7:
        raise ValueError(
            f"Unsupported tables.json schema_version={schema_version} in {tables_path}; expected 7"
        )
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        raise ValueError(f"Invalid tables list in {tables_path}")
    return payload


# ---------------------------------------------------------------------------
# Quarter normalization
# ---------------------------------------------------------------------------


def normalize_quarter(value: str) -> str:
    text = str(value or "").strip().lower()
    match = re.search(r"([qt])\s*([1-4])", text)
    if match:
        return f"t{match.group(2)}"
    raise ValueError(f"Trimestre invalide: {value!r}. Attendu: t1, t2, t3 ou t4.")


def resolve_reference_period(
    year_current: int,
    quarter_current: str,
) -> tuple[int, str]:
    quarter = normalize_quarter(quarter_current)
    year = int(year_current)
    if quarter == "t2":
        return year, "t1"
    if quarter == "t3":
        return year, "t2"
    if quarter == "t1":
        return year - 1, "t3"
    if quarter == "t4":
        return year - 1, "t4"
    raise ValueError(f"Trimestre invalide: {quarter_current!r}")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_footnotes(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not fid and not text:
            continue
        normalized.append({"id": fid, "text": text})
    return normalized


def _normalize_extraction_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if (
        status
        in _BUSINESS_EXTRACTION_STATUSES
        | _ARTIFACT_EXTRACTION_STATUSES
        | _SUSPECT_EXTRACTION_STATUSES
    ):
        return status
    return "ok"


# ---------------------------------------------------------------------------
# Table view builders
# ---------------------------------------------------------------------------


def _table_card(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
    row_count = int(entry.get("row_count", len(indicators)) or 0)

    header_source = entry.get("headers", [])
    header_list = list(header_source) if isinstance(header_source, list) else []
    headers = []
    for h in header_list:
        cell = str(h).strip()
        if cell:
            headers.append(cell)
    page_raw = entry.get("page")
    page: int | None
    try:
        page = (
            int(page_raw)
            if page_raw is not None and str(page_raw).strip() != ""
            else None
        )
    except (TypeError, ValueError):
        page = None

    footnotes = _normalize_footnotes(entry.get("footnotes", []))

    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": page,
        "row_count": row_count,
        "first_indicator": indicators[0] if indicators else "",
        "footnote_count": len(footnotes),
        "headers": headers,
        "indicators": indicators,
        "footnotes": footnotes,
    }


def _table_detail(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": entry.get("page"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [
            str(value).strip()
            for value in list(entry.get("headers", []) or [])
            if str(value).strip()
        ],
        "indicators": indicators,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _table_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "extraction_status": str(entry.get("extraction_status", "") or "ok"),
        "page": entry.get("page"),
        "section": str(entry.get("section", "") or "unknown_section"),
        "bbox": entry.get("bbox"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [
            str(value).strip()
            for value in list(entry.get("headers", []) or [])
            if str(value).strip()
        ],
        "indicators": indicators,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


# ---------------------------------------------------------------------------
# Table partitioning
# ---------------------------------------------------------------------------


def _partition_tables_by_status(
    tables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    business: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    suspects: list[dict[str, str]] = []
    ghost_ids: list[str] = []
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        table_id = str(entry.get("table_id", "") or "").strip()
        if not table_id:
            continue
        status = _normalize_extraction_status(entry.get("extraction_status"))
        # Artifact status takes priority over ghost filter — explicitly
        # classified tables must be reported even with 0 indicators/headers.
        if status in _ARTIFACT_EXTRACTION_STATUSES:
            artifacts.append(
                {
                    "table_id": table_id,
                    "reason": "Excluded from business matching by extraction_status=confirmed_no_table.",
                }
            )
            continue
        # Ghost table filter: 0 real indicators = not a real table
        if _is_ghost_table(entry):
            ghost_ids.append(table_id)
            continue
        if status in _BUSINESS_EXTRACTION_STATUSES:
            business.append(entry)
        else:
            suspects.append(
                {
                    "table_id": table_id,
                    "reason": (
                        "Excluded from business matching: extraction_status not classified "
                        "as business or artifact (unexpected)."
                    ),
                }
            )
    if ghost_ids:
        logger.info(
            "Filtered %d ghost table(s) (0 real indicators): %s",
            len(ghost_ids),
            ", ".join(ghost_ids),
        )
    return business, artifacts, suspects


def _merge_extraction_suspect_side(
    tables: list[dict[str, Any]],
    partition_suspect_refs: list[dict[str, str]],
    snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build review-queue rows for extraction suspects: partition edge cases plus
    all tables with extraction_status suspect_unresolved (still matched).
    """
    by_id: dict[str, dict[str, str]] = {}
    for item in partition_suspect_refs:
        tid = str(item.get("table_id", "") or "").strip()
        if tid:
            by_id[tid] = {
                "table_id": tid,
                "reason": str(item.get("reason", "") or ""),
            }
    suspect_reason = (
        "extraction_status=suspect_unresolved; table included in matching and "
        "flagged for extraction review."
    )
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        tid = str(entry.get("table_id", "") or "").strip()
        if not tid:
            continue
        if (
            _normalize_extraction_status(entry.get("extraction_status"))
            != "suspect_unresolved"
        ):
            continue
        by_id.setdefault(
            tid,
            {"table_id": tid, "reason": suspect_reason},
        )
    return [
        {**by_id[tid], **(snapshots.get(tid) or {})} for tid in sorted(by_id.keys())
    ]


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _archive_pdf(source_path: str | None, out_dir: Path, filename: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    source = Path(text)
    if not source.exists() or not source.is_file():
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    try:
        if source.resolve() != target.resolve():
            # Ne pas écraser si le fichier existe déjà (on garde le PDF original validé)
            if not target.exists():
                shutil.copy2(source, target)
        return str(target)
    except OSError:
        return ""


def _require_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required string field: {field}")
    return text


def _extract_usage_metrics(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    try:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        prompt = 0
    try:
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        completion = 0
    try:
        total = int(getattr(usage, "total_tokens", 0) or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        total = prompt + completion
    return prompt, completion, total
