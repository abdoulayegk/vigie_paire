"""Writer for Vision extraction audit: indicators.json and footnotes.json."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..extraction.section_taxonomy import canonicalize_section
from ..models.table_models import (
    get_canonical_footnotes,
    get_comparison_indicators,
    get_vision_raw_indicators,
)
from ..utils.footnotes_utils import (
    count_stringified_dict_suspects,
    footnotes_list_to_dict,
)

logger = logging.getLogger(__name__)

COMPACT_REPORT_SCHEMA_VERSION = 3


def _table_section(table: Any) -> str:
    """Return a human-readable section value for a table-like object."""
    return str(getattr(table, "section", "") or "").strip() or "unknown_section"


def _compact_table_section(table: Any) -> str:
    """Return canonical compact section value with a defensive fallback."""
    raw = str(getattr(table, "section", "") or "").strip()
    if not raw:
        return "unknown_section"
    try:
        normalized = canonicalize_section(raw)
    except Exception:
        normalized = raw
    return str(normalized or "unknown_section").strip() or "unknown_section"


def _table_title(table: Any) -> str:
    """Return a readable title fallback for a table-like object."""
    title = getattr(table, "title_clean", None) or getattr(table, "title", None) or ""
    return str(title or "").strip() or "(sans titre)"


def _table_page(table: Any) -> int:
    """Return the PDF page number for a table-like object."""
    return int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)


def _compact_table_title(table: Any) -> str:
    """Return compact title value; empty string is preserved when no title exists."""
    title = getattr(table, "title_clean", None)
    if title is None:
        title = getattr(table, "title", None)
    return str(title or "")


def _compact_created_at(meta: dict[str, Any] | None = None) -> str:
    metadata = dict(meta or {})
    return str(
        metadata.get("created_at")
        or metadata.get("extracted_at")
        or datetime.now().isoformat(timespec="seconds")
    )


def _compact_top_level(
    *,
    bank_code: str,
    year: int,
    quarter: str,
    created_at: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(meta or {})
    return {
        "bank_code": str(bank_code),
        "year": int(year),
        "quarter": str(quarter),
        "created_at": str(created_at),
        "schema_version": int(metadata.get("schema_version") or COMPACT_REPORT_SCHEMA_VERSION),
        "model_version": str(metadata.get("model_version") or ""),
        "prompt_version": str(metadata.get("prompt_version") or ""),
    }


def _stable_sort_tables(tables: list[Any]) -> list[Any]:
    indexed = list(enumerate(tables))
    indexed.sort(
        key=lambda item: (
            _table_page(item[1]),
            int(getattr(item[1], "table_index_on_page", 10**9) or 10**9),
            item[0],
        )
    )
    return [table for _, table in indexed]


def _compact_footnotes(table: Any) -> list[dict[str, str]]:
    return [
        {
            "id": str(item.get("id") or "").strip(),
            "text": str(item.get("text") or "").strip(),
        }
        for item in get_canonical_footnotes(table)
        if str(item.get("id") or "").strip() and str(item.get("text") or "").strip()
    ]


def _compact_table_common_entry(table: Any) -> dict[str, Any]:
    return {
        "table_id": str(getattr(table, "table_id", "") or ""),
        "page": _table_page(table),
        "section": _compact_table_section(table),
        "title": _compact_table_title(table),
    }


def _compact_table_entry(table: Any) -> dict[str, Any]:
    entry = _compact_table_common_entry(table)
    entry["headers"] = [
        str(value) for value in list(getattr(table, "headers", []) or [])
    ]
    rows = []
    for row in list(getattr(table, "rows", []) or []):
        if isinstance(row, (list, tuple)):
            rows.append([str(value) for value in row])
        else:
            rows.append([str(row)])
    entry["rows"] = rows
    entry["indicators_raw"] = [
        str(value).strip()
        for value in get_vision_raw_indicators(table)
        if str(value).strip()
    ]
    entry["indicators_normalized"] = [
        str(value).strip()
        for value in get_comparison_indicators(table)
        if str(value).strip()
    ]
    entry["footnotes"] = _compact_footnotes(table)
    return entry


def _compact_indicator_entry(table: Any) -> dict[str, Any]:
    entry = _compact_table_common_entry(table)
    entry["indicators_raw"] = [
        str(value).strip()
        for value in get_vision_raw_indicators(table)
        if str(value).strip()
    ]
    entry["indicators_normalized"] = [
        str(value).strip()
        for value in get_comparison_indicators(table)
        if str(value).strip()
    ]
    return entry


def _compact_footnote_entry(table: Any) -> dict[str, Any]:
    entry = _compact_table_common_entry(table)
    entry["footnotes"] = _compact_footnotes(table)
    return entry


def _atomic_write_json(out_path: Path, payload: dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)
    return out_path


def _table_indicators_for_text(table: Any) -> list[str]:
    """Return report-friendly indicators, preferring raw Vision labels."""
    indicators = get_vision_raw_indicators(table) or get_comparison_indicators(table)
    return [str(x).strip() for x in indicators if str(x).strip()]


def _table_footnotes_for_text(table: Any) -> list[dict[str, str]]:
    """Return canonical footnotes for human-readable report exports."""
    return get_canonical_footnotes(table)


def _report_statistics(tables: list[Any]) -> dict[str, Any]:
    """Compute summary statistics for one extracted report."""
    section_distribution: dict[str, int] = {}
    tables_with_indicators = 0
    indicators_total = 0
    tables_with_footnotes = 0
    footnote_entries_total = 0

    for table in tables:
        section = _table_section(table)
        section_distribution[section] = section_distribution.get(section, 0) + 1

        indicators = _table_indicators_for_text(table)
        if indicators:
            tables_with_indicators += 1
            indicators_total += len(indicators)

        footnotes = _table_footnotes_for_text(table)
        if footnotes:
            tables_with_footnotes += 1
            footnote_entries_total += len(footnotes)

    return {
        "tables_total": len(tables),
        "sections_detected": sorted(section_distribution.keys()),
        "tables_with_indicators": tables_with_indicators,
        "indicators_total": indicators_total,
        "tables_with_footnotes": tables_with_footnotes,
        "footnote_entries_total": footnote_entries_total,
        "section_distribution": section_distribution,
    }


def _report_summary_payload(
    tables: list[Any],
    *,
    bank_code: str,
    year: int,
    quarter: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical report-centric summary payload."""
    statistics = _report_statistics(tables)
    payload: dict[str, Any] = {
        **_report_payload_metadata(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            meta=meta,
        ),
        "summary": statistics,
    }
    warnings: list[dict[str, Any]] = []
    repr_suspect_count = 0
    for table in tables:
        repr_suspect_count += count_stringified_dict_suspects(
            getattr(table, "footnotes", None) or []
        )
    if repr_suspect_count > 0:
        warnings.append(
            {
                "code": "repr_suspect_detected",
                "message": "Stringified-dict footnotes detected in extracted tables.",
                "count": int(repr_suspect_count),
            }
        )
    if warnings:
        payload["warnings"] = warnings
    return payload


def _lines_to_text(lines: list[str]) -> str:
    """Join text lines with a trailing newline for file outputs."""
    return "\n".join(lines).rstrip() + "\n"


def _table_entry_indicators(
    table: Any,
    source: str,
) -> dict[str, Any]:
    """Build indicators entry for one table."""
    table_id = str(getattr(table, "table_id", "") or "")
    title = getattr(table, "title_clean", None) or getattr(table, "title", None) or ""
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    indicators = get_comparison_indicators(table)
    indicators_raw = get_vision_raw_indicators(table) or indicators
    unit_context = getattr(table, "unit_context", None) or ""

    sections: list[dict[str, Any]] = []
    if indicators_raw:
        sections.append(
            {
                "section": title or "Indicateurs",
                "indicators": [
                    str(x).strip() for x in indicators_raw if str(x).strip()
                ],
            }
        )

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
    title = getattr(table, "title_clean", None) or getattr(table, "title", None) or ""
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    footnotes_source = getattr(table, "footnotes", None) or []
    footnotes_raw = get_canonical_footnotes(table)
    fn_dict = footnotes_list_to_dict(footnotes_raw)
    footnote_markers = list(fn_dict.keys())
    repr_suspects = count_stringified_dict_suspects(footnotes_source)

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
        "artifact_role": "audit_only",
        "authoritative_source": "table_artifacts_for_comparison",
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
    footnote_entries_total = sum(
        len(e.get("footnotes_content", {}) or {}) for e in entries
    )
    repr_suspect_count = sum(int(e.get("_repr_suspect_count", 0) or 0) for e in entries)

    for entry in entries:
        entry.pop("_repr_suspect_count", None)

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "artifact_role": "audit_only",
        "authoritative_source": "table_artifacts_for_comparison",
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


def _report_payload_metadata(
    *,
    bank_code: str,
    year: int,
    quarter: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build common metadata for report-centric extraction artifacts."""
    metadata = dict(meta or {})
    created_at = str(
        metadata.get("created_at")
        or metadata.get("extracted_at")
        or datetime.now().isoformat(timespec="seconds")
    )
    return {
        "bank_code": bank_code,
        "year": int(year),
        "quarter": str(quarter),
        "created_at": created_at,
        "artifact_role": "report_canonical",
        "authoritative_source": "stored_table_artifacts",
        "pdf_fingerprint": str(metadata.get("pdf_fingerprint") or ""),
        "pipeline_version": str(metadata.get("pipeline_version") or ""),
        "schema_version": metadata.get("schema_version"),
        "model_version": str(metadata.get("model_version") or ""),
        "prompt_version": str(metadata.get("prompt_version") or ""),
    }


def write_report_indicators_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report-centric indicators.json for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [_table_entry_indicators(table, str(quarter)) for table in tables]
    indicators_total = sum(
        len(section.get("indicators", []) or [])
        for entry in entries
        for section in list(entry.get("sections") or [])
        if isinstance(section, dict)
    )
    payload: dict[str, Any] = {
        **_report_payload_metadata(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            meta=meta,
        ),
        "meta": {
            "tables_total": len(entries),
            "indicators_total": indicators_total,
        },
        "tables": entries,
    }
    out_path = out_dir / "indicators.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote report indicators.json to %s", out_path)
    return out_path


def write_report_indicators_txt(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report-centric indicators.txt for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _report_payload_metadata(
        bank_code=bank_code,
        year=year,
        quarter=quarter,
        meta=meta,
    )
    lines = [
        "Rapport des indicateurs extraits",
        f"Banque: {metadata['bank_code']}",
        f"Annee: {metadata['year']}",
        f"Trimestre: {metadata['quarter']}",
        f"Date de generation: {metadata['created_at']}",
        "",
    ]

    included = 0
    for table in tables:
        indicators = _table_indicators_for_text(table)
        if not indicators:
            continue
        included += 1
        lines.extend(
            [
                f"Section: {_table_section(table)}",
                f"Tableau: {_table_title(table)}",
                f"Page: {_table_page(table)}",
                "Indicateurs:",
            ]
        )
        for idx, indicator in enumerate(indicators, start=1):
            lines.append(f"{idx}. {indicator}")
        lines.append("")

    if included == 0:
        lines.append("Aucun indicateur extrait pour ce rapport.")

    out_path = out_dir / "indicators.txt"
    out_path.write_text(_lines_to_text(lines), encoding="utf-8")
    logger.debug("Wrote report indicators.txt to %s", out_path)
    return out_path


def write_report_footnotes_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report-centric footnotes.json for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [_table_entry_footnotes(table, str(quarter)) for table in tables]
    tables_total = len(entries)
    tables_with_footnotes = sum(1 for e in entries if bool(e.get("has_footnotes")))
    footnote_entries_total = sum(
        len(e.get("footnotes_content", {}) or {}) for e in entries
    )
    repr_suspect_count = sum(int(e.get("_repr_suspect_count", 0) or 0) for e in entries)
    for entry in entries:
        entry.pop("_repr_suspect_count", None)
    payload: dict[str, Any] = {
        **_report_payload_metadata(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            meta=meta,
        ),
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
    logger.debug("Wrote report footnotes.json to %s", out_path)
    return out_path


def write_report_footnotes_txt(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report-centric footnotes.txt for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _report_payload_metadata(
        bank_code=bank_code,
        year=year,
        quarter=quarter,
        meta=meta,
    )
    lines = [
        "Rapport des footnotes extraites",
        f"Banque: {metadata['bank_code']}",
        f"Annee: {metadata['year']}",
        f"Trimestre: {metadata['quarter']}",
        f"Date de generation: {metadata['created_at']}",
        "",
    ]

    included = 0
    for table in tables:
        footnotes = _table_footnotes_for_text(table)
        if not footnotes:
            continue
        included += 1
        lines.extend(
            [
                f"Section: {_table_section(table)}",
                f"Tableau: {_table_title(table)}",
                f"Page: {_table_page(table)}",
                "Footnotes:",
            ]
        )
        for footnote in footnotes:
            fid = str(footnote.get("id", "") or "").strip() or "-"
            text = str(footnote.get("text", "") or "").strip()
            lines.append(f"- [{fid}] {text}")
        lines.append("")

    if included == 0:
        lines.append("Aucune footnote extraite pour ce rapport.")

    out_path = out_dir / "footnotes.txt"
    out_path.write_text(_lines_to_text(lines), encoding="utf-8")
    logger.debug("Wrote report footnotes.txt to %s", out_path)
    return out_path


def write_compact_tables_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write the compact canonical tables.json used by the minimal extraction flow."""
    created_at = _compact_created_at(meta)
    ordered_tables = _stable_sort_tables(tables)
    payload: dict[str, Any] = {
        **_compact_top_level(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            created_at=created_at,
            meta=meta,
        ),
        "tables": [_compact_table_entry(table) for table in ordered_tables],
    }
    out_path = out_dir / "tables.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact tables.json to %s", out_path)
    return out_path


def write_compact_indicators_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write the compact indicators.json used by the minimal extraction flow."""
    created_at = _compact_created_at(meta)
    ordered_tables = _stable_sort_tables(tables)
    payload: dict[str, Any] = {
        **_compact_top_level(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            created_at=created_at,
            meta=meta,
        ),
        "tables": [_compact_indicator_entry(table) for table in ordered_tables],
    }
    out_path = out_dir / "indicators.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact indicators.json to %s", out_path)
    return out_path


def write_compact_footnotes_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write the compact footnotes.json used by the minimal extraction flow."""
    created_at = _compact_created_at(meta)
    ordered_tables = _stable_sort_tables(tables)
    payload: dict[str, Any] = {
        **_compact_top_level(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            created_at=created_at,
            meta=meta,
        ),
        "tables": [_compact_footnote_entry(table) for table in ordered_tables],
    }
    out_path = out_dir / "footnotes.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact footnotes.json to %s", out_path)
    return out_path


def write_compact_report_artifacts(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write tables.json, indicators.json, and footnotes.json from the same in-memory tables."""
    metadata = dict(meta or {})
    metadata["created_at"] = _compact_created_at(metadata)
    return {
        "tables": write_compact_tables_json(
            tables, out_dir, bank_code, year, quarter, meta=metadata
        ),
        "indicators": write_compact_indicators_json(
            tables, out_dir, bank_code, year, quarter, meta=metadata
        ),
        "footnotes": write_compact_footnotes_json(
            tables, out_dir, bank_code, year, quarter, meta=metadata
        ),
    }


def write_report_summary_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report_summary.json for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _report_summary_payload(
        tables,
        bank_code=bank_code,
        year=year,
        quarter=quarter,
        meta=meta,
    )
    out_path = out_dir / "report_summary.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote report_summary.json to %s", out_path)
    return out_path


def write_report_summary_txt(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write report_summary.txt for one extracted quarter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _report_summary_payload(
        tables,
        bank_code=bank_code,
        year=year,
        quarter=quarter,
        meta=meta,
    )
    summary = payload.get("summary", {}) or {}
    lines = [
        "Resume final du rapport extrait",
        f"Banque: {payload['bank_code']}",
        f"Annee: {payload['year']}",
        f"Trimestre: {payload['quarter']}",
        f"Date de generation: {payload['created_at']}",
        "",
        (
            f"Le rapport contient {summary.get('tables_total', 0)} tableau(x), "
            f"{summary.get('indicators_total', 0)} indicateur(s) et "
            f"{summary.get('footnote_entries_total', 0)} footnote(s) extraits."
        ),
        "",
        "Synthese chiffree:",
        f"- Tables total: {summary.get('tables_total', 0)}",
        f"- Sections detectees: {len(summary.get('sections_detected', []) or [])}",
        f"- Tables avec indicateurs: {summary.get('tables_with_indicators', 0)}",
        f"- Indicateurs total: {summary.get('indicators_total', 0)}",
        f"- Tables avec footnotes: {summary.get('tables_with_footnotes', 0)}",
        f"- Footnotes total: {summary.get('footnote_entries_total', 0)}",
        "",
        "Repartition par section:",
    ]
    section_distribution = summary.get("section_distribution", {}) or {}
    if section_distribution:
        for section, count in sorted(section_distribution.items()):
            lines.append(f"- {section}: {count} tableau(x)")
    else:
        lines.append("- Aucune section detectee")

    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings extraction:"])
        for warning in warnings:
            message = str(warning.get("message", "") or "").strip()
            count = warning.get("count")
            suffix = f" ({count})" if count is not None else ""
            lines.append(f"- {message}{suffix}")

    out_path = out_dir / "report_summary.txt"
    out_path.write_text(_lines_to_text(lines), encoding="utf-8")
    logger.debug("Wrote report_summary.txt to %s", out_path)
    return out_path
