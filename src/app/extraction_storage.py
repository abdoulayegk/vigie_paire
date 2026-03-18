"""Persist canonical ``TableArtifact`` extractions for replayable comparison.

This storage layer persists the canonical in-memory table objects used by the
comparison engine. It is distinct from the canonical Dash/UI payload written by
``app.comparison_canonical``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.models.table_models import TableArtifact, infer_content_source
from vigilance.utils.indicator_cleaner import (
    normalize_indicator_for_comparison,
    post_normalize_indicator,
)
from vigilance.utils.table_page_structure import derive_page_local_structure

logger = logging.getLogger(__name__)

STORAGE_SCHEMA_VERSION = 3
TABLES_FILENAME = "tables.json"
META_FILENAME = "meta.json"
REPORT_SUMMARY_JSON_FILENAME = "report_summary.json"
REPORT_SUMMARY_TXT_FILENAME = "report_summary.txt"
REPORT_INDICATORS_FILENAME = "indicators.json"
REPORT_INDICATORS_TXT_FILENAME = "indicators.txt"
REPORT_FOOTNOTES_FILENAME = "footnotes.json"
REPORT_FOOTNOTES_TXT_FILENAME = "footnotes.txt"

# Contract versions for cache compatibility. Bump to reject older stored artifacts.
ARTIFACT_CONTRACT_VERSION = 1
EXTRACTION_METRICS_VERSION = 1
QUALITY_POLICY_VERSION = 1


def build_extraction_manifest(
    pdf_path: str,
    section_ranges: list[dict[str, Any]],
    extraction_mode: str = "vision_full_gpt4o",
) -> dict[str, Any]:
    """Build manifest for cache compatibility (provenance + contract versions)."""
    try:
        path_bytes = Path(pdf_path).resolve().as_posix().encode("utf-8")
        pdf_fingerprint = hashlib.sha256(path_bytes).hexdigest()[:16]
    except Exception:
        pdf_fingerprint = ""
    try:
        ranges_bytes = json.dumps(section_ranges, sort_keys=True).encode("utf-8")
        section_fingerprint = hashlib.sha256(ranges_bytes).hexdigest()[:16]
    except Exception:
        section_fingerprint = ""
    return {
        "pdf_path": pdf_path,
        "pdf_fingerprint": pdf_fingerprint,
        "section_ranges_fingerprint": section_fingerprint,
        "extraction_mode": extraction_mode,
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "extraction_metrics_version": EXTRACTION_METRICS_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
    }


def is_stored_manifest_compatible(
    stored_meta: dict[str, Any],
    expected_manifest: dict[str, Any],
) -> bool:
    """
    Return True if stored extraction is safe to reuse (provenance + contract).
    If stored has no manifest (legacy cache), return True for backward compatibility.
    """
    stored_manifest = stored_meta.get("extraction_manifest") if stored_meta else None
    if not stored_manifest or not isinstance(stored_manifest, dict):
        return True
    for key in ("artifact_contract_version", "extraction_metrics_version", "quality_policy_version"):
        stored_v = stored_manifest.get(key)
        expected_v = expected_manifest.get(key)
        if expected_v is not None and stored_v is not None:
            try:
                if int(stored_v) < int(expected_v):
                    logger.debug(
                        "Stored extraction incompatible: %s stored=%s expected=%s",
                        key,
                        stored_v,
                        expected_v,
                    )
                    return False
            except (TypeError, ValueError):
                return False
    if expected_manifest.get("pdf_fingerprint") and stored_manifest.get("pdf_fingerprint"):
        if stored_manifest.get("pdf_fingerprint") != expected_manifest.get("pdf_fingerprint"):
            logger.debug(
                "Stored extraction incompatible: pdf_fingerprint mismatch"
            )
            return False
    if expected_manifest.get("section_ranges_fingerprint") and stored_manifest.get("section_ranges_fingerprint"):
        if stored_manifest.get("section_ranges_fingerprint") != expected_manifest.get("section_ranges_fingerprint"):
            logger.debug(
                "Stored extraction incompatible: section_ranges_fingerprint mismatch"
            )
            return False
    return True


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


def get_extraction_artifact_paths(
    bank_code: str,
    year: int,
    quarter: str,
    base_dir: Path,
) -> dict[str, Path]:
    """Return canonical filesystem paths for one stored extraction."""
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    return {
        "dir": target_dir,
        "snapshot": target_dir / EXTRACTION_SNAPSHOT_FILENAME,
        "tables": target_dir / TABLES_FILENAME,
        "meta": target_dir / META_FILENAME,
        "report_summary_json": target_dir / REPORT_SUMMARY_JSON_FILENAME,
        "report_summary_txt": target_dir / REPORT_SUMMARY_TXT_FILENAME,
        "indicators": target_dir / REPORT_INDICATORS_FILENAME,
        "indicators_txt": target_dir / REPORT_INDICATORS_TXT_FILENAME,
        "footnotes": target_dir / REPORT_FOOTNOTES_FILENAME,
        "footnotes_txt": target_dir / REPORT_FOOTNOTES_TXT_FILENAME,
    }


def describe_extraction_artifacts(
    bank_code: str,
    year: int,
    quarter: str,
    base_dir: Path,
) -> dict[str, Any]:
    """Describe report-centric extraction artifacts for provenance and observability."""
    paths = get_extraction_artifact_paths(bank_code, year, quarter, base_dir)
    return {
        "bank_code": str(bank_code),
        "year": int(year),
        "quarter": _normalize_storage_quarter(quarter),
        "artifact_dir": str(paths["dir"]),
        "snapshot_path": str(paths["snapshot"]),
        "tables_path": str(paths["tables"]),
        "meta_path": str(paths["meta"]),
        "report_summary_json_path": str(paths["report_summary_json"]),
        "report_summary_txt_path": str(paths["report_summary_txt"]),
        "indicators_path": str(paths["indicators"]),
        "indicators_txt_path": str(paths["indicators_txt"]),
        "footnotes_path": str(paths["footnotes"]),
        "footnotes_txt_path": str(paths["footnotes_txt"]),
        "artifacts_present": {
            "snapshot": paths["snapshot"].exists(),
            "tables": paths["tables"].exists(),
            "meta": paths["meta"].exists(),
            "report_summary_json": paths["report_summary_json"].exists(),
            "report_summary_txt": paths["report_summary_txt"].exists(),
            "indicators": paths["indicators"].exists(),
            "indicators_txt": paths["indicators_txt"].exists(),
            "footnotes": paths["footnotes"].exists(),
            "footnotes_txt": paths["footnotes_txt"].exists(),
        },
    }


def _footnote_to_canonical(item: dict[str, Any]) -> dict[str, str]:
    """Normalize a footnote item to stored canonical form: ``{"id": str, "text": str}``."""
    fid = (item.get("id") or item.get("marker") or "").strip()
    text = str(item.get("text") or "").strip()
    return {"id": fid, "text": text}


def _normalize_legacy_debug_metrics(dm: dict[str, Any]) -> dict[str, Any]:
    """Copy legacy vision_primary_* keys into canonical vision_extraction_* when missing."""
    out = dict(dm)
    if out.get("vision_extraction_confidence") is None and out.get("vision_primary_confidence") is not None:
        out["vision_extraction_confidence"] = out["vision_primary_confidence"]
    if out.get("vision_extraction_applied") is None and out.get("vision_primary_applied") is not None:
        out["vision_extraction_applied"] = out["vision_primary_applied"]
    if out.get("vision_extraction_attempted") is None and out.get("vision_primary_attempted") is not None:
        out["vision_extraction_attempted"] = out["vision_primary_attempted"]
    return out


def _backfill_page_local_structure(tables: list[TableArtifact]) -> None:
    """Recompute and set page-local structure on each table (for backward compatibility)."""
    structure = derive_page_local_structure(tables)
    for t in tables:
        key = (t.table_id, t.page_pdf)
        if key not in structure:
            continue
        info = structure[key]
        t.table_index_on_page = info.get("table_index_on_page")
        t.tables_on_page = info.get("tables_on_page")
        t.bbox_top = info.get("bbox_top")
        t.page_local_role = info.get("page_local_role")


def table_artifact_from_dict(d: dict[str, Any]) -> TableArtifact:
    """Reconstruct the canonical comparison ``TableArtifact`` from stored JSON."""
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
    extraction_method = str(d.get("extraction_method") or "docling")
    content_source = infer_content_source(
        extraction_method,
        d.get("content_source"),
    )
    first_column_indicators_raw_from_storage = d.get("first_column_indicators_raw")
    raw_indicators_stored = list(d.get("first_column_indicators") or [])
    use_raw_reconstruction = (
        content_source == "vision_gpt4o"
        and first_column_indicators_raw_from_storage
        and isinstance(first_column_indicators_raw_from_storage, list)
    )
    if use_raw_reconstruction:
        raw_list = [
            str(ind or "").strip()
            for ind in first_column_indicators_raw_from_storage
            if str(ind or "").strip()
        ]
        if raw_list:
            first_column_indicators = []
            for ind in raw_list:
                fixed, _, _ = post_normalize_indicator(
                    normalize_indicator_for_comparison(ind)
                )
                if fixed and normalize_indicator_for_comparison(fixed):
                    first_column_indicators.append(fixed)
        else:
            first_column_indicators = [
                n
                for ind in raw_indicators_stored
                if (n := normalize_indicator_for_comparison(str(ind or "").strip()))
            ]
    elif first_column_indicators_raw_from_storage and isinstance(
        first_column_indicators_raw_from_storage, list
    ):
        first_column_indicators = [
            n
            for ind in first_column_indicators_raw_from_storage
            if (n := normalize_indicator_for_comparison(str(ind or "").strip()))
        ]
    else:
        first_column_indicators = [
            n
            for ind in raw_indicators_stored
            if (n := normalize_indicator_for_comparison(str(ind or "").strip()))
        ]
    title_clean = d.get("title_clean")
    title_raw = d.get("title_raw")
    table_number = d.get("table_number")
    bbox = d.get("bbox")
    quarter = d.get("quarter")
    pdf_path = d.get("pdf_path")
    first_column_indicators_raw = d.get("first_column_indicators_raw")
    first_column_groups = d.get("first_column_groups")
    hierarchical_indicator_signature = d.get("hierarchical_indicator_signature")
    title_reliability = d.get("title_reliability")
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
    if debug_metrics:
        debug_metrics = _normalize_legacy_debug_metrics(debug_metrics)
    table_index_on_page = d.get("table_index_on_page")
    if table_index_on_page is not None and not isinstance(table_index_on_page, int):
        table_index_on_page = None
    tables_on_page = d.get("tables_on_page")
    if tables_on_page is not None and not isinstance(tables_on_page, int):
        tables_on_page = None
    bbox_top = d.get("bbox_top")
    if bbox_top is not None:
        try:
            bbox_top = float(bbox_top)
        except (TypeError, ValueError):
            bbox_top = None
    page_local_role = d.get("page_local_role")
    if page_local_role is not None and not isinstance(page_local_role, str):
        page_local_role = None
    # comparison_eligible and comparison_blockers are recomputed by
    # TableArtifact.__post_init__ from current state — do not pass
    # stale values from stored JSON.

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
        table_index_on_page=table_index_on_page,
        tables_on_page=tables_on_page,
        bbox_top=bbox_top,
        page_local_role=page_local_role,
        quarter=quarter,
        pdf_path=pdf_path,
        first_column_indicators_raw=first_column_indicators_raw,
        first_column_groups=first_column_groups,
        hierarchical_indicator_signature=hierarchical_indicator_signature,
        title_reliability=title_reliability,
        footnotes=footnotes,
        fragmentation_detected=fragmentation_detected,
        debug_metrics=debug_metrics,
        content_source=content_source,
    )


# Atomic snapshot file so manifest and payload cannot drift. Future layout may use
# fingerprint-based paths (e.g. .../quarter/{pdf_fp}_{section_fp}/) with a "latest" pointer.
EXTRACTION_SNAPSHOT_FILENAME = "extraction_snapshot.json"


def _ensure_report_view_artifacts(
    *,
    bank_code: str,
    year: int,
    quarter: str,
    target_dir: Path,
    tables: list[TableArtifact],
    meta: dict[str, Any],
) -> None:
    """Ensure all report-centric JSON/TXT artifacts exist for this extraction."""
    report_summary_json_path = target_dir / REPORT_SUMMARY_JSON_FILENAME
    report_summary_txt_path = target_dir / REPORT_SUMMARY_TXT_FILENAME
    indicators_path = target_dir / REPORT_INDICATORS_FILENAME
    indicators_txt_path = target_dir / REPORT_INDICATORS_TXT_FILENAME
    footnotes_path = target_dir / REPORT_FOOTNOTES_FILENAME
    footnotes_txt_path = target_dir / REPORT_FOOTNOTES_TXT_FILENAME
    report_summary_json_missing = not report_summary_json_path.exists()
    report_summary_txt_missing = not report_summary_txt_path.exists()
    indicators_missing = not indicators_path.exists()
    indicators_txt_missing = not indicators_txt_path.exists()
    footnotes_missing = not footnotes_path.exists()
    footnotes_txt_missing = not footnotes_txt_path.exists()
    if (
        not report_summary_json_missing
        and not report_summary_txt_missing
        and not indicators_missing
        and not indicators_txt_missing
        and not footnotes_missing
        and not footnotes_txt_missing
    ):
        return
    from vigilance.extraction.vision_extraction_writer import (
        write_report_footnotes_txt,
        write_report_footnotes_json,
        write_report_indicators_txt,
        write_report_indicators_json,
        write_report_summary_json,
        write_report_summary_txt,
    )

    if report_summary_json_missing:
        write_report_summary_json(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
        )
    if report_summary_txt_missing:
        write_report_summary_txt(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
        )
    if indicators_missing:
        write_report_indicators_json(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
        )
    if indicators_txt_missing:
        write_report_indicators_txt(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
        )
    if footnotes_missing:
        write_report_footnotes_json(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
        )
    if footnotes_txt_missing:
        write_report_footnotes_txt(
            tables,
            target_dir,
            bank_code,
            year,
            quarter,
            meta=meta,
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

    Writes extraction_snapshot.json atomically (tables + meta in one file so they cannot drift),
    then tables.json and meta.json for backward compatibility.
    Layout: outputs/extractions/{bank_code}/{year}/{quarter}/.
    Returns the directory path.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    target_dir.mkdir(parents=True, exist_ok=True)

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

    meta_payload = dict(meta or {})
    meta_payload.setdefault("schema_version", STORAGE_SCHEMA_VERSION)
    meta_payload.setdefault("bank_code", bank_code)
    meta_payload.setdefault("year", year)
    meta_payload.setdefault("quarter", quarter_norm)
    meta_payload.setdefault("extracted_at", datetime.now(timezone.utc).isoformat())
    meta_payload.setdefault("created_at", meta_payload.get("extracted_at"))
    meta_payload.setdefault("pipeline_version", "")
    meta_payload.setdefault("model_version", "")
    meta_payload.setdefault("prompt_version", "")
    try:
        manifest = build_extraction_manifest(
            pdf_path=str(meta_payload.get("pdf_path") or ""),
            section_ranges=list(meta_payload.get("section_ranges") or []),
            extraction_mode=str(meta_payload.get("extraction_method") or "vision_full_gpt4o"),
        )
        meta_payload["extraction_manifest"] = manifest
        meta_payload.setdefault("pdf_fingerprint", manifest.get("pdf_fingerprint", ""))
    except Exception:
        pass

    snapshot = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "bank_code": bank_code,
        "year": year,
        "quarter": quarter_norm,
        "tables": tables_payload["tables"],
        "meta": meta_payload,
    }
    snapshot_path = target_dir / EXTRACTION_SNAPSHOT_FILENAME
    tmp_path = target_dir / (EXTRACTION_SNAPSHOT_FILENAME + ".tmp")
    tmp_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(snapshot_path)

    tables_path = target_dir / TABLES_FILENAME
    tables_path.write_text(
        json.dumps(tables_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = target_dir / META_FILENAME
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _ensure_report_view_artifacts(
        bank_code=bank_code,
        year=year,
        quarter=quarter_norm,
        target_dir=target_dir,
        tables=tables,
        meta=meta_payload,
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

    Prefers extraction_snapshot.json (atomic tables+meta) when present; otherwise
    reads tables.json and meta.json. Returns (tables, meta) or None if not found or invalid.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    paths = get_extraction_artifact_paths(bank_code, year, quarter_norm, base_dir)
    target_dir = paths["dir"]
    snapshot_path = paths["snapshot"]
    tables_path = paths["tables"]
    meta_path = paths["meta"]

    if snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            tables_data = snapshot.get("tables", [])
            meta = snapshot.get("meta") or {}
            if not isinstance(tables_data, list):
                return None
            stored_version = snapshot.get("schema_version")
            if stored_version is not None:
                try:
                    v = int(stored_version)
                    if v < STORAGE_SCHEMA_VERSION:
                        logger.debug(
                            "load_extraction stale schema_version bank=%s year=%s quarter=%s stored=%s current=%s",
                            bank_code,
                            year,
                            quarter_norm,
                            v,
                            STORAGE_SCHEMA_VERSION,
                        )
                        return None
                except (TypeError, ValueError):
                    pass
            tables = [table_artifact_from_dict(t) for t in tables_data]
            if not tables:
                logger.debug(
                    "load_extraction empty tables bank=%s year=%s quarter=%s",
                    bank_code,
                    year,
                    quarter_norm,
                )
                return None
            _backfill_page_local_structure(tables)
            _ensure_report_view_artifacts(
                bank_code=bank_code,
                year=year,
                quarter=quarter_norm,
                target_dir=target_dir,
                tables=tables,
                meta=meta,
            )
            return (tables, meta)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug("Load extraction snapshot failed %s: %s", target_dir, e)

    if not tables_path.exists():
        return None
    try:
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        stored_version = payload.get("schema_version")
        if stored_version is not None:
            try:
                v = int(stored_version)
                if v < STORAGE_SCHEMA_VERSION:
                    logger.debug(
                        "load_extraction stale schema_version bank=%s year=%s quarter=%s stored=%s current=%s",
                        bank_code,
                        year,
                        quarter_norm,
                        v,
                        STORAGE_SCHEMA_VERSION,
                    )
                    return None
            except (TypeError, ValueError):
                pass
        tables_data = payload.get("tables", [])
        if not isinstance(tables_data, list):
            return None
        tables = [table_artifact_from_dict(t) for t in tables_data]
        if not tables:
            logger.debug(
                "load_extraction empty tables bank=%s year=%s quarter=%s",
                bank_code,
                year,
                quarter_norm,
            )
            return None
        _backfill_page_local_structure(tables)
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _ensure_report_view_artifacts(
            bank_code=bank_code,
            year=year,
            quarter=quarter_norm,
            target_dir=target_dir,
            tables=tables,
            meta=meta,
        )
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
