"""Run current-vs-previous quarter comparison from uploaded PDFs and section ranges."""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None  # type: ignore[assignment]

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]

# Fallback defaults when config keys absent (PASS 2 wires from get_matching_thresholds)
_INDICATOR_DEFAULTS = {
    "indicator_rename_min_score": 0.86,
    "indicator_gate_min_len_ratio": 0.55,
    "indicator_gate_min_token_overlap": 1,
}

from app.comparison_canonical import (
    compute_changed_tables_t1,
    compute_changed_tables_t2,
)
from app.quarter_utils import build_quarter_context
from app.ui_config import INDICATOR_COMPARISON_DIR, LOGS_DIR

_SEMANTIC_JUDGE_LOG = LOGS_DIR / "semantic_judge_decisions.jsonl"
_VALIDATION_LOG = LOGS_DIR / "validation.jsonl"
from vigilance.compare import run_strict_intra_section_compare
from vigilance.compare.footnote_comparator import FootnoteComparator
from vigilance.compare.table_fragment_merger import merge_table_fragments
from vigilance.config import (
    get_matching_thresholds,
    get_quality_gate_config,
    get_validation_config,
)
from vigilance.models.table_models import (
    EXTRACTION_STATUS_BLOCKED,
    EXTRACTION_STATUS_CERTIFIED,
    EXTRACTION_STATUS_REVIEW_REQUIRED,
    VISION_CONTENT_SOURCE,
    TableArtifact,
    derive_extraction_blockers,
    get_canonical_footnotes,
    get_comparison_indicators,
    get_extraction_confidence,
    get_extraction_quality_flags,
    get_extraction_status,
    get_vision_raw_indicators,
    infer_content_source,
)
from vigilance.utils.footnotes_utils import (
    footnotes_list_to_dict,
    normalize_footnotes_to_canonical,
)
from vigilance.utils.indicator_cleaner import (
    dedupe_indicators,
    merge_line_split_indicators,
    normalize_indicator_for_comparison,
    post_normalize_indicator,
)
from vigilance.utils.indicator_normalizer import (
    get_canonical_text,
    get_token_sorted_text,
    strip_footnote_markers_from_indicator,
)
from vigilance.utils.matching_normalizer import _classify_excluded_line
from vigilance.utils.rbc_table_signals import (
    build_rbc_first_column_signals,
    classify_rbc_title_reliability,
    is_rbc_bank,
)

_MATCH_DECISIONS_LOG = LOGS_DIR / "match_decisions.jsonl"
_INDICATOR_DIFF_DEBUG_LOG = LOGS_DIR / "indicator_diff_debug.jsonl"


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return None


def _resolve_vision_extraction_enabled(
    bank_code: str,
    explicit: bool | None,
    *,
    allow_env_legacy: bool = True,
) -> bool:
    """Resolution order: explicit > env > bank config."""
    if explicit is not None:
        return bool(explicit)
    if allow_env_legacy:
        env_choice = _env_bool("VIGILANCE_VISION_EXTRACTION_ENABLED")
        if env_choice is not None:
            return env_choice
    try:
        from vigilance.config import get_vision_extraction_config

        cfg = get_vision_extraction_config(bank_code=bank_code) or {}
        if "enabled" in cfg:
            return bool(cfg.get("enabled"))
    except Exception:
        pass
    return False


def _write_match_decision_log(record: dict[str, Any]) -> None:
    """Append a structured audit record to the JSONL match-decision log."""
    try:
        with open(_MATCH_DECISIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("match_decision log write failed: %s", exc)


def _write_validation_log(record: dict[str, Any], *, run_id: str | None = None) -> None:
    """Append a structured validation record to the JSONL validation log."""
    try:
        payload = dict(record or {})
        if run_id and not payload.get("run_id"):
            payload["run_id"] = run_id
        with open(_VALIDATION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("validation log write failed: %s", exc)


def _compare_table_footnotes(t1: TableArtifact, t2: TableArtifact) -> dict[str, Any]:
    """Compare footnotes between two matched TableArtifacts.

    Returns a dict with keys: added, removed, modified, counts.
    """
    dict1 = footnotes_list_to_dict(get_canonical_footnotes(t1))
    dict2 = footnotes_list_to_dict(get_canonical_footnotes(t2))
    if not dict1 and not dict2:
        return {
            "added": [],
            "removed": [],
            "modified": [],
            "counts": {"added": 0, "removed": 0, "modified": 0},
        }

    comparator = FootnoteComparator()
    table_id = t1.table_id or t2.table_id
    changes = comparator.compare_footnotes(dict1, dict2, table_id)

    added = [c.to_dict() for c in changes if c.change_type == "new_footnote"]
    removed = [c.to_dict() for c in changes if c.change_type == "removed_footnote"]
    modified = [c.to_dict() for c in changes if c.change_type == "modified_footnote"]
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "counts": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
        },
    }


def _section_uid(table: TableArtifact) -> str:
    return f"{table.section}|{table.table_id}|p{table.page_pdf}"


def _canonical_section_name(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return "unknown_section"
    try:
        from vigilance.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(value)
    except Exception:
        lowered = value.lower().replace("é", "e").replace("è", "e")
        if "capital" in lowered or "fonds propres" in lowered:
            return "gestion_capital"
        if "risque" in lowered:
            return "gestion_risques"
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        return lowered or "unknown_section"


def _normalize_ranges(sections: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in sections or []:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_page", 0) or 0)
        end = int(item.get("end_page", start) or start)
        if start <= 0:
            continue
        if end < start:
            end = start
        section_raw = str(
            item.get("type") or item.get("section") or item.get("label") or ""
        )
        section = _canonical_section_name(section_raw)
        result.append({"section": section, "start": start, "end": end})
    result.sort(key=lambda row: (int(row["start"]), str(row["section"])))
    return result


def _infer_year(*paths: str) -> int:
    for value in paths:
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            return int(match.group(0))
    return datetime.now().year


def _table_to_artifact(
    table: Any,
    *,
    bank_code: str,
    quarter: str,
    pdf_path: str,
    table_index_on_page: int | None = None,
    tables_on_page: int | None = None,
    bbox_top: float | None = None,
    page_local_role: str | None = None,
) -> TableArtifact:
    rows = [list(row) for row in (getattr(table, "rows", []) or [])]
    headers = [str(h) for h in (getattr(table, "headers", []) or []) if h is not None]
    extraction_method = getattr(table, "extraction_method", None) or "docling"
    content_source = infer_content_source(
        extraction_method,
        getattr(table, "content_source", None),
    )
    vision_raw_source = getattr(table, "first_column_indicators_raw", None)
    vision_raw_indicators = (
        [str(x).strip() for x in vision_raw_source if str(x).strip()]
        if vision_raw_source is not None
        else []
    )
    if content_source != VISION_CONTENT_SOURCE:
        vision_raw_indicators = []

    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    if is_rbc_bank(bank_code) and vision_raw_indicators:
        rbc_signals = build_rbc_first_column_signals(
            rows=rows,
            raw_indicators=vision_raw_indicators,
        )
        first_column_groups = list(rbc_signals.groups_raw)
        hierarchical_indicator_signature = list(
            rbc_signals.hierarchical_indicator_signature
        )

    # Quality pass 1: line-split merge (deterministic)
    vision_raw_indicators, line_merge_count = merge_line_split_indicators(
        vision_raw_indicators
    )
    if line_merge_count > 0:
        logger.info(
            "indicators_line_merge: table=%s page=%s merges=%d",
            getattr(table, "table_id", ""),
            getattr(table, "page_number", 0),
            line_merge_count,
        )

    # Quality pass 2: dedupe (if duplicate_ratio >= 0.15)
    vision_raw_indicators, duplicate_ratio, dup_removed = dedupe_indicators(
        vision_raw_indicators
    )
    if dup_removed > 0:
        logger.info(
            "indicators_dedupe: table=%s page=%s removed=%d duplicate_ratio=%.3f",
            getattr(table, "table_id", ""),
            getattr(table, "page_number", 0),
            dup_removed,
            duplicate_ratio,
        )

    fragmentation_from_post_norm = False
    comparison_normalized_indicators: list[str] = []
    for ind in vision_raw_indicators:
        fixed, camel_hit, tag_hit = post_normalize_indicator(
            normalize_indicator_for_comparison(ind)
        )
        if camel_hit or tag_hit:
            fragmentation_from_post_norm = True
        comparison_normalized_indicators.append(fixed)

    footnotes_raw = getattr(table, "footnotes", None)
    canonical_footnotes = None
    # comparison_blockers are recomputed by TableArtifact.__post_init__
    # from content_source, first_column_indicators_raw, and footnotes.

    if footnotes_raw is None:
        canonical_footnotes = None
    else:
        canonical_footnotes = normalize_footnotes_to_canonical(footnotes_raw)

    # Part C: propagate fragmentation flag (merge or post-norm corrections)
    frag_from_extraction = bool(getattr(table, "fragmentation_detected", False))
    fragmentation_detected = frag_from_extraction or fragmentation_from_post_norm

    section = _canonical_section_name(str(getattr(table, "section", "")))
    dm_raw = getattr(table, "debug_metrics", None)
    debug_metrics = dict(dm_raw) if isinstance(dm_raw, dict) else None

    title_raw = getattr(table, "title_raw", None) or getattr(table, "title", None)
    title_clean = getattr(table, "title_clean", None)
    title_display = title_clean or getattr(table, "title", None)
    title_reliability = getattr(
        table, "title_reliability", None
    ) or classify_rbc_title_reliability(
        title_display or title_raw,
        bank_code=bank_code,
    )

    return TableArtifact(
        bank_code=bank_code,
        section=section,
        page_pdf=int(getattr(table, "page_number", 0) or 0),
        table_id=str(getattr(table, "table_id", "")),
        title=title_display,
        headers=headers,
        title_clean=title_clean,
        title_raw=title_raw,
        rows=rows,
        first_column_indicators=comparison_normalized_indicators,
        first_column_indicators_raw=vision_raw_indicators,
        extraction_method=extraction_method,
        table_number=getattr(table, "table_number", None),
        bbox=getattr(table, "bbox", None),
        table_index_on_page=table_index_on_page,
        tables_on_page=tables_on_page,
        bbox_top=bbox_top,
        page_local_role=page_local_role,
        quarter=quarter,
        pdf_path=pdf_path,
        first_column_groups=first_column_groups,
        hierarchical_indicator_signature=hierarchical_indicator_signature,
        title_reliability=title_reliability,
        footnotes=canonical_footnotes,
        fragmentation_detected=fragmentation_detected,
        debug_metrics=debug_metrics,
        content_source=content_source,
    )


def _extract_tables(
    *,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    section_ranges: list[dict[str, Any]],
    api_key: str | None,
    use_vision_extraction: bool | None = None,
    use_stored_extraction_if_available: bool = False,
    extraction_base_dir: str | None = None,
    return_provenance: bool = False,
) -> list[TableArtifact] | tuple[list[TableArtifact], dict[str, Any]]:
    from pathlib import Path as _Path

    from vigilance.extraction.docling_processor import (
        extract_tables_docling_by_sections,
    )

    base_dir = _Path(extraction_base_dir or "outputs/extractions")

    def _build_extraction_source(
        mode: str,
        *,
        selected_reason: str = "",
    ) -> dict[str, Any]:
        from app.extraction_storage import describe_extraction_artifacts

        info = describe_extraction_artifacts(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            base_dir=base_dir,
        )
        info["mode"] = mode
        if selected_reason:
            info["selected_reason"] = selected_reason
        return info

    def _return(
        tables: list[TableArtifact],
        provenance: dict[str, Any],
    ) -> list[TableArtifact] | tuple[list[TableArtifact], dict[str, Any]]:
        if return_provenance:
            return (tables, provenance)
        return tables

    def _extraction_snapshot(tables: list[TableArtifact]) -> dict[str, int]:
        comparable = sum(
            1 for t in tables if bool(getattr(t, "comparison_eligible", False))
        )
        tables_with_raw = sum(1 for t in tables if get_vision_raw_indicators(t))
        raw_indicators = sum(len(get_vision_raw_indicators(t)) for t in tables)
        return {
            "tables": len(tables),
            "comparable": comparable,
            "tables_with_raw": tables_with_raw,
            "raw_indicators": raw_indicators,
        }

    def _prefer_stored_over_fresh(
        fresh_tables: list[TableArtifact],
        stored_tables: list[TableArtifact],
    ) -> bool:
        fresh = _extraction_snapshot(fresh_tables)
        stored = _extraction_snapshot(stored_tables)
        if fresh["tables"] == 0:
            return bool(stored["tables"] > 0)
        if fresh["tables_with_raw"] == 0 and stored["tables_with_raw"] > 0:
            return True
        if fresh["comparable"] == 0 and stored["comparable"] > 0:
            return True
        if fresh["raw_indicators"] == 0 and stored["raw_indicators"] > 0:
            return True
        return False

    # Step 4: Load stored extraction if available (avoids re-running Vision)
    if use_stored_extraction_if_available:
        try:
            from app.extraction_storage import (
                build_extraction_manifest,
                is_stored_manifest_compatible,
                load_extraction,
            )

            stored = load_extraction(bank_code, year, quarter, base_dir)
            if stored is not None:
                stored_tables, stored_meta = stored
                if not stored_tables:
                    logger.info(
                        "extraction_stored_ignored_empty bank=%s year=%s quarter=%s",
                        bank_code,
                        year,
                        quarter,
                    )
                else:
                    expected_manifest = build_extraction_manifest(
                        pdf_path=pdf_path,
                        section_ranges=section_ranges,
                        extraction_mode="vision_full_gpt4o",
                    )
                    if not is_stored_manifest_compatible(stored_meta, expected_manifest):
                        logger.info(
                            "extraction_stored_rejected_stale_cache bank=%s year=%s quarter=%s",
                            bank_code,
                            year,
                            quarter,
                        )
                    else:
                        logger.info(
                            "Loaded stored extraction: %s/%s/%s (%d tables)",
                            bank_code,
                            year,
                            quarter,
                            len(stored_tables),
                        )
                        logger.info(
                            "extraction_quarter_tables count=%d bank=%s year=%s quarter=%s source=stored",
                            len(stored_tables),
                            bank_code,
                            year,
                            quarter,
                        )
                        return _return(
                            stored_tables,
                            _build_extraction_source(
                                "stored",
                                selected_reason="compatible_cache",
                            ),
                        )
        except Exception as e:
            logger.debug("Could not load stored extraction: %s", e)

    use_vision_extraction = _resolve_vision_extraction_enabled(
        bank_code,
        use_vision_extraction,
        allow_env_legacy=True,
    )
    del api_key

    raw_tables = extract_tables_docling_by_sections(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        section_ranges=section_ranges,
        use_vision_extraction=use_vision_extraction,
    )

    from vigilance.utils.table_page_structure import derive_page_local_structure

    page_structure = derive_page_local_structure(raw_tables)

    def _page_for_raw(t: Any) -> int:
        return int(getattr(t, "page_number", 0) or getattr(t, "page_pdf", 0) or 0)

    artifacts = [
        _table_to_artifact(
            table,
            bank_code=bank_code,
            quarter=quarter,
            pdf_path=pdf_path,
            table_index_on_page=page_structure.get((getattr(table, "table_id", ""), _page_for_raw(table)), {}).get("table_index_on_page"),
            tables_on_page=page_structure.get((getattr(table, "table_id", ""), _page_for_raw(table)), {}).get("tables_on_page"),
            bbox_top=page_structure.get((getattr(table, "table_id", ""), _page_for_raw(table)), {}).get("bbox_top"),
            page_local_role=page_structure.get((getattr(table, "table_id", ""), _page_for_raw(table)), {}).get("page_local_role"),
        )
        for table in raw_tables
    ]

    # Step 4: Save extraction systematically after every run (do not persist empty)
    try:
        from app.extraction_storage import (
            build_extraction_manifest,
            is_stored_manifest_compatible,
            load_extraction,
            save_extraction,
        )

        stored = load_extraction(bank_code, year, quarter, base_dir)
        if stored is not None:
            stored_tables, stored_meta = stored
            expected_manifest = build_extraction_manifest(
                pdf_path=pdf_path,
                section_ranges=section_ranges,
                extraction_mode="vision_full_gpt4o",
            )
            if is_stored_manifest_compatible(stored_meta, expected_manifest) and _prefer_stored_over_fresh(
                artifacts, stored_tables
            ):
                logger.warning(
                    "Fresh extraction degraded for %s/%s/%s; reusing stored extraction "
                    "(fresh comparable=%d raw_tables=%d raw_indicators=%d, stored comparable=%d raw_tables=%d raw_indicators=%d)",
                    bank_code,
                    year,
                    quarter,
                    _extraction_snapshot(artifacts)["comparable"],
                    _extraction_snapshot(artifacts)["tables_with_raw"],
                    _extraction_snapshot(artifacts)["raw_indicators"],
                    _extraction_snapshot(stored_tables)["comparable"],
                    _extraction_snapshot(stored_tables)["tables_with_raw"],
                    _extraction_snapshot(stored_tables)["raw_indicators"],
                )
                return _return(
                    stored_tables,
                    _build_extraction_source(
                        "stored",
                        selected_reason="fresh_degraded_reused_stored",
                    ),
                )

        if not artifacts:
            logger.warning(
                "extraction_save_skipped_empty bank=%s year=%s quarter=%s tables_count=0",
                bank_code,
                year,
                quarter,
            )
        else:
            save_extraction(
                bank_code=bank_code,
                year=year,
                quarter=quarter,
                tables=artifacts,
                meta={
                    "pdf_path": pdf_path,
                    "use_vision_extraction": use_vision_extraction,
                    "extraction_method": "vision_full_gpt4o",
                    "section_ranges": section_ranges,
                    "schema_version": 2,
                },
                base_dir=base_dir,
            )
    except Exception as e:
        logger.warning("Could not save extraction: %s", e)

    logger.info(
        "extraction_quarter_tables count=%d bank=%s year=%s quarter=%s source=fresh",
        len(artifacts),
        bank_code,
        year,
        quarter,
    )
    return _return(
        artifacts,
        _build_extraction_source("fresh", selected_reason="fresh_extraction"),
    )


def _canonical_indicator_key(text: str) -> str:
    """Canonical key for indicator comparison (shared with structural_comparator)."""
    return normalize_indicator_for_comparison(text)


# --- Part D: Pre-Diff Safety Check ---

_SUSPICIOUS_OVERLAP_THRESHOLD = 0.15
_SUSPICIOUS_SIZE_DIFF_RATIO = 0.60


def _compute_indicator_overlap(t1: TableArtifact, t2: TableArtifact) -> float:
    """Jaccard overlap of canonical indicator keys between two tables."""

    def _keys(t: TableArtifact) -> set[str]:
        result: set[str] = set()
        for ind in get_comparison_indicators(t):
            s = str(ind).strip()
            if not s:
                continue
            key = _canonical_indicator_key(s)
            if key:
                result.add(key)
        return result

    a, b = _keys(t1), _keys(t2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_quality_flags(table: TableArtifact) -> list[str]:
    """Build quality flag strings from canonical table_models for comparison output."""
    flags_map = get_extraction_quality_flags(table)
    flags: list[str] = []
    if flags_map.get("crop_rejected"):
        flags.append("crop_rejected")
    if flags_map.get("recrop_failed_incomplete"):
        flags.append("recrop_failed_incomplete")
    if flags_map.get("recrop_used"):
        flags.append("recrop_used")
    if flags_map.get("recrop_attempted"):
        flags.append("recrop_attempted")
    if not flags_map.get("vision_extraction_applied", True):
        flags.append("vision_extraction_not_applied")
    if flags_map.get("appears_truncated"):
        flags.append("appears_truncated")
    if table.fragmentation_detected:
        flags.append("fragmentation")
    dm = table.debug_metrics or {}
    dup = dm.get("duplicate_ratio", 0)
    if isinstance(dup, (int, float)) and dup > 0.20:
        flags.append(f"high_duplicate_ratio({dup:.2f})")
    hlr = dm.get("header_like_ratio", 0)
    if isinstance(hlr, (int, float)) and hlr > 0.20:
        flags.append(f"high_header_like({hlr:.2f})")
    return flags


def _extraction_confidence(table: TableArtifact) -> str:
    """Return extraction confidence level from canonical table_models: high, medium, low, or unknown."""
    conf = get_extraction_confidence(table)
    if conf >= 0.75:
        return "high"
    if conf >= 0.5:
        return "medium"
    if conf > 0.0:
        return "low"
    return "unknown"


def _compute_extraction_kpis(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    comparisons: list[dict[str, Any]],
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate extraction reliability KPIs and certification summary for the comparison output."""
    all_tables = list(tables_t1) + list(tables_t2)
    total = len(all_tables) or 1

    vision_attempted = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_fallback_attempted")
    )
    vision_applied = sum(
        1 for t in all_tables if (t.debug_metrics or {}).get("vision_fallback_applied")
    )
    vision_extraction_attempted = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_extraction_attempted")
    )
    vision_extraction_applied = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_extraction_applied")
    )
    vision_schema_contract_fail_count = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_schema_contract_failed")
    )
    vision_extraction_disabled_reason = ""
    for t in all_tables:
        reason = (t.debug_metrics or {}).get("vision_extraction_disabled_reason")
        if isinstance(reason, str) and reason.strip():
            vision_extraction_disabled_reason = reason.strip()
            break

    disagree_count = 0
    for t in all_tables:
        arb = (t.debug_metrics or {}).get("vision_arbitration")
        if isinstance(arb, dict):
            agreement = arb.get("agreement_signals", {}).get("agreement", "")
            if agreement in ("disagree", "strong_disagree"):
                disagree_count += 1

    matched_with_changes = sum(
        1
        for c in comparisons
        if c.get("added_indicators") or c.get("removed_indicators")
    )
    matched_total = len(comparisons) or 1
    noise_rate = matched_with_changes / matched_total

    renamed_total = sum(len(c.get("renamed_indicators", [])) for c in comparisons)
    add_remove_total = sum(
        len(c.get("added_indicators", [])) for c in comparisons
    ) + sum(len(c.get("removed_indicators", [])) for c in comparisons)
    rename_conversion = (
        renamed_total / (renamed_total + add_remove_total)
        if (renamed_total + add_remove_total) > 0
        else 0.0
    )

    tables_certified = sum(1 for t in all_tables if get_extraction_status(t) == EXTRACTION_STATUS_CERTIFIED)
    tables_review_required = sum(
        1 for t in all_tables if get_extraction_status(t) == EXTRACTION_STATUS_REVIEW_REQUIRED
    )
    tables_blocked = sum(1 for t in all_tables if get_extraction_status(t) == EXTRACTION_STATUS_BLOCKED)
    tables_crop_rejected = sum(
        1 for t in all_tables if (get_extraction_quality_flags(t).get("crop_rejected"))
    )
    tables_low_confidence = sum(
        1 for t in all_tables if get_extraction_confidence(t) < 0.5
    )
    tables_vision_not_applied = sum(
        1 for t in all_tables if not get_extraction_quality_flags(t).get("vision_extraction_applied", True)
    )
    tables_budget_exhausted = sum(
        1 for t in all_tables if (t.debug_metrics or {}).get("vision_budget_exhausted")
    )

    return {
        "vision_attempt_rate": round(vision_attempted / total, 3),
        "vision_applied_rate": round(vision_applied / total, 3),
        "docling_vision_disagreement_rate": round(disagree_count / total, 3),
        "added_removed_noise_rate_on_matched_tables": round(noise_rate, 3),
        "rename_conversion_rate": round(rename_conversion, 3),
        "tables_total": len(all_tables),
        "vision_attempted_count": vision_attempted,
        "vision_applied_count": vision_applied,
        "vision_extraction_attempted_count": vision_extraction_attempted,
        "vision_extraction_applied_count": vision_extraction_applied,
        "vision_schema_contract_fail_count": vision_schema_contract_fail_count,
        "vision_extraction_disabled_reason": vision_extraction_disabled_reason or None,
        "disagreement_count": disagree_count,
        "incertain_count": sum(
            1 for c in comparisons if c.get("table_status") == "incertain"
        ),
        "tables_certified": tables_certified,
        "tables_review_required": tables_review_required,
        "tables_blocked": tables_blocked,
        "tables_crop_rejected": tables_crop_rejected,
        "tables_low_confidence": tables_low_confidence,
        "tables_vision_not_applied": tables_vision_not_applied,
        "tables_budget_exhausted": tables_budget_exhausted,
    }


def _pre_diff_safety_check(
    t1: TableArtifact, t2: TableArtifact, indicator_overlap: float
) -> tuple[bool, str]:
    """Flag suspicious pairs before calling _indicator_diff.

    Returns (suspicious_low_overlap, reason_string).
    Does NOT block matching -- only exposes the signal.
    """
    n1 = len(get_comparison_indicators(t1))
    n2 = len(get_comparison_indicators(t2))
    size_diff_ratio = abs(n1 - n2) / max(n1, n2, 1)

    if (
        indicator_overlap < _SUSPICIOUS_OVERLAP_THRESHOLD
        and size_diff_ratio > _SUSPICIOUS_SIZE_DIFF_RATIO
    ):
        reason = (
            f"indicator_overlap={indicator_overlap:.3f} < {_SUSPICIOUS_OVERLAP_THRESHOLD}, "
            f"size_diff_ratio={size_diff_ratio:.3f} > {_SUSPICIOUS_SIZE_DIFF_RATIO}"
        )
        logger.warning(
            "suspicious_low_overlap: t1=%s t2=%s %s",
            t1.table_id,
            t2.table_id,
            reason,
        )
        return True, reason
    return False, ""


def _build_unmatched_candidate_maps(
    strict: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    row_map: dict[str, list[dict[str, Any]]] = {}
    for item in strict.get("debug_unmatched_candidates", []) or []:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("t1_uid", "")).strip()
        if uid:
            row_map[uid] = list(item.get("candidates", []) or [])

    col_map: dict[str, list[dict[str, Any]]] = {}
    for item in strict.get("debug_unmatched_candidates_t2", []) or []:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("t2_uid", "")).strip()
        if uid:
            col_map[uid] = list(item.get("candidates", []) or [])
    return row_map, col_map


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_pair_priority(entry: dict[str, Any]) -> tuple[int, float]:
    source = str(entry.get("source", "") or "")
    if source == "suspicious_pair":
        source_rank = 0
    elif source == "ambiguous_unmatched":
        source_rank = 1
    elif source == "confirmed_unmatched":
        source_rank = 2
    else:
        source_rank = 3
    return (
        source_rank,
        -(
            _safe_float(entry.get("score"))
            + 0.20 * _safe_float(entry.get("coverage_min"))
            + 0.10 * _safe_float(entry.get("distinctive_overlap_score"))
        ),
    )


def _iter_unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _table_rescue_indicator_sequence(table: TableArtifact) -> list[str]:
    indicators = _all_indicators_value_clean_ordered(table)
    out: list[str] = []
    seen: set[str] = set()
    for value in indicators:
        normalized = normalize_indicator_for_comparison(str(value or "").strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _table_rescue_title_text(table: TableArtifact) -> str:
    return normalize_indicator_for_comparison(str(getattr(table, "title", "") or ""))


def _table_rescue_headers_text(table: TableArtifact) -> str:
    headers = [
        str(h).strip()
        for h in (getattr(table, "headers", None) or [])
        if str(h).strip()
    ]
    return normalize_indicator_for_comparison(" | ".join(headers[:6]))


def _table_rescue_fingerprint(table: TableArtifact) -> str:
    seq = _table_rescue_indicator_sequence(table)
    if not seq:
        content = ""
    else:
        n = len(seq)
        top = seq[:3]
        mid_start = max(0, (n - 3) // 2)
        middle = seq[mid_start : mid_start + 3]
        tail = seq[-3:]
        content = " | ".join(_iter_unique_preserve_order(top + middle + tail))
    title = _table_rescue_title_text(table)
    headers = _table_rescue_headers_text(table)
    return f"{title} || {headers} || {content}".strip()


def _sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _indicator_jaccard_from_tables(
    table_a: TableArtifact, table_b: TableArtifact
) -> float:
    seq_a = set(_table_rescue_indicator_sequence(table_a))
    seq_b = set(_table_rescue_indicator_sequence(table_b))
    if not seq_a or not seq_b:
        return 0.0
    return len(seq_a & seq_b) / max(len(seq_a | seq_b), 1)


def _table_rescue_row_count_ratio(
    table_a: TableArtifact, table_b: TableArtifact
) -> float:
    rows_a = len(getattr(table_a, "rows", None) or [])
    rows_b = len(getattr(table_b, "rows", None) or [])
    if rows_a <= 0 or rows_b <= 0:
        return 1.0
    return max(rows_a, rows_b) / max(min(rows_a, rows_b), 1)


def _table_rescue_page_proximity(
    table_a: TableArtifact, table_b: TableArtifact
) -> float:
    page_a = _safe_int(getattr(table_a, "page_pdf", 0), 0)
    page_b = _safe_int(getattr(table_b, "page_pdf", 0), 0)
    if page_a <= 0 or page_b <= 0:
        return 0.0
    diff = abs(page_a - page_b)
    return max(0.0, 1.0 - (min(diff, 12) / 12.0))


def _candidate_gap(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    return _safe_float(candidates[0].get("score")) - _safe_float(
        candidates[1].get("score")
    )


def _rerank_vision_candidate(
    *,
    source_table: TableArtifact,
    target_table: TableArtifact,
    candidate_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    title_ratio = _sequence_ratio(
        _table_rescue_title_text(source_table),
        _table_rescue_title_text(target_table),
    )
    header_ratio = _sequence_ratio(
        _table_rescue_headers_text(source_table),
        _table_rescue_headers_text(target_table),
    )
    fingerprint_ratio = _sequence_ratio(
        _table_rescue_fingerprint(source_table),
        _table_rescue_fingerprint(target_table),
    )
    indicator_jaccard = _indicator_jaccard_from_tables(source_table, target_table)
    page_proximity = _table_rescue_page_proximity(source_table, target_table)
    row_ratio = _table_rescue_row_count_ratio(source_table, target_table)
    row_ratio_score = 1.0 / max(row_ratio, 1.0)

    base_score = _safe_float((candidate_payload or {}).get("score", 0.0))
    coverage_min = _safe_float((candidate_payload or {}).get("coverage_min", 0.0))
    distinctive = _safe_float(
        (candidate_payload or {}).get("distinctive_overlap_score", 0.0)
    )
    top_overlap = _safe_float((candidate_payload or {}).get("top_overlap", 0.0))
    tail_overlap = _safe_float((candidate_payload or {}).get("tail_overlap", 0.0))
    suspicion_flags = list((candidate_payload or {}).get("suspicion_flags", []) or [])

    rerank_score = (
        0.28 * base_score
        + 0.18 * max(coverage_min, indicator_jaccard)
        + 0.14 * title_ratio
        + 0.10 * header_ratio
        + 0.16 * fingerprint_ratio
        + 0.08 * page_proximity
        + 0.06 * row_ratio_score
    )
    if "prefix_bias" in suspicion_flags:
        rerank_score -= 0.03
    if "subset_superset" in suspicion_flags:
        rerank_score -= 0.04
    if top_overlap >= 0.75 and tail_overlap < 0.25:
        rerank_score -= 0.05

    return {
        "score": round(rerank_score, 6),
        "base_score": round(base_score, 6),
        "coverage_min": round(max(coverage_min, indicator_jaccard), 6),
        "coverage_gap": round(
            _safe_float((candidate_payload or {}).get("coverage_gap", 0.0)), 6
        ),
        "top_overlap": round(max(top_overlap, indicator_jaccard), 6),
        "tail_overlap": round(tail_overlap, 6),
        "distinctive_overlap_score": round(max(distinctive, fingerprint_ratio), 6),
        "row_count_ratio": round(
            _safe_float(
                (candidate_payload or {}).get("row_count_ratio", row_ratio), row_ratio
            ),
            6,
        ),
        "title_similarity": round(title_ratio, 6),
        "header_similarity": round(header_ratio, 6),
        "fingerprint_similarity": round(fingerprint_ratio, 6),
        "page_proximity": round(page_proximity, 6),
        "indicator_jaccard": round(indicator_jaccard, 6),
        "suspicion_flags": suspicion_flags,
    }


def _collect_vision_rescue_candidates(
    *,
    strict: dict[str, Any],
    t1_by_uid: dict[str, TableArtifact],
    t2_by_uid: dict[str, TableArtifact],
    max_candidates_per_table: int,
    max_tables_per_run: int,
    cross_section_rescue_enabled: bool = False,
    cross_section_rerank_min: float = 0.30,
) -> tuple[list[dict[str, Any]], int]:
    row_candidates_map, col_candidates_map = _build_unmatched_candidate_maps(strict)
    pairs = strict.get("pairs", []) or []
    paired_t1: set[str] = set()
    paired_t2: set[str] = set()
    for pair in pairs:
        paired_t1.add(str(pair.get("t1_uid", "")))
        paired_t2.add(str(pair.get("t2_uid", "")))
        for extra_uid in pair.get("merge_members_t1", []) or []:
            paired_t1.add(str(extra_uid))
        for extra_uid in pair.get("split_members_t2", []) or []:
            paired_t2.add(str(extra_uid))

    ambiguous_t1 = {
        str(item.get("t1_uid", "")): item
        for item in strict.get("ambiguous_unmatched_previous", []) or []
        if str(item.get("t1_uid", "")).strip()
    }
    ambiguous_t2 = {
        str(item.get("t2_uid", "")): item
        for item in strict.get("ambiguous_unmatched_current", []) or []
        if str(item.get("t2_uid", "")).strip()
    }
    confirmed_removed = {
        str(item.get("t1_uid", "")): item
        for item in strict.get("removed_tables", []) or []
        if str(item.get("t1_uid", "")).strip()
    }
    confirmed_added = {
        str(item.get("t2_uid", "")): item
        for item in strict.get("added_tables", []) or []
        if str(item.get("t2_uid", "")).strip()
    }

    def _rescue_target_allowed(table: TableArtifact | None) -> bool:
        if table is None:
            return False
        return get_extraction_status(table) != EXTRACTION_STATUS_BLOCKED

    section_t2_unpaired = {
        str(uid): table
        for uid, table in t2_by_uid.items()
        if str(uid) not in paired_t2 and _rescue_target_allowed(table)
    }
    section_t1_unpaired = {
        str(uid): table
        for uid, table in t1_by_uid.items()
        if str(uid) not in paired_t1 and _rescue_target_allowed(table)
    }

    source_specs: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()

    def _add_source(side: str, uid: str, source: str) -> None:
        uid = str(uid or "").strip()
        if not uid:
            return
        key = (side, uid)
        if key in seen_sources:
            return
        if side == "t1":
            if uid in paired_t1 or uid not in t1_by_uid:
                return
        else:
            if uid in paired_t2 or uid not in t2_by_uid:
                return
        seen_sources.add(key)
        source_specs.append({"side": side, "uid": uid, "source": source})

    for entry in strict.get("suspicious_pairs", []) or []:
        if not isinstance(entry, dict):
            continue
        _add_source("t1", str(entry.get("t1_uid", "")).strip(), "suspicious_pair")
        _add_source("t2", str(entry.get("t2_uid", "")).strip(), "suspicious_pair")
    for uid in ambiguous_t1:
        _add_source("t1", uid, "ambiguous_unmatched")
    for uid in ambiguous_t2:
        _add_source("t2", uid, "ambiguous_unmatched")
    for uid in confirmed_removed:
        _add_source("t1", uid, "confirmed_unmatched")
    for uid in confirmed_added:
        _add_source("t2", uid, "confirmed_unmatched")

    source_specs.sort(
        key=lambda item: (
            0
            if item["source"] == "suspicious_pair"
            else 1
            if item["source"] == "ambiguous_unmatched"
            else 2,
            item["uid"],
        )
    )
    if max_tables_per_run > 0:
        source_specs = source_specs[:max_tables_per_run]

    candidate_entries: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for spec in source_specs:
        side = str(spec["side"])
        source = str(spec["source"])
        if side == "t1":
            source_uid = str(spec["uid"])
            source_table = t1_by_uid.get(source_uid)
            if source_table is None:
                continue
            section = str(getattr(source_table, "section", "") or "").strip()
            payload_by_uid = {
                str(item.get("t2_uid", "")).strip(): item
                for item in row_candidates_map.get(source_uid, [])
                if str(item.get("t2_uid", "")).strip()
            }
            for entry in strict.get("suspicious_pairs", []) or []:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("t1_uid", "")).strip() == source_uid:
                    t2_uid = str(entry.get("t2_uid", "")).strip()
                    if t2_uid:
                        payload_by_uid[t2_uid] = dict(entry)
            all_candidates = []
            for t2_uid, target_table in section_t2_unpaired.items():
                target_section = str(getattr(target_table, "section", "") or "").strip()
                is_cross_section = target_section != section
                if is_cross_section:
                    if not (
                        cross_section_rescue_enabled and source == "confirmed_unmatched"
                    ):
                        continue
                candidate_payload = payload_by_uid.get(t2_uid)
                reranked = _rerank_vision_candidate(
                    source_table=source_table,
                    target_table=target_table,
                    candidate_payload=candidate_payload,
                )
                rerank_threshold = (
                    cross_section_rerank_min if is_cross_section else 0.18
                )
                if reranked["score"] < rerank_threshold:
                    continue
                reranked["is_cross_section"] = is_cross_section
                all_candidates.append(
                    {
                        "t1_uid": source_uid,
                        "t2_uid": t2_uid,
                        "source": source,
                        **reranked,
                    }
                )
            all_candidates.sort(key=lambda item: item["score"], reverse=True)
            for candidate in all_candidates[:max_candidates_per_table]:
                pair_key = (candidate["t1_uid"], candidate["t2_uid"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidate_entries.append(candidate)
        else:
            source_uid = str(spec["uid"])
            source_table = t2_by_uid.get(source_uid)
            if source_table is None:
                continue
            section = str(getattr(source_table, "section", "") or "").strip()
            payload_by_uid = {
                str(item.get("t1_uid", "")).strip(): item
                for item in col_candidates_map.get(source_uid, [])
                if str(item.get("t1_uid", "")).strip()
            }
            for entry in strict.get("suspicious_pairs", []) or []:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("t2_uid", "")).strip() == source_uid:
                    t1_uid = str(entry.get("t1_uid", "")).strip()
                    if t1_uid:
                        payload_by_uid[t1_uid] = dict(entry)
            all_candidates = []
            for t1_uid, target_table in section_t1_unpaired.items():
                target_section = str(getattr(target_table, "section", "") or "").strip()
                is_cross_section = target_section != section
                if is_cross_section:
                    if not (
                        cross_section_rescue_enabled and source == "confirmed_unmatched"
                    ):
                        continue
                candidate_payload = payload_by_uid.get(t1_uid)
                reranked = _rerank_vision_candidate(
                    source_table=target_table,
                    target_table=source_table,
                    candidate_payload=candidate_payload,
                )
                rerank_threshold = (
                    cross_section_rerank_min if is_cross_section else 0.18
                )
                if reranked["score"] < rerank_threshold:
                    continue
                reranked["is_cross_section"] = is_cross_section
                all_candidates.append(
                    {
                        "t1_uid": t1_uid,
                        "t2_uid": source_uid,
                        "source": source,
                        **reranked,
                    }
                )
            all_candidates.sort(key=lambda item: item["score"], reverse=True)
            for candidate in all_candidates[:max_candidates_per_table]:
                pair_key = (candidate["t1_uid"], candidate["t2_uid"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidate_entries.append(candidate)

    candidate_entries.sort(key=_candidate_pair_priority)
    return candidate_entries, len(source_specs)


def _rebuild_strict_unmatched_state(
    *,
    strict: dict[str, Any],
    rescued_t1_uids: set[str],
    rescued_t2_uids: set[str],
    uncertain_t1_uids: set[str],
    uncertain_t2_uids: set[str],
) -> None:
    unmatched_t1 = []
    for item in strict.get("unmatched_t1", []) or []:
        t1_uid = str(item.get("t1_uid", "")).strip()
        if not t1_uid or t1_uid in rescued_t1_uids:
            continue
        cloned = dict(item)
        if t1_uid in uncertain_t1_uids:
            cloned["unmatched_status"] = "ambiguous"
            cloned["reason"] = "vision_rescue_uncertain"
            flags = list(cloned.get("suspicion_flags", []) or [])
            if "vision_rescue_uncertain" not in flags:
                flags.append("vision_rescue_uncertain")
            cloned["suspicion_flags"] = flags
        unmatched_t1.append(cloned)

    unmatched_t2 = []
    for item in strict.get("unmatched_t2", []) or []:
        t2_uid = str(item.get("t2_uid", "")).strip()
        if not t2_uid or t2_uid in rescued_t2_uids:
            continue
        cloned = dict(item)
        if t2_uid in uncertain_t2_uids:
            cloned["unmatched_status"] = "ambiguous"
            cloned["reason"] = "vision_rescue_uncertain"
            flags = list(cloned.get("suspicion_flags", []) or [])
            if "vision_rescue_uncertain" not in flags:
                flags.append("vision_rescue_uncertain")
            cloned["suspicion_flags"] = flags
        unmatched_t2.append(cloned)

    strict["unmatched_t1"] = unmatched_t1
    strict["unmatched_t2"] = unmatched_t2
    strict["unmatched_confirmed_t1"] = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "confirmed"
    ]
    strict["unmatched_ambiguous_t1"] = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "ambiguous"
    ]
    strict["unmatched_confirmed_t2"] = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "confirmed"
    ]
    strict["unmatched_ambiguous_t2"] = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "ambiguous"
    ]
    strict["ambiguous_unmatched_previous"] = list(strict["unmatched_ambiguous_t1"])
    strict["ambiguous_unmatched_current"] = list(strict["unmatched_ambiguous_t2"])

    strict["added_tables"] = [
        item
        for item in strict.get("added_tables", []) or []
        if str(item.get("t2_uid", "")).strip() not in rescued_t2_uids
        and str(item.get("t2_uid", "")).strip() not in uncertain_t2_uids
    ]
    strict["removed_tables"] = [
        item
        for item in strict.get("removed_tables", []) or []
        if str(item.get("t1_uid", "")).strip() not in rescued_t1_uids
        and str(item.get("t1_uid", "")).strip() not in uncertain_t1_uids
    ]
    strict["added_tables_confirmed"] = list(strict.get("added_tables", []) or [])
    strict["removed_tables_confirmed"] = list(strict.get("removed_tables", []) or [])
    strict["ambiguous_tables"] = [
        {
            "side": "previous",
            "uid": str(item.get("t1_uid", "")).strip(),
            "table_id": item.get("t1_table_id"),
            "title": item.get("title_t1"),
            "page": item.get("page_t1"),
            "section": item.get("section", ""),
            "reason": item.get("reason", ""),
            "suspicion_flags": list(item.get("suspicion_flags", []) or []),
        }
        for item in strict["ambiguous_unmatched_previous"]
    ] + [
        {
            "side": "current",
            "uid": str(item.get("t2_uid", "")).strip(),
            "table_id": item.get("t2_table_id"),
            "title": item.get("title_t2"),
            "page": item.get("page_t2"),
            "section": item.get("section", ""),
            "reason": item.get("reason", ""),
            "suspicion_flags": list(item.get("suspicion_flags", []) or []),
        }
        for item in strict["ambiguous_unmatched_current"]
    ]


def _normalize_bbox_ltrb_norm(bbox: Any) -> list[float] | None:
    """Normalize bbox to [l, t, r, b] in 0..1. Returns None if invalid or outside [0,1]."""
    if bbox is None:
        return None
    try:
        l_val, t_val, r_val, b_val = 0.0, 0.0, 1.0, 1.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            l_val, t_val, r_val, b_val = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        elif isinstance(bbox, dict):
            if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
                l_val = float(bbox["x0"])
                t_val = float(bbox["y0"])
                r_val = float(bbox["x1"])
                b_val = float(bbox["y1"])
            elif all(k in bbox for k in ("l", "t", "r", "b")):
                l_val = float(bbox["l"])
                t_val = float(bbox["t"])
                r_val = float(bbox["r"])
                b_val = float(bbox["b"])
            elif all(k in bbox for k in ("x", "y", "width", "height")):
                l_val = float(bbox["x"])
                t_val = float(bbox["y"])
                r_val = l_val + float(bbox["width"])
                b_val = t_val + float(bbox["height"])
            else:
                return None
        else:
            return None
        if r_val <= l_val or b_val <= t_val:
            return None
        if l_val < -0.05 or t_val < -0.05 or r_val > 1.05 or b_val > 1.05:
            return None
        return [
            max(0.0, min(1.0, l_val)),
            max(0.0, min(1.0, t_val)),
            max(0.0, min(1.0, r_val)),
            max(0.0, min(1.0, b_val)),
        ]
    except (TypeError, ValueError):
        return None


def _all_indicators_value_clean_ordered(table: TableArtifact) -> list[str]:
    """Return value_clean indicators in table order (same string space as added/removed)."""
    raw = get_vision_raw_indicators(table)
    result: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if _classify_excluded_line(s):
            continue
        cleaned = strip_footnote_markers_from_indicator(s)
        key = _canonical_indicator_key(cleaned)
        if not key:
            continue
        result.append(cleaned)
    return result


def _build_clean_to_raw_indicator_lookup(table: TableArtifact) -> dict[str, str]:
    """Build stable mapping from canonical clean key to raw display indicator text."""
    clean_values = list(getattr(table, "first_column_indicators", None) or [])
    raw_values = get_vision_raw_indicators(table)
    if not raw_values:
        raw_values = clean_values

    def _display_text(raw_text: str, fallback: str) -> str:
        cleaned = strip_footnote_markers_from_indicator(str(raw_text).strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            return cleaned
        return str(fallback).strip()

    raw_lookup: dict[str, str] = {}
    for raw_item in raw_values:
        raw_text = str(raw_item).strip()
        if not raw_text:
            continue
        value_clean = strip_footnote_markers_from_indicator(raw_text)
        key = _canonical_indicator_key(value_clean)
        if key and key not in raw_lookup:
            raw_lookup[key] = _display_text(raw_text, value_clean)

    # When arrays drift in length, positional zip is unsafe; prefer raw-derived mapping.
    if len(clean_values) != len(raw_values):
        return raw_lookup

    lookup: dict[str, str] = {}
    for idx, clean_item in enumerate(clean_values):
        clean_text = str(clean_item).strip()
        if not clean_text:
            continue
        value_clean = strip_footnote_markers_from_indicator(clean_text)
        key = _canonical_indicator_key(value_clean)
        if not key or key in lookup:
            continue
        raw_text = str(raw_values[idx]).strip() if idx < len(raw_values) else clean_text
        raw_key = _canonical_indicator_key(
            strip_footnote_markers_from_indicator(raw_text)
        )
        if raw_key == key:
            lookup[key] = _display_text(raw_text, clean_text)
            continue
        if key in raw_lookup:
            lookup[key] = raw_lookup[key]
            continue
        lookup[key] = _display_text(clean_text, clean_text)

    for key, value in raw_lookup.items():
        lookup.setdefault(key, value)
    return lookup


def _clean_values_to_raw_display(
    clean_values: list[str], lookup: dict[str, str]
) -> list[str]:
    """Convert clean indicator values to raw display values using canonical key lookup."""
    result: list[str] = []
    for value in clean_values:
        clean_text = str(value).strip()
        if not clean_text:
            continue
        value_clean = strip_footnote_markers_from_indicator(clean_text)
        key = _canonical_indicator_key(value_clean)
        result.append(str(lookup.get(key) or clean_text))
    return result


def _structural_header_keys_from_rows(table: TableArtifact) -> set[str]:
    """Return empty-valued row labels that behave like structural group headers."""
    result: set[str] = set()
    for row in getattr(table, "rows", None) or []:
        if not isinstance(row, (list, tuple)) or not row:
            continue
        label = str(row[0]).strip()
        if not label or _classify_excluded_line(label):
            continue
        other_cells = [str(cell).strip() for cell in row[1:]]
        if any(cell for cell in other_cells):
            continue
        cleaned = strip_footnote_markers_from_indicator(label)
        key = _canonical_indicator_key(cleaned)
        if key:
            result.add(key)
    return result


_DONT_STRUCTURAL_RE = re.compile(r"\bdont\b\s*:?\s*$", re.IGNORECASE)
_ROLLFORWARD_HEADER_CHILD_PREFIXES = (
    "solde debut",
    "nouvelle emission d instrument admissible a titre de fonds propre",
    "rachat de fonds propre",
    "autre y compri",
    "solde a la fin",
)


def _normalize_value_cells_for_structure(row: list[Any] | tuple[Any, ...]) -> list[str]:
    values: list[str] = []
    for cell in list(row)[1:]:
        text = re.sub(r"\s+", " ", str(cell or "").strip())
        if text:
            values.append(text)
    return values


def _structural_rollforward_header_keys_from_rows(table: TableArtifact) -> set[str]:
    """Return subgroup headers that introduce a rollforward block.

    These labels are section/group headers even when OCR leaks a numeric value into the
    same row (for example "Autres elements de fonds propres de categorie 1").
    """
    rows = [
        row
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    result: set[str] = set()
    for idx in range(len(rows) - 2):
        current = list(rows[idx])
        label = str(current[0] if current else "").strip()
        if not label or _classify_excluded_line(label):
            continue
        current_key = _canonical_indicator_key(
            strip_footnote_markers_from_indicator(label)
        )
        if not current_key or current_key.startswith("solde "):
            continue

        next_keys: list[str] = []
        for lookahead in rows[idx + 1 : idx + 6]:
            next_label = str(list(lookahead)[0] if lookahead else "").strip()
            if not next_label or _classify_excluded_line(next_label):
                continue
            next_key = _canonical_indicator_key(
                strip_footnote_markers_from_indicator(next_label)
            )
            if next_key:
                next_keys.append(next_key)
        if not next_keys or not next_keys[0].startswith("solde debut"):
            continue
        child_signal_count = sum(
            1
            for next_key in next_keys[:4]
            if any(
                next_key.startswith(prefix)
                for prefix in _ROLLFORWARD_HEADER_CHILD_PREFIXES
            )
        )
        if child_signal_count >= 3:
            result.add(current_key)
    return result


def _structural_duplicate_value_keys_from_rows(table: TableArtifact) -> set[str]:
    """Return ``dont:`` rows that duplicate the following child row values."""
    rows = [
        row
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    result: set[str] = set()
    for idx in range(len(rows) - 1):
        current = list(rows[idx])
        nxt = list(rows[idx + 1])
        label = str(current[0] if current else "").strip()
        next_label = str(nxt[0] if nxt else "").strip()
        if not label or not next_label:
            continue
        if _classify_excluded_line(label) or _classify_excluded_line(next_label):
            continue
        if not _DONT_STRUCTURAL_RE.search(label):
            continue
        current_values = _normalize_value_cells_for_structure(current)
        next_values = _normalize_value_cells_for_structure(nxt)
        if len(current_values) < 2 or current_values != next_values:
            continue
        key = _canonical_indicator_key(strip_footnote_markers_from_indicator(label))
        if key:
            result.add(key)
    return result


_PAGE_REF_HEADER_RE = re.compile(r"\bpages?\b", re.IGNORECASE)
_LEADING_ORDINAL_RE = re.compile(r"^\s*\d{1,3}\s+\S")
_PAGE_REF_ALLOWED_RE = re.compile(r"^(?:notes?\s+)?[\d,\s\-àaet]+$", re.IGNORECASE)


def _looks_like_page_reference_cell(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not value:
        return False
    value = re.sub(r"\(\d+\)\s*$", "", value).strip()
    if not value or len(value) > 80:
        return False
    alpha_tokens = re.findall(r"[a-zà-ÿ]+", value, flags=re.IGNORECASE)
    if any(token not in {"note", "notes", "et", "a", "à"} for token in alpha_tokens):
        return False
    return bool(_PAGE_REF_ALLOWED_RE.fullmatch(value))


def _is_page_reference_table(table: TableArtifact) -> bool:
    headers = [str(h or "").strip() for h in (getattr(table, "headers", None) or [])]
    if not any(_PAGE_REF_HEADER_RE.search(header) for header in headers):
        return False

    rows = [
        list(row)
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    if len(rows) < 10:
        return False

    ordinal_rows = 0
    populated_value_cells = 0
    page_ref_cells = 0
    for row in rows:
        label = str(row[0] if row else "").strip()
        if label and _LEADING_ORDINAL_RE.match(label):
            ordinal_rows += 1
        for cell in row[1:]:
            cell_text = str(cell or "").strip()
            if not cell_text:
                continue
            populated_value_cells += 1
            if _looks_like_page_reference_cell(cell_text):
                page_ref_cells += 1

    if ordinal_rows < 8 or populated_value_cells == 0:
        return False
    return (page_ref_cells / populated_value_cells) >= 0.6


def _lcs_pair_indices(
    left_seq: list[str],
    right_seq: list[str],
    *,
    band_window: int | None = None,
) -> list[tuple[int, int]]:
    """Return (i, j) pairs where left_seq[i] == right_seq[j] in an LCS.
    If band_window is set, only fill a band |i - j| <= band_window (for large sequences)."""
    if not left_seq or not right_seq:
        return []
    rows, cols = len(left_seq), len(right_seq)

    if band_window is not None:
        # Banded DP: only (i, j) with |i - j| <= band_window
        band = band_window
        # dp[i][j] with j in [max(0, i-band), min(cols, i+band+1)]
        # Store as dp[i][j - (i - band)] for i in range(rows+1), j in range(max(0,i-band), min(cols+1, i+band+2))
        # Simpler: use a dict (i, j) -> value
        dp: dict[tuple[int, int], int] = {}
        for i in range(rows + 1):
            for j in range(max(0, i - band), min(cols + 1, i + band + 2)):
                if i == 0 or j == 0:
                    dp[i, j] = 0
                elif left_seq[i - 1] == right_seq[j - 1]:
                    dp[i, j] = dp.get((i - 1, j - 1), 0) + 1
                else:
                    dp[i, j] = max(
                        dp.get((i - 1, j), 0),
                        dp.get((i, j - 1), 0),
                    )
        # Backtrack to get pairs (only step to cells that were filled)
        pairs: list[tuple[int, int]] = []
        i, j = rows, cols
        while i > 0 and j > 0 and (i, j) in dp:
            if left_seq[i - 1] == right_seq[j - 1]:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            else:
                v_prev_i = dp.get((i - 1, j), -1)
                v_prev_j = dp.get((i, j - 1), -1)
                if v_prev_i >= v_prev_j and (i - 1, j) in dp:
                    i -= 1
                elif (i, j - 1) in dp:
                    j -= 1
                else:
                    break
        pairs.reverse()
        return pairs
    else:
        dp = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if left_seq[i - 1] == right_seq[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        pairs = []
        i, j = rows, cols
        while i > 0 and j > 0:
            if left_seq[i - 1] == right_seq[j - 1]:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        pairs.reverse()
        return pairs


def _order_aware_stable_pairs(
    left_order: list[str],
    right_order: list[str],
    removed_keys: set[str],
    added_keys: set[str],
    *,
    th: dict[str, Any],
) -> set[tuple[str, str]]:
    """Pairs (left_k, right_k) that are order-aligned and similar enough to treat as stable.
    Uses LCS to get unmatched positions, then pairs by index with ratio threshold."""
    if not removed_keys or not added_keys:
        return set()
    min_ratio = float(th.get("indicator_order_aware_min_ratio", 0.85))
    band = int(th.get("indicator_order_aware_band_window", 50))
    n, m = len(left_order), len(right_order)
    use_band = n * m > 10000
    pair_indices = _lcs_pair_indices(
        left_order,
        right_order,
        band_window=band if use_band else None,
    )
    lcs_left = {i for i, _ in pair_indices}
    lcs_right = {j for _, j in pair_indices}
    left_unmatched_keys = [left_order[i] for i in range(len(left_order)) if i not in lcs_left]
    right_unmatched_keys = [right_order[j] for j in range(len(right_order)) if j not in lcs_right]
    left_candidates = [k for k in left_unmatched_keys if k in removed_keys]
    right_candidates = [k for k in right_unmatched_keys if k in added_keys]
    stable: set[tuple[str, str]] = set()
    if rapidfuzz_fuzz is None:
        return stable
    for idx in range(min(len(left_candidates), len(right_candidates))):
        lk, rk = left_candidates[idx], right_candidates[idx]
        score = rapidfuzz_fuzz.ratio(lk, rk) / 100.0
        if score >= min_ratio:
            stable.add((lk, rk))
    return stable


def _ordered_indicator_keys(
    values: list[str],
    *,
    excluded_keys: set[str] | None = None,
) -> list[str]:
    """Return canonical indicator keys in source order, deduplicated."""
    result: list[str] = []
    seen: set[str] = set()
    excluded = excluded_keys or set()
    for value in values:
        kind = _classify_excluded_line(value)
        if kind:
            continue
        cleaned = strip_footnote_markers_from_indicator(value)
        key = _canonical_indicator_key(cleaned)
        if not key or key in seen or key in excluded:
            continue
        seen.add(key)
        result.append(key)
    return result


def _is_likely_extraction_split(
    added_key: str,
    prev_key: str,
    next_key: str,
) -> bool:
    """True if added_key looks like a fragment of prev_key or next_key (extraction split).

    When the added key has two or more tokens that appear in neither neighbor, treat it
    as a genuine new line (e.g. "option de remplacement relative a l acquisition de cwb")
    and do not filter it as neighbor-aligned noise.
    """
    atokens = set(added_key.split())
    ptokens = set(prev_key.split())
    ntokens = set(next_key.split())
    tokens_in_neither = atokens - ptokens - ntokens
    return len(tokens_in_neither) < 2


def _filter_neighbor_aligned_candidates(
    candidate_keys: set[str],
    *,
    source_order: list[str],
    target_order: list[str],
) -> set[str]:
    """Drop singleton additions/removals bounded by shared neighbors on the other side.

    Only filters when the candidate looks like an extraction split (tokens mostly
    in prev/next). Semantically distinct additions (e.g. CWB-specific sub-lines)
    are kept as ADDED.
    """
    if not candidate_keys:
        return set()
    target_pos = {key: idx for idx, key in enumerate(target_order)}
    filtered: set[str] = set()
    for idx, key in enumerate(source_order):
        if key not in candidate_keys:
            continue
        block_start = idx
        while block_start > 0 and source_order[block_start - 1] in candidate_keys:
            block_start -= 1
        block_end = idx
        while (
            block_end + 1 < len(source_order)
            and source_order[block_end + 1] in candidate_keys
        ):
            block_end += 1
        if block_end > block_start:
            continue
        prev_key = next(
            (
                source_order[j]
                for j in range(idx - 1, -1, -1)
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        next_key = next(
            (
                source_order[j]
                for j in range(idx + 1, len(source_order))
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        if (
            prev_key
            and next_key
            and prev_key in target_pos
            and next_key in target_pos
            and target_pos[prev_key] < target_pos[next_key]
            and _is_likely_extraction_split(key, prev_key, next_key)
        ):
            filtered.add(key)
    return filtered


_INDICATOR_STOPWORDS = frozenset(
    {
        "de",
        "du",
        "des",
        "la",
        "le",
        "les",
        "et",
        "ou",
        "and",
        "the",
        "of",
        "to",
        "en",
        "au",
        "aux",
        "a",
        "an",
    }
)
_INDICATOR_UNIT_TOKENS = frozenset(
    {"%", "million", "millions", "milliard", "milliards", "dollars", "cad", "usd"}
)
# Acronyms for overlap gate: if both strings contain same one, gate passes
_INDICATOR_ACRONYM_RE = re.compile(
    r"\b(cet[-]?1|at[-]?1|tlac|rwa|ifrs[-]?9|tier[-]?\s*1|tier[-]?\s*2|bale[-]?\s*iii|pillar[-]?\s*3)\b",
    re.IGNORECASE,
)


def _indicator_strong_tokens(text: str) -> set[str]:
    """Tokenize for overlap gate: normalize hyphens/slashes, drop stopwords/units, drop pure numbers except 1-9."""
    if not text:
        return set()
    normalized = re.sub(r"[-/]", " ", (text or "").lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    tokens: set[str] = set()
    for t in normalized.split():
        if not t:
            continue
        if t in _INDICATOR_STOPWORDS or t in _INDICATOR_UNIT_TOKENS:
            continue
        if t.isdigit():
            if len(t) > 1:
                continue
            if t in ("1", "2", "3", "9"):
                tokens.add(t)
            continue
        tokens.add(t)
    return tokens


def _indicator_acronyms(text: str) -> set[str]:
    """Extract allowlisted acronyms (CET1, TLAC, RWA, etc.) for overlap gate."""
    if not text:
        return set()
    return {
        m.group(1).lower().replace(" ", "").replace("-", "")
        for m in _INDICATOR_ACRONYM_RE.finditer(text or "")
    }


_PREFILTER_MATRIX_CAP = 25_000
_PREFILTER_TOP_K_PER_REMOVED = 50

# Order-invariant matching and embedding gating (config overridable)
_INDICATOR_EMB_MIN = 0.45
_INDICATOR_MIN_TOKENS = 6
_INDICATOR_MIN_ALPHA_RATIO = 0.40


# Tokens that often appear in unit/qualifier suffixes; subset-only diff with these is reformulation, not parent/child
_PARENT_CHILD_UNIT_QUALIFIER_TOKENS = frozenset(
    {
        "en",
        "de",
        "des",
        "du",
        "million",
        "millions",
        "milliard",
        "milliers",
        "dollars",
        "canadiens",
        "cad",
        "usd",
        # Structural prefixes: "Total des X" vs "X" is a reformulation, not parent/child
        "total",
        "sous",
        "net",
        "brut",
        "montant",
    }
)


def _is_parent_child_pair(removed_label: str, added_label: str, norm_fn: Any) -> bool:
    """True if one label is strict token-subset of the other (parent-child, not a rename)."""
    a = norm_fn(removed_label)
    b = norm_fn(added_label)
    if not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    if len(ta) <= 1 or len(tb) <= 1:
        return False
    if ta < tb:
        extra = tb - ta
        if extra <= _PARENT_CHILD_UNIT_QUALIFIER_TOKENS:
            return False
        return True
    if tb < ta:
        extra = ta - tb
        if extra <= _PARENT_CHILD_UNIT_QUALIFIER_TOKENS:
            return False
        return True
    return False


def _hungarian_pair_added_removed(
    removed_items: list[str],
    added_items: list[str],
    *,
    th: dict[str, Any] | None = None,
    embedding_service: Any = None,
) -> tuple[list[str], list[str], list[tuple[str, str]], dict[str, Any]]:
    """
    Global 1-to-1 pairing between removed and added indicators (renames).
    Uses global one-to-one assignment for rename pairing.
    Returns (added_restant, removed_restant, list of (removed_text, added_text), debug_dict).
    Debug dict is for logging only; never written to JSON.
    """
    th = th or {}
    min_score = float(
        th.get(
            "indicator_rename_min_score",
            _INDICATOR_DEFAULTS["indicator_rename_min_score"],
        )
    )
    min_len_ratio = float(
        th.get(
            "indicator_gate_min_len_ratio",
            _INDICATOR_DEFAULTS["indicator_gate_min_len_ratio"],
        )
    )
    min_token_overlap = int(
        th.get(
            "indicator_gate_min_token_overlap",
            _INDICATOR_DEFAULTS["indicator_gate_min_token_overlap"],
        )
    )
    weights_raw = th.get("indicator_similarity_weights")
    weights: dict[str, float] | None = (
        weights_raw if isinstance(weights_raw, dict) else None
    )

    if not removed_items or not added_items or rapidfuzz_fuzz is None:
        return (
            list(added_items),
            list(removed_items),
            [],
            {"gated_out_pairs": 0, "accepted_renames": 0},
        )

    large_matrix_cap = int(th.get("indicator_hungarian_large_matrix_cap", 500))
    large_matrix_min_score = float(
        th.get("indicator_hungarian_large_matrix_min_score", 0.88)
    )
    matrix_size_for_cap = len(removed_items) * len(added_items)
    is_large_matrix = matrix_size_for_cap > large_matrix_cap
    effective_min_score = (
        max(min_score, large_matrix_min_score) if is_large_matrix else min_score
    )
    min_score_pct = int(effective_min_score * 100)

    def _norm_for_sort(s: str) -> str:
        return _canonical_indicator_key(strip_footnote_markers_from_indicator(s))

    removed = sorted(removed_items, key=_norm_for_sort)
    added = sorted(added_items, key=_norm_for_sort)

    gate_len_ratio = min_len_ratio
    gate_token_overlap = min_token_overlap
    if is_large_matrix:
        gate_len_ratio = float(
            th.get(
                "indicator_hungarian_large_matrix_min_len_ratio",
                min_len_ratio,
            )
        )
        gate_token_overlap = int(
            th.get(
                "indicator_hungarian_large_matrix_min_token_overlap",
                min_token_overlap,
            )
        )

    def _length_ratio_ok(a: str, r: str) -> bool:
        la, lr = len(_norm_for_sort(a)), len(_norm_for_sort(r))
        if max(la, lr) <= 0:
            return True
        return (min(la, lr) / max(la, lr)) >= gate_len_ratio

    def _token_overlap_ok(a: str, r: str) -> bool:
        na, nr = _norm_for_sort(a), _norm_for_sort(r)
        ta = _indicator_strong_tokens(na)
        tr = _indicator_strong_tokens(nr)
        if len(ta & tr) >= gate_token_overlap:
            return True
        acro_a = _indicator_acronyms(na)
        acro_r = _indicator_acronyms(nr)
        return len(acro_a & acro_r) > 0

    def _similarity(a: str, r: str) -> float:
        ratio_score = rapidfuzz_fuzz.ratio(a, r)
        token_score = rapidfuzz_fuzz.token_set_ratio(a, r)
        if weights:
            return (
                weights.get("ratio", 0.4) * ratio_score
                + weights.get("token_set", 0.6) * token_score
            )
        return max(ratio_score, token_score)

    use_token_sorted = bool(th.get("use_indicator_token_sorted_matching", True))
    min_tokens = int(th.get("indicator_embed_min_tokens", _INDICATOR_MIN_TOKENS))
    emb_min = float(th.get("indicator_embed_min_sim", _INDICATOR_EMB_MIN))
    min_alpha_ratio = float(
        th.get("indicator_embed_min_alpha_ratio", _INDICATOR_MIN_ALPHA_RATIO)
    )

    def _lex_similarity_both_forms(a: str, r: str) -> tuple[float, float, float]:
        """Lex similarity on canonical and token_sorted; returns (lex_canon, lex_ts, lex_final)."""
        lex_canon = _similarity(a, r)
        if not use_token_sorted:
            return lex_canon, 0.0, lex_canon
        canon_a, ts_a = get_canonical_text(a), get_token_sorted_text(a)
        canon_r, ts_r = get_canonical_text(r), get_token_sorted_text(r)
        lex_ts = 0.0
        if ts_a and ts_r:
            lex_ts = max(
                rapidfuzz_fuzz.ratio(ts_a, ts_r),
                rapidfuzz_fuzz.token_set_ratio(ts_a, ts_r),
            )
        return lex_canon, lex_ts, max(lex_canon, lex_ts)

    def _embed_gate_ok(a: str, r: str, emb_sim: float) -> bool:
        if emb_sim < emb_min:
            return False
        ts_a, ts_r = get_token_sorted_text(a), get_token_sorted_text(r)
        tokens_a = [t for t in ts_a.split() if t] if ts_a else []
        tokens_r = [t for t in ts_r.split() if t] if ts_r else []
        if len(tokens_a) < min_tokens or len(tokens_r) < min_tokens:
            return False
        for text in (a, r):
            if not text:
                continue
            alpha = sum(1 for c in text if c.isalpha())
            if (alpha / len(text)) < min_alpha_ratio:
                return False
        return True

    n_rem, n_add = len(removed), len(added)
    use_emb = (
        bool(th.get("use_embeddings", False))
        and embedding_service
        and getattr(embedding_service, "available", False)
    )
    embed_weight = float(th.get("embedding_weight_indicator", 0.35)) if use_emb else 0.0

    def _norm_for_embed(s: str) -> str:
        c = _canonical_indicator_key(strip_footnote_markers_from_indicator(s))
        return c if c else (s or " ").strip()[:200]

    embed_matrix_canon = None
    embed_matrix_ts = None
    if use_emb and n_rem > 0 and n_add > 0:
        try:
            import numpy as np

            texts_rem_canon = [_norm_for_embed(removed[i]) for i in range(n_rem)]
            texts_add_canon = [_norm_for_embed(added[j]) for j in range(n_add)]
            embed_matrix_canon = embedding_service.get_pairwise_cosine(
                texts_rem_canon, texts_add_canon
            )
            if use_token_sorted:
                texts_rem_ts = [
                    get_token_sorted_text(removed[i]) or " " for i in range(n_rem)
                ]
                texts_add_ts = [
                    get_token_sorted_text(added[j]) or " " for j in range(n_add)
                ]
                embed_matrix_ts = embedding_service.get_pairwise_cosine(
                    texts_rem_ts, texts_add_ts
                )
        except Exception as e:
            logger.debug("Indicator embedding batch failed: %s", e)
            embed_matrix_canon = None
            embed_matrix_ts = None
    matrix_size = n_rem * n_add
    prefilter_used = matrix_size > _PREFILTER_MATRIX_CAP
    candidate_set: set[tuple[int, int]] | None = None
    if prefilter_used:
        candidate_set = set()
        for i in range(n_rem):
            scored: list[tuple[float, int]] = []
            for j in range(n_add):
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(
                    added[j], removed[i]
                ):
                    sc = rapidfuzz_fuzz.token_set_ratio(added[j], removed[i])
                    scored.append((sc, j))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, j in scored[:_PREFILTER_TOP_K_PER_REMOVED]:
                candidate_set.add((i, j))

    gated_out = 0
    accepted_scores: list[float] = []

    if linear_sum_assignment is not None:
        import numpy as np

        scores = np.full((n_rem, n_add), -1e9, dtype=np.float64)
        for i in range(n_rem):
            for j in range(n_add):
                if candidate_set is not None and (i, j) not in candidate_set:
                    gated_out += 1
                    continue
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(
                    added[j], removed[i]
                ):
                    lex_canon, lex_ts, lex = _lex_similarity_both_forms(
                        added[j], removed[i]
                    )
                    emb_sim_canon = (
                        float(embed_matrix_canon[i, j])
                        if embed_matrix_canon is not None
                        else 0.0
                    )
                    emb_sim_ts = (
                        float(embed_matrix_ts[i, j])
                        if embed_matrix_ts is not None
                        else 0.0
                    )
                    emb_sim = (
                        max(emb_sim_canon, emb_sim_ts)
                        if (
                            embed_matrix_canon is not None
                            or embed_matrix_ts is not None
                        )
                        else 0.0
                    )
                    embed_ok = embed_weight > 0 and _embed_gate_ok(
                        added[j], removed[i], emb_sim
                    )
                    w_eff = embed_weight if embed_ok else 0.0
                    calibrated_embed = (emb_sim * 100.0) if emb_sim >= emb_min else 0.0
                    scores[i, j] = (1.0 - w_eff) * lex + w_eff * calibrated_embed
                else:
                    gated_out += 1
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        renamed_pairs: list[tuple[str, str]] = []
        renamed_indices: list[tuple[int, int]] = []
        used_rem: set[int] = set()
        used_add: set[int] = set()
        for k in range(len(row_ind)):
            i, j = int(row_ind[k]), int(col_ind[k])
            if i >= n_rem or j >= n_add:
                continue
            sc = float(scores[i, j])
            if sc >= min_score_pct and not _is_parent_child_pair(
                removed[i], added[j], _norm_for_sort
            ):
                renamed_pairs.append((removed[i], added[j]))
                renamed_indices.append((i, j))
                accepted_scores.append(sc)
                used_rem.add(i)
                used_add.add(j)
        added_restant = [added[j] for j in range(n_add) if j not in used_add]
        removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem]

        def _debug_dict() -> dict[str, Any]:
            asc = sorted(accepted_scores) if accepted_scores else []
            unmatched_candidates: list[dict[str, Any]] = []
            for i in range(n_rem):
                if i in used_rem:
                    continue
                r = removed[i]
                cand: list[tuple[str, float]] = []
                for j in range(n_add):
                    if j in used_add:
                        continue
                    sc = float(scores[i, j])
                    if sc > -1e8:
                        cand.append((added[j], sc))
                cand.sort(key=lambda x: x[1], reverse=True)
                unmatched_candidates.append({"removed": r, "top3": cand[:3]})
            rename_pair_debug: list[dict[str, Any]] = []
            for (r, a), (i, j) in zip(renamed_pairs, renamed_indices):
                lc, lts, _ = _lex_similarity_both_forms(a, r)
                ec = (
                    float(embed_matrix_canon[i, j])
                    if embed_matrix_canon is not None
                    else 0.0
                )
                ets = (
                    float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                )
                reasons: list[str] = []
                if not _embed_gate_ok(a, r, max(ec, ets)):
                    reasons.append("embed_gated")
                rename_pair_debug.append(
                    {
                        "lex_canonical": round(lc, 2),
                        "lex_token_sorted": round(lts, 2),
                        "embed_canonical": round(ec, 3),
                        "embed_token_sorted": round(ets, 3),
                        "final_score": round(float(scores[i, j]), 2),
                        "reasons": reasons or ["ok"],
                    }
                )
            return {
                "gated_out_pairs": gated_out,
                "accepted_renames": len(renamed_pairs),
                "prefilter_used": prefilter_used,
                "rename_pair_debug": rename_pair_debug,
                "score_distribution": {
                    "min": min(asc) if asc else None,
                    "max": max(asc) if asc else None,
                    "mean": sum(asc) / len(asc) if asc else None,
                    "median": asc[len(asc) // 2] if asc else None,
                },
                "unmatched_removed_with_candidates": unmatched_candidates,
            }

        if logger.isEnabledFor(logging.DEBUG) and renamed_pairs:
            for (r, a), (i, j) in zip(renamed_pairs, renamed_indices):
                lc, lts, _ = _lex_similarity_both_forms(a, r)
                ec = (
                    float(embed_matrix_canon[i, j])
                    if embed_matrix_canon is not None
                    else 0.0
                )
                ets = (
                    float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                )
                chosen = float(scores[i, j])
                logger.debug(
                    "indicator_rename removed=%r added=%r lex_canon=%.1f lex_ts=%.1f chosen=%.1f embed_canon=%.3f embed_ts=%.3f",
                    r[:60] if r else "",
                    a[:60] if a else "",
                    lc,
                    lts,
                    chosen,
                    ec,
                    ets,
                )

        return added_restant, removed_restant, renamed_pairs, _debug_dict()
    else:
        # Greedy fallback: process in sorted order, pick best above threshold with same gating
        used_add_f: set[int] = set()
        used_rem_f: set[int] = set()
        renamed_pairs = []
        for i, r in enumerate(removed):
            best_j = -1
            best_score = -1.0
            for j, a in enumerate(added):
                if j in used_add_f:
                    continue
                if not _length_ratio_ok(a, r) or not _token_overlap_ok(a, r):
                    gated_out += 1
                    continue
                _, _, lex_final = _lex_similarity_both_forms(a, r)
                sc = lex_final
                if sc >= min_score_pct and sc > best_score:
                    best_score = sc
                    best_j = j
            if best_j >= 0 and not _is_parent_child_pair(
                r, added[best_j], _norm_for_sort
            ):
                renamed_pairs.append((r, added[best_j]))
                _, _, lex_f = _lex_similarity_both_forms(added[best_j], r)
                accepted_scores.append(lex_f)
                used_rem_f.add(i)
                used_add_f.add(best_j)
        added_restant = [added[j] for j in range(n_add) if j not in used_add_f]
        removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem_f]
        asc = sorted(accepted_scores) if accepted_scores else []
        debug = {
            "gated_out_pairs": gated_out,
            "accepted_renames": len(renamed_pairs),
            "prefilter_used": prefilter_used,
            "score_distribution": {
                "min": min(asc) if asc else None,
                "max": max(asc) if asc else None,
                "mean": sum(asc) / len(asc) if asc else None,
                "median": asc[len(asc) // 2] if asc else None,
            },
            "unmatched_removed_with_candidates": [
                {"removed": removed[i], "top3": []}
                for i in range(n_rem)
                if i not in used_rem_f
            ],
        }
        if logger.isEnabledFor(logging.DEBUG) and renamed_pairs and rapidfuzz_fuzz:
            for r, a in renamed_pairs:
                ratio_sc = rapidfuzz_fuzz.ratio(a, r)
                token_sc = rapidfuzz_fuzz.token_set_ratio(a, r)
                chosen = _similarity(a, r)
                logger.debug(
                    "indicator_rename removed=%r added=%r ratio=%.1f token=%.1f chosen_score=%.1f embed_sim=%.3f",
                    r[:60] if r else "",
                    a[:60] if a else "",
                    ratio_sc,
                    token_sc,
                    chosen,
                    0.0 if not use_emb else 0.0,
                )
        return added_restant, removed_restant, renamed_pairs, debug


def _adaptive_fusion_threshold(concat_token_count: int) -> float:
    if concat_token_count >= 8:
        return 0.88
    if concat_token_count < 5:
        return 0.94
    return 0.92


def _fusion_split_score(single_norm: str, k1: str, k2: str) -> tuple[float, int]:
    """Order-invariant similarity of single_norm to concat(k1,k2) in either order."""
    from difflib import SequenceMatcher

    c_fwd = f"{k1} {k2}".strip()
    c_rev = f"{k2} {k1}".strip()
    ntok = max(len(c_fwd.split()), 1)
    if not c_fwd:
        return 0.0, ntok
    if rapidfuzz_fuzz is not None:
        s_fwd = rapidfuzz_fuzz.token_set_ratio(single_norm, c_fwd) / 100.0
        s_rev = rapidfuzz_fuzz.token_set_ratio(single_norm, c_rev) / 100.0
        best = max(s_fwd, s_rev)
    else:
        best = max(
            SequenceMatcher(None, single_norm, c_fwd).ratio(),
            SequenceMatcher(None, single_norm, c_rev).ratio(),
        )
    # Truncation: last token of k1 or k2 may be OCR-truncated (e.g. amort vs amorti)
    toks_single = single_norm.split()
    for frag_key in (k1, k2):
        ft = frag_key.split()
        if not ft or len(ft[-1]) < 4:
            continue
        last = ft[-1]
        for w in toks_single:
            if len(w) >= len(last) and w.startswith(last) and w != last:
                repaired = " ".join(ft[:-1] + [w])
                other = k2 if frag_key is k1 else k1
                c1 = f"{repaired} {other}".strip()
                c2 = f"{other} {repaired}".strip()
                if rapidfuzz_fuzz is not None:
                    best = max(
                        best,
                        rapidfuzz_fuzz.token_set_ratio(single_norm, c1) / 100.0,
                        rapidfuzz_fuzz.token_set_ratio(single_norm, c2) / 100.0,
                    )
                else:
                    best = max(
                        best,
                        SequenceMatcher(None, single_norm, c1).ratio(),
                        SequenceMatcher(None, single_norm, c2).ratio(),
                    )
    return best, ntok


def _detect_fusion_split(
    added: list[str], removed: list[str], ratio_threshold: float = 0.92
) -> tuple[list[str], list[str], bool]:
    """Merge fusion/split: 1 added = concat of 2 removed (or vice versa).
    Uses token_set_ratio (order-invariant) and adaptive thresholds.
    Returns (added, removed, had_fusion_split).
    """
    added = list(added)
    removed = list(removed)
    had_fusion_split = False

    def _merge_added_from_removed() -> None:
        nonlocal added, removed, had_fusion_split
        for a in added[:]:
            a_norm = _canonical_indicator_key(a)
            if not a_norm:
                continue
            for j, r1 in enumerate(removed):
                k1 = _canonical_indicator_key(r1)
                if not k1:
                    continue
                for k, r2 in enumerate(removed):
                    if j >= k:
                        continue
                    k2 = _canonical_indicator_key(r2)
                    if not k2:
                        continue
                    score, ntok = _fusion_split_score(a_norm, k1, k2)
                    thresh = _adaptive_fusion_threshold(ntok)
                    if score >= thresh or f"{k1} {k2}".strip() == a_norm or f"{k2} {k1}".strip() == a_norm:
                        added.remove(a)
                        removed.remove(r2)
                        removed.remove(r1)
                        had_fusion_split = True
                        return
            if not added:
                break

    def _merge_removed_from_added() -> None:
        nonlocal added, removed, had_fusion_split
        for r in removed[:]:
            r_norm = _canonical_indicator_key(r)
            if not r_norm:
                continue
            for j, a1 in enumerate(added):
                k1 = _canonical_indicator_key(a1)
                if not k1:
                    continue
                for k, a2 in enumerate(added):
                    if j >= k:
                        continue
                    k2 = _canonical_indicator_key(a2)
                    if not k2:
                        continue
                    score, ntok = _fusion_split_score(r_norm, k1, k2)
                    thresh = _adaptive_fusion_threshold(ntok)
                    if score >= thresh or f"{k1} {k2}".strip() == r_norm or f"{k2} {k1}".strip() == r_norm:
                        removed.remove(r)
                        added.remove(a2)
                        added.remove(a1)
                        had_fusion_split = True
                        return
            if not removed:
                break

    changed = True
    while changed:
        changed = False
        before_a, before_r = len(added), len(removed)
        _merge_added_from_removed()
        if len(added) != before_a or len(removed) != before_r:
            changed = True
            continue
        _merge_removed_from_added()
        if len(added) != before_a or len(removed) != before_r:
            changed = True

    return added, removed, had_fusion_split


def _apply_short_indicator_guard(
    added_keys: set[str],
    removed_keys: set[str],
    stable_keys: set[str],
    th: dict[str, Any],
    excluded_counts: dict[str, int],
) -> None:
    """Drop short keys that are strict token-subsets of a long stable line (false split)."""
    if not th.get("indicator_short_guard_enabled", True):
        return
    max_t = int(th.get("indicator_short_guard_max_tokens", 3))
    min_stable = int(th.get("indicator_short_guard_min_stable_tokens", 5))
    stable_list = [sk for sk in stable_keys if len(sk.split()) >= min_stable]

    def _maybe_drop(key: str, which: str) -> bool:
        parts = key.split()
        if len(parts) < 2 or len(parts) > max_t:
            return False
        tk = frozenset(parts)
        for sk in stable_list:
            st = set(sk.split())
            if tk < st:
                if which == "added":
                    added_keys.discard(key)
                else:
                    removed_keys.discard(key)
                excluded_counts["short_indicator_guard"] = (
                    excluded_counts.get("short_indicator_guard", 0) + 1
                )
                return True
        return False

    for k in list(added_keys):
        _maybe_drop(k, "added")
    for k in list(removed_keys):
        _maybe_drop(k, "removed")


def _indicator_diff(
    t1: TableArtifact,
    t2: TableArtifact,
    *,
    neighbor_aligned_filter_enabled: bool = True,
    return_debug: bool = False,
    th: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], bool, dict[str, int], dict[str, Any] | None]:
    if _is_page_reference_table(t1) and _is_page_reference_table(t2):
        return [], [], False, {"page_reference_table": 1}, None

    # Matching/diff use first_column_indicators (clean) only; UI display prefers raw.
    left = get_comparison_indicators(t1)
    right = get_comparison_indicators(t2)
    left_all_keys = set(_ordered_indicator_keys(left))
    right_all_keys = set(_ordered_indicator_keys(right))
    left_structural_keys = (
        _structural_header_keys_from_rows(t1)
        | _structural_rollforward_header_keys_from_rows(t1)
        | _structural_duplicate_value_keys_from_rows(t1)
    ) - right_all_keys
    right_structural_keys = (
        _structural_header_keys_from_rows(t2)
        | _structural_rollforward_header_keys_from_rows(t2)
        | _structural_duplicate_value_keys_from_rows(t2)
    ) - left_all_keys

    def _norm(
        values: list[str],
        *,
        structural_keys: set[str],
    ) -> tuple[dict[str, str], dict[str, int]]:
        mapped: dict[str, str] = {}
        excluded: dict[str, int] = {}
        for value in values:
            kind = _classify_excluded_line(value)
            if kind:
                excluded[kind] = excluded.get(kind, 0) + 1
                continue
            value_clean = strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key in structural_keys:
                excluded["structural"] = excluded.get("structural", 0) + 1
                continue
            if key and key not in mapped:
                mapped[key] = value_clean
        return mapped, excluded

    left_map, left_excluded = _norm(left, structural_keys=left_structural_keys)
    right_map, right_excluded = _norm(right, structural_keys=right_structural_keys)
    excluded_counts: dict[str, int] = {}
    for k in set(left_excluded) | set(right_excluded):
        excluded_counts[k] = left_excluded.get(k, 0) + right_excluded.get(k, 0)

    left_order = _ordered_indicator_keys(left, excluded_keys=left_structural_keys)
    right_order = _ordered_indicator_keys(right, excluded_keys=right_structural_keys)

    added_keys = set(right_map.keys() - left_map.keys())
    removed_keys = set(left_map.keys() - right_map.keys())

    th = th or {}
    if th.get("indicator_order_aware_alignment_enabled", False):
        order_stable = _order_aware_stable_pairs(
            left_order,
            right_order,
            removed_keys,
            added_keys,
            th=th,
        )
        added_keys -= {r for _l, r in order_stable}
        removed_keys -= {l for l, _r in order_stable}
        if order_stable:
            excluded_counts["order_aware_stable"] = len(order_stable)

    # --- NOUVELLE PHASE : RÉSOLUTION NEAR-STABLE ---
    # On recherche les clés qui ont raté le match exact d'un cheveu
    total_keys = len(left_map) + len(right_map)
    large_table_min = int(th.get("indicator_near_stable_large_table_min_indicators", 40))
    if total_keys >= large_table_min:
        near_stable_threshold = float(
            th.get("indicator_near_stable_large_table_threshold", 0.92)
        )
    else:
        near_stable_threshold = float(
            th.get("indicator_near_stable_threshold", 0.95)
        )
    use_token_set = bool(th.get("indicator_near_stable_use_token_set", False))

    resolved_removed = set()
    resolved_added = set()

    for r_key in list(removed_keys):
        best_match = None
        best_score = 0.0

        for a_key in list(added_keys - resolved_added):
            if rapidfuzz_fuzz is not None:
                ratio_score = rapidfuzz_fuzz.ratio(r_key, a_key) / 100.0
                if use_token_set:
                    token_score = rapidfuzz_fuzz.token_set_ratio(r_key, a_key) / 100.0
                    score = max(ratio_score, token_score)
                else:
                    score = ratio_score
            else:
                # Fallback to jaccard if rapidfuzz not available
                lt = set(r_key.split())
                rt = set(a_key.split())
                score = len(lt & rt) / len(lt | rt) if (lt | rt) else 0.0

            if score > best_score:
                best_score = score
                best_match = a_key

        if best_score >= near_stable_threshold and best_match:
            resolved_removed.add(r_key)
            resolved_added.add(best_match)

    added_keys -= resolved_added
    removed_keys -= resolved_removed

    if len(resolved_added) > 0:
        excluded_counts["near_stable"] = len(resolved_added)

    stable_keys = set(left_map.keys()) & set(right_map.keys())
    _apply_short_indicator_guard(added_keys, removed_keys, stable_keys, th, excluded_counts)

    added = [right_map[key] for key in added_keys]
    removed = [left_map[key] for key in removed_keys]
    added.sort()
    removed.sort()
    added, removed, had_fusion_split = _detect_fusion_split(added, removed)
    remaining_added_keys = {
        key
        for value in added
        if (
            key := _canonical_indicator_key(
                strip_footnote_markers_from_indicator(value)
            )
        )
    }
    remaining_removed_keys = {
        key
        for value in removed
        if (
            key := _canonical_indicator_key(
                strip_footnote_markers_from_indicator(value)
            )
        )
    }
    filtered_added: set[str] = set()
    filtered_removed: set[str] = set()
    if neighbor_aligned_filter_enabled:
        filtered_added = _filter_neighbor_aligned_candidates(
            remaining_added_keys,
            source_order=right_order,
            target_order=left_order,
        )
        filtered_removed = _filter_neighbor_aligned_candidates(
            remaining_removed_keys,
            source_order=left_order,
            target_order=right_order,
        )
    if filtered_added:
        excluded_counts["neighbor_aligned"] = excluded_counts.get(
            "neighbor_aligned", 0
        ) + len(filtered_added)
        added = [
            value
            for value in added
            if _canonical_indicator_key(strip_footnote_markers_from_indicator(value))
            not in filtered_added
        ]
    if filtered_removed:
        excluded_counts["neighbor_aligned"] = excluded_counts.get(
            "neighbor_aligned", 0
        ) + len(filtered_removed)
        removed = [
            value
            for value in removed
            if _canonical_indicator_key(strip_footnote_markers_from_indicator(value))
            not in filtered_removed
        ]
    added.sort()
    removed.sort()
    diff_debug_info: dict[str, Any] | None = None
    if return_debug:
        diff_debug_info = {
            "left_map": left_map,
            "right_map": right_map,
        }
    return added, removed, had_fusion_split, excluded_counts, diff_debug_info


def _build_indicator_diff_debug(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    left_map: dict[str, str],
    right_map: dict[str, str],
    added: list[str],
    removed: list[str],
    renamed_pairs: list[tuple[str, str]],
    t1_clean_to_raw: dict[str, str],
    t2_clean_to_raw: dict[str, str],
    indicator_debug: dict[str, Any] | None,
    th: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-indicator decision log for debugging false positives.

    Each entry has: side (t1|t2), raw, clean, canonical_key, status (stable|added|removed|renamed),
    matched_to (for stable/renamed), score (if renamed), reason, threshold_used.
    """
    decisions: list[dict[str, Any]] = []
    rename_pair_scores: dict[tuple[str, str], float] = {}
    if indicator_debug:
        rpd = indicator_debug.get("rename_pair_debug") or []
        for (r, a), dbg in zip(
            renamed_pairs,
            rpd[: len(renamed_pairs)],
        ):
            try:
                score = float(dbg.get("final_score", 0.0))
                if score > 1.0:
                    score = score / 100.0
                rename_pair_scores[(r, a)] = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                pass

    min_score = float(
        th.get(
            "indicator_rename_min_score",
            _INDICATOR_DEFAULTS["indicator_rename_min_score"],
        )
    )

    # T1 indicators
    for key, value_clean in left_map.items():
        raw = t1_clean_to_raw.get(key) or value_clean
        if key in right_map:
            status = "stable"
            matched_to = right_map[key]
            reason = "exact_canonical_match"
            score = 100.0
        else:
            pair = next(((r, a) for (r, a) in renamed_pairs if r == value_clean), None)
            if pair:
                status = "renamed"
                _, matched_to = pair
                score = rename_pair_scores.get(pair)
                reason = "fuzzy_rename"
                if score is not None and score < min_score:
                    reason = f"fuzzy_rename_below_threshold_{min_score}"
            else:
                status = "removed"
                matched_to = None
                score = None
                reason = "no_match_after_rename_pairing"
        decisions.append(
            {
                "side": "t1",
                "raw": raw[:200] if raw else "",
                "clean": value_clean[:200] if value_clean else "",
                "canonical_key": key[:200] if key else "",
                "status": status,
                "matched_to": (matched_to[:200] if matched_to else None)
                if matched_to
                else None,
                "score": round(score, 4) if score is not None else None,
                "reason": reason,
                "threshold_used": min_score if status == "renamed" else None,
            }
        )

    # T2 indicators
    for key, value_clean in right_map.items():
        raw = t2_clean_to_raw.get(key) or value_clean
        if key in left_map:
            status = "stable"
            matched_to = left_map[key]
            reason = "exact_canonical_match"
            score = 100.0
        else:
            pair = next(((r, a) for (r, a) in renamed_pairs if a == value_clean), None)
            if pair:
                status = "renamed"
                matched_to, _ = pair
                score = rename_pair_scores.get(pair)
                reason = "fuzzy_rename"
                if score is not None and score < min_score:
                    reason = f"fuzzy_rename_below_threshold_{min_score}"
            else:
                status = "added"
                matched_to = None
                score = None
                reason = "no_match_after_rename_pairing"
        decisions.append(
            {
                "side": "t2",
                "raw": raw[:200] if raw else "",
                "clean": value_clean[:200] if value_clean else "",
                "canonical_key": key[:200] if key else "",
                "status": status,
                "matched_to": (matched_to[:200] if matched_to else None)
                if matched_to
                else None,
                "score": round(score, 4) if score is not None else None,
                "reason": reason,
                "threshold_used": min_score if status == "renamed" else None,
            }
        )

    return decisions


def _fuzzy_pair_added_removed(
    added: list[str],
    removed: list[str],
    bank_code: str | None,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """
    Appariement 1-1 flou entre indicateurs ajoutes et supprimes (reformulations/renommages).

    Retourne (added_restant, removed_restant, paires (removed_text, added_text)).
    Si rapidfuzz est indisponible, retourne (added, removed, []).
    """
    if not added or not removed or rapidfuzz_fuzz is None:
        return added, removed, []

    th = get_matching_thresholds(bank_code=bank_code)
    threshold = float(th.get("indicator_similarity_threshold", 0.88))
    token_threshold = float(th.get("indicator_fuzzy_token_threshold", 0.85))
    threshold_pct = int(threshold * 100)
    token_threshold_pct = int(token_threshold * 100)

    candidates: list[tuple[str, str, float]] = []
    for a in added:
        for r in removed:
            ratio_score = rapidfuzz_fuzz.ratio(a, r)
            token_score = rapidfuzz_fuzz.token_set_ratio(a, r)
            score = max(ratio_score, token_score)
            if score >= threshold_pct or token_score >= token_threshold_pct:
                candidates.append((a, r, float(score)))

    candidates.sort(key=lambda x: x[2], reverse=True)

    def _norm(s: str) -> str:
        return _canonical_indicator_key(strip_footnote_markers_from_indicator(s))

    used_added: set[str] = set()
    used_removed: set[str] = set()
    renamed_pairs: list[tuple[str, str]] = []
    for a, r, _ in candidates:
        if (
            a not in used_added
            and r not in used_removed
            and not _is_parent_child_pair(r, a, _norm)
        ):
            renamed_pairs.append((r, a))
            used_added.add(a)
            used_removed.add(r)

    added_restant = [x for x in added if x not in used_added]
    removed_restant = [x for x in removed if x not in used_removed]
    return added_restant, removed_restant, renamed_pairs


def _derive_table_status(
    *,
    rescue_type: str | None,
    extraction_low_confidence: bool,
    added: list[str],
    removed: list[str],
    renamed_indicators: list[dict[str, str]],
) -> tuple[str, bool]:
    """Derive table status for Dash/review queue.

    Important: had_fusion_split from indicator normalization is not a structural change.
    Only explicit table-level split/merge rescue marks structure_change.
    """
    rescue = str(rescue_type or "").strip().lower()
    structure_change_detected = rescue == "split_merge_rescue"
    has_changes = bool(added or removed or renamed_indicators)

    if structure_change_detected:
        return "structure_change", True
    if extraction_low_confidence and (added or removed) and not renamed_indicators:
        return "incertain", False
    return ("modifie" if has_changes else "stable"), False


def _empty_result(
    bank_code: str,
    year: int,
    reason: str,
    *,
    quarter_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    current_label = (
        str(quarter_context.get("current", {}).get("label", ""))
        if isinstance(quarter_context, dict)
        else ""
    )
    previous_label = (
        str(quarter_context.get("previous", {}).get("label", ""))
        if isinstance(quarter_context, dict)
        else ""
    )
    current_code = (
        str(quarter_context.get("current", {}).get("code", "t2"))
        if isinstance(quarter_context, dict)
        else "t2"
    )
    previous_code = (
        str(quarter_context.get("previous", {}).get("code", "t1"))
        if isinstance(quarter_context, dict)
        else "t1"
    )
    return {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": previous_label or "t1",
        "quarter_to": current_label or "t2",
        "previous_quarter": previous_label or "",
        "current_quarter": current_label or "",
        "comparison_direction": "current_vs_previous",
        "year": year,
        "summary": {
            "tables_t1": 0,
            "tables_t2": 0,
            "tables_matched": 0,
            "tables_added": 0,
            "tables_removed": 0,
            "rescued_matches_count": 0,
            "split_merge_rescues_count": 0,
            "total_added_indicators": 0,
            "total_removed_indicators": 0,
            "total_renamed_indicators": 0,
            "status_counts": {
                "stable": 0,
                "modifie": 0,
                "renommage_probable": 0,
                "incertain": 0,
                "needs_review": 0,
                "structure_change": 0,
                "ajoute": 0,
                "supprime": 0,
            },
        },
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
        "meta": {
            "generated_at": now,
            "provenance": "comparison_runner",
            "source_format": "empty",
            "error": reason,
            "quarter_context": quarter_context or {},
            "extraction_sources": {
                "previous": {
                    "quarter": previous_code or "t1",
                    "label": previous_label or "",
                    "year": int(
                        (quarter_context or {}).get("previous", {}).get("year", year)
                    ),
                    "mode": "unknown",
                    "artifact_dir": "",
                    "tables_path": "",
                    "indicators_path": "",
                    "footnotes_path": "",
                    "meta_path": "",
                    "snapshot_path": "",
                    "artifacts_present": {
                        "snapshot": False,
                        "tables": False,
                        "meta": False,
                        "indicators": False,
                        "footnotes": False,
                    },
                },
                "current": {
                    "quarter": current_code or "t2",
                    "label": current_label or "",
                    "year": int(
                        (quarter_context or {}).get("current", {}).get("year", year)
                    ),
                    "mode": "unknown",
                    "artifact_dir": "",
                    "tables_path": "",
                    "indicators_path": "",
                    "footnotes_path": "",
                    "meta_path": "",
                    "snapshot_path": "",
                    "artifacts_present": {
                        "snapshot": False,
                        "tables": False,
                        "meta": False,
                        "indicators": False,
                        "footnotes": False,
                    },
                },
            },
            "executive_summary": {
                "content": "Aucun resultat produit. " + reason,
            },
            "embedding_debug": {
                "embedding_enabled": False,
                "embedding_table_used": False,
                "embedding_indicator_used": False,
                "embedding_api_calls": 0,
                "embedding_cache_hits": 0,
                "embedding_batch_sizes": [],
                "embedding_errors": 0,
                "config_use_embeddings": False,
                "fallback_vision_used": False,
                "hungarian_table": False,
                "hungarian_indicator": True,
                "table_pair_count": 0,
                "indicator_rename_count": 0,
                "table_pair_debug": [],
                "rename_pair_debug": [],
            },
        },
    }


def run_comparison_with_sections(
    *,
    pdf_path_t1: str | None = None,
    pdf_path_t2: str | None = None,
    pdf_path_previous: str | None = None,
    pdf_path_current: str | None = None,
    bank_code: str,
    sections_t1: list[dict[str, Any]] | None = None,
    sections_t2: list[dict[str, Any]] | None = None,
    sections_previous: list[dict[str, Any]] | None = None,
    sections_current: list[dict[str, Any]] | None = None,
    current_quarter: str | None = None,
    previous_quarter: str | None = None,
    current_year: int | None = None,
    previous_year: int | None = None,
    use_genai: bool = False,
    api_key: str | None = None,
    generate_visual_proofs: bool = False,
    use_vision_extraction: bool | None = None,
    use_vision_extraction_override: bool | None = None,
    include_footnotes: bool = False,
    include_genai_classification: bool = False,
    use_stored_extraction_if_available: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end comparison used by the Dash Analyze callback.

    Args:
        use_vision_extraction: If True/False, overrides config vision_extraction.enabled
            for this run. If None, config is used.
        use_stored_extraction_if_available: If True, load extraction from disk when
            available (faster). If False, always re-run extraction (default for now).
    """
    del use_genai, generate_visual_proofs  # kept for backward-compatible signature
    if use_vision_extraction is not None and use_vision_extraction_override is not None:
        if bool(use_vision_extraction) != bool(use_vision_extraction_override):
            raise ValueError(
                "Conflicting values for use_vision_extraction and use_vision_extraction_override."
            )
    if use_vision_extraction is None and use_vision_extraction_override is not None:
        use_vision_extraction = use_vision_extraction_override

    pdf_path_previous = pdf_path_previous or pdf_path_t1
    pdf_path_current = pdf_path_current or pdf_path_t2
    if not pdf_path_previous or not pdf_path_current:
        raise ValueError(
            "Both pdf_path_current and pdf_path_previous are required for comparison."
        )
    # Legacy internal aliases: T1 = previous quarter, T2 = current quarter.
    pdf_path_t1 = pdf_path_previous
    pdf_path_t2 = pdf_path_current

    ranges_t1 = _normalize_ranges(sections_previous or sections_t1)
    ranges_t2 = _normalize_ranges(sections_current or sections_t2)

    inferred_year = _infer_year(pdf_path_previous, pdf_path_current)
    quarter_context = build_quarter_context(
        current_quarter or "Q2",
        year=current_year or inferred_year,
        previous_quarter=previous_quarter,
    )
    if previous_year is not None:
        quarter_context["previous"]["year"] = int(previous_year)
        quarter_context["previous"]["label"] = (
            f"Q{quarter_context['previous']['quarter']}-{int(previous_year)}"
        )
        quarter_context["comparison_label"] = (
            f"{quarter_context['current']['label']} vs {quarter_context['previous']['label']}"
        )
    year = int(quarter_context["current"]["year"])
    previous_quarter_code = str(quarter_context["previous"]["code"])
    current_quarter_code = str(quarter_context["current"]["code"])
    previous_year_val = int(quarter_context["previous"]["year"])
    current_year_val = int(quarter_context["current"]["year"])

    if not ranges_t1 or not ranges_t2:
        return _empty_result(
            bank_code,
            year,
            "Aucune section valide fournie.",
            quarter_context=quarter_context,
        )

    try:
        # Run both report extractions in parallel to cut total runtime (Docling + Vision per PDF)
        def _coerce_tables_with_provenance(
            value: Any,
            *,
            quarter_code: str,
            quarter_label: str,
            year_value: int,
        ) -> tuple[list[TableArtifact], dict[str, Any]]:
            from pathlib import Path as _Path

            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], list)
                and isinstance(value[1], dict)
            ):
                tables = value[0]
                provenance = dict(value[1])
            else:
                tables = list(value or [])
                provenance = {}
            base_extraction_dir = _Path("outputs/extractions") / str(bank_code) / str(year_value) / str(quarter_code)
            provenance.setdefault("quarter", quarter_code)
            provenance.setdefault("label", quarter_label)
            provenance.setdefault("year", year_value)
            provenance.setdefault("mode", "unknown")
            provenance.setdefault("artifact_dir", str(base_extraction_dir))
            provenance.setdefault("snapshot_path", str(base_extraction_dir / "extraction_snapshot.json"))
            provenance.setdefault("tables_path", str(base_extraction_dir / "tables.json"))
            provenance.setdefault("indicators_path", str(base_extraction_dir / "indicators.json"))
            provenance.setdefault("footnotes_path", str(base_extraction_dir / "footnotes.json"))
            provenance.setdefault("meta_path", str(base_extraction_dir / "meta.json"))
            provenance.setdefault(
                "artifacts_present",
                {
                    "snapshot": False,
                    "tables": False,
                    "meta": False,
                    "indicators": False,
                    "footnotes": False,
                },
            )
            return tables, provenance

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_t1 = executor.submit(
                _extract_tables,
                pdf_path=pdf_path_previous,
                bank_code=bank_code,
                quarter=previous_quarter_code,
                year=previous_year_val,
                section_ranges=ranges_t1,
                api_key=api_key,
                use_vision_extraction=use_vision_extraction,
                use_stored_extraction_if_available=use_stored_extraction_if_available,
                return_provenance=True,
            )
            fut_t2 = executor.submit(
                _extract_tables,
                pdf_path=pdf_path_current,
                bank_code=bank_code,
                quarter=current_quarter_code,
                year=current_year_val,
                section_ranges=ranges_t2,
                api_key=api_key,
                use_vision_extraction=use_vision_extraction,
                use_stored_extraction_if_available=use_stored_extraction_if_available,
                return_provenance=True,
            )
            tables_t1, extraction_source_previous = _coerce_tables_with_provenance(
                fut_t1.result(),
                quarter_code=previous_quarter_code,
                quarter_label=str(quarter_context["previous"]["label"]),
                year_value=previous_year_val,
            )
            tables_t2, extraction_source_current = _coerce_tables_with_provenance(
                fut_t2.result(),
                quarter_code=current_quarter_code,
                quarter_label=str(quarter_context["current"]["label"]),
                year_value=current_year_val,
            )
    except Exception as exc:
        if "Vision schema contract invalid" in str(exc):
            raise
        return _empty_result(
            bank_code,
            year,
            f"Extraction impossible: {exc}",
            quarter_context=quarter_context,
        )

    if not tables_t1 and not tables_t2:
        return _empty_result(
            bank_code,
            year,
            "Aucun tableau extrait depuis les sections selectionnees.",
            quarter_context=quarter_context,
        )

    try:
        from vigilance.config import get_matching_thresholds

        cfg = get_matching_thresholds(bank_code=bank_code) or {}
        algorithm_used = "symmetric_assignment"
    except Exception:
        cfg = {}
        algorithm_used = "symmetric_assignment"

    raw_tables_t1_count = len(tables_t1)
    raw_tables_t2_count = len(tables_t2)
    logger.info(
        "comparison_input_tables raw_t1=%d raw_t2=%d bank=%s year=%s",
        raw_tables_t1_count,
        raw_tables_t2_count,
        bank_code,
        year,
    )
    fragment_merges_t1: list[dict[str, Any]] = []
    fragment_merges_t2: list[dict[str, Any]] = []
    if bool(cfg.get("matching_v2_enabled", True)):
        try:
            merge_score_min = float(cfg.get("merge_fragment_score_min", 0.85) or 0.85)
        except (TypeError, ValueError):
            merge_score_min = 0.85
        tables_t1, fragment_merges_t1 = merge_table_fragments(
            tables_t1,
            merge_score_min=merge_score_min,
        )
        tables_t2, fragment_merges_t2 = merge_table_fragments(
            tables_t2,
            merge_score_min=merge_score_min,
        )
        logger.info(
            "comparison_input_tables tables_t1=%d tables_t2=%d bank=%s year=%s (after_fragment_merge)",
            len(tables_t1),
            len(tables_t2),
            bank_code,
            year,
        )

    extraction_run_id: str | None = None
    extraction_out_dir: str | None = None
    quality_gate_status: dict[str, Any] = {
        "enabled": False,
        "status": "SKIPPED",
        "eligible_for_review": True,
        "fail_reasons": [],
    }

    # Sauvegarde indicators.json et footnotes.json pour audit + Quality Gate
    qg_enabled = False
    try:
        from vigilance.config import get_vision_extraction_config

        vec = get_vision_extraction_config(bank_code=bank_code)
        qg_cfg = get_quality_gate_config(bank_code=bank_code) or {}
        qg_enabled = bool(qg_cfg.get("enabled", False))
        should_write_extraction_audit = (
            bool(vec.get("save_indicators_footnotes_json")) or qg_enabled
        )

        if should_write_extraction_audit:
            from app.ui_config import OUTPUT_DIR
            from vigilance.extraction.vision_extraction_writer import (
                write_footnotes_json,
                write_indicators_json,
            )

            extraction_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = (
                OUTPUT_DIR
                / "comparisons"
                / bank_code
                / "extractions"
                / extraction_run_id
            )
            indicators_path = write_indicators_json(
                tables_t1, tables_t2, out_dir, bank_code, extraction_run_id
            )
            footnotes_path = write_footnotes_json(
                tables_t1, tables_t2, out_dir, bank_code, extraction_run_id
            )
            extraction_out_dir = str(out_dir)

            if qg_enabled:
                from vigilance.quality.quality_gate import run_quality_gate

                qg_result = run_quality_gate(
                    indicators_path=indicators_path,
                    footnotes_path=footnotes_path,
                    out_dir=out_dir,
                    bank_code=bank_code,
                    run_id=extraction_run_id,
                    config=qg_cfg,
                )
                quality_gate_status = {
                    "enabled": True,
                    **qg_result,
                }
    except Exception as exc:
        if qg_enabled:
            quality_gate_status = {
                "enabled": True,
                "status": "FAIL",
                "eligible_for_review": True,
                "fail_reasons": [f"quality_gate_execution_error({exc})"],
            }
        logger.warning("Extraction writer/quality gate skipped: %s", exc)

    # Extraction certification: always compute for diagnostics; enforce (fail run) only when gate enabled.
    try:
        from vigilance.quality.quality_gate import evaluate_extraction_quality

        extraction_report = evaluate_extraction_quality(
            list(tables_t1) + list(tables_t2),
            config=get_quality_gate_config(bank_code=bank_code) if qg_enabled else None,
        )
        quality_gate_status["extraction_certification"] = extraction_report.get("summary", {})
        quality_gate_status["blocker_breakdown"] = extraction_report.get("blocker_breakdown") or {}
        quality_gate_status["blocked_table_evidence"] = extraction_report.get("blocked_table_evidence") or []
        if qg_enabled and extraction_report.get("status") == "FAIL":
            quality_gate_status["status"] = "FAIL"
            quality_gate_status["fail_reasons"] = list(
                quality_gate_status.get("fail_reasons", [])
            ) + list(extraction_report.get("fail_reasons", []))
        # eligible_for_review is never set to False: quality gate is diagnostic only, does not block review queue or exports.
    except Exception as exc:
        logger.warning("Extraction certification evaluation skipped: %s", exc)

    embedding_service = None
    if cfg.get("use_embeddings") and api_key:
        try:
            from vigilance.embedding import EmbeddingService

            embedding_service = EmbeddingService(
                api_key=api_key,
                model=str(cfg.get("embedding_model", "text-embedding-3-small")),
            )
        except Exception as e:
            logger.warning("EmbeddingService init failed: %s", e)
            embedding_service = None

    strict = run_strict_intra_section_compare(
        tables_t1=tables_t1,
        tables_t2=tables_t2,
        bank_code=bank_code,
        embedding_service=embedding_service,
        api_key=api_key,
    )

    t1_by_uid = {_section_uid(table): table for table in tables_t1}
    t2_by_uid = {_section_uid(table): table for table in tables_t2}

    # --- Phase 2: Semantic Judge (GPT-4o) for allowed banks only ---
    semantic_judge_enabled = False
    semantic_judge_results: dict[str, dict[str, Any]] = {}
    semantic_judge_stats = {"calls": 0, "errors": 0, "overrides": 0}
    val_cfg_early = get_validation_config(bank_code=bank_code) or {}
    sj_config_enabled = val_cfg_early.get("semantic_judge_enabled")
    sj_banks = val_cfg_early.get("semantic_judge_banks")
    if api_key:
        try:
            from vigilance.genai.semantic_judge import (
                _needs_semantic_validation,
                _needs_semantic_validation_unmatched,
                is_bank_allowed,
                run_semantic_judge_for_pair,
                run_semantic_judge_for_unmatched,
            )

            if sj_config_enabled is False:
                semantic_judge_enabled = False
            elif sj_config_enabled is True:
                semantic_judge_enabled = (
                    is_bank_allowed(bank_code, allowed_banks=sj_banks)
                    if sj_banks
                    else is_bank_allowed(bank_code)
                )
            else:
                semantic_judge_enabled = is_bank_allowed(bank_code)
        except ImportError:
            semantic_judge_enabled = False

    if semantic_judge_enabled:
        logger.info("Semantic judge enabled for bank=%s", bank_code)
        for pair in strict.get("pairs", []):
            t1_uid = str(pair.get("t1_uid", ""))
            t2_uid = str(pair.get("t2_uid", ""))
            tbl_t1 = t1_by_uid.get(t1_uid)
            tbl_t2 = t2_by_uid.get(t2_uid)
            if tbl_t1 is None or tbl_t2 is None:
                continue

            pair_overlap = _compute_indicator_overlap(tbl_t1, tbl_t2)
            susp, _ = _pre_diff_safety_check(tbl_t1, tbl_t2, pair_overlap)

            if not _needs_semantic_validation(pair, pair_overlap, susp):
                continue

            try:
                judge_result = run_semantic_judge_for_pair(
                    bank_code=bank_code,
                    api_key=api_key,
                    table_t1=tbl_t1,
                    table_t2=tbl_t2,
                    pair=pair,
                    indicator_overlap=pair_overlap,
                    suspicious_low_overlap=susp,
                    t1_uid=t1_uid,
                    t2_uid=t2_uid,
                    strict_result=strict,
                    t2_by_uid=t2_by_uid,
                    log_path=_SEMANTIC_JUDGE_LOG,
                )
                semantic_judge_results[t1_uid] = judge_result
                semantic_judge_stats["calls"] += 1
                if judge_result.get("guard_action") == "structural_override":
                    semantic_judge_stats["overrides"] += 1
            except Exception as exc:
                logger.warning("Semantic judge failed for t1=%s: %s", t1_uid, exc)
                semantic_judge_stats["errors"] += 1

        for item in strict.get("added_tables", []):
            if not _needs_semantic_validation_unmatched("table_added"):
                continue
            t2_uid_added = str(item.get("t2_uid", ""))
            tbl = t2_by_uid.get(t2_uid_added)
            if tbl is None:
                continue
            try:
                judge_result = run_semantic_judge_for_unmatched(
                    bank_code=bank_code,
                    api_key=api_key,
                    unmatched_table=tbl,
                    change_type="table_added",
                    opposite_tables=tables_t1,
                    strict_result=strict,
                    t_by_uid=t1_by_uid,
                    log_path=_SEMANTIC_JUDGE_LOG,
                )
                semantic_judge_results[f"added_{t2_uid_added}"] = judge_result
                semantic_judge_stats["calls"] += 1
            except Exception as exc:
                logger.warning(
                    "Semantic judge (added) failed for t2=%s: %s", t2_uid_added, exc
                )
                semantic_judge_stats["errors"] += 1

        for item in strict.get("removed_tables", []):
            if not _needs_semantic_validation_unmatched("table_removed"):
                continue
            t1_uid_removed = str(item.get("t1_uid", ""))
            tbl = t1_by_uid.get(t1_uid_removed)
            if tbl is None:
                continue
            try:
                judge_result = run_semantic_judge_for_unmatched(
                    bank_code=bank_code,
                    api_key=api_key,
                    unmatched_table=tbl,
                    change_type="table_removed",
                    opposite_tables=tables_t2,
                    strict_result=strict,
                    t_by_uid=t2_by_uid,
                    log_path=_SEMANTIC_JUDGE_LOG,
                )
                semantic_judge_results[f"removed_{t1_uid_removed}"] = judge_result
                semantic_judge_stats["calls"] += 1
            except Exception as exc:
                logger.warning(
                    "Semantic judge (removed) failed for t1=%s: %s", t1_uid_removed, exc
                )
                semantic_judge_stats["errors"] += 1

        if semantic_judge_stats["calls"] > 0:
            logger.info(
                "Semantic judge complete: %d calls, %d errors, %d structural overrides",
                semantic_judge_stats["calls"],
                semantic_judge_stats["errors"],
                semantic_judge_stats["overrides"],
            )

    comparisons: list[dict[str, Any]] = []
    rejected_by_vision_pair: list[dict[str, Any]] = []
    vision_rejected_added_items: list[dict[str, Any]] = []
    vision_rejected_removed_items: list[dict[str, Any]] = []
    vision_pair_stats: dict[str, int] = {
        "calls": 0,
        "rejected": 0,
        "accepted": 0,
        "errors": 0,
    }
    rename_validator_stats: dict[str, int | float] = {
        "calls": 0,
        "pairs_validated": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "candidates_in_band": 0,
        "auto_accepted_out_of_band": 0,
    }
    added_table_validator_stats: dict[str, int] = {
        "calls": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
    }
    indicator_validator_stats: dict[str, Any] = {
        "enabled": False,
        "calls": 0,
        "filtered_added": 0,
        "filtered_removed": 0,
        "errors": 0,
        "use_vision": False,
        "vision_fallback_count": 0,
    }
    table_pair_embed_debug: list[dict[str, Any]] = []
    all_rename_pair_debug: list[dict[str, Any]] = []
    all_unmatched_indicator_candidates: list[dict[str, Any]] = []
    try:
        from vigilance.config import get_vision_extraction_config as _gvec

        vec = _gvec(bank_code=bank_code) or {}
    except Exception:
        vec = {}
    val_cfg = get_validation_config(bank_code=bank_code) or {}
    vision_pair_validation = val_cfg.get(
        "vision_pair_validation", vec.get("vision_pair_validation", False)
    )
    vision_pair_confidence_min = float(val_cfg.get("vision_pair_confidence_min", 0.75))
    rename_validator_enabled = bool(val_cfg.get("rename_validator_enabled", False))
    rename_validator_confidence_min = float(
        val_cfg.get("rename_validator_confidence_min", 0.8)
    )
    rename_validator_batch_size = int(val_cfg.get("rename_validator_batch_size", 10))
    rename_band_raw = val_cfg.get("rename_validator_uncertain_score_band", [0.85, 0.95])
    rename_band_min = 0.85
    rename_band_max = 0.95
    if isinstance(rename_band_raw, (list, tuple)) and len(rename_band_raw) == 2:
        try:
            rename_band_min = float(rename_band_raw[0])
            rename_band_max = float(rename_band_raw[1])
        except (TypeError, ValueError):
            rename_band_min, rename_band_max = 0.85, 0.95
    rename_band_min = max(0.0, min(1.0, rename_band_min))
    rename_band_max = max(0.0, min(1.0, rename_band_max))
    if rename_band_min > rename_band_max:
        rename_band_min, rename_band_max = rename_band_max, rename_band_min
    added_table_validator_enabled = bool(
        val_cfg.get("added_table_validator_enabled", False)
    )
    added_table_validator_confidence_min = float(
        val_cfg.get("added_table_validator_confidence_min", 0.75)
    )
    indicator_validator_enabled = bool(
        val_cfg.get("indicator_validator_enabled", False)
    )
    indicator_validator_use_vision = bool(
        val_cfg.get("indicator_validator_use_vision", True)
    )
    indicator_validator_confidence_min = float(
        val_cfg.get("indicator_validator_confidence_min", 0.8)
    )
    indicator_validator_batch_size = int(
        val_cfg.get("indicator_validator_batch_size", 8)
    )
    bottom_ext = float(vec.get("bottom_extension_footnotes", 0.12))

    # Optional post-matching rescue over unmatched tables (usually small set).
    vision_unmatched_rescue_enabled = bool(
        val_cfg.get("vision_unmatched_rescue_enabled", True)
    )
    vision_unmatched_rescue_confidence_min = float(
        val_cfg.get(
            "vision_unmatched_rescue_confidence_min",
            vision_pair_confidence_min,
        )
    )
    vision_unmatched_rescue_max_pairs = int(
        val_cfg.get("vision_unmatched_rescue_max_pairs", 25)
    )
    vision_unmatched_rescue_max_candidates_per_table = int(
        val_cfg.get("vision_unmatched_rescue_max_candidates_per_table", 3)
    )
    vision_unmatched_rescue_max_tables_per_run = int(
        val_cfg.get("vision_unmatched_rescue_max_tables_per_run", 20)
    )
    cross_section_rescue_enabled = bool(
        val_cfg.get("cross_section_rescue_enabled", False)
    )
    cross_section_rescue_rerank_min = float(
        val_cfg.get("cross_section_rescue_rerank_min", 0.30)
    )
    cross_section_rescue_vision_confidence_min = float(
        val_cfg.get("cross_section_rescue_vision_confidence_min", 0.85)
    )
    vision_unmatched_rescue_summary: dict[str, Any] = {
        "enabled": bool(vision_unmatched_rescue_enabled),
        "candidate_pairs_tested": 0,
        "candidate_matches": 0,
        "rescued_pairs": 0,
        "conflicts": 0,
        "errors": 0,
        "candidate_pairs_considered": 0,
        "candidate_tables_considered": 0,
        "vision_rejected_pairs": 0,
        "vision_unresolved_pairs": 0,
        "metrics": {},
    }

    if vision_unmatched_rescue_enabled and api_key:
        try:
            from vigilance.extraction.vision_pair_validator import (
                AssignmentEntry,
                VisionMetrics,
                resolve_bijective_assignment,
                validate_pair_full,
            )

            candidate_pairs, source_tables_considered = (
                _collect_vision_rescue_candidates(
                    strict=strict,
                    t1_by_uid=t1_by_uid,
                    t2_by_uid=t2_by_uid,
                    max_candidates_per_table=max(
                        1, vision_unmatched_rescue_max_candidates_per_table
                    ),
                    max_tables_per_run=max(
                        0, vision_unmatched_rescue_max_tables_per_run
                    ),
                    cross_section_rescue_enabled=cross_section_rescue_enabled,
                    cross_section_rerank_min=cross_section_rescue_rerank_min,
                )
            )
            if candidate_pairs:
                pairs_budget = max(0, vision_unmatched_rescue_max_pairs)
                if pairs_budget > 0:
                    candidate_pairs = candidate_pairs[:pairs_budget]
                tested_pairs = 0
                scored_candidates: list[AssignmentEntry] = []
                metrics = VisionMetrics()
                rescue_results: list[dict[str, Any]] = []
                attempted_t1_uids: set[str] = set()
                attempted_t2_uids: set[str] = set()
                uncertain_t1_uids: set[str] = set()
                uncertain_t2_uids: set[str] = set()
                rejected_pairs: list[dict[str, Any]] = []

                for candidate in candidate_pairs:
                    t1_uid_r = str(candidate.get("t1_uid", "")).strip()
                    t2_uid_a = str(candidate.get("t2_uid", "")).strip()
                    t1_tbl = t1_by_uid.get(t1_uid_r)
                    t2_tbl = t2_by_uid.get(t2_uid_a)
                    if t1_tbl is None or t2_tbl is None:
                        continue
                    bbox_t1 = _normalize_bbox_ltrb_norm(getattr(t1_tbl, "bbox", None))
                    bbox_t2 = _normalize_bbox_ltrb_norm(getattr(t2_tbl, "bbox", None))
                    pdf_t1 = t1_tbl.pdf_path or pdf_path_t1
                    pdf_t2 = t2_tbl.pdf_path or pdf_path_t2
                    if not bbox_t1 or not bbox_t2 or not pdf_t1 or not pdf_t2:
                        continue

                    vd = validate_pair_full(
                        pdf_t1,
                        t1_tbl.page_pdf,
                        bbox_t1,
                        pdf_t2,
                        t2_tbl.page_pdf,
                        bbox_t2,
                        api_key,
                        bottom_extension=bottom_ext,
                        title_t1=t1_tbl.title or None,
                        title_t2=t2_tbl.title or None,
                        section_t1=getattr(t1_tbl, "section", None),
                        section_t2=getattr(t2_tbl, "section", None),
                    )
                    tested_pairs += 1
                    attempted_t1_uids.add(t1_uid_r)
                    attempted_t2_uids.add(t2_uid_a)
                    metrics.record_decision(vd, is_rescue=True)

                    result_entry = {
                        **candidate,
                        "decision": vd.decision,
                        "confidence": round(float(vd.confidence or 0.0), 4),
                        "reason_code": getattr(vd, "reason_code", "") or "",
                        "analysis": dict(getattr(vd, "analysis", {}) or {}),
                    }
                    rescue_results.append(result_entry)

                    if (
                        vd.decision == "match"
                        and vd.confidence >= vision_unmatched_rescue_confidence_min
                    ):
                        from vigilance.models.table_models import (
                            get_extraction_quality_flags,
                        )

                        is_cross = candidate.get("is_cross_section", False)
                        flags1 = get_extraction_quality_flags(t1_tbl)
                        flags2 = get_extraction_quality_flags(t2_tbl)
                        low_q1 = (
                            flags1.get("crop_rejected")
                            or flags1.get("recrop_failed_incomplete")
                            or not flags1.get("vision_extraction_applied", True)
                        )
                        low_q2 = (
                            flags2.get("crop_rejected")
                            or flags2.get("recrop_failed_incomplete")
                            or not flags2.get("vision_extraction_applied", True)
                        )
                        allow_rescue = True
                        if low_q1 or low_q2:
                            if is_cross:
                                allow_rescue = (
                                    vd.confidence
                                    >= cross_section_rescue_vision_confidence_min
                                )
                            else:
                                table_num_match = (
                                    str(
                                        getattr(t1_tbl, "table_number", "") or ""
                                    ).strip()
                                    == str(
                                        getattr(t2_tbl, "table_number", "") or ""
                                    ).strip()
                                    and (
                                        getattr(t1_tbl, "table_number", None) or ""
                                    ).strip()
                                )
                                section_match = (
                                    str(getattr(t1_tbl, "section", "") or "").strip()
                                    == str(
                                        getattr(t2_tbl, "section", "") or ""
                                    ).strip()
                                )
                                allow_rescue = table_num_match and section_match
                        if allow_rescue:
                            rescue_source = (
                                "cross_section_vision_rescue"
                                if is_cross
                                else "vision_unmatched_rescue"
                            )
                            scored_candidates.append(
                                AssignmentEntry(
                                    t1_uid=t1_uid_r,
                                    t2_uid=t2_uid_a,
                                    confidence=float(vd.confidence),
                                    decision=vd,
                                    source=rescue_source,
                                )
                            )
                    elif vd.decision == "no_match":
                        rejected_pairs.append(result_entry)
                    else:
                        uncertain_t1_uids.add(t1_uid_r)
                        uncertain_t2_uids.add(t2_uid_a)

                vision_unmatched_rescue_summary["candidate_pairs_considered"] = len(
                    candidate_pairs
                )
                vision_unmatched_rescue_summary["candidate_tables_considered"] = int(
                    source_tables_considered
                )
                vision_unmatched_rescue_summary["candidate_pairs_tested"] = tested_pairs
                vision_unmatched_rescue_summary["candidate_matches"] = len(
                    scored_candidates
                )
                vision_unmatched_rescue_summary["vision_rejected_pairs"] = len(
                    rejected_pairs
                )

                rescued_pairs: list[dict[str, Any]] = []
                cross_section_rescued_pairs: list[dict[str, Any]] = []
                rescued_t1_uids: set[str] = set()
                rescued_t2_uids: set[str] = set()

                if scored_candidates:
                    bijection = resolve_bijective_assignment(scored_candidates)
                    metrics.record_bijection(bijection)
                    strict["vision_unmatched_rescue_conflicts"] = list(
                        bijection.conflicts
                    )
                    vision_unmatched_rescue_summary["conflicts"] = len(
                        bijection.conflicts
                    )
                    for ass in bijection.assigned_pairs:
                        rescued_t1_uids.add(ass.t1_uid)
                        rescued_t2_uids.add(ass.t2_uid)
                        is_cross = ass.source == "cross_section_vision_rescue"
                        pair_entry = {
                            "t1_uid": ass.t1_uid,
                            "t2_uid": ass.t2_uid,
                            "score": float(ass.confidence),
                            "reason": (
                                "cross_section_vision_rescue"
                                if is_cross
                                else "vision_unmatched_rescue"
                            ),
                            "rescue_type": (
                                "cross_section_vision_rescue"
                                if is_cross
                                else "vision_unmatched_rescue"
                            ),
                            "decision_level": "rescue",
                            "match_source": (
                                "cross_section_vision_rescue"
                                if is_cross
                                else "vision_unmatched_rescue"
                            ),
                            "match_stage": (
                                "cross_section_rescue"
                                if is_cross
                                else "vision_rescue"
                            ),
                        }
                        rescued_pairs.append(pair_entry)
                        if is_cross:
                            cross_section_rescued_pairs.append(pair_entry)

                strict["cross_section_rescued_pairs"] = cross_section_rescued_pairs
                uncertain_t1_uids.difference_update(rescued_t1_uids)
                uncertain_t2_uids.difference_update(rescued_t2_uids)

                if rescued_pairs:
                    strict["pairs"] = list(strict.get("pairs", [])) + rescued_pairs
                    strict["rescued_matches_count"] = int(
                        strict.get("rescued_matches_count", 0) or 0
                    ) + len(rescued_pairs)
                vision_unmatched_rescue_summary["rescued_pairs"] = len(rescued_pairs)
                vision_unmatched_rescue_summary["vision_unresolved_pairs"] = sum(
                    1 for item in rescue_results if item.get("decision") == "unknown"
                )
                strict["vision_rescued_pairs"] = rescued_pairs
                strict["vision_unmatched_rescue_candidates"] = candidate_pairs
                strict["vision_unmatched_rescue_results"] = rescue_results
                strict["vision_rejected_pairs"] = rejected_pairs
                _rebuild_strict_unmatched_state(
                    strict=strict,
                    rescued_t1_uids=rescued_t1_uids,
                    rescued_t2_uids=rescued_t2_uids,
                    uncertain_t1_uids=uncertain_t1_uids,
                    uncertain_t2_uids=uncertain_t2_uids,
                )
                vision_unmatched_rescue_summary["metrics"] = metrics.as_dict()
        except Exception as exc:
            vision_unmatched_rescue_summary["errors"] = (
                vision_unmatched_rescue_summary.get("errors", 0) + 1
            )
            logger.warning("Vision unmatched rescue failed: %s", exc)

    strict.setdefault("vision_rescued_pairs", [])
    strict.setdefault("cross_section_rescued_pairs", [])
    strict.setdefault("vision_unmatched_rescue_candidates", [])
    strict.setdefault("vision_unmatched_rescue_results", [])
    strict.setdefault("vision_rejected_pairs", [])
    strict.setdefault(
        "added_tables_confirmed", list(strict.get("added_tables", []) or [])
    )
    strict.setdefault(
        "removed_tables_confirmed", list(strict.get("removed_tables", []) or [])
    )
    strict.setdefault(
        "ambiguous_tables",
        [
            {
                "side": "previous",
                "uid": str(item.get("t1_uid", "")).strip(),
                "table_id": item.get("t1_table_id"),
                "title": item.get("title_t1"),
                "page": item.get("page_t1"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
                "suspicion_flags": list(item.get("suspicion_flags", []) or []),
            }
            for item in strict.get("ambiguous_unmatched_previous", []) or []
        ]
        + [
            {
                "side": "current",
                "uid": str(item.get("t2_uid", "")).strip(),
                "table_id": item.get("t2_table_id"),
                "title": item.get("title_t2"),
                "page": item.get("page_t2"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
                "suspicion_flags": list(item.get("suspicion_flags", []) or []),
            }
            for item in strict.get("ambiguous_unmatched_current", []) or []
        ],
    )

    for pair in strict.get("pairs", []):
        t1_uid = str(pair.get("t1_uid", ""))
        t2_uid = str(pair.get("t2_uid", ""))
        table_t1 = t1_by_uid.get(t1_uid)
        table_t2 = t2_by_uid.get(t2_uid)
        if table_t1 is None or table_t2 is None:
            continue

        # Part D & E: compute overlap and run pre-diff safety check
        pair_indicator_overlap = _compute_indicator_overlap(table_t1, table_t2)
        suspicious_low_overlap, suspicious_reason = _pre_diff_safety_check(
            table_t1, table_t2, pair_indicator_overlap
        )
        rescue_type = pair.get("rescue_type")

        # Validate table pair with Vision before indicator diff.
        vision_rejected = False
        if (
            vision_pair_validation
            and api_key
            and (pair_indicator_overlap < 0.5 or rescue_type)
            and rescue_type
            != "vision_unmatched_rescue"  # already validated during rescue
        ):
            bbox_t1 = _normalize_bbox_ltrb_norm(getattr(table_t1, "bbox", None))
            bbox_t2 = _normalize_bbox_ltrb_norm(getattr(table_t2, "bbox", None))
            pdf_t1 = table_t1.pdf_path or pdf_path_t1
            pdf_t2 = table_t2.pdf_path or pdf_path_t2
            if bbox_t1 and bbox_t2 and pdf_t1 and pdf_t2:
                try:
                    from vigilance.extraction.vision_pair_validator import (
                        validate_pair_same_concept,
                    )

                    same_concept, confidence = validate_pair_same_concept(
                        pdf_t1,
                        table_t1.page_pdf,
                        bbox_t1,
                        pdf_t2,
                        table_t2.page_pdf,
                        bbox_t2,
                        api_key,
                        bottom_extension=bottom_ext,
                        title_t1=table_t1.title or None,
                        title_t2=table_t2.title or None,
                        section_t1=getattr(table_t1, "section", None),
                        section_t2=getattr(table_t2, "section", None),
                    )
                    vision_pair_stats["calls"] += 1
                    if same_concept:
                        vision_pair_stats["accepted"] += 1
                    elif confidence >= vision_pair_confidence_min:
                        vision_rejected = True
                        vision_pair_stats["rejected"] += 1
                        rejected_by_vision_pair.append(
                            {
                                "table_id_t1": table_t1.table_id,
                                "table_id_t2": table_t2.table_id,
                                "title_t1": table_t1.title or "",
                                "title_t2": table_t2.title or "",
                                "indicator_overlap": pair_indicator_overlap,
                                "rescue_type": rescue_type,
                                "confidence": round(confidence, 3),
                            }
                        )
                        _write_validation_log(
                            {
                                "validator": "vision_pair",
                                "bank": bank_code,
                                "t1_uid": t1_uid,
                                "t2_uid": t2_uid,
                                "decision": "rejected",
                                "same_concept": same_concept,
                                "confidence": round(confidence, 3),
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                            },
                            run_id=extraction_run_id,
                        )
                        vision_rejected_added_items.append(
                            {
                                "t2_uid": t2_uid,
                                "t2_table_id": table_t2.table_id,
                                "section": table_t1.section or table_t2.section or "",
                                "page_t2": table_t2.page_pdf,
                                "title_t2": table_t2.title or "",
                                "reason": "unmatched",
                                "source_reason": "vision_pair_rejected",
                                "first_column_indicators": list(
                                    getattr(table_t2, "first_column_indicators", [])
                                    or []
                                ),
                                "first_column_indicators_raw": list(
                                    getattr(
                                        table_t2, "first_column_indicators_raw", None
                                    )
                                    or []
                                ),
                            }
                        )
                        vision_rejected_removed_items.append(
                            {
                                "t1_uid": t1_uid,
                                "t1_table_id": table_t1.table_id,
                                "section": table_t1.section or table_t2.section or "",
                                "page_t1": table_t1.page_pdf,
                                "title_t1": table_t1.title or "",
                                "reason": "unmatched",
                                "source_reason": "vision_pair_rejected",
                                "first_column_indicators": list(
                                    getattr(table_t1, "first_column_indicators", [])
                                    or []
                                ),
                                "first_column_indicators_raw": list(
                                    getattr(
                                        table_t1, "first_column_indicators_raw", None
                                    )
                                    or []
                                ),
                            }
                        )
                    else:
                        vision_pair_stats["accepted"] += 1
                except Exception as exc:
                    vision_pair_stats["errors"] += 1
                    logger.debug("Vision pair validation error: %s", exc)

        if vision_rejected:
            continue

        indicator_diff_debug_enabled = (
            cfg.get("indicator_diff_debug", False)
            or os.environ.get("INDICATOR_DIFF_DEBUG", "").strip().lower() in _ENV_TRUE
        )
        added, removed, had_fusion_split, excluded_counts, diff_debug_info = (
            _indicator_diff(
                table_t1,
                table_t2,
                neighbor_aligned_filter_enabled=cfg.get(
                    "neighbor_aligned_filter_enabled", True
                ),
                return_debug=indicator_diff_debug_enabled,
                th=cfg,
            )
        )
        t1_clean_to_raw = _build_clean_to_raw_indicator_lookup(table_t1)
        t2_clean_to_raw = _build_clean_to_raw_indicator_lookup(table_t2)
        use_hungarian = cfg.get("indicator_hungarian_enabled", True)
        indicator_debug: dict[str, Any] | None = None
        if use_hungarian:
            added, removed, renamed_pairs, indicator_debug = (
                _hungarian_pair_added_removed(
                    removed, added, th=cfg, embedding_service=embedding_service
                )
            )
            if indicator_debug:
                rpd = indicator_debug.get("rename_pair_debug") or []
                for e in rpd:
                    e_with_ctx = dict(e)
                    e_with_ctx["table_id_t1"] = table_t1.table_id
                    e_with_ctx["table_id_t2"] = table_t2.table_id
                    all_rename_pair_debug.append(e_with_ctx)
                unmatched = (
                    indicator_debug.get("unmatched_removed_with_candidates") or []
                )
                if unmatched:
                    all_unmatched_indicator_candidates.append(
                        {
                            "table_id_t1": table_t1.table_id,
                            "table_id_t2": table_t2.table_id,
                            "unmatched_removed_with_top_candidates": unmatched,
                        }
                    )
            if indicator_debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "indicator_pairing %s:%s gated=%s renames=%s scores=%s",
                    table_t1.section,
                    table_t1.table_id,
                    indicator_debug.get("gated_out_pairs"),
                    indicator_debug.get("accepted_renames"),
                    indicator_debug.get("score_distribution"),
                )
        else:
            added, removed, renamed_pairs = _fuzzy_pair_added_removed(
                added, removed, bank_code
            )

        if rename_validator_enabled and api_key and renamed_pairs:
            try:
                from vigilance.genai import validate_rename_pairs

                pair_score_map: dict[tuple[str, str], float] = {}
                if use_hungarian and indicator_debug:
                    rpd = indicator_debug.get("rename_pair_debug") or []
                    if len(rpd) == len(renamed_pairs):
                        for (old_l, new_l), dbg in zip(renamed_pairs, rpd):
                            try:
                                score = float(dbg.get("final_score", 0.0))
                                if score > 1.0:
                                    score = score / 100.0
                                pair_score_map[(old_l, new_l)] = max(
                                    0.0, min(1.0, score)
                                )
                            except (TypeError, ValueError):
                                continue

                candidates_for_validation: list[tuple[str, str]] = []
                auto_accepted_pairs: list[tuple[str, str]] = []
                original_renamed_pairs = list(renamed_pairs)
                for old_l, new_l in renamed_pairs:
                    pair_key = (old_l, new_l)
                    score = pair_score_map.get(pair_key)
                    if score is not None and not (
                        rename_band_min <= score <= rename_band_max
                    ):
                        auto_accepted_pairs.append(pair_key)
                    else:
                        candidates_for_validation.append(pair_key)

                rename_validator_stats["candidates_in_band"] = (
                    rename_validator_stats.get("candidates_in_band", 0)
                    + len(candidates_for_validation)
                )
                rename_validator_stats["auto_accepted_out_of_band"] = (
                    rename_validator_stats.get("auto_accepted_out_of_band", 0)
                    + len(auto_accepted_pairs)
                )

                accepted_pairs: list[tuple[str, str]] = []
                rejected_pairs: list[tuple[str, str]] = []
                rv_stats: dict[str, Any] = {
                    "calls": 0,
                    "pairs_validated": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "errors": 0,
                }
                if candidates_for_validation:
                    accepted_pairs, rejected_pairs, rv_stats = validate_rename_pairs(
                        candidates_for_validation,
                        api_key=api_key,
                        batch_size=rename_validator_batch_size,
                        confidence_min=rename_validator_confidence_min,
                    )

                rename_validator_stats["calls"] = rename_validator_stats.get(
                    "calls", 0
                ) + rv_stats.get("calls", 0)
                rename_validator_stats["pairs_validated"] = rename_validator_stats.get(
                    "pairs_validated", 0
                ) + rv_stats.get("pairs_validated", 0)
                rename_validator_stats["accepted"] = rename_validator_stats.get(
                    "accepted", 0
                ) + rv_stats.get("accepted", 0)
                rename_validator_stats["rejected"] = rename_validator_stats.get(
                    "rejected", 0
                ) + rv_stats.get("rejected", 0)
                rename_validator_stats["errors"] = rename_validator_stats.get(
                    "errors", 0
                ) + rv_stats.get("errors", 0)

                accepted_set = set(accepted_pairs)
                auto_accepted_set = set(auto_accepted_pairs)
                renamed_pairs = []
                for pair_candidate in original_renamed_pairs:
                    if (
                        pair_candidate in auto_accepted_set
                        or pair_candidate in accepted_set
                    ):
                        renamed_pairs.append(pair_candidate)

                for r_label, a_label in rejected_pairs:
                    added.append(a_label)
                    removed.append(r_label)
                if rejected_pairs:
                    _write_validation_log(
                        {
                            "validator": "rename",
                            "bank": bank_code,
                            "table_id_t1": table_t1.table_id,
                            "table_id_t2": table_t2.table_id,
                            "rejected_count": len(rejected_pairs),
                            "sample_rejected": [
                                {"from": r[:60], "to": a[:60]}
                                for (r, a) in rejected_pairs[:3]
                            ],
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        },
                        run_id=extraction_run_id,
                    )
            except Exception as exc:
                logger.debug("Rename validator error: %s", exc)
                rename_validator_stats["errors"] = (
                    rename_validator_stats.get("errors", 0) + 1
                )

        if indicator_validator_enabled and api_key and (added or removed):
            indicator_validator_stats["enabled"] = True
            indicator_validator_stats["use_vision"] = indicator_validator_use_vision
            added_count_before = len(added)
            removed_count_before = len(removed)
            all_t1 = get_comparison_indicators(table_t1)
            all_t2 = get_comparison_indicators(table_t2)
            pdf_t1 = table_t1.pdf_path or pdf_path_t1
            pdf_t2 = table_t2.pdf_path or pdf_path_t2
            try:
                if indicator_validator_use_vision:
                    from vigilance.extraction.vision_indicator_added_validator import (
                        try_vision_validate_indicators,
                    )

                    (
                        added,
                        removed,
                        vision_stats,
                    ) = try_vision_validate_indicators(
                        added,
                        removed,
                        table_t1,
                        table_t2,
                        pdf_t1,
                        pdf_t2,
                        api_key,
                        indicator_validator_confidence_min,
                    )
                    indicator_validator_stats["calls"] += vision_stats.get(
                        "vision_calls", 0
                    )
                    indicator_validator_stats["filtered_added"] += vision_stats.get(
                        "vision_filtered_added", 0
                    )
                    indicator_validator_stats["filtered_removed"] += vision_stats.get(
                        "vision_filtered_removed", 0
                    )
                    if vision_stats.get("vision_fallback_reason"):
                        indicator_validator_stats["vision_fallback_count"] = (
                            indicator_validator_stats.get("vision_fallback_count", 0)
                            + 1
                        )
                    could_validate_added = vision_stats.get(
                        "could_validate_added", False
                    )
                    could_validate_removed = vision_stats.get(
                        "could_validate_removed", False
                    )
                    need_genai_fallback = (added and not could_validate_added) or (
                        removed and not could_validate_removed
                    )
                    if need_genai_fallback and (added or removed):
                        from vigilance.genai import validate_indicator_added_removed

                        added, removed, genai_stats = validate_indicator_added_removed(
                            added,
                            removed,
                            all_t1,
                            all_t2,
                            api_key=api_key,
                            batch_size=indicator_validator_batch_size,
                            confidence_min=indicator_validator_confidence_min,
                        )
                        indicator_validator_stats["calls"] += genai_stats.get(
                            "calls", 0
                        )
                        indicator_validator_stats["filtered_added"] += genai_stats.get(
                            "filtered_added", 0
                        )
                        indicator_validator_stats["filtered_removed"] += (
                            genai_stats.get("filtered_removed", 0)
                        )
                        indicator_validator_stats["errors"] += genai_stats.get(
                            "errors", 0
                        )
                else:
                    from vigilance.genai import validate_indicator_added_removed

                    added, removed, genai_stats = validate_indicator_added_removed(
                        added,
                        removed,
                        all_t1,
                        all_t2,
                        api_key=api_key,
                        batch_size=indicator_validator_batch_size,
                        confidence_min=indicator_validator_confidence_min,
                    )
                    indicator_validator_stats["calls"] += genai_stats.get("calls", 0)
                    indicator_validator_stats["filtered_added"] += genai_stats.get(
                        "filtered_added", 0
                    )
                    indicator_validator_stats["filtered_removed"] += genai_stats.get(
                        "filtered_removed", 0
                    )
                    indicator_validator_stats["errors"] += genai_stats.get("errors", 0)
                filtered_added_this = added_count_before - len(added)
                filtered_removed_this = removed_count_before - len(removed)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "indicator_validator pair t1=%s t2=%s before: added=%d removed=%d -> after: added=%d removed=%d (filtered_added=%d filtered_removed=%d)",
                        table_t1.table_id,
                        table_t2.table_id,
                        added_count_before,
                        removed_count_before,
                        len(added),
                        len(removed),
                        filtered_added_this,
                        filtered_removed_this,
                    )
                if filtered_added_this or filtered_removed_this:
                    _write_validation_log(
                        {
                            "validator": "indicator_added_removed",
                            "bank": bank_code,
                            "table_id_t1": table_t1.table_id,
                            "table_id_t2": table_t2.table_id,
                            "filtered_added": filtered_added_this,
                            "filtered_removed": filtered_removed_this,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        },
                        run_id=extraction_run_id,
                    )
            except Exception as exc:
                logger.debug("Indicator validator error: %s", exc)
                indicator_validator_stats["errors"] = (
                    indicator_validator_stats.get("errors", 0) + 1
                )

        renamed_indicators = [{"from": r, "to": a} for (r, a) in renamed_pairs]
        added_indicators_raw = _clean_values_to_raw_display(added, t2_clean_to_raw)
        removed_indicators_raw = _clean_values_to_raw_display(removed, t1_clean_to_raw)
        renamed_indicators_raw = [
            {
                "from": str(
                    t1_clean_to_raw.get(
                        _canonical_indicator_key(
                            strip_footnote_markers_from_indicator(removed_clean)
                        )
                    )
                    or removed_clean
                ),
                "to": str(
                    t2_clean_to_raw.get(
                        _canonical_indicator_key(
                            strip_footnote_markers_from_indicator(added_clean)
                        )
                    )
                    or added_clean
                ),
                "from_clean": removed_clean,
                "to_clean": added_clean,
            }
            for (removed_clean, added_clean) in renamed_pairs
        ]

        table_pair_embed_debug.append(
            {
                "t1_uid": t1_uid,
                "t2_uid": t2_uid,
                "embed_sim_canonical": round(
                    float(pair.get("embed_sim_canon", 0) or 0), 3
                ),
                "embed_sim_token_sorted": round(
                    float(pair.get("embed_sim_token_sorted", 0) or 0), 3
                ),
                "gating_decision": str(pair.get("table_fp_gating") or ""),
                "fingerprint_token_count": int(
                    pair.get("fingerprint_token_count", 0) or 0
                ),
            }
        )
        match_decision_level = str(pair.get("decision_level") or "match")
        conf_t1 = _extraction_confidence(table_t1)
        conf_t2 = _extraction_confidence(table_t2)
        extraction_low_confidence = conf_t1 == "low" or conf_t2 == "low"

        table_status, structure_change_detected = _derive_table_status(
            rescue_type=str(rescue_type or ""),
            extraction_low_confidence=extraction_low_confidence,
            added=added,
            removed=removed,
            renamed_indicators=renamed_indicators,
        )

        uncertain_diff = table_status == "incertain"

        indicator_decisions: list[dict[str, Any]] = []
        if indicator_diff_debug_enabled and diff_debug_info is not None:
            indicator_decisions = _build_indicator_diff_debug(
                table_t1,
                table_t2,
                diff_debug_info["left_map"],
                diff_debug_info["right_map"],
                added,
                removed,
                renamed_pairs,
                t1_clean_to_raw,
                t2_clean_to_raw,
                indicator_debug,
                cfg,
            )
            if logger.isEnabledFor(logging.DEBUG) and indicator_decisions:
                logger.debug(
                    "indicator_diff_debug table_t1=%s table_t2=%s decisions_count=%d",
                    table_t1.table_id,
                    table_t2.table_id,
                    len(indicator_decisions),
                )
            if (
                indicator_decisions
                and os.environ.get("INDICATOR_DIFF_DEBUG_LOG") == "1"
            ):
                try:
                    _INDICATOR_DIFF_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
                    run_id = extraction_run_id or datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    with open(
                        _INDICATOR_DIFF_DEBUG_LOG,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "table_id_t1": table_t1.table_id,
                                    "table_id_t2": table_t2.table_id,
                                    "section": table_t1.section or table_t2.section,
                                    "indicator_decisions": indicator_decisions,
                                    "threshold_rename_min": float(
                                        cfg.get(
                                            "indicator_rename_min_score",
                                            _INDICATOR_DEFAULTS[
                                                "indicator_rename_min_score"
                                            ],
                                        )
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except OSError as e:
                    logger.debug("Could not write indicator_diff_debug log: %s", e)

        # Part E: effective_label_overlap from pair if available
        effective_label_overlap = float(
            pair.get("soft_indicator_overlap", pair_indicator_overlap)
            or pair_indicator_overlap
        )

        qf_t1 = _extract_quality_flags(table_t1)
        qf_t2 = _extract_quality_flags(table_t2)

        dm_t1 = table_t1.debug_metrics or {}
        dm_t2 = table_t2.debug_metrics or {}
        arb_t1 = dm_t1.get("vision_arbitration")
        arb_t2 = dm_t2.get("vision_arbitration")

        comparisons.append(
            {
                "table_id_t1": table_t1.table_id,
                "table_id_t2": table_t2.table_id,
                "title_t1": table_t1.title or "",
                "title_t2": table_t2.title or "",
                "table_title_raw": (
                    getattr(table_t1, "title_raw", None) or table_t1.title or ""
                ),
                "table_number": getattr(table_t2, "table_number", None)
                or getattr(table_t1, "table_number", None),
                "page_t1": table_t1.page_pdf,
                "page_t2": table_t2.page_pdf,
                "section": table_t1.section or table_t2.section,
                "match_score": float(pair.get("score", 0.0) or 0.0),
                "match_quality": "high"
                if float(pair.get("score", 0.0) or 0.0) >= 0.7
                else "medium",
                "match_reason": pair.get("reason", ""),
                "match_decision_level": match_decision_level,
                "rescue_type": rescue_type,
                "added_indicators": added,
                "removed_indicators": removed,
                "renamed_indicators": renamed_indicators,
                "added_indicators_raw": added_indicators_raw,
                "removed_indicators_raw": removed_indicators_raw,
                "renamed_indicators_raw": renamed_indicators_raw,
                "renamed_probable_indicators": [],
                "all_indicators_t1": _all_indicators_value_clean_ordered(table_t1),
                "all_indicators_t2": _all_indicators_value_clean_ordered(table_t2),
                "bbox_t1": _normalize_bbox_ltrb_norm(getattr(table_t1, "bbox", None)),
                "bbox_t2": _normalize_bbox_ltrb_norm(getattr(table_t2, "bbox", None)),
                "indicator_decisions": indicator_decisions,
                "review_reasons": [],
                "uncertain_diff": uncertain_diff,
                "structure_change_detected": structure_change_detected,
                "table_status": table_status,
                "counts": {
                    "added": len(added),
                    "removed": len(removed),
                    "renamed": len(renamed_indicators),
                    "renamed_probable": 0,
                    "excluded_totals": excluded_counts.get("total", 0),
                    "excluded_units": excluded_counts.get("unit", 0),
                    "excluded_dates": excluded_counts.get("date", 0),
                },
                "source_method_t1": table_t1.extraction_method,
                "source_method_t2": table_t2.extraction_method,
                "quality_flags_t1": qf_t1,
                "quality_flags_t2": qf_t2,
                "source_pdf_t1": table_t1.pdf_path or "",
                "source_pdf_t2": table_t2.pdf_path or "",
                "match_metadata": {
                    "indicator_overlap": round(pair_indicator_overlap, 4),
                    "effective_label_overlap": round(effective_label_overlap, 4),
                    "fragmentation_detected_t1": table_t1.fragmentation_detected,
                    "fragmentation_detected_t2": table_t2.fragmentation_detected,
                    "suspicious_low_overlap": suspicious_low_overlap,
                    "suspicious_reason": suspicious_reason
                    if suspicious_low_overlap
                    else None,
                    "semantic_judge": semantic_judge_results.get(t1_uid)
                    if semantic_judge_enabled
                    else None,
                    "extraction_confidence_t1": conf_t1,
                    "extraction_confidence_t2": conf_t2,
                    "quality_flags_t1": qf_t1,
                    "quality_flags_t2": qf_t2,
                    "vision_arbitration_t1": arb_t1,
                    "vision_arbitration_t2": arb_t2,
                    "fusion_split_normalization_applied": had_fusion_split,
                },
            }
        )

        if include_footnotes:
            fn_result = _compare_table_footnotes(table_t1, table_t2)
            comparisons[-1]["footnotes_diff"] = fn_result
            comparisons[-1]["footnotes_counts"] = fn_result["counts"]

        # Part F: structured audit log for each matched pair
        _write_match_decision_log(
            {
                "bank": bank_code,
                "t1_id": table_t1.table_id,
                "t2_id": table_t2.table_id,
                "score": round(float(pair.get("score", 0.0) or 0.0), 4),
                "indicator_overlap": round(pair_indicator_overlap, 4),
                "match_reason": pair.get("reason", ""),
                "fragmentation_detected_t1": table_t1.fragmentation_detected,
                "fragmentation_detected_t2": table_t2.fragmentation_detected,
                "suspicious_low_overlap_flag": suspicious_low_overlap,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def _source_method(uid: str, by_uid: dict) -> str:
        t = by_uid.get(uid)
        return (getattr(t, "extraction_method", None) or "docling") if t else "docling"

    added_tables_sources = (
        list(strict.get("added_tables_confirmed", strict.get("added_tables", [])))
        + vision_rejected_added_items
    )
    removed_tables_sources = (
        list(strict.get("removed_tables_confirmed", strict.get("removed_tables", [])))
        + vision_rejected_removed_items
    )

    tables_added = []
    for item in added_tables_sources:
        t2_uid_added = str(item.get("t2_uid", ""))
        t2_t = t2_by_uid.get(t2_uid_added)
        entry: dict[str, Any] = {
            "table_status": "ajoute",
            "table_id": str(item.get("t2_table_id", "")),
            "title": item.get("title_t2", ""),
            "table_number": getattr(t2_t, "table_number", None) if t2_t else None,
            "page": item.get("page_t2"),
            "section": item.get("section", ""),
            "source_reason": str(item.get("source_reason", "")),
            "source_method": _source_method(t2_uid_added, t2_by_uid),
            "quality_flags": _extract_quality_flags(t2_t) if t2_t else [],
            "extraction_confidence": _extraction_confidence(t2_t)
            if t2_t
            else "unknown",
            "indicators": list(item.get("first_column_indicators", []) or []),
            "first_column_indicators_raw": list(
                item.get("first_column_indicators_raw")
                or (getattr(t2_t, "first_column_indicators_raw", None) or [])
            ),
            "all_indicators_t1": [],
            "all_indicators_t2": _all_indicators_value_clean_ordered(t2_t)
            if t2_t
            else [],
            "bbox_t1": None,
            "bbox_t2": _normalize_bbox_ltrb_norm(getattr(t2_t, "bbox", None))
            if t2_t
            else None,
        }
        sj = semantic_judge_results.get(f"added_{t2_uid_added}")
        if sj is not None:
            entry["semantic_judge"] = sj

        if (
            added_table_validator_enabled
            and api_key
            and entry.get("bbox_t2")
            and entry.get("page")
        ):
            pdf_added = (t2_t.pdf_path if t2_t else None) or pdf_path_t2
            if pdf_added:
                try:
                    from vigilance.extraction.vision_added_table_validator import (
                        validate_added_table,
                    )

                    is_real_new, conf = validate_added_table(
                        pdf_added,
                        int(entry["page"]),
                        entry["bbox_t2"],
                        api_key,
                        bottom_extension=bottom_ext,
                        title=str(entry.get("title", ""))[:200],
                    )
                    added_table_validator_stats["calls"] += 1
                    if not is_real_new and conf >= added_table_validator_confidence_min:
                        added_table_validator_stats["rejected"] += 1
                        _write_validation_log(
                            {
                                "validator": "added_table",
                                "bank": bank_code,
                                "table_id": entry.get("table_id"),
                                "title": str(entry.get("title", ""))[:80],
                                "decision": "rejected",
                                "is_real_new": is_real_new,
                                "confidence": round(conf, 3),
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                            },
                            run_id=extraction_run_id,
                        )
                        continue
                    added_table_validator_stats["accepted"] += 1
                except Exception as exc:
                    added_table_validator_stats["errors"] += 1
                    logger.debug("Added table validator error: %s", exc)

        tables_added.append(entry)
    tables_removed = []
    for item in removed_tables_sources:
        t1_uid_removed = str(item.get("t1_uid", ""))
        t1_t = t1_by_uid.get(t1_uid_removed)
        entry_r: dict[str, Any] = {
            "table_status": "supprime",
            "table_id": str(item.get("t1_table_id", "")),
            "title": item.get("title_t1", ""),
            "table_number": getattr(t1_t, "table_number", None) if t1_t else None,
            "page": item.get("page_t1"),
            "section": item.get("section", ""),
            "source_reason": str(item.get("source_reason", "")),
            "source_method": _source_method(t1_uid_removed, t1_by_uid),
            "quality_flags": _extract_quality_flags(t1_t) if t1_t else [],
            "extraction_confidence": _extraction_confidence(t1_t)
            if t1_t
            else "unknown",
            "indicators": list(item.get("first_column_indicators", []) or []),
            "first_column_indicators_raw": list(
                item.get("first_column_indicators_raw")
                or (getattr(t1_t, "first_column_indicators_raw", None) or [])
            ),
            "all_indicators_t1": _all_indicators_value_clean_ordered(t1_t)
            if t1_t
            else [],
            "all_indicators_t2": [],
            "bbox_t1": _normalize_bbox_ltrb_norm(getattr(t1_t, "bbox", None))
            if t1_t
            else None,
            "bbox_t2": None,
        }
        sj_r = semantic_judge_results.get(f"removed_{t1_uid_removed}")
        if sj_r is not None:
            entry_r["semantic_judge"] = sj_r
        tables_removed.append(entry_r)

    # -- POST-MATCHING GenAI classification (does NOT alter matching results) --
    _GENAI_ANALYSIS_FALLBACK: dict[str, Any] = {
        "relevance": "NON_CLASSIFIE",
        "risk_level": "FAIBLE",
        "confidence": 0.0,
        "justification": "Classification non disponible (erreur ou limite API).",
    }

    if include_genai_classification and api_key:
        try:
            from vigilance.genai import GenAIChangeClassifier

            classifier = GenAIChangeClassifier(api_key=api_key)

            def _has_changes(c: dict[str, Any]) -> bool:
                return bool(
                    c.get("added_indicators")
                    or c.get("removed_indicators")
                    or c.get("renamed_indicators")
                    or c.get("footnotes_counts", {}).get("added", 0)
                    or c.get("footnotes_counts", {}).get("removed", 0)
                    or c.get("footnotes_counts", {}).get("modified", 0)
                )

            to_classify = [(i, c) for i, c in enumerate(comparisons) if _has_changes(c)]
            if to_classify:
                batch_results = classifier.classify_batch([c for _, c in to_classify])
                for (idx, _comp), analysis in zip(to_classify, batch_results):
                    comparisons[idx]["genai_analysis"] = (
                        analysis
                        if isinstance(analysis, dict) and analysis.get("relevance")
                        else _GENAI_ANALYSIS_FALLBACK
                    )
                for idx, _ in to_classify:
                    if not comparisons[idx].get("genai_analysis", {}).get("relevance"):
                        comparisons[idx]["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK

            # Classify added/removed tables (synthetic payloads for same classifier)
            def _synthetic_comp_added(t: dict[str, Any]) -> dict[str, Any]:
                return {
                    "section": t.get("section", ""),
                    "table_title": t.get("title", "") or t.get("table_id", ""),
                    "title_t1": "",
                    "title_t2": t.get("title", "") or t.get("table_id", ""),
                    "table_status": "ajoute",
                    "added_indicators": list(
                        t.get("indicators", []) or t.get("all_indicators_t2", [])
                    )[:30],
                    "removed_indicators": [],
                    "renamed_indicators": [],
                }

            def _synthetic_comp_removed(t: dict[str, Any]) -> dict[str, Any]:
                return {
                    "section": t.get("section", ""),
                    "table_title": t.get("title", "") or t.get("table_id", ""),
                    "title_t1": t.get("title", "") or t.get("table_id", ""),
                    "title_t2": "",
                    "table_status": "supprime",
                    "added_indicators": [],
                    "removed_indicators": list(
                        t.get("indicators", []) or t.get("all_indicators_t1", [])
                    )[:30],
                    "renamed_indicators": [],
                }

            if tables_added:
                added_payloads = [_synthetic_comp_added(t) for t in tables_added]
                added_analyses = classifier.classify_batch(added_payloads)
                for t, analysis in zip(tables_added, added_analyses):
                    t["genai_analysis"] = (
                        analysis
                        if isinstance(analysis, dict) and analysis.get("relevance")
                        else _GENAI_ANALYSIS_FALLBACK
                    )
                for t in tables_added:
                    if not t.get("genai_analysis", {}).get("relevance"):
                        t["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK
            if tables_removed:
                removed_payloads = [_synthetic_comp_removed(t) for t in tables_removed]
                removed_analyses = classifier.classify_batch(removed_payloads)
                for t, analysis in zip(tables_removed, removed_analyses):
                    t["genai_analysis"] = (
                        analysis
                        if isinstance(analysis, dict) and analysis.get("relevance")
                        else _GENAI_ANALYSIS_FALLBACK
                    )
                for t in tables_removed:
                    if not t.get("genai_analysis", {}).get("relevance"):
                        t["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK

            logger.info(
                "GenAI classification: %d calls, %d errors, %.1fs total, "
                "circuit_breaker=%s",
                classifier.stats["calls"],
                classifier.stats["errors"],
                classifier.stats["total_latency"],
                "OPEN" if classifier.circuit_open else "closed",
            )
        except Exception as exc:
            logger.warning("GenAI classification layer failed: %s", exc)

            def _has_changes_fallback(c: dict[str, Any]) -> bool:
                return bool(
                    c.get("added_indicators")
                    or c.get("removed_indicators")
                    or c.get("renamed_indicators")
                    or c.get("footnotes_counts", {}).get("added", 0)
                    or c.get("footnotes_counts", {}).get("removed", 0)
                    or c.get("footnotes_counts", {}).get("modified", 0)
                )

            for i, c in enumerate(comparisons):
                if _has_changes_fallback(c) and not c.get("genai_analysis", {}).get(
                    "relevance"
                ):
                    comparisons[i]["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK
            for t in tables_added:
                if not t.get("genai_analysis", {}).get("relevance"):
                    t["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK
            for t in tables_removed:
                if not t.get("genai_analysis", {}).get("relevance"):
                    t["genai_analysis"] = _GENAI_ANALYSIS_FALLBACK

    ambiguous_tables = list(strict.get("ambiguous_tables", []) or [])
    added_tables_confirmed = list(
        strict.get("added_tables_confirmed", []) or tables_added
    )
    removed_tables_confirmed = list(
        strict.get("removed_tables_confirmed", []) or tables_removed
    )
    vision_rescued_pairs = list(strict.get("vision_rescued_pairs", []) or [])
    cross_section_rescued_pairs = list(
        strict.get("cross_section_rescued_pairs", []) or []
    )
    tables_comparable_t1 = int(
        strict.get(
            "tables_comparable_t1",
            sum(
                1 for table in tables_t1 if getattr(table, "comparison_eligible", False)
            ),
        )
        or 0
    )
    tables_comparable_t2 = int(
        strict.get(
            "tables_comparable_t2",
            sum(
                1 for table in tables_t2 if getattr(table, "comparison_eligible", False)
            ),
        )
        or 0
    )
    pairing_coverage = float(
        strict.get(
            "pairing_coverage",
            len(comparisons) / max(min(tables_comparable_t1, tables_comparable_t2), 1),
        )
        or 0.0
    )
    indicator_change_pairs = sum(
        1
        for c in comparisons
        if c.get("added_indicators")
        or c.get("removed_indicators")
        or c.get("renamed_indicators")
    )
    footnote_change_pairs = sum(
        1
        for c in comparisons
        if any(
            int((c.get("footnotes_counts", {}) or {}).get(key, 0) or 0) > 0
            for key in ("added", "removed", "modified")
        )
    )
    pairing_low_confidence = pairing_coverage < 0.75 or bool(ambiguous_tables)

    total_added = sum(len(c.get("added_indicators", [])) for c in comparisons)
    total_removed = sum(len(c.get("removed_indicators", [])) for c in comparisons)
    total_renamed = sum(len(c.get("renamed_indicators", [])) for c in comparisons)

    total_fn_added = sum(
        c.get("footnotes_counts", {}).get("added", 0) for c in comparisons
    )
    total_fn_removed = sum(
        c.get("footnotes_counts", {}).get("removed", 0) for c in comparisons
    )
    total_fn_modified = sum(
        c.get("footnotes_counts", {}).get("modified", 0) for c in comparisons
    )

    status_counts = {
        "stable": sum(1 for c in comparisons if c.get("table_status") == "stable"),
        "modifie": sum(1 for c in comparisons if c.get("table_status") == "modifie"),
        "renommage_probable": 0,
        "incertain": sum(1 for c in comparisons if c.get("table_status") == "incertain")
        + len(ambiguous_tables),
        "needs_review": 0,
        "structure_change": sum(
            1 for c in comparisons if c.get("table_status") == "structure_change"
        ),
        "ajoute": len(added_tables_confirmed),
        "supprime": len(removed_tables_confirmed),
    }

    now = datetime.now().isoformat(timespec="seconds")
    current_label = str(quarter_context["current"]["label"])
    previous_label = str(quarter_context["previous"]["label"])
    summary_text = (
        f"Comparaison {current_label} vs {previous_label}: "
        f"{len(comparisons)} tableaux apparies sur {min(tables_comparable_t1, tables_comparable_t2)} comparables, "
        f"{total_added} ajouts d'indicateurs, {total_removed} suppressions. "
    )
    if pairing_low_confidence:
        summary_text += (
            f"{len(added_tables_confirmed)} tableaux non apparies cote courant et "
            f"{len(removed_tables_confirmed)} cote precedent restent a confirmer."
        )
    else:
        summary_text += (
            f"{len(added_tables_confirmed)} tableaux ajoutes confirmes dans le trimestre courant, "
            f"{len(removed_tables_confirmed)} tableaux retires confirmes depuis le trimestre precedent."
        )
    if ambiguous_tables:
        summary_text += (
            f" {len(ambiguous_tables)} tableau(x) restent ambigus apres le rescue."
        )
    if vision_rescued_pairs:
        summary_text += (
            f" {len(vision_rescued_pairs)} tableau(x) recuperes par validation Vision."
        )
    if cross_section_rescued_pairs:
        summary_text += (
            f" {len(cross_section_rescued_pairs)} tableau(x) recuperes par rescue cross-section."
        )

    extraction_quality_kpis = _compute_extraction_kpis(
        tables_t1,
        tables_t2,
        comparisons,
        added_tables_confirmed,
        removed_tables_confirmed,
    )
    logger.info(
        "comparison_result_summary tables_t1=%d tables_t2=%d tables_matched=%d bank=%s",
        len(tables_t1),
        len(tables_t2),
        len(comparisons),
        bank_code,
    )

    result: dict[str, Any] = {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": previous_label,
        "quarter_to": current_label,
        "previous_quarter": previous_label,
        "current_quarter": current_label,
        "comparison_direction": "current_vs_previous",
        "year": year,
        "summary": {
            "tables_t1": len(tables_t1),
            "tables_t2": len(tables_t2),
            "tables_extracted_t1": len(tables_t1),
            "tables_extracted_t2": len(tables_t2),
            "tables_comparable_t1": tables_comparable_t1,
            "tables_comparable_t2": tables_comparable_t2,
            "tables_matched": len(comparisons),
            "tables_added": len(added_tables_confirmed),
            "tables_removed": len(removed_tables_confirmed),
            "tables_added_confirmed": len(added_tables_confirmed),
            "tables_removed_confirmed": len(removed_tables_confirmed),
            "ambiguous_tables": len(ambiguous_tables),
            "ambiguous_pairs": len(strict.get("ambiguous_pairs", []) or []),
            "vision_rescued_pairs": len(vision_rescued_pairs),
            "cross_section_rescued_pairs": len(cross_section_rescued_pairs),
            "rescued_matches_count": int(strict.get("rescued_matches_count", 0) or 0),
            "split_merge_rescues_count": int(
                strict.get("split_merge_rescues_count", 0) or 0
            ),
            "pairing_coverage": round(pairing_coverage, 6),
            "indicator_change_pairs": indicator_change_pairs,
            "footnote_change_pairs": footnote_change_pairs,
            "pairing_low_confidence": pairing_low_confidence,
            "total_added_indicators": total_added,
            "total_removed_indicators": total_removed,
            "total_renamed_indicators": total_renamed,
            "total_footnotes_added": total_fn_added,
            "total_footnotes_removed": total_fn_removed,
            "total_footnotes_modified": total_fn_modified,
            "total_changements_reglementaires": sum(
                1
                for c in comparisons
                if c.get("genai_analysis", {}).get("relevance") == "REGLEMENTAIRE"
            ),
            "total_changements_non_significatifs": sum(
                1
                for c in comparisons
                if c.get("genai_analysis", {}).get("relevance") == "NON_SIGNIFICATIF"
            ),
            "status_counts": status_counts,
        },
        "displaced_indicators": [],
        "table_comparisons": comparisons,
        "tables_added": added_tables_confirmed,
        "tables_removed": removed_tables_confirmed,
        "tables_added_confirmed": added_tables_confirmed,
        "tables_removed_confirmed": removed_tables_confirmed,
        "ambiguous_tables": ambiguous_tables,
        "vision_rescued_pairs": vision_rescued_pairs,
        "probable_pairs": list(strict.get("probable_pairs", [])),
        "rejected_by_vision_pair": rejected_by_vision_pair,
        "debug_unmatched_candidates": list(
            strict.get("debug_unmatched_candidates", [])
        ),
        "meta": {
            "generated_at": now,
            "provenance": "comparison_runner",
            "source_format": "strict_intra_section",
            "quarter_context": quarter_context,
            "algorithm_used": algorithm_used,
            "raw_tables_t1": raw_tables_t1_count,
            "raw_tables_t2": raw_tables_t2_count,
            "fragment_merges_t1_count": len(fragment_merges_t1),
            "fragment_merges_t2_count": len(fragment_merges_t2),
            "fragment_merges_t1": fragment_merges_t1,
            "fragment_merges_t2": fragment_merges_t2,
            "executive_summary": {"content": summary_text},
            "embedding_debug": {
                "embedding_enabled": bool(cfg.get("use_embeddings", False)),
                "embedding_table_used": (
                    embedding_service is not None
                    and (
                        embedding_service.stats.api_calls > 0
                        or embedding_service.stats.cache_hits > 0
                    )
                    and cfg.get("use_embeddings")
                ),
                "embedding_indicator_used": (
                    embedding_service is not None and cfg.get("use_embeddings")
                ),
                "embedding_api_calls": embedding_service.stats.api_calls
                if embedding_service
                else 0,
                "embedding_cache_hits": embedding_service.stats.cache_hits
                if embedding_service
                else 0,
                "embedding_batch_sizes": list(embedding_service.stats.batch_sizes)
                if embedding_service
                else [],
                "embedding_errors": embedding_service.stats.errors
                if embedding_service
                else 0,
                "config_use_embeddings": bool(cfg.get("use_embeddings", False)),
                "fallback_vision_used": False,
                "hungarian_table": True,
                "table_matcher_engine": algorithm_used,
                "hungarian_indicator": cfg.get("indicator_hungarian_enabled", True),
                "table_pair_count": len(comparisons),
                "indicator_rename_count": total_renamed,
                "table_pair_debug": table_pair_embed_debug,
                "rename_pair_debug": all_rename_pair_debug,
                "indicator_unmatched_with_candidates": all_unmatched_indicator_candidates,
            },
            "semantic_judge_debug": {
                "enabled": semantic_judge_enabled,
                "bank_allowed": semantic_judge_enabled,
                "total_calls": semantic_judge_stats["calls"],
                "total_errors": semantic_judge_stats["errors"],
                "structural_overrides": semantic_judge_stats["overrides"],
            },
            "validation_summary": {
                "vision_pair": {
                    "enabled": vision_pair_validation,
                    "calls": vision_pair_stats["calls"],
                    "accepted": vision_pair_stats["accepted"],
                    "rejected": vision_pair_stats["rejected"],
                    "errors": vision_pair_stats["errors"],
                },
                "vision_unmatched_rescue": {
                    "enabled": vision_unmatched_rescue_summary.get("enabled", False),
                    "candidate_pairs_tested": vision_unmatched_rescue_summary.get(
                        "candidate_pairs_tested", 0
                    ),
                    "candidate_pairs_considered": vision_unmatched_rescue_summary.get(
                        "candidate_pairs_considered", 0
                    ),
                    "candidate_tables_considered": vision_unmatched_rescue_summary.get(
                        "candidate_tables_considered", 0
                    ),
                    "candidate_matches": vision_unmatched_rescue_summary.get(
                        "candidate_matches", 0
                    ),
                    "rescued_pairs": vision_unmatched_rescue_summary.get(
                        "rescued_pairs", 0
                    ),
                    "vision_rejected_pairs": vision_unmatched_rescue_summary.get(
                        "vision_rejected_pairs", 0
                    ),
                    "vision_unresolved_pairs": vision_unmatched_rescue_summary.get(
                        "vision_unresolved_pairs", 0
                    ),
                    "conflicts": vision_unmatched_rescue_summary.get("conflicts", 0),
                    "errors": vision_unmatched_rescue_summary.get("errors", 0),
                    "metrics": vision_unmatched_rescue_summary.get("metrics", {}),
                },
                "strict_matcher": {
                    "pairs": len(strict.get("pairs", [])),
                    "matched_pairs": len(
                        strict.get("matched_pairs", strict.get("pairs", []))
                    ),
                    "probable_pairs": len(strict.get("probable_pairs", [])),
                    "suspicious_pairs": len(strict.get("suspicious_pairs", [])),
                    "ambiguous_pairs": len(strict.get("ambiguous_pairs", [])),
                    "rescued_matches_count": strict.get("rescued_matches_count", 0),
                    "split_merge_rescues_count": strict.get(
                        "split_merge_rescues_count", 0
                    ),
                    "tables_comparable_t1": tables_comparable_t1,
                    "tables_comparable_t2": tables_comparable_t2,
                    "pairing_coverage": round(pairing_coverage, 6),
                    "unmatched_confirmed_t1": len(
                        strict.get("unmatched_confirmed_t1", [])
                    ),
                    "unmatched_ambiguous_t1": len(
                        strict.get("unmatched_ambiguous_t1", [])
                    ),
                    "unmatched_confirmed_t2": len(
                        strict.get("unmatched_confirmed_t2", [])
                    ),
                    "unmatched_ambiguous_t2": len(
                        strict.get("unmatched_ambiguous_t2", [])
                    ),
                    "ambiguous_unmatched_current": len(
                        strict.get("ambiguous_unmatched_current", [])
                    ),
                    "ambiguous_unmatched_previous": len(
                        strict.get("ambiguous_unmatched_previous", [])
                    ),
                    "suspicious_pairs_payload": strict.get("suspicious_pairs", []),
                    "vision_rescued_pairs_payload": vision_rescued_pairs,
                    "cross_section_rescued_pairs_payload": cross_section_rescued_pairs,
                    "ambiguous_tables_payload": ambiguous_tables,
                    "matching_diagnostics": strict.get("matching_diagnostics", {}),
                },
                "semantic_judge": {
                    "enabled": semantic_judge_enabled,
                    "calls": semantic_judge_stats["calls"],
                    "errors": semantic_judge_stats["errors"],
                    "structural_overrides": semantic_judge_stats["overrides"],
                },
                "rename_validator": {
                    "enabled": rename_validator_enabled,
                    "uncertain_score_band": [
                        round(rename_band_min, 4),
                        round(rename_band_max, 4),
                    ],
                    "calls": rename_validator_stats.get("calls", 0),
                    "pairs_validated": rename_validator_stats.get("pairs_validated", 0),
                    "candidates_in_band": rename_validator_stats.get(
                        "candidates_in_band", 0
                    ),
                    "auto_accepted_out_of_band": rename_validator_stats.get(
                        "auto_accepted_out_of_band", 0
                    ),
                    "accepted": rename_validator_stats.get("accepted", 0),
                    "rejected": rename_validator_stats.get("rejected", 0),
                    "errors": rename_validator_stats.get("errors", 0),
                },
                "added_table_validator": {
                    "enabled": added_table_validator_enabled,
                    "calls": added_table_validator_stats.get("calls", 0),
                    "accepted": added_table_validator_stats.get("accepted", 0),
                    "rejected": added_table_validator_stats.get("rejected", 0),
                    "errors": added_table_validator_stats.get("errors", 0),
                },
                "indicator_validator": {
                    "enabled": indicator_validator_stats.get("enabled", False),
                    "calls": indicator_validator_stats.get("calls", 0),
                    "filtered_added": indicator_validator_stats.get(
                        "filtered_added", 0
                    ),
                    "filtered_removed": indicator_validator_stats.get(
                        "filtered_removed", 0
                    ),
                    "errors": indicator_validator_stats.get("errors", 0),
                    "use_vision": indicator_validator_stats.get("use_vision", False),
                    "vision_fallback_count": indicator_validator_stats.get(
                        "vision_fallback_count", 0
                    ),
                },
            },
            "extraction_kpis": extraction_quality_kpis,
            "extraction_quality": extraction_quality_kpis,
            "extraction_quality_summary": {
                "tables_certified": extraction_quality_kpis.get("tables_certified", 0),
                "tables_review_required": extraction_quality_kpis.get("tables_review_required", 0),
                "tables_blocked": extraction_quality_kpis.get("tables_blocked", 0),
                "tables_crop_rejected": extraction_quality_kpis.get("tables_crop_rejected", 0),
                "tables_low_confidence": extraction_quality_kpis.get("tables_low_confidence", 0),
                "tables_budget_exhausted": extraction_quality_kpis.get("tables_budget_exhausted", 0),
            },
            "quality_gate": quality_gate_status,
            "extraction_artifacts": {
                "run_id": extraction_run_id,
                "out_dir": extraction_out_dir,
            },
            "extraction_sources": {
                "previous": extraction_source_previous,
                "current": extraction_source_current,
            },
        },
    }

    result["summary"]["tables_changed_t1"] = compute_changed_tables_t1(result)
    result["summary"]["tables_changed_t2"] = compute_changed_tables_t2(result)
    result["summary"]["eligible_for_review"] = True
    result["eligible_for_review"] = True

    # -- GenAI executive summary enrichment (feature-flagged via bank_profiles.yaml) --
    if api_key:
        try:
            from app.genai_summary import enrich_result_with_genai

            result = enrich_result_with_genai(result)
        except Exception as exc:
            logger.warning("GenAI executive summary enrichment failed: %s", exc)

    INDICATOR_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_slug = current_label.lower().replace(" ", "_")
    previous_slug = previous_label.lower().replace(" ", "_")
    out_path = INDICATOR_COMPARISON_DIR / (
        f"{bank_code}_{current_slug}_vs_{previous_slug}_{stamp}.json"
    )
    result["meta"]["compare_path"] = str(out_path)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return result
