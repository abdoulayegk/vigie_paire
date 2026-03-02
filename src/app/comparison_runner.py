"""Run T1/T2 comparison from uploaded PDFs and section ranges."""

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

from app.comparison_canonical import compute_changed_tables_t1, compute_changed_tables_t2
from app.ui_config import INDICATOR_COMPARISON_DIR, LOGS_DIR

_SEMANTIC_JUDGE_LOG = LOGS_DIR / "semantic_judge_decisions.jsonl"
_VALIDATION_LOG = LOGS_DIR / "validation.jsonl"
from vigilance.compare import run_strict_intra_section_compare
from vigilance.compare.table_fragment_merger import merge_table_fragments
from vigilance.comparison.footnote_comparator import FootnoteComparator
from vigilance.config import (
    get_matching_thresholds,
    get_quality_gate_config,
    get_validation_config,
)
from vigilance.models.table_models import TableArtifact
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
)
from vigilance.utils.matching_normalizer import _classify_excluded_line


_MATCH_DECISIONS_LOG = LOGS_DIR / "match_decisions.jsonl"


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


def _resolve_vision_primary_mode(
    bank_code: str,
    explicit: bool | None,
    *,
    allow_env_legacy: bool = True,
) -> bool:
    """Resolution order: explicit > legacy env > bank config."""
    if explicit is not None:
        return bool(explicit)
    if allow_env_legacy:
        env_choice = _env_bool("VIGILANCE_VISION_PRIMARY")
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
    dict1 = footnotes_list_to_dict(t1.footnotes or [])
    dict2 = footnotes_list_to_dict(t2.footnotes or [])
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
    table: Any, *, bank_code: str, quarter: str, pdf_path: str
) -> TableArtifact:
    rows = [list(row) for row in (getattr(table, "rows", []) or [])]
    headers = [str(h) for h in (getattr(table, "headers", []) or []) if h is not None]
    raw = getattr(table, "first_column_indicators_raw", None)
    if raw is not None:
        raw = [str(x).strip() for x in raw if str(x).strip()]
    else:
        indicators_fallback = [
            str(item).strip()
            for item in (getattr(table, "first_column_indicators", []) or [])
            if str(item).strip()
        ]
        raw = indicators_fallback
    if not raw:
        for row in rows:
            if row and str(row[0]).strip():
                raw.append(str(row[0]).strip())

    # Quality pass 1: line-split merge (deterministic)
    raw, line_merge_count = merge_line_split_indicators(raw)
    if line_merge_count > 0:
        logger.info(
            "indicators_line_merge: table=%s page=%s merges=%d",
            getattr(table, "table_id", ""),
            getattr(table, "page_number", 0),
            line_merge_count,
        )

    # Quality pass 2: dedupe (if duplicate_ratio >= 0.15)
    raw, duplicate_ratio, dup_removed = dedupe_indicators(raw)
    if dup_removed > 0:
        logger.info(
            "indicators_dedupe: table=%s page=%s removed=%d duplicate_ratio=%.3f",
            getattr(table, "table_id", ""),
            getattr(table, "page_number", 0),
            dup_removed,
            duplicate_ratio,
        )

    # Parts A & B: post-normalize cleaned indicators (camelCase split + tag-space fix).
    fragmentation_from_post_norm = False
    post_normed: list[str] = []
    for ind in raw:
        fixed, camel_hit, tag_hit = post_normalize_indicator(
            normalize_indicator_for_comparison(ind)
        )
        if camel_hit or tag_hit:
            fragmentation_from_post_norm = True
        post_normed.append(fixed)
    indicators = post_normed

    footnotes_raw = getattr(table, "footnotes", None)
    if footnotes_raw:
        footnotes_out = normalize_footnotes_to_canonical(footnotes_raw)
    else:
        footnotes_out = None

    # Part C: propagate fragmentation flag (merge or post-norm corrections)
    frag_from_extraction = bool(getattr(table, "fragmentation_detected", False))
    fragmentation_detected = frag_from_extraction or fragmentation_from_post_norm

    section = _canonical_section_name(str(getattr(table, "section", "")))
    dm_raw = getattr(table, "debug_metrics", None)
    debug_metrics = dict(dm_raw) if isinstance(dm_raw, dict) else None

    title_raw = getattr(table, "title_raw", None) or getattr(table, "title", None)
    title_clean = getattr(table, "title_clean", None)
    title_display = title_clean or getattr(table, "title", None)

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
        first_column_indicators=indicators,
        first_column_indicators_raw=raw,
        extraction_method=getattr(table, "extraction_method", None) or "docling",
        table_number=getattr(table, "table_number", None),
        bbox=getattr(table, "bbox", None),
        quarter=quarter,
        pdf_path=pdf_path,
        footnotes=footnotes_out,
        fragmentation_detected=fragmentation_detected,
        debug_metrics=debug_metrics,
    )


def _extract_tables(
    *,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    section_ranges: list[dict[str, Any]],
    use_vision_fallback: bool,
    api_key: str | None,
    use_vision_primary: bool | None = None,
) -> list[TableArtifact]:
    from vigilance.extraction.docling_processor import (
        extract_tables_docling_by_sections,
    )

    use_vision_primary = _resolve_vision_primary_mode(
        bank_code,
        use_vision_primary,
        allow_env_legacy=True,
    )
    del api_key

    raw_tables = extract_tables_docling_by_sections(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        section_ranges=section_ranges,
        use_vision_primary=use_vision_primary,
        use_vision_fallback=use_vision_fallback,
    )

    return [
        _table_to_artifact(
            table, bank_code=bank_code, quarter=quarter, pdf_path=pdf_path
        )
        for table in raw_tables
    ]


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
        for ind in t.first_column_indicators:
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
    """Build quality flag strings from debug_metrics for comparison output."""
    dm = table.debug_metrics or {}
    flags: list[str] = []
    if dm.get("quality_suspect_for_vision"):
        flags.append("quality_suspect")
    if dm.get("vision_fallback_applied"):
        flags.append("vision_applied")
    elif dm.get("vision_fallback_attempted"):
        flags.append("vision_attempted_not_applied")
    arb = dm.get("vision_arbitration")
    if isinstance(arb, dict):
        decision = arb.get("decision", "")
        if "rejected" in decision:
            flags.append(f"vision_{decision}")
        agreement = arb.get("agreement_signals", {}).get("agreement", "")
        if agreement == "strong_disagree":
            flags.append("extraction_strong_disagree")
    if table.fragmentation_detected:
        flags.append("fragmentation")
    dup = dm.get("duplicate_ratio", 0)
    if isinstance(dup, (int, float)) and dup > 0.20:
        flags.append(f"high_duplicate_ratio({dup:.2f})")
    hlr = dm.get("header_like_ratio", 0)
    if isinstance(hlr, (int, float)) and hlr > 0.20:
        flags.append(f"high_header_like({hlr:.2f})")
    return flags


def _extraction_confidence(table: TableArtifact) -> str:
    """Return extraction confidence level: high, medium, low, or unknown."""
    dm = table.debug_metrics or {}
    arb = dm.get("vision_arbitration")
    if isinstance(arb, dict):
        decision = arb.get("decision", "")
        agreement = arb.get("agreement_signals", {}).get("agreement", "")
        if decision.startswith("accepted") and agreement == "agree":
            return "high"
        if decision.startswith("accepted"):
            return "medium"
        if "rejected" in decision and dm.get("quality_suspect_for_vision"):
            return "low"
    quality = dm.get("table_quality_score")
    if isinstance(quality, (int, float)):
        if quality >= 0.75:
            return "high"
        if quality >= 0.50:
            return "medium"
        return "low"
    return "unknown"


def _compute_extraction_kpis(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    comparisons: list[dict[str, Any]],
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate extraction reliability KPIs for the comparison output."""
    all_tables = list(tables_t1) + list(tables_t2)
    total = len(all_tables) or 1

    vision_attempted = sum(
        1 for t in all_tables
        if (t.debug_metrics or {}).get("vision_fallback_attempted")
    )
    vision_applied = sum(
        1 for t in all_tables
        if (t.debug_metrics or {}).get("vision_fallback_applied")
    )
    vision_primary_attempted = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_primary_attempted")
    )
    vision_primary_applied = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_primary_applied")
    )
    vision_schema_contract_fail_count = sum(
        1
        for t in all_tables
        if (t.debug_metrics or {}).get("vision_schema_contract_failed")
    )
    vision_primary_disabled_reason = ""
    for t in all_tables:
        reason = (t.debug_metrics or {}).get("vision_primary_disabled_reason")
        if isinstance(reason, str) and reason.strip():
            vision_primary_disabled_reason = reason.strip()
            break

    disagree_count = 0
    for t in all_tables:
        arb = (t.debug_metrics or {}).get("vision_arbitration")
        if isinstance(arb, dict):
            agreement = arb.get("agreement_signals", {}).get("agreement", "")
            if agreement in ("disagree", "strong_disagree"):
                disagree_count += 1

    matched_with_changes = sum(
        1 for c in comparisons
        if c.get("added_indicators") or c.get("removed_indicators")
    )
    matched_total = len(comparisons) or 1
    noise_rate = matched_with_changes / matched_total

    renamed_total = sum(len(c.get("renamed_indicators", [])) for c in comparisons)
    add_remove_total = (
        sum(len(c.get("added_indicators", [])) for c in comparisons)
        + sum(len(c.get("removed_indicators", [])) for c in comparisons)
    )
    rename_conversion = (
        renamed_total / (renamed_total + add_remove_total)
        if (renamed_total + add_remove_total) > 0
        else 0.0
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
        "vision_primary_attempted_count": vision_primary_attempted,
        "vision_primary_applied_count": vision_primary_applied,
        "vision_schema_contract_fail_count": vision_schema_contract_fail_count,
        "vision_primary_disabled_reason": vision_primary_disabled_reason or None,
        "disagreement_count": disagree_count,
        "incertain_count": sum(1 for c in comparisons if c.get("table_status") == "incertain"),
    }


def _pre_diff_safety_check(
    t1: TableArtifact, t2: TableArtifact, indicator_overlap: float
) -> tuple[bool, str]:
    """Flag suspicious pairs before calling _indicator_diff.

    Returns (suspicious_low_overlap, reason_string).
    Does NOT block matching -- only exposes the signal.
    """
    n1 = len(t1.first_column_indicators)
    n2 = len(t2.first_column_indicators)
    size_diff_ratio = abs(n1 - n2) / max(n1, n2, 1)

    if indicator_overlap < _SUSPICIOUS_OVERLAP_THRESHOLD and size_diff_ratio > _SUSPICIOUS_SIZE_DIFF_RATIO:
        reason = (
            f"indicator_overlap={indicator_overlap:.3f} < {_SUSPICIOUS_OVERLAP_THRESHOLD}, "
            f"size_diff_ratio={size_diff_ratio:.3f} > {_SUSPICIOUS_SIZE_DIFF_RATIO}"
        )
        logger.warning(
            "suspicious_low_overlap: t1=%s t2=%s %s",
            t1.table_id, t2.table_id, reason,
        )
        return True, reason
    return False, ""


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
    raw = getattr(table, "first_column_indicators_raw", None) or []
    if not raw:
        raw = getattr(table, "first_column_indicators", None) or []
    result: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if _classify_excluded_line(s):
            continue
        cleaned = _strip_footnote_markers_from_indicator(s)
        key = _canonical_indicator_key(cleaned)
        if not key:
            continue
        result.append(cleaned)
    return result


def _build_clean_to_raw_indicator_lookup(table: TableArtifact) -> dict[str, str]:
    """Build stable mapping from canonical clean key to raw display indicator text."""
    clean_values = list(getattr(table, "first_column_indicators", None) or [])
    raw_values = list(getattr(table, "first_column_indicators_raw", None) or [])
    if not raw_values:
        raw_values = clean_values

    lookup: dict[str, str] = {}
    for idx, clean_item in enumerate(clean_values):
        clean_text = str(clean_item).strip()
        if not clean_text:
            continue
        value_clean = _strip_footnote_markers_from_indicator(clean_text)
        key = _canonical_indicator_key(value_clean)
        if not key or key in lookup:
            continue
        raw_text = str(raw_values[idx]).strip() if idx < len(raw_values) else clean_text
        lookup[key] = raw_text or clean_text

    # Defensive fallback when clean/raw arrays drift.
    for raw_item in raw_values:
        raw_text = str(raw_item).strip()
        if not raw_text:
            continue
        value_clean = _strip_footnote_markers_from_indicator(raw_text)
        key = _canonical_indicator_key(value_clean)
        if key and key not in lookup:
            lookup[key] = raw_text
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
        value_clean = _strip_footnote_markers_from_indicator(clean_text)
        key = _canonical_indicator_key(value_clean)
        result.append(str(lookup.get(key) or clean_text))
    return result


# Trailing footnote/reference patterns (do not remove semantic numbers: Tier 1, CET1, Bâle III, Pillar 3, IFRS 9)
_INDICATOR_TRAILING_SUPER = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*$")
_INDICATOR_TRAILING_STARS = re.compile(r"\s*[*\u2020\u2021\u00A7]+\s*$")  # * ** † ‡ §
_INDICATOR_TRAILING_PAREN_NUM = re.compile(r"\s*[\(\[]\d+[\)\]]\s*$")
_INDICATOR_TRAILING_NOTE_NUM = re.compile(r"\s+Note\s+\d+\s*\.?\s*$", re.IGNORECASE)
_INDICATOR_TRAILING_COMMA_NUMS = re.compile(r"\s*,\s*\d+(?:\s*,\s*\d+)*\s*$")
_INDICATOR_TRAILING_SPACE_NUMS_COMMA = re.compile(r"\s+\d+(?:\s*,\s*\d+)+\s*$")


def _strip_footnote_markers_from_indicator(text: str) -> str:
    """Remove trailing footnote markers and refs from indicator label. Preserves semantic numbers (Tier 1, CET1, etc.)."""
    if not text:
        return ""
    value = (text or "").strip()
    while True:
        prev = value
        value = _INDICATOR_TRAILING_SUPER.sub("", value)
        value = _INDICATOR_TRAILING_STARS.sub("", value)
        value = _INDICATOR_TRAILING_PAREN_NUM.sub("", value)
        value = _INDICATOR_TRAILING_NOTE_NUM.sub("", value)
        value = _INDICATOR_TRAILING_COMMA_NUMS.sub("", value)
        value = _INDICATOR_TRAILING_SPACE_NUMS_COMMA.sub("", value)
        value = re.sub(r"\s+", " ", value).strip()
        if value == prev:
            break
    return value


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


def _hungarian_pair_added_removed(
    removed_items: list[str],
    added_items: list[str],
    *,
    th: dict[str, Any] | None = None,
    embedding_service: Any = None,
) -> tuple[list[str], list[str], list[tuple[str, str]], dict[str, Any]]:
    """
    Global 1-to-1 pairing between removed and added indicators (renames).
    Uses Hungarian assignment when scipy is available; otherwise deterministic greedy.
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

    min_score_pct = int(min_score * 100)

    def _norm_for_sort(s: str) -> str:
        return _canonical_indicator_key(_strip_footnote_markers_from_indicator(s))

    removed = sorted(removed_items, key=_norm_for_sort)
    added = sorted(added_items, key=_norm_for_sort)

    def _length_ratio_ok(a: str, r: str) -> bool:
        la, lr = len(_norm_for_sort(a)), len(_norm_for_sort(r))
        if max(la, lr) <= 0:
            return True
        return (min(la, lr) / max(la, lr)) >= min_len_ratio

    def _token_overlap_ok(a: str, r: str) -> bool:
        na, nr = _norm_for_sort(a), _norm_for_sort(r)
        ta = _indicator_strong_tokens(na)
        tr = _indicator_strong_tokens(nr)
        if len(ta & tr) >= min_token_overlap:
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
        c = _canonical_indicator_key(_strip_footnote_markers_from_indicator(s))
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
            if sc >= min_score_pct:
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
            if best_j >= 0:
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


def _detect_fusion_split(
    added: list[str], removed: list[str], ratio_threshold: float = 0.92
) -> tuple[list[str], list[str], bool]:
    """Merge fusion/split: 1 added = concat of 2 removed (or vice versa).
    Returns (added, removed, had_fusion_split).
    """
    from difflib import SequenceMatcher

    added = list(added)
    removed = list(removed)
    had_fusion_split = False

    def _merge_added_from_removed() -> None:
        nonlocal added, removed, had_fusion_split
        for i, a in enumerate(added[:]):
            a_norm = _canonical_indicator_key(a)
            for j, r1 in enumerate(removed):
                for k, r2 in enumerate(removed):
                    if j >= k:
                        continue
                    concat = (
                        _canonical_indicator_key(r1)
                        + " "
                        + _canonical_indicator_key(r2)
                    )
                    if not concat.strip():
                        continue
                    ratio = SequenceMatcher(None, a_norm, concat).ratio()
                    if ratio >= ratio_threshold or concat == a_norm:
                        added.remove(a)
                        removed.remove(r2)
                        removed.remove(r1)
                        had_fusion_split = True
                        return
            if not added:
                break

    def _merge_removed_from_added() -> None:
        nonlocal added, removed, had_fusion_split
        for i, r in enumerate(removed[:]):
            r_norm = _canonical_indicator_key(r)
            for j, a1 in enumerate(added):
                for k, a2 in enumerate(added):
                    if j >= k:
                        continue
                    concat = (
                        _canonical_indicator_key(a1)
                        + " "
                        + _canonical_indicator_key(a2)
                    )
                    if not concat.strip():
                        continue
                    ratio = SequenceMatcher(None, r_norm, concat).ratio()
                    if ratio >= ratio_threshold or concat == r_norm:
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


def _indicator_diff(
    t1: TableArtifact, t2: TableArtifact
) -> tuple[list[str], list[str], bool, dict[str, int]]:
    # Matching/diff use first_column_indicators (clean) only; UI display prefers raw.
    left = [
        str(item).strip() for item in t1.first_column_indicators if str(item).strip()
    ]
    right = [
        str(item).strip() for item in t2.first_column_indicators if str(item).strip()
    ]

    def _norm(values: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        mapped: dict[str, str] = {}
        excluded: dict[str, int] = {}
        for value in values:
            kind = _classify_excluded_line(value)
            if kind:
                excluded[kind] = excluded.get(kind, 0) + 1
                continue
            value_clean = _strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key and key not in mapped:
                mapped[key] = value_clean
        return mapped, excluded

    left_map, left_excluded = _norm(left)
    right_map, right_excluded = _norm(right)
    excluded_counts: dict[str, int] = {}
    for k in set(left_excluded) | set(right_excluded):
        excluded_counts[k] = left_excluded.get(k, 0) + right_excluded.get(k, 0)

    added = [right_map[key] for key in right_map.keys() - left_map.keys()]
    removed = [left_map[key] for key in left_map.keys() - right_map.keys()]
    added.sort()
    removed.sort()
    added, removed, had_fusion_split = _detect_fusion_split(added, removed)
    added.sort()
    removed.sort()
    return added, removed, had_fusion_split, excluded_counts


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

    used_added: set[str] = set()
    used_removed: set[str] = set()
    renamed_pairs: list[tuple[str, str]] = []
    for a, r, _ in candidates:
        if a not in used_added and r not in used_removed:
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


def _empty_result(bank_code: str, year: int, reason: str) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": "t1",
        "quarter_to": "t2",
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
    pdf_path_t1: str,
    pdf_path_t2: str,
    bank_code: str,
    sections_t1: list[dict[str, Any]] | None,
    sections_t2: list[dict[str, Any]] | None,
    use_genai: bool = False,
    api_key: str | None = None,
    generate_visual_proofs: bool = False,
    use_vision_fallback: bool = False,
    use_vision_primary: bool | None = None,
    use_vision_primary_override: bool | None = None,
    include_footnotes: bool = False,
    include_genai_classification: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end comparison used by the Dash Analyze callback.

    Args:
        use_vision_primary: If True/False, overrides config vision_extraction.enabled
            for this run. If None, config is used.
    """
    del use_genai, generate_visual_proofs  # kept for backward-compatible signature
    if use_vision_primary is not None and use_vision_primary_override is not None:
        if bool(use_vision_primary) != bool(use_vision_primary_override):
            raise ValueError(
                "Conflicting values for use_vision_primary and use_vision_primary_override."
            )
    if use_vision_primary is None and use_vision_primary_override is not None:
        use_vision_primary = use_vision_primary_override

    year = _infer_year(pdf_path_t1, pdf_path_t2)
    ranges_t1 = _normalize_ranges(sections_t1)
    ranges_t2 = _normalize_ranges(sections_t2)

    if not ranges_t1 or not ranges_t2:
        return _empty_result(bank_code, year, "Aucune section valide fournie.")

    try:
        # Run both report extractions in parallel to cut total runtime (Docling + Vision per PDF)
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_t1 = executor.submit(
                _extract_tables,
                pdf_path=pdf_path_t1,
                bank_code=bank_code,
                quarter="t1",
                year=year,
                section_ranges=ranges_t1,
                use_vision_fallback=use_vision_fallback,
                api_key=api_key,
                use_vision_primary=use_vision_primary,
            )
            fut_t2 = executor.submit(
                _extract_tables,
                pdf_path=pdf_path_t2,
                bank_code=bank_code,
                quarter="t2",
                year=year,
                section_ranges=ranges_t2,
                use_vision_fallback=use_vision_fallback,
                api_key=api_key,
                use_vision_primary=use_vision_primary,
            )
            tables_t1 = fut_t1.result()
            tables_t2 = fut_t2.result()
    except Exception as exc:
        if "Vision schema contract invalid" in str(exc):
            raise
        return _empty_result(bank_code, year, f"Extraction impossible: {exc}")

    if not tables_t1 and not tables_t2:
        return _empty_result(
            bank_code, year, "Aucun tableau extrait depuis les sections selectionnees."
        )

    try:
        from vigilance.config import get_matching_thresholds

        cfg = get_matching_thresholds(bank_code=bank_code) or {}
        algorithm_used = "hungarian" if cfg.get("use_hungarian_matching") else "greedy"
    except Exception:
        cfg = {}
        algorithm_used = "greedy"

    raw_tables_t1_count = len(tables_t1)
    raw_tables_t2_count = len(tables_t2)
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
        should_write_extraction_audit = bool(
            vec.get("save_indicators_footnotes_json")
        ) or qg_enabled

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
                "eligible_for_review": False,
                "fail_reasons": [f"quality_gate_execution_error({exc})"],
            }
        logger.warning("Extraction writer/quality gate skipped: %s", exc)

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
            from vigilance.comparison.semantic_judge import (
                is_bank_allowed,
                _needs_semantic_validation,
                run_semantic_judge_for_pair,
                run_semantic_judge_for_unmatched,
                _needs_semantic_validation_unmatched,
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
                logger.warning("Semantic judge (added) failed for t2=%s: %s", t2_uid_added, exc)
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
                logger.warning("Semantic judge (removed) failed for t1=%s: %s", t1_uid_removed, exc)
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
    vision_pair_confidence_min = float(
        val_cfg.get("vision_pair_confidence_min", 0.75)
    )
    rename_validator_enabled = bool(val_cfg.get("rename_validator_enabled", False))
    rename_validator_confidence_min = float(
        val_cfg.get("rename_validator_confidence_min", 0.8)
    )
    rename_validator_batch_size = int(
        val_cfg.get("rename_validator_batch_size", 10)
    )
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
                    )
                    vision_pair_stats["calls"] += 1
                    if same_concept:
                        vision_pair_stats["accepted"] += 1
                    elif confidence >= vision_pair_confidence_min:
                        vision_rejected = True
                        vision_pair_stats["rejected"] += 1
                        rejected_by_vision_pair.append({
                            "table_id_t1": table_t1.table_id,
                            "table_id_t2": table_t2.table_id,
                            "title_t1": table_t1.title or "",
                            "title_t2": table_t2.title or "",
                            "indicator_overlap": pair_indicator_overlap,
                            "rescue_type": rescue_type,
                            "confidence": round(confidence, 3),
                        })
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
                        vision_rejected_added_items.append({
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
                                getattr(table_t2, "first_column_indicators_raw", None)
                                or []
                            ),
                        })
                        vision_rejected_removed_items.append({
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
                                getattr(table_t1, "first_column_indicators_raw", None)
                                or []
                            ),
                        })
                    else:
                        vision_pair_stats["accepted"] += 1
                except Exception as exc:
                    vision_pair_stats["errors"] += 1
                    logger.debug("Vision pair validation error: %s", exc)

        if vision_rejected:
            continue

        added, removed, had_fusion_split, excluded_counts = _indicator_diff(
            table_t1, table_t2
        )
        t1_clean_to_raw = _build_clean_to_raw_indicator_lookup(table_t1)
        t2_clean_to_raw = _build_clean_to_raw_indicator_lookup(table_t2)
        use_hungarian = cfg.get("indicator_hungarian_enabled", True)
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
                unmatched = indicator_debug.get("unmatched_removed_with_candidates") or []
                if unmatched:
                    all_unmatched_indicator_candidates.append({
                        "table_id_t1": table_t1.table_id,
                        "table_id_t2": table_t2.table_id,
                        "unmatched_removed_with_top_candidates": unmatched,
                    })
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

        if (
            rename_validator_enabled
            and api_key
            and renamed_pairs
        ):
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

                rename_validator_stats["calls"] = (
                    rename_validator_stats.get("calls", 0) + rv_stats.get("calls", 0)
                )
                rename_validator_stats["pairs_validated"] = (
                    rename_validator_stats.get("pairs_validated", 0)
                    + rv_stats.get("pairs_validated", 0)
                )
                rename_validator_stats["accepted"] = (
                    rename_validator_stats.get("accepted", 0)
                    + rv_stats.get("accepted", 0)
                )
                rename_validator_stats["rejected"] = (
                    rename_validator_stats.get("rejected", 0)
                    + rv_stats.get("rejected", 0)
                )
                rename_validator_stats["errors"] = (
                    rename_validator_stats.get("errors", 0)
                    + rv_stats.get("errors", 0)
                )

                accepted_set = set(accepted_pairs)
                auto_accepted_set = set(auto_accepted_pairs)
                renamed_pairs = []
                for pair_candidate in original_renamed_pairs:
                    if (
                        pair_candidate in auto_accepted_set
                        or pair_candidate in accepted_set
                    ):
                        renamed_pairs.append(pair_candidate)

                for (r_label, a_label) in rejected_pairs:
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

        if (
            indicator_validator_enabled
            and api_key
            and (added or removed)
        ):
            indicator_validator_stats["enabled"] = True
            indicator_validator_stats["use_vision"] = indicator_validator_use_vision
            added_count_before = len(added)
            removed_count_before = len(removed)
            all_t1 = list(table_t1.first_column_indicators or [])
            all_t2 = list(table_t2.first_column_indicators or [])
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
                            indicator_validator_stats.get(
                                "vision_fallback_count", 0
                            )
                            + 1
                        )
                    if (
                        vision_stats.get("vision_fallback_reason")
                        and (added or removed)
                    ):
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
                filtered_added_this = added_count_before - len(added)
                filtered_removed_this = removed_count_before - len(removed)
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
                            _strip_footnote_markers_from_indicator(removed_clean)
                        )
                    )
                    or removed_clean
                ),
                "to": str(
                    t2_clean_to_raw.get(
                        _canonical_indicator_key(
                            _strip_footnote_markers_from_indicator(added_clean)
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

        # Part E: effective_label_overlap from pair if available
        effective_label_overlap = float(
            pair.get("soft_indicator_overlap", pair_indicator_overlap) or pair_indicator_overlap
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
                "table_title_raw": (getattr(table_t1, "title_raw", None) or table_t1.title or ""),
                "table_number": getattr(table_t2, "table_number", None) or getattr(table_t1, "table_number", None),
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
                "indicator_decisions": [],
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
                    "suspicious_reason": suspicious_reason if suspicious_low_overlap else None,
                    "semantic_judge": semantic_judge_results.get(t1_uid) if semantic_judge_enabled else None,
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
        _write_match_decision_log({
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
        })

    def _source_method(uid: str, by_uid: dict) -> str:
        t = by_uid.get(uid)
        return (getattr(t, "extraction_method", None) or "docling") if t else "docling"

    added_tables_sources = list(strict.get("added_tables", [])) + vision_rejected_added_items
    removed_tables_sources = list(strict.get("removed_tables", [])) + vision_rejected_removed_items

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
            "extraction_confidence": _extraction_confidence(t2_t) if t2_t else "unknown",
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

        if added_table_validator_enabled and api_key and entry.get("bbox_t2") and entry.get("page"):
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
            "extraction_confidence": _extraction_confidence(t1_t) if t1_t else "unknown",
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
                    comparisons[idx]["genai_analysis"] = analysis

            # Classify added/removed tables (synthetic payloads for same classifier)
            def _synthetic_comp_added(t: dict[str, Any]) -> dict[str, Any]:
                return {
                    "section": t.get("section", ""),
                    "table_title": t.get("title", "") or t.get("table_id", ""),
                    "title_t1": "",
                    "title_t2": t.get("title", "") or t.get("table_id", ""),
                    "table_status": "ajoute",
                    "added_indicators": list(t.get("indicators", []) or t.get("all_indicators_t2", []))[:30],
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
                    "removed_indicators": list(t.get("indicators", []) or t.get("all_indicators_t1", []))[:30],
                    "renamed_indicators": [],
                }

            if tables_added:
                added_payloads = [_synthetic_comp_added(t) for t in tables_added]
                added_analyses = classifier.classify_batch(added_payloads)
                for t, analysis in zip(tables_added, added_analyses):
                    t["genai_analysis"] = analysis
            if tables_removed:
                removed_payloads = [_synthetic_comp_removed(t) for t in tables_removed]
                removed_analyses = classifier.classify_batch(removed_payloads)
                for t, analysis in zip(tables_removed, removed_analyses):
                    t["genai_analysis"] = analysis

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
        "incertain": sum(1 for c in comparisons if c.get("table_status") == "incertain"),
        "needs_review": 0,
        "structure_change": sum(
            1 for c in comparisons if c.get("table_status") == "structure_change"
        ),
        "ajoute": len(tables_added),
        "supprime": len(tables_removed),
    }

    now = datetime.now().isoformat(timespec="seconds")
    summary_text = (
        f"{len(comparisons)} tableaux apparies, {total_added} ajouts d'indicateurs, "
        f"{total_removed} suppressions, {len(tables_added)} tableaux ajoutes, {len(tables_removed)} supprimes."
    )

    extraction_quality_kpis = _compute_extraction_kpis(
        tables_t1, tables_t2, comparisons, tables_added, tables_removed
    )

    result: dict[str, Any] = {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": "t1",
        "quarter_to": "t2",
        "year": year,
        "summary": {
            "tables_t1": len(tables_t1),
            "tables_t2": len(tables_t2),
            "tables_matched": len(comparisons),
            "tables_added": len(tables_added),
            "tables_removed": len(tables_removed),
            "rescued_matches_count": int(strict.get("rescued_matches_count", 0) or 0),
            "split_merge_rescues_count": int(
                strict.get("split_merge_rescues_count", 0) or 0
            ),
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
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "probable_pairs": list(strict.get("probable_pairs", [])),
        "rejected_by_vision_pair": rejected_by_vision_pair,
        "debug_unmatched_candidates": list(
            strict.get("debug_unmatched_candidates", [])
        ),
        "meta": {
            "generated_at": now,
            "provenance": "comparison_runner",
            "source_format": "strict_intra_section",
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
                "fallback_vision_used": use_vision_fallback,
                "hungarian_table": cfg.get("use_hungarian_matching", False),
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
            "quality_gate": quality_gate_status,
            "extraction_artifacts": {
                "run_id": extraction_run_id,
                "out_dir": extraction_out_dir,
            },
        },
    }

    result["summary"]["tables_changed_t1"] = compute_changed_tables_t1(result)
    result["summary"]["tables_changed_t2"] = compute_changed_tables_t2(result)
    eligible_for_review = bool(quality_gate_status.get("eligible_for_review", True))
    result["summary"]["eligible_for_review"] = eligible_for_review
    result["eligible_for_review"] = eligible_for_review

    # -- GenAI executive summary enrichment (feature-flagged via bank_profiles.yaml) --
    if api_key:
        try:
            from app.genai_summary import enrich_result_with_genai

            result = enrich_result_with_genai(result)
        except Exception as exc:
            logger.warning("GenAI executive summary enrichment failed: %s", exc)

    INDICATOR_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = INDICATOR_COMPARISON_DIR / f"{bank_code}_t1_vs_t2_{year}_{stamp}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["meta"]["compare_path"] = str(out_path)

    return result
