"""Strict intra-section comparator for T1/T2 tables."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

try:
    from vigilance.embedding import EmbeddingService
except ImportError:
    EmbeddingService = None  # type: ignore[misc, assignment]

import numpy as np
from scipy.optimize import linear_sum_assignment

from vigilance.models.table_models import TableArtifact
from vigilance.utils.indicator_cleaner import (
    is_header_footer_table_title,
    normalize_indicator_for_comparison,
    strip_dates_from_table_title,
    strip_note_refs_from_title,
    strip_units_from_table_title,
)
from vigilance.utils.indicator_normalizer import get_token_sorted_text
from vigilance.utils.matching_normalizer import (
    header_schema_similarity,
    is_date_only_line,
    is_generic_title,
    is_non_indicator_line,
    normalize_for_matching,
)
from vigilance.utils.rbc_table_signals import (
    build_rbc_first_column_signals,
    classify_rbc_title_reliability,
    is_rbc_bank,
)

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (overridable via configs/bank_profiles.yaml)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, float] = {
    "overlap_threshold": 0.55,
    "overlap_floor_min": 0.35,
    "margin_threshold": 0.10,
    "unknown_section_penalty": 0.15,
    "title_similarity_min_indicator_match": 0.50,
    "borderline_score_threshold": 0.65,
    "match_score_v2": 0.70,
    "probable_score_v2": 0.62,
    "unknown_match_min_containment": 0.65,
    "unknown_match_min_score": 0.74,
    "unknown_match_min_title_similarity": 0.75,
    "unknown_match_min_structure": 0.65,
    "rescue_single_min_containment": 0.65,
    "rescue_single_min_title_similarity": 0.40,
    "rescue_split_merge_min_union_containment": 0.80,
    "rescue_split_merge_min_header_schema": 0.65,
    "title_override_min_similarity": 0.85,
    "title_override_body_similarity": 0.92,
    "title_override_min_structure": 0.50,
    "title_override_min_overlap": 0.55,
    "title_override_max_size_ratio": 0.25,
    "hash_exact_min_title_similarity": 0.85,
    "hash_exact_min_body_similarity": 0.92,
    "title_match_min_similarity": 0.88,
    "title_match_min_structure": 0.50,
    "table_number_low_overlap_header_title_min_similarity": 0.88,
    "table_number_low_overlap_header_structure_min": 0.50,
    "date_title_match_min_structure": 0.30,
    "date_title_match_min_position": 0.25,
    "soft_overlap_weight": 0.5,
    "min_label_overlap_reject": 0.55,
    "use_plan_score_formula": False,
    "weight_s_labels": 0.70,
    "weight_s_anchors": 0.15,
    "weight_s_title": 0.10,
    "weight_s_size": 0.05,
    "size_mismatch_reject_threshold": 0.60,
    "ignore_table_id_for_number": 0.0,
    "include_explanation": False,
    "split_diagnostic_max_candidates": 5.0,
    "use_post_hungarian_threshold": False,
    "hungarian_min_score": 0.62,
}


def _load_compare_thresholds(bank_code: str | None = None) -> dict[str, float]:
    """Read thresholds from config with hard-coded fallbacks."""
    try:
        from vigilance.config import get_matching_thresholds

        cfg = get_matching_thresholds(bank_code=bank_code)
    except Exception:
        cfg = {}
    result = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in cfg:
            try:
                result[key] = float(cfg[key])
            except (TypeError, ValueError):
                pass
    if "generic_titles" in cfg and isinstance(cfg.get("generic_titles"), list):
        result["generic_titles"] = list(cfg["generic_titles"])
    if "split_diagnostic_max_candidates" in cfg:
        try:
            result["split_diagnostic_max_candidates"] = int(
                cfg["split_diagnostic_max_candidates"]
            )
        except (TypeError, ValueError):
            pass
    if "table_fp_min_content_tokens" in cfg:
        try:
            result["table_fp_min_content_tokens"] = int(
                cfg["table_fp_min_content_tokens"]
            )
        except (TypeError, ValueError):
            pass
    return result


def _infer_bank_code(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
) -> str | None:
    """Infer a unique bank code from table artifacts when available."""
    candidates: set[str] = set()
    for table in [*tables_t1, *tables_t2]:
        code = str(getattr(table, "bank_code", "") or "").strip().lower()
        if code:
            candidates.add(code)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


TABLE_NUMBER_RE = re.compile(
    r"\b(?:tableau|table)[\s_\-]*([0-9]+)([a-z]?)\b", re.IGNORECASE
)
TABLE_NUMBER_SHORT_RE = re.compile(r"\bT[\s_\-]*([0-9]+)([A-Za-z]?)\b")
HEADER_FOOTER_NUMBER_LEADING_RE = re.compile(r"^\s*(\d{1,3})\b")
HEADER_FOOTER_NUMBER_TRAILING_RE = re.compile(r"\b(\d{1,3})\s*$")
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}


def _is_known_section(value: str | None) -> bool:
    return bool(value and value.strip().lower() not in UNKNOWN_SECTIONS)


def _table_uid(table: TableArtifact) -> str:
    return f"{table.section}|{table.table_id}|p{table.page_pdf}"


_RBC_SIGNALS_CACHE: dict[str, Any] = {}
_RBC_SIGNALS_CACHE_MAX = 500


def _table_cache_key(table: TableArtifact) -> str:
    return "|".join(
        [
            _table_uid(table),
            str(getattr(table, "title", "") or ""),
            str(len(getattr(table, "rows", None) or [])),
            str(len(getattr(table, "headers", None) or [])),
            str(len(getattr(table, "first_column_indicators", None) or [])),
        ]
    )


def _get_rbc_signals(table: TableArtifact) -> Any | None:
    bank_code = str(getattr(table, "bank_code", "") or "").strip().lower()
    if not is_rbc_bank(bank_code):
        return None
    uid = _table_cache_key(table)
    if uid in _RBC_SIGNALS_CACHE:
        return _RBC_SIGNALS_CACHE[uid]

    signals = build_rbc_first_column_signals(
        rows=list(getattr(table, "rows", None) or []),
        raw_indicators=list(
            getattr(table, "first_column_indicators_raw", None)
            or getattr(table, "first_column_indicators", None)
            or []
        ),
    )

    if len(_RBC_SIGNALS_CACHE) >= _RBC_SIGNALS_CACHE_MAX:
        _RBC_SIGNALS_CACHE.clear()
    _RBC_SIGNALS_CACHE[uid] = signals
    return signals


def _table_title_reliability(
    table: TableArtifact,
    bank_code: str | None = None,
) -> str:
    resolved_bank = bank_code or getattr(table, "bank_code", None)
    cached = getattr(table, "title_reliability", None)
    if cached:
        return str(cached)
    return classify_rbc_title_reliability(table.title or table.title_raw, bank_code=resolved_bank)


def _use_rbc_first_column_enrichment(table: TableArtifact) -> bool:
    if not is_rbc_bank(getattr(table, "bank_code", None)):
        return False
    explicit_groups = list(getattr(table, "first_column_groups", None) or [])
    explicit_hier = list(getattr(table, "hierarchical_indicator_signature", None) or [])
    if explicit_groups or any(" > " in str(item) for item in explicit_hier):
        return True
    signals = _get_rbc_signals(table)
    if signals is None:
        return False
    return bool(
        signals.groups_raw
        or any(" > " in str(item) for item in (signals.hierarchical_indicator_signature or []))
    )


@dataclass(slots=True, frozen=True)
class TableLabel:
    base: str
    suffix: str

    @property
    def full(self) -> str:
        return f"{self.base}{self.suffix}"


_TABLE_NUMBER_SPLIT_RE = re.compile(r"^(\d+)([a-zA-Z]?)$")


def _extract_table_label(table: TableArtifact) -> TableLabel | None:
    raw_num = getattr(table, "table_number", None)
    if raw_num and str(raw_num).strip():
        m = _TABLE_NUMBER_SPLIT_RE.match(str(raw_num).strip())
        if m:
            return TableLabel(base=str(int(m.group(1))), suffix=m.group(2).lower())

    text = (table.title or "").strip()
    if not text:
        return None

    match = TABLE_NUMBER_RE.search(text)
    if match:
        return TableLabel(base=str(int(match.group(1))), suffix=match.group(2).lower())
    match = TABLE_NUMBER_SHORT_RE.search(text)
    if match:
        return TableLabel(base=str(int(match.group(1))), suffix=match.group(2).lower())

    if is_header_footer_table_title(text, getattr(table, "bank_code", None)):
        match = HEADER_FOOTER_NUMBER_LEADING_RE.search(text)
        if match:
            return TableLabel(base=str(int(match.group(1))), suffix="")
        match = HEADER_FOOTER_NUMBER_TRAILING_RE.search(text)
        if match:
            return TableLabel(base=str(int(match.group(1))), suffix="")
    return None


def _canonical_indicator_label(value: str | None) -> str:
    """Unified canonical key for indicator labels; delegates to normalize_indicator_for_comparison."""
    return normalize_indicator_for_comparison(str(value or "").strip())


def _indicator_set(table: TableArtifact) -> set[str]:
    if _use_rbc_first_column_enrichment(table):
        signals = _get_rbc_signals(table)
        if signals and signals.indicators_clean:
            return set(signals.indicators_clean)

    values: set[str] = set()
    if table.rows:
        for row in table.rows:
            if not row:
                continue
            canonical = _canonical_indicator_label(row[0])
            if canonical:
                values.add(canonical)
    if not values and getattr(table, "first_column_indicators", None):
        for label in table.first_column_indicators:
            canonical = _canonical_indicator_label(label)
            if canonical:
                values.add(canonical)
    return values


# ---------------------------------------------------------------------------
# Table features (anchors, indicator_set_hash) for plan-aligned scoring
# ---------------------------------------------------------------------------

_FEATURES_CACHE: dict[str, tuple[list[str], str]] = {}
_FEATURES_CACHE_MAX = 500


def _get_table_features(table: TableArtifact) -> tuple[list[str], str]:
    """Return (anchors, indicator_set_hash).

    Anchors are aligned with _indicator_set (canonical labels) for consistency with
    matching logic. Hash uses vigie_extract_schema for compatibility and fast path.
    Cached by table UID for Hungarian loop efficiency.
    """
    uid = _table_cache_key(table)
    if uid in _FEATURES_CACHE:
        return _FEATURES_CACHE[uid]
    anchors = sorted(_indicator_set(table))
    hash_val = ""
    if _use_rbc_first_column_enrichment(table):
        signals = _get_rbc_signals(table)
        indicators = list(signals.indicators_raw) if signals and signals.indicators_raw else []
    else:
        indicators = list(getattr(table, "first_column_indicators", None) or [])
    indicators = [str(x).strip() for x in indicators if str(x).strip()]
    if not indicators and table.rows:
        for row in table.rows:
            if row and str(row[0]).strip():
                indicators.append(str(row[0]).strip())
    try:
        from vigilance.report.vigie_extract_schema import (
            compute_features,
            parse_first_column,
        )

        first_col = parse_first_column(indicators)
        feats = compute_features(first_col)
        hash_val = feats["indicator_set_hash"]
    except Exception:
        pass
    result = (anchors, hash_val)
    if len(_FEATURES_CACHE) < _FEATURES_CACHE_MAX:
        _FEATURES_CACHE[uid] = result
    return result


def _jaccard_anchors(table_a: TableArtifact, table_b: TableArtifact) -> float:
    """Jaccard similarity of anchor sets (normalized first-column labels)."""
    anchors_a, _ = _get_table_features(table_a)
    anchors_b, _ = _get_table_features(table_b)
    set_a = set(anchors_a)
    set_b = set(anchors_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _detect_split_diagnostic(
    table_t1: TableArtifact,
    candidates_t2: list[tuple[TableArtifact, float]],
    *,
    s_labels_min: float = 0.35,
    s_labels_max: float = 0.55,
    coverage_min: float = 0.80,
) -> list[tuple[TableArtifact, float]] | None:
    """Detect split: T1 has 2 T2 candidates with s_labels ~0.45 each, union covers T1 (plan Phase 6).

    Returns the 2 filtered candidates when split is probable, else None.
    """
    filtered = [
        (t2, sl) for t2, sl in candidates_t2 if s_labels_min <= sl <= s_labels_max
    ]
    if len(filtered) != 2:
        return None
    t2a, t2b = filtered[0][0], filtered[1][0]
    labels_t1 = _indicator_set(table_t1)
    labels_t2a = _indicator_set(t2a)
    labels_t2b = _indicator_set(t2b)
    union_t2 = labels_t2a | labels_t2b
    if not labels_t1:
        return None
    coverage = len(labels_t1 & union_t2) / len(labels_t1)
    return filtered if coverage >= coverage_min else None


def explain_match(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    score: float,
) -> dict[str, Any]:
    """Produce audit-ready explanation for a match (plan Phase 5)."""
    set_a = _indicator_set(table_t1)
    set_b = _indicator_set(table_t2)
    s_labels = _soft_label_overlap(table_t1, table_t2)
    s_anchors = _jaccard_anchors(table_t1, table_t2)
    s_title = _title_similarity(table_t1, table_t2, bank_code=None)
    s_size = _structure_similarity(table_t1, table_t2)

    common = sorted(set_a & set_b, key=lambda x: (-len(x), x))[:5]
    missing_in_t2 = sorted(set_a - set_b, key=lambda x: (-len(x), x))[:5]
    missing_in_t1 = sorted(set_b - set_a, key=lambda x: (-len(x), x))[:5]

    return {
        "score": round(score, 4),
        "subscores": {
            "labels": round(s_labels, 4),
            "anchors": round(s_anchors, 4),
            "title": round(s_title, 4),
            "size": round(s_size, 4),
        },
        "top5_common_labels": common,
        "top5_missing_in_t2": missing_in_t2,
        "top5_missing_in_t1": missing_in_t1,
        "n_indicators_t1": len(set_a),
        "n_indicators_t2": len(set_b),
    }


def _jaccard(table_a: TableArtifact, table_b: TableArtifact) -> float:
    a = _indicator_set(table_a)
    b = _indicator_set(table_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _indicator_containment(table_a: TableArtifact, table_b: TableArtifact) -> float:
    a = _indicator_set(table_a)
    b = _indicator_set(table_b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(min(len(a), len(b)), 1)


def _label_similarity(left: str, right: str) -> float:
    """Single label pair similarity, 0-1. Uses rapidfuzz token_set_ratio if available."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if rapidfuzz_fuzz is not None:
        return rapidfuzz_fuzz.token_set_ratio(left, right) / 100.0
    return SequenceMatcher(None, left, right).ratio()


def _soft_label_overlap(table_a: TableArtifact, table_b: TableArtifact) -> float:
    """
    Soft overlap of indicator labels: for each label in T1, best match in T2 (fuzzy),
    then average; symmetric for T2->T1. Handles reorder, concat, small text variations.
    """
    a = _indicator_set(table_a)
    b = _indicator_set(table_b)
    if not a or not b:
        return 0.0
    list_a = list(a)
    list_b = list(b)
    # T1 -> T2: for each label in A, max similarity with any label in B, then average
    sum_a = 0.0
    for la in list_a:
        best = max((_label_similarity(la, lb) for lb in list_b), default=0.0)
        sum_a += best
    avg_a = sum_a / len(list_a)
    # T2 -> T1
    sum_b = 0.0
    for lb in list_b:
        best = max((_label_similarity(lb, la) for la in list_a), default=0.0)
        sum_b += best
    avg_b = sum_b / len(list_b)
    return (avg_a + avg_b) / 2.0


def _normalize_title(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_for_matching(value: str | None, bank_code: str | None = None) -> str:
    """Normalize title for matching: strip notes, dates, units, then normalize."""
    value = strip_note_refs_from_title(value or "")
    value = strip_dates_from_table_title(value)
    value = strip_units_from_table_title(value, bank_code=bank_code)
    return _normalize_title(value)


def _is_date_only_title(value: str | None) -> bool:
    """Return True when title is effectively a standalone date (after note cleanup)."""
    cleaned = strip_note_refs_from_title((value or "").strip())
    if not cleaned:
        return False
    return is_date_only_line(cleaned)


def _title_similarity(
    table_a: TableArtifact, table_b: TableArtifact, bank_code: str | None = None
) -> float:
    effective_bank_code = bank_code or _infer_bank_code([table_a], [table_b])
    if is_header_footer_table_title(
        table_a.title, effective_bank_code
    ) or is_header_footer_table_title(table_b.title, effective_bank_code):
        return 0.0
    left = _title_for_matching(table_a.title, effective_bank_code)
    right = _title_for_matching(table_b.title, effective_bank_code)
    if not left and not right:
        return 1.0  # both empty (e.g. date-only CIBC titles) -> same table
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _title_body_without_table_number(
    value: str | None, bank_code: str | None = None
) -> str:
    """Remove table numbering tokens to compare semantic title bodies."""
    text = _title_for_matching(value, bank_code)
    if not text:
        return ""
    text = TABLE_NUMBER_RE.sub(" ", text)
    text = TABLE_NUMBER_SHORT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" -_")


def _table_shape(table: TableArtifact) -> tuple[int, int]:
    header_cols = len(table.headers or [])
    if header_cols <= 0:
        header_cols = max((len(row) for row in (table.rows or []) if row), default=0)
    max_row_cols = max((len(row) for row in (table.rows or []) if row), default=0)
    cols = max(header_cols, max_row_cols)
    row_count = len(table.rows or [])
    return cols, row_count


def _structure_similarity(table_a: TableArtifact, table_b: TableArtifact) -> float:
    cols_a, rows_a = _table_shape(table_a)
    cols_b, rows_b = _table_shape(table_b)

    if cols_a == 0 and cols_b == 0 and rows_a == 0 and rows_b == 0:
        return 0.0

    col_den = max(cols_a, cols_b, 1)
    row_den = max(rows_a, rows_b, 1)
    col_score = 1.0 - (abs(cols_a - cols_b) / col_den)
    row_score = 1.0 - (abs(rows_a - rows_b) / row_den)
    return max(0.0, min(1.0, (0.7 * col_score) + (0.3 * row_score)))


def _position_proximity(table_a: TableArtifact, table_b: TableArtifact) -> float:
    page_a = int(table_a.page_pdf or 0)
    page_b = int(table_b.page_pdf or 0)
    if page_a <= 0 or page_b <= 0:
        return 0.0
    diff = abs(page_a - page_b)
    # 1.0 if same page, gradually down to 0.0 when diff >= 12 pages.
    return max(0.0, 1.0 - (min(diff, 12) / 12.0))


def _context_heading(table: TableArtifact) -> str:
    headers = [str(h).strip() for h in (table.headers or []) if str(h).strip()]
    if headers:
        return " | ".join(headers[:3])
    return ""


def _context_heading_similarity(
    table_a: TableArtifact, table_b: TableArtifact
) -> float:
    left = _normalize_title(_context_heading(table_a))
    right = _normalize_title(_context_heading(table_b))
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _section_state(table_t1: TableArtifact, table_t2: TableArtifact) -> str:
    left_known = _is_known_section(table_t1.section)
    right_known = _is_known_section(table_t2.section)
    if left_known and right_known:
        return (
            "same_known" if table_t1.section == table_t2.section else "mismatch_known"
        )
    return "unknown_present"


def _sections_candidate_compatible(
    table_t1: TableArtifact, table_t2: TableArtifact
) -> bool:
    return _section_state(table_t1, table_t2) != "mismatch_known"


def _table_indicators_after_exclusions(table: TableArtifact) -> list[str]:
    """Indicators (first column) after excluding date/unit/total lines; preserves order."""
    if _use_rbc_first_column_enrichment(table):
        signals = _get_rbc_signals(table)
        raw = list(signals.indicators_raw) if signals and signals.indicators_raw else []
    else:
        raw = list(getattr(table, "first_column_indicators", None) or [])
    return [
        str(x).strip()
        for x in raw
        if x
        and str(x).strip()
        and not is_date_only_line(str(x))
        and not is_non_indicator_line(str(x))
    ]


def _table_fingerprint_sample(
    indicators: list[str], top_n: int = 3, mid_n: int = 3, bottom_n: int = 3
) -> list[str]:
    """Sample: top_n + middle mid_n + bottom bottom_n (deterministic)."""
    if not indicators:
        return []
    n = len(indicators)
    if n <= top_n + bottom_n:
        return (
            indicators[:top_n] + indicators[-bottom_n:]
            if n > top_n
            else indicators[:top_n]
        )
    top = indicators[:top_n]
    mid_start = (n - mid_n) // 2
    mid = indicators[mid_start : mid_start + mid_n]
    bottom = indicators[-bottom_n:]
    seen: set[str] = set()
    out: list[str] = []
    for part in (top, mid, bottom):
        for x in part:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _table_fingerprint_text(table: TableArtifact) -> str:
    """Build fingerprint text for embedding: title + headers + sampled indicators (top 3 + middle 3 + bottom 3)."""
    title = (table.title or "").strip() or "Untitled"
    if is_rbc_bank(getattr(table, "bank_code", None)) and _table_title_reliability(
        table,
        getattr(table, "bank_code", None),
    ) != "reliable":
        title = "Untitled"
    headers = " | ".join((table.headers or [])[:10])
    indicators = _table_indicators_after_exclusions(table)
    sampled = _table_fingerprint_sample(indicators)
    ind_text = " | ".join(str(x).strip() for x in sampled if x and str(x).strip())
    return f"Title: {title}. Headers: {headers}. Content: {ind_text}"


def _table_fingerprint_token_sorted(table: TableArtifact) -> str:
    """Order-invariant fingerprint: content part is token-sorted for matching fragmented/permuted labels."""
    title = (table.title or "").strip() or "Untitled"
    if is_rbc_bank(getattr(table, "bank_code", None)) and _table_title_reliability(
        table,
        getattr(table, "bank_code", None),
    ) != "reliable":
        title = "Untitled"
    headers = " | ".join((table.headers or [])[:10])
    indicators = _table_indicators_after_exclusions(table)
    sampled = _table_fingerprint_sample(indicators)
    content_raw = " ".join(str(x).strip() for x in sampled if x and str(x).strip())
    content_ts = get_token_sorted_text(content_raw) if content_raw else ""
    return f"Title: {title}. Headers: {headers}. Content: {content_ts}"


_TABLE_FP_BOILERPLATE = frozenset(
    {"tableau", "table", "en", "millions", "milliards", "dollars", "cad"}
)


def _table_fingerprint_embed_gate_ok(
    fingerprint_text: str, min_content_tokens: int = 5
) -> bool:
    """Only use table embedding if fingerprint has enough content tokens and is not mostly boilerplate."""
    content = ""
    if ". Content: " in fingerprint_text:
        content = fingerprint_text.split(". Content: ", 1)[-1].strip().lower()
    if not content:
        return False
    tokens = [t for t in content.split() if t and t not in _TABLE_FP_BOILERPLATE]
    return len(tokens) >= min_content_tokens


def _rbc_structural_bonus(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    *,
    bank_code: str | None,
    indicator_containment: float,
    header_schema_similarity_score: float,
) -> float:
    if not is_rbc_bank(bank_code):
        return 0.0
    if not (
        _use_rbc_first_column_enrichment(table_t1)
        or _use_rbc_first_column_enrichment(table_t2)
    ):
        return 0.0
    if indicator_containment < 0.55 or header_schema_similarity_score < 0.45:
        return 0.0

    signals_t1 = _get_rbc_signals(table_t1)
    signals_t2 = _get_rbc_signals(table_t2)
    if signals_t1 is None or signals_t2 is None:
        return 0.0

    group_a = set(signals_t1.groups_clean or [])
    group_b = set(signals_t2.groups_clean or [])
    hier_a = {
        normalize_indicator_for_comparison(item)
        for item in (signals_t1.hierarchical_indicator_signature or [])
        if str(item).strip()
    }
    hier_b = {
        normalize_indicator_for_comparison(item)
        for item in (signals_t2.hierarchical_indicator_signature or [])
        if str(item).strip()
    }

    group_score = len(group_a & group_b) / len(group_a | group_b) if group_a and group_b else 0.0
    hier_score = len(hier_a & hier_b) / len(hier_a | hier_b) if hier_a and hier_b else 0.0
    return min(0.10, (0.04 * group_score) + (0.06 * hier_score))


def _composite_score(
    *,
    table_label_score: float,
    indicator_containment: float,
    title_similarity: float,
    structure_similarity: float,
    indicator_overlap: float,
    header_schema_similarity_score: float,
    context_heading_similarity: float,
    position_proximity: float,
    embed_similarity: float = 0.0,
    thresholds: dict[str, float] | None = None,
) -> float:
    # V2 weights: label > containment > indicator > title > structure > header_schema > embed > context > position.
    th = thresholds or {}
    weights = {
        "label": float(th.get("weight_label", 0.25)),
        "containment": float(th.get("weight_containment", 0.22)),
        "indicator": float(
            th.get("weight_label_overlap", th.get("weight_indicator", 0.18))
        ),
        "title": float(th.get("weight_title", 0.14)),
        "structure": float(th.get("weight_structure", 0.10)),
        "header_schema": float(th.get("weight_header_schema", 0.08)),
        "embed": float(th.get("embedding_weight_table", th.get("weight_embed", 0.12))),
        "context": float(th.get("weight_context", 0.00)),
        "position": float(th.get("weight_position", 0.03)),
    }
    # When embedding is used, scale other weights so total approx 1.0
    embed_w = weights["embed"] if embed_similarity > 0 else 0.0
    if embed_w > 0:
        other_sum = sum(v for k, v in weights.items() if k != "embed")
        if other_sum > 0:
            scale = (1.0 - embed_w) / other_sum
            weights = {
                k: (v * scale if k != "embed" else embed_w) for k, v in weights.items()
            }
    return (
        (weights["label"] * table_label_score)
        + (weights["containment"] * indicator_containment)
        + (weights["indicator"] * indicator_overlap)
        + (weights["title"] * title_similarity)
        + (weights["structure"] * structure_similarity)
        + (weights["header_schema"] * header_schema_similarity_score)
        + (weights["embed"] * embed_similarity)
        + (weights["context"] * context_heading_similarity)
        + (weights["position"] * position_proximity)
    )


@dataclass(slots=True)
class ScoreResult:
    """Result of pair scoring for Hungarian matrix; guard rails only, no business thresholds."""

    score: float
    is_blocked: bool
    block_reason: str | None
    decision: "MatchDecision"


@dataclass(slots=True)
class MatchDecision:
    is_match: bool
    reason: str
    score: float
    section_match: bool
    table_number_match: bool
    table_label_base_match: bool
    table_label_suffix_diff: bool
    indicator_overlap: float
    title_similarity: float
    structure_similarity: float
    context_heading_similarity: float
    position_proximity: float
    t1_uid: str
    t2_uid: str
    t1_table_id: str
    t2_table_id: str
    indicator_containment: float = 0.0
    header_schema_similarity: float = 0.0
    section_state: str = "unknown_present"
    decision_level: str = "no_match"  # match | probable | no_match
    rescue_type: str | None = None
    soft_indicator_overlap: float = 0.0
    embed_sim: float = 0.0
    embed_sim_canon: float = 0.0
    embed_sim_token_sorted: float = 0.0
    table_fp_gating: str = ""
    fingerprint_token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Guard rail reasons that block a pair from Hungarian (score = -inf)
_GUARD_RAIL_REASONS = frozenset(
    {
        "cross_section_forbidden",
        "table_number_conflict",
        "low_label_overlap_reject",
        "size_mismatch_reject",
    }
)


def _compute_pair_score_with_guard_rails(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    *,
    overlap_threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
    bank_code: str | None = None,
    embedding_service: EmbeddingService | None = None,
) -> ScoreResult:
    """Compute raw score and detect guard-rail blocks. No business thresholds applied.
    Used by Hungarian post-threshold mode: pairs blocked by guard rails get -inf;
    all others get their real score for global optimization."""
    decision = match_decision(
        table_t1,
        table_t2,
        overlap_threshold=overlap_threshold,
        thresholds=thresholds,
        bank_code=bank_code,
        embedding_service=embedding_service,
    )
    is_blocked = decision.reason in _GUARD_RAIL_REASONS
    score = -1e9 if is_blocked else decision.score
    return ScoreResult(
        score=score,
        is_blocked=is_blocked,
        block_reason=decision.reason if is_blocked else None,
        decision=decision,
    )


def _top_candidates_for_row(
    *,
    row_index: int,
    scores: np.ndarray,
    decisions: list[list[MatchDecision]],
    t2_list: list[TableArtifact],
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        [
            (float(scores[row_index, j]), j)
            for j in range(len(t2_list))
            if float(scores[row_index, j]) > -1e8
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    payload: list[dict[str, Any]] = []
    for score, j in ranked[:limit]:
        decision = decisions[row_index][j]
        candidate = t2_list[j]
        payload.append(
            {
                "t2_uid": decision.t2_uid,
                "t2_table_id": candidate.table_id,
                "score": round(score, 4),
                "decision_level": decision.decision_level,
                "reason": decision.reason,
                "indicator_overlap": round(decision.indicator_overlap, 4),
                "indicator_containment": round(decision.indicator_containment, 4),
                "section_state": decision.section_state,
            }
        )
    return payload


def _top_candidates_for_column(
    *,
    col_index: int,
    scores: np.ndarray,
    decisions: list[list[MatchDecision]],
    t1_list: list[TableArtifact],
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        [
            (float(scores[i, col_index]), i)
            for i in range(len(t1_list))
            if float(scores[i, col_index]) > -1e8
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    payload: list[dict[str, Any]] = []
    for score, i in ranked[:limit]:
        decision = decisions[i][col_index]
        candidate = t1_list[i]
        payload.append(
            {
                "t1_uid": decision.t1_uid,
                "t1_table_id": candidate.table_id,
                "score": round(score, 4),
                "decision_level": decision.decision_level,
                "reason": decision.reason,
                "indicator_overlap": round(decision.indicator_overlap, 4),
                "indicator_containment": round(decision.indicator_containment, 4),
                "section_state": decision.section_state,
            }
        )
    return payload


def _score_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    return float(candidates[0]["score"]) - float(candidates[1]["score"])


def _rebuild_decision_with_level(
    decision: MatchDecision,
    *,
    decision_level: str,
) -> MatchDecision:
    return MatchDecision(
        is_match=(decision_level == "match"),
        reason=decision.reason,
        score=decision.score,
        section_match=decision.section_match,
        table_number_match=decision.table_number_match,
        table_label_base_match=decision.table_label_base_match,
        table_label_suffix_diff=decision.table_label_suffix_diff,
        indicator_overlap=decision.indicator_overlap,
        title_similarity=decision.title_similarity,
        structure_similarity=decision.structure_similarity,
        context_heading_similarity=decision.context_heading_similarity,
        position_proximity=decision.position_proximity,
        t1_uid=decision.t1_uid,
        t2_uid=decision.t2_uid,
        t1_table_id=decision.t1_table_id,
        t2_table_id=decision.t2_table_id,
        indicator_containment=decision.indicator_containment,
        header_schema_similarity=decision.header_schema_similarity,
        section_state=decision.section_state,
        decision_level=decision_level,
        rescue_type=decision.rescue_type,
        soft_indicator_overlap=decision.soft_indicator_overlap,
        embed_sim=getattr(decision, "embed_sim", 0.0),
        embed_sim_canon=getattr(decision, "embed_sim_canon", 0.0),
        embed_sim_token_sorted=getattr(decision, "embed_sim_token_sorted", 0.0),
        table_fp_gating=getattr(decision, "table_fp_gating", ""),
        fingerprint_token_count=getattr(decision, "fingerprint_token_count", 0),
    )


def _run_hungarian_assignment(
    *,
    t1_list: list[TableArtifact],
    t2_list: list[TableArtifact],
    th: dict[str, float],
    overlap_threshold: float,
    bank_code: str | None,
    margin_threshold: float,
    borderline_score: float,
    embedding_service: EmbeddingService | None,
    use_post_threshold: bool,
) -> dict[str, Any]:
    if not t1_list or not t2_list:
        return {
            "assignments": [],
            "unmatched_t1_indices": list(range(len(t1_list))),
            "unmatched_t2_indices": list(range(len(t2_list))),
            "ambiguous_t1_indices": set(),
            "ambiguous_t2_indices": set(),
            "diagnostics": {
                "score_matrix": [],
                "row_top_candidates": {},
                "column_top_candidates": {},
                "rejected_by_margin": [],
            },
        }

    match_score_v2 = th.get("match_score_v2", 0.70)
    probable_score_v2 = th.get("probable_score_v2", 0.62)
    hungarian_min_score = th.get("hungarian_min_score", probable_score_v2)

    n, m = len(t1_list), len(t2_list)
    decisions: list[list[MatchDecision]] = []
    scores = np.full((n, m), -1e9, dtype=np.float64)

    for i, t1 in enumerate(t1_list):
        row_decisions: list[MatchDecision] = []
        for j, t2 in enumerate(t2_list):
            if use_post_threshold:
                score_result = _compute_pair_score_with_guard_rails(
                    t1,
                    t2,
                    overlap_threshold=overlap_threshold,
                    thresholds=th,
                    bank_code=bank_code,
                    embedding_service=embedding_service,
                )
                row_decisions.append(score_result.decision)
                if not score_result.is_blocked:
                    scores[i, j] = score_result.score
            else:
                decision = match_decision(
                    t1,
                    t2,
                    overlap_threshold=overlap_threshold,
                    thresholds=th,
                    bank_code=bank_code,
                    embedding_service=embedding_service,
                )
                row_decisions.append(decision)
                if decision.is_match:
                    scores[i, j] = decision.score
        decisions.append(row_decisions)

    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)

    assignments: list[tuple[int, int, MatchDecision]] = []
    assigned_t1: set[int] = set()
    assigned_t2: set[int] = set()
    ambiguous_t1_indices: set[int] = set()
    ambiguous_t2_indices: set[int] = set()
    rejected_by_margin: list[dict[str, Any]] = []

    row_top_candidates = {
        str(_table_uid(t1_list[i])): _top_candidates_for_row(
            row_index=i,
            scores=scores,
            decisions=decisions,
            t2_list=t2_list,
        )
        for i in range(n)
    }
    column_top_candidates = {
        str(_table_uid(t2_list[j])): _top_candidates_for_column(
            col_index=j,
            scores=scores,
            decisions=decisions,
            t1_list=t1_list,
        )
        for j in range(m)
    }

    for k in range(len(row_ind)):
        i, j = int(row_ind[k]), int(col_ind[k])
        decision = decisions[i][j]
        if float(scores[i, j]) <= -1e8:
            continue
        assigned_t1.add(i)
        assigned_t2.add(j)
        assignments.append((i, j, decision))

    filtered_assignments: list[tuple[int, int, MatchDecision]] = []
    for i, j, decision in assignments:
        row_candidates = row_top_candidates.get(str(_table_uid(t1_list[i])), [])
        col_candidates = column_top_candidates.get(str(_table_uid(t2_list[j])), [])
        row_margin = _score_margin(row_candidates)
        col_margin = _score_margin(col_candidates)
        row_ambiguous = row_margin is not None and row_margin < margin_threshold
        col_ambiguous = col_margin is not None and col_margin < margin_threshold
        if row_ambiguous or col_ambiguous:
            assigned_t1.discard(i)
            assigned_t2.discard(j)
            ambiguous_t1_indices.add(i)
            ambiguous_t2_indices.add(j)
            if row_ambiguous and len(row_candidates) > 1:
                ambiguous_t2_indices.add(
                    next(
                        idx
                        for idx, table in enumerate(t2_list)
                        if _table_uid(table) == str(row_candidates[1]["t2_uid"])
                    )
                )
            if col_ambiguous and len(col_candidates) > 1:
                ambiguous_t1_indices.add(
                    next(
                        idx
                        for idx, table in enumerate(t1_list)
                        if _table_uid(table) == str(col_candidates[1]["t1_uid"])
                    )
                )
            rejected_by_margin.append(
                {
                    "t1_uid": decision.t1_uid,
                    "t2_uid": decision.t2_uid,
                    "score": round(decision.score, 4),
                    "row_margin": None if row_margin is None else round(row_margin, 4),
                    "column_margin": None
                    if col_margin is None
                    else round(col_margin, 4),
                    "row_candidates": row_candidates,
                    "column_candidates": col_candidates,
                }
            )
            continue

        if use_post_threshold:
            score = decision.score
            if score < hungarian_min_score:
                assigned_t1.discard(i)
                assigned_t2.discard(j)
                continue
            decision_level = (
                "match"
                if score >= match_score_v2
                else "probable"
                if score >= probable_score_v2
                else "no_match"
            )
            if decision_level == "no_match":
                assigned_t1.discard(i)
                assigned_t2.discard(j)
                continue
            decision = _rebuild_decision_with_level(
                decision,
                decision_level=decision_level,
            )

        filtered_assignments.append((i, j, decision))

    for _, _, decision in filtered_assignments:
        if decision.score < borderline_score:
            logger.warning(
                "Borderline match (score=%.3f): %s <-> %s  reason=%s  io=%.3f  ts=%.3f",
                decision.score,
                decision.t1_uid,
                decision.t2_uid,
                decision.reason,
                decision.indicator_overlap,
                decision.title_similarity,
            )

    unmatched_t1_indices = [i for i in range(n) if i not in assigned_t1]
    unmatched_t2_indices = [j for j in range(m) if j not in assigned_t2]
    score_matrix = [
        [
            None if float(scores[i, j]) <= -1e8 else round(float(scores[i, j]), 4)
            for j in range(m)
        ]
        for i in range(n)
    ]
    return {
        "assignments": filtered_assignments,
        "unmatched_t1_indices": unmatched_t1_indices,
        "unmatched_t2_indices": unmatched_t2_indices,
        "ambiguous_t1_indices": ambiguous_t1_indices,
        "ambiguous_t2_indices": ambiguous_t2_indices,
        "diagnostics": {
            "score_matrix": score_matrix,
            "row_top_candidates": row_top_candidates,
            "column_top_candidates": column_top_candidates,
            "rejected_by_margin": rejected_by_margin,
        },
    }


def match_decision(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    *,
    overlap_threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
    bank_code: str | None = None,
    embedding_service: EmbeddingService | None = None,
) -> MatchDecision:
    """Return match decision for one pair, with hard cross-section blocking."""
    th = thresholds or _load_compare_thresholds(bank_code=bank_code)
    if overlap_threshold is None:
        overlap_threshold = th["overlap_threshold"]
    overlap_floor_min = th["overlap_floor_min"]
    title_sim_min_ind = th["title_similarity_min_indicator_match"]
    match_score_v2 = th.get("match_score_v2", 0.70)
    probable_score_v2 = th.get("probable_score_v2", 0.62)
    unknown_match_min_containment = th.get("unknown_match_min_containment", 0.65)
    unknown_match_min_score = th.get("unknown_match_min_score", 0.74)
    unknown_section_penalty = th.get("unknown_section_penalty", 0.15)
    min_label_overlap_reject = th.get("min_label_overlap_reject", 0.55)

    t1_uid = _table_uid(table_t1)
    t2_uid = _table_uid(table_t2)
    section_state = _section_state(table_t1, table_t2)

    # Fast path: indicator_set_hash exact match (plan Phase 2)
    # Skip if table_number conflict: different bases block even with same indicators.
    label_t1 = _extract_table_label(table_t1)
    label_t2 = _extract_table_label(table_t2)
    ignore_table_id_for_number = bool(th.get("ignore_table_id_for_number", False))
    table_number_conflict = bool(
        label_t1
        and label_t2
        and label_t1.base != label_t2.base
        and not ignore_table_id_for_number
    )
    anchors_t1, hash_t1 = _get_table_features(table_t1)
    anchors_t2, hash_t2 = _get_table_features(table_t2)
    if (
        not table_number_conflict
        and anchors_t1
        and anchors_t2
        and hash_t1
        and hash_t2
        and hash_t1 == hash_t2
    ):
        if section_state == "same_known":
            title_sim = _title_similarity(table_t1, table_t2, bank_code=bank_code)
            body_t1 = _title_body_without_table_number(table_t1.title, bank_code)
            body_t2 = _title_body_without_table_number(table_t2.title, bank_code)
            body_similarity = (
                SequenceMatcher(None, body_t1, body_t2).ratio()
                if body_t1 and body_t2
                else 0.0
            )
            hash_title_min = float(
                th.get(
                    "hash_exact_min_title_similarity",
                    th.get("title_override_min_similarity", 0.85),
                )
            )
            hash_body_min = float(
                th.get(
                    "hash_exact_min_body_similarity",
                    th.get("title_override_body_similarity", 0.92),
                )
            )
            # Guard rail: hash alone must not force match when titles diverge semantically.
            if title_sim >= hash_title_min and body_similarity >= hash_body_min:
                tn_match = bool(
                    label_t1 and label_t2 and label_t1.full == label_t2.full
                )
                tl_base = bool(label_t1 and label_t2 and label_t1.base == label_t2.base)
                return MatchDecision(
                    is_match=True,
                    reason="indicator_set_hash_exact",
                    score=1.0,
                    section_match=True,
                    table_number_match=tn_match,
                    table_label_base_match=tl_base,
                    table_label_suffix_diff=tl_base and not tn_match,
                    indicator_overlap=1.0,
                    title_similarity=title_sim,
                    structure_similarity=_structure_similarity(table_t1, table_t2),
                    context_heading_similarity=_context_heading_similarity(
                        table_t1, table_t2
                    ),
                    position_proximity=_position_proximity(table_t1, table_t2),
                    t1_uid=t1_uid,
                    t2_uid=t2_uid,
                    t1_table_id=table_t1.table_id,
                    t2_table_id=table_t2.table_id,
                    indicator_containment=1.0,
                    header_schema_similarity=header_schema_similarity(
                        table_t1.headers or [], table_t2.headers or []
                    ),
                    section_state=section_state,
                    decision_level="match",
                    soft_indicator_overlap=1.0,
                )
            logger.debug(
                "Skip hash_exact for %s <-> %s: title_sim=%.3f body_sim=%.3f (min %.3f/%.3f)",
                t1_uid,
                t2_uid,
                title_sim,
                body_similarity,
                hash_title_min,
                hash_body_min,
            )
    section_match = section_state == "same_known"
    table_number_match = bool(label_t1 and label_t2 and label_t1.full == label_t2.full)
    table_label_base_match = bool(
        label_t1 and label_t2 and label_t1.base == label_t2.base
    )
    table_label_suffix_diff = bool(table_label_base_match and not table_number_match)
    indicator_overlap = _jaccard(table_t1, table_t2)
    soft_indicator_overlap = _soft_label_overlap(table_t1, table_t2)
    soft_overlap_weight = th.get("soft_overlap_weight", 0.5)
    effective_label_overlap = (
        1.0 - soft_overlap_weight
    ) * indicator_overlap + soft_overlap_weight * soft_indicator_overlap
    indicator_containment = _indicator_containment(table_t1, table_t2)
    title_similarity = _title_similarity(table_t1, table_t2, bank_code=bank_code)
    titles_reliable_for_score = True
    if is_rbc_bank(bank_code or getattr(table_t1, "bank_code", None)):
        titles_reliable_for_score = (
            _table_title_reliability(table_t1, bank_code) == "reliable"
            and _table_title_reliability(table_t2, bank_code) == "reliable"
        )
    effective_title_similarity = (
        title_similarity if titles_reliable_for_score else 0.0
    )
    structure_similarity = _structure_similarity(table_t1, table_t2)
    header_schema_similarity_score = header_schema_similarity(
        table_t1.headers or [],
        table_t2.headers or [],
    )
    context_heading_similarity = _context_heading_similarity(table_t1, table_t2)
    position_proximity = _position_proximity(table_t1, table_t2)

    embed_sim = 0.0
    embed_sim_canon = 0.0
    embed_sim_ts = 0.0
    table_fp_gating = "skip"
    fp_token_count = 0
    use_ts_fp = bool(th.get("use_table_token_sorted_fingerprint", True))
    if (
        th.get("use_embeddings")
        and embedding_service
        and getattr(embedding_service, "available", False)
    ):
        try:
            min_fp_tokens = int(th.get("table_fp_min_content_tokens", 5))
            txt1 = _table_fingerprint_text(table_t1)
            txt2 = _table_fingerprint_text(table_t2)
            gate_ok = _table_fingerprint_embed_gate_ok(
                txt1, min_fp_tokens
            ) and _table_fingerprint_embed_gate_ok(txt2, min_fp_tokens)
            content1 = txt1.split(". Content: ", 1)[-1] if ". Content: " in txt1 else ""
            fp_token_count = len([t for t in content1.split() if t])
            if gate_ok:
                embed_sim_canon = float(
                    embedding_service.get_single_pair_cosine(txt1, txt2)
                )
                embed_sim = embed_sim_canon
                if use_ts_fp:
                    ts1 = _table_fingerprint_token_sorted(table_t1)
                    ts2 = _table_fingerprint_token_sorted(table_t2)
                    if _table_fingerprint_embed_gate_ok(
                        ts1, min_fp_tokens
                    ) and _table_fingerprint_embed_gate_ok(ts2, min_fp_tokens):
                        embed_sim_ts = float(
                            embedding_service.get_single_pair_cosine(ts1, ts2)
                        )
                        embed_sim = max(embed_sim_canon, embed_sim_ts)
                table_fp_gating = "ok"
            else:
                table_fp_gating = "boilerplate_or_short"
        except Exception as e:
            logger.debug("Table embedding failed for %s/%s: %s", t1_uid, t2_uid, e)
            table_fp_gating = "error"

    header_footer_pair = is_header_footer_table_title(
        table_t1.title or "", bank_code
    ) and is_header_footer_table_title(table_t2.title or "", bank_code)
    # Table number is not a key for match decision; do not let it drive the score.
    table_label_score = 0.0

    use_plan_formula = bool(th.get("use_plan_score_formula", False))
    if use_plan_formula:
        s_labels = effective_label_overlap
        s_anchors = _jaccard_anchors(table_t1, table_t2)
        s_title = effective_title_similarity
        s_size = structure_similarity
        w_labels = float(
            th.get("weight_s_labels") or th.get("weight_label_overlap", 0.70)
        )
        w_anchors = float(th.get("weight_s_anchors", 0.15))
        w_title = float(th.get("weight_s_title") or th.get("weight_title", 0.10))
        w_size = float(th.get("weight_s_size") or th.get("weight_structure", 0.05))
        w_embed = (
            float(th.get("embedding_weight_table", 0.12)) if embed_sim > 0 else 0.0
        )
        composite_score = (
            w_labels * s_labels
            + w_anchors * s_anchors
            + w_title * s_title
            + w_size * s_size
        )
        if w_embed > 0:
            composite_score = (1.0 - w_embed) * composite_score + w_embed * embed_sim
    else:
        composite_score = _composite_score(
            table_label_score=table_label_score,
            indicator_containment=indicator_containment,
            title_similarity=effective_title_similarity,
            structure_similarity=structure_similarity,
            indicator_overlap=effective_label_overlap,
            header_schema_similarity_score=header_schema_similarity_score,
            context_heading_similarity=context_heading_similarity,
            position_proximity=position_proximity,
            embed_similarity=embed_sim,
            thresholds=th,
        )
    composite_score = min(
        1.0,
        composite_score
        + _rbc_structural_bonus(
            table_t1,
            table_t2,
            bank_code=bank_code or getattr(table_t1, "bank_code", None),
            indicator_containment=indicator_containment,
            header_schema_similarity_score=header_schema_similarity_score,
        ),
    )
    if section_state == "unknown_present":
        composite_score *= max(0.0, 1.0 - unknown_section_penalty)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "table_pair_score t1=%s t2=%s title_sim=%.3f structure_sim=%.3f "
            "header_schema=%.3f context=%.3f label=%.3f containment=%.3f composite=%.3f embed_sim=%.3f embed_canon=%.3f embed_ts=%.3f gating=%s",
            t1_uid,
            t2_uid,
            title_similarity,
            structure_similarity,
            header_schema_similarity_score,
            context_heading_similarity,
            effective_label_overlap,
            indicator_containment,
            composite_score,
            embed_sim,
            embed_sim_canon,
            embed_sim_ts,
            table_fp_gating,
        )

    set_a = _indicator_set(table_t1)
    set_b = _indicator_set(table_t2)
    max_indicators = max(len(set_a), len(set_b))
    few_max = int(th.get("few_indicators_max_count", 6))
    use_few_floor = max_indicators <= few_max and (
        (bank_code or "").lower() == "rbc"
        or (
            is_header_footer_table_title(table_t1.title, bank_code)
            and is_header_footer_table_title(table_t2.title, bank_code)
        )
    )
    effective_floor_min = (
        th.get("overlap_floor_min_few_indicators", 0.25)
        if use_few_floor
        else overlap_floor_min
    )
    overlap_floor = max(min(overlap_threshold, 0.30), effective_floor_min)

    raw_generic_titles = th.get("generic_titles") or []
    generic_titles_set = (
        frozenset(normalize_for_matching(str(t), "title") for t in raw_generic_titles)
        if isinstance(raw_generic_titles, list) and raw_generic_titles
        else None
    )
    both_generic = bool(
        generic_titles_set is not None
        and is_generic_title(table_t1.title or "", generic_titles_set)
        and is_generic_title(table_t2.title or "", generic_titles_set)
    )
    generic_title_min_containment = float(th.get("generic_title_min_containment", 0.70))
    generic_title_min_score = float(th.get("generic_title_min_score", 0.75))

    def _build_decision(
        *,
        is_match: bool,
        reason: str,
        score: float,
        force_no_probable: bool = False,
        rescue_type: str | None = None,
    ) -> MatchDecision:
        final_match = is_match
        final_reason = reason
        # Anti-false-match: reject if label overlap is too low (no title/number override).
        # Skip when both tables have no indicators (empty sets => overlap undefined).
        max_indicators = max(len(set_a), len(set_b))
        if (
            final_match
            and max_indicators > 0
            and soft_indicator_overlap < min_label_overlap_reject
        ):
            final_match = False
            final_reason = "low_label_overlap_reject"
        # Garde-fou taille (plan Phase 4): reject if size ratio > threshold
        size_mismatch_threshold = th.get("size_mismatch_reject_threshold", 0.60)
        n1, n2 = len(set_a), len(set_b)
        size_ratio = abs(n1 - n2) / max(n1, n2, 1)
        if final_match and size_ratio > size_mismatch_threshold:
            final_match = False
            final_reason = "size_mismatch_reject"
        decision_level = "match" if final_match else "no_match"

        if section_state == "unknown_present":
            if final_match and (
                indicator_containment < unknown_match_min_containment
                or score < unknown_match_min_score
            ):
                final_match = False
                final_reason = "unknown_section_penalized"
            if final_match and (
                title_similarity < th.get("unknown_match_min_title_similarity", 0.75)
                and structure_similarity < th.get("unknown_match_min_structure", 0.65)
            ):
                final_match = False
                final_reason = "unknown_section_penalized"
            if (
                not final_match
                and not force_no_probable
                and score >= probable_score_v2
                and indicator_containment >= 0.45
            ):
                decision_level = "probable"
                final_reason = "unknown_section_penalized"
            elif not final_match:
                decision_level = "no_match"
        else:
            if (
                final_match
                and score < match_score_v2
                and reason
                not in {
                    "table_number_match",
                    "table_number_low_overlap_rescue",
                    "date_title_structure_rescue",
                    "title_override_match",
                }
            ):
                final_match = False
            if (
                not final_match
                and not force_no_probable
                and reason not in {"cross_section_forbidden", "table_number_conflict"}
                and score >= probable_score_v2
                and indicator_containment >= 0.45
            ):
                decision_level = "probable"
            elif final_match:
                decision_level = "match"
            else:
                decision_level = "no_match"

        return MatchDecision(
            is_match=final_match,
            reason=final_reason,
            score=score,
            section_match=section_match,
            table_number_match=table_number_match,
            table_label_base_match=table_label_base_match,
            table_label_suffix_diff=table_label_suffix_diff,
            indicator_overlap=indicator_overlap,
            title_similarity=title_similarity,
            structure_similarity=structure_similarity,
            context_heading_similarity=context_heading_similarity,
            position_proximity=position_proximity,
            t1_uid=t1_uid,
            t2_uid=t2_uid,
            t1_table_id=table_t1.table_id,
            t2_table_id=table_t2.table_id,
            indicator_containment=indicator_containment,
            header_schema_similarity=header_schema_similarity_score,
            section_state=section_state,
            decision_level=decision_level,
            rescue_type=rescue_type,
            soft_indicator_overlap=soft_indicator_overlap,
            embed_sim=embed_sim,
            embed_sim_canon=embed_sim_canon,
            embed_sim_token_sorted=embed_sim_ts,
            table_fp_gating=table_fp_gating,
            fingerprint_token_count=fp_token_count,
        )

    if section_state == "mismatch_known":
        return _build_decision(
            is_match=False,
            reason="cross_section_forbidden",
            score=0.0,
            force_no_probable=True,
        )

    title_override_sim = th.get("title_override_min_similarity", 0.85)
    title_override_struct = th.get("title_override_min_structure", 0.50)
    title_override_overlap = max(
        overlap_threshold,
        th.get("title_override_min_overlap", overlap_threshold),
    )
    title_override_body_sim = th.get("title_override_body_similarity", 0.92)

    if label_t1 and label_t2 and not table_label_base_match and not header_footer_pair:
        body_t1 = _title_body_without_table_number(table_t1.title, bank_code)
        body_t2 = _title_body_without_table_number(table_t2.title, bank_code)
        body_similarity = (
            SequenceMatcher(None, body_t1, body_t2).ratio()
            if body_t1 and body_t2
            else 0.0
        )
        body_is_specific = (
            len(body_t1) >= 12
            and len(body_t2) >= 12
            and len(body_t1.split()) >= 3
            and len(body_t2.split()) >= 3
        )
        title_override_max_size_ratio = float(
            th.get("title_override_max_size_ratio", 0.25)
        )
        size_ratio_override = abs(len(set_a) - len(set_b)) / max(
            len(set_a), len(set_b), 1
        )
        title_can_override = (
            title_similarity >= title_override_sim
            and body_similarity >= title_override_body_sim
            and body_is_specific
            and effective_label_overlap >= title_override_overlap
            and structure_similarity >= title_override_struct
            and size_ratio_override <= title_override_max_size_ratio
        )
        if title_can_override:
            logger.warning(
                "Title override on number conflict: %s <-> %s  title_sim=%.3f  body_sim=%.3f  io=%.3f",
                t1_uid,
                t2_uid,
                title_similarity,
                body_similarity,
                indicator_overlap,
            )
            return _build_decision(
                is_match=True,
                reason="title_override_match",
                score=max(
                    composite_score, title_similarity * 0.85, indicator_containment
                ),
            )
        return _build_decision(
            is_match=False,
            reason="table_number_conflict",
            score=0.0,
            force_no_probable=True,
        )

    title_match_sim = th.get("title_match_min_similarity", 0.88)
    title_match_struct = th.get("title_match_min_structure", 0.50)

    # Indicator overlap is decisive unless title/structure provide a robust fallback.
    if effective_label_overlap < overlap_floor:
        low_overlap_header_title_min = th.get(
            "table_number_low_overlap_header_title_min_similarity",
            title_match_sim,
        )
        low_overlap_header_struct_min = th.get(
            "table_number_low_overlap_header_structure_min",
            title_match_struct,
        )
        raw_header_title_similarity = SequenceMatcher(
            None,
            _normalize_title(table_t1.title),
            _normalize_title(table_t2.title),
        ).ratio()
        # Rescue: strong title + structure only (table number is not a key).
        if (
            raw_header_title_similarity >= low_overlap_header_title_min
            and structure_similarity >= low_overlap_header_struct_min
        ):
            logger.debug(
                "Title/structure rescue on low label overlap: %s <-> %s  raw_title_sim=%.3f  ss=%.3f  eo=%.3f",
                t1_uid,
                t2_uid,
                raw_header_title_similarity,
                structure_similarity,
                effective_label_overlap,
            )
            return _build_decision(
                is_match=True,
                reason="table_number_low_overlap_rescue",
                score=max(composite_score, raw_header_title_similarity * 0.85, 0.80),
            )

        date_title_struct_min = th.get("date_title_match_min_structure", 0.30)
        date_title_pos_min = th.get("date_title_match_min_position", 0.25)
        if (
            _is_date_only_title(table_t1.title)
            and _is_date_only_title(table_t2.title)
            and title_similarity >= 0.98
            and structure_similarity >= date_title_struct_min
            and position_proximity >= date_title_pos_min
        ):
            logger.warning(
                "Date-title rescue on low indicators: %s <-> %s  ss=%.3f  pos=%.3f  io=%.3f",
                t1_uid,
                t2_uid,
                structure_similarity,
                position_proximity,
                indicator_overlap,
            )
            return _build_decision(
                is_match=True,
                reason="date_title_structure_rescue",
                score=max(composite_score, 0.76),
            )

        if (
            title_similarity >= title_match_sim
            and structure_similarity >= title_match_struct
        ):
            logger.warning(
                "Title override on low indicators: %s <-> %s  title_sim=%.3f  ss=%.3f  io=%.3f",
                t1_uid,
                t2_uid,
                title_similarity,
                structure_similarity,
                indicator_overlap,
            )
            return _build_decision(
                is_match=True,
                reason="title_override_match",
                score=max(composite_score, title_similarity * 0.85),
            )
        return _build_decision(
            is_match=False,
            reason="low_containment",
            score=composite_score,
        )

    # Decision is content-only: overlap + title, no table number gate.
    if (
        indicator_containment >= overlap_threshold
        and title_similarity >= title_sim_min_ind
    ):
        if both_generic and (
            indicator_containment < generic_title_min_containment
            or composite_score < generic_title_min_score
        ):
            return _build_decision(
                is_match=False,
                reason="generic_title_insufficient_signals",
                score=composite_score,
            )
        return _build_decision(
            is_match=True,
            reason="indicator_overlap_match",
            score=max(indicator_containment, composite_score),
        )

    if (
        title_similarity >= 0.90
        and structure_similarity >= 0.60
        and context_heading_similarity >= 0.35
    ) or (
        title_similarity >= 0.72
        and composite_score >= 0.72
        and effective_label_overlap >= overlap_floor
    ):
        if both_generic and (
            indicator_containment < generic_title_min_containment
            or composite_score < generic_title_min_score
        ):
            return _build_decision(
                is_match=False,
                reason="generic_title_insufficient_signals",
                score=composite_score,
            )
        return _build_decision(
            is_match=True,
            reason="multi_signal_match",
            score=composite_score,
        )

    if (
        use_few_floor
        and indicator_containment >= overlap_floor
        and structure_similarity >= 0.50
    ):
        if both_generic and (
            indicator_containment < generic_title_min_containment
            or composite_score < generic_title_min_score
        ):
            return _build_decision(
                is_match=False,
                reason="generic_title_insufficient_signals",
                score=composite_score,
            )
        return _build_decision(
            is_match=True,
            reason="few_indicators_header_footer_match",
            score=composite_score,
        )

    return _build_decision(
        is_match=False,
        reason="weak_signals",
        score=composite_score,
    )


def _match_section_hungarian(
    t1_list: list[TableArtifact],
    t2_list: list[TableArtifact],
    *,
    th: dict[str, float],
    overlap_threshold: float,
    bank_code: str | None,
    margin_threshold: float,
    borderline_score: float,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Optimal assignment via Hungarian algorithm for one candidate set."""
    return _run_hungarian_assignment(
        t1_list=t1_list,
        t2_list=t2_list,
        th=th,
        overlap_threshold=overlap_threshold,
        bank_code=bank_code,
        margin_threshold=margin_threshold,
        borderline_score=borderline_score,
        embedding_service=embedding_service,
        use_post_threshold=False,
    )


def _match_section_hungarian_post_threshold(
    t1_list: list[TableArtifact],
    t2_list: list[TableArtifact],
    *,
    th: dict[str, float],
    overlap_threshold: float,
    bank_code: str | None,
    margin_threshold: float,
    borderline_score: float,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Hungarian with post-threshold over all non-blocked candidates."""
    return _run_hungarian_assignment(
        t1_list=t1_list,
        t2_list=t2_list,
        th=th,
        overlap_threshold=overlap_threshold,
        bank_code=bank_code,
        margin_threshold=margin_threshold,
        borderline_score=borderline_score,
        embedding_service=embedding_service,
        use_post_threshold=True,
    )


def _match_tables_hungarian(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    th: dict[str, float],
    overlap_threshold: float,
    effective_bank_code: str | None,
    margin_threshold: float,
    borderline_score: float,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Hungarian assignment on known sections + symmetric residual assignment."""
    pairs: list[dict[str, Any]] = []
    probable_pairs: list[dict[str, Any]] = []
    debug_unmatched_candidates: list[dict[str, Any]] = []
    debug_unmatched_candidates_t2: list[dict[str, Any]] = []
    used_t1_uids: set[str] = set()
    used_t2_uids: set[str] = set()
    matcher_diagnostics: dict[str, Any] = {
        "phases": [],
        "rejected_by_margin": [],
    }

    use_post_threshold = bool(th.get("use_post_hungarian_threshold", False))

    sections = {
        t.section for t in tables_t1 + tables_t2 if _is_known_section(t.section)
    }

    for section in sorted(sections):
        t1_section = [t for t in tables_t1 if t.section == section]
        t2_section = [t for t in tables_t2 if t.section == section]
        if not t1_section or not t2_section:
            continue

        if use_post_threshold:
            section_matcher = _match_section_hungarian_post_threshold
        else:
            section_matcher = _match_section_hungarian

        section_result = section_matcher(
            t1_section,
            t2_section,
            th=th,
            overlap_threshold=overlap_threshold,
            bank_code=effective_bank_code,
            margin_threshold=margin_threshold,
            borderline_score=borderline_score,
            embedding_service=embedding_service,
        )
        matcher_diagnostics["phases"].append(
            {
                "phase": "hungarian_main",
                "section": section,
                "t1_uids": [_table_uid(item) for item in t1_section],
                "t2_uids": [_table_uid(item) for item in t2_section],
                **section_result["diagnostics"],
            }
        )
        matcher_diagnostics["rejected_by_margin"].extend(
            section_result["diagnostics"].get("rejected_by_margin", [])
        )

        for i, j, d in section_result["assignments"]:
            t1_obj = t1_section[i]
            t2_obj = t2_section[j]
            pair = d.to_dict()
            pair["section"] = section
            pair["page_t1"] = t1_obj.page_pdf
            pair["title_t1"] = t1_obj.title
            pair["t1_uid"] = d.t1_uid
            pair["page_t2"] = t2_obj.page_pdf
            pair["title_t2"] = t2_obj.title
            pair["t2_uid"] = d.t2_uid
            pair["match_source"] = "hungarian_main"
            used_t2_uids.add(d.t2_uid)
            used_t1_uids.add(d.t1_uid)
            if use_post_threshold and d.decision_level == "probable":
                probable_pairs.append(pair)
            else:
                pairs.append(pair)

    remaining_t1 = [t for t in tables_t1 if _table_uid(t) not in used_t1_uids]
    remaining_t2 = [t for t in tables_t2 if _table_uid(t) not in used_t2_uids]

    residual_result = (
        (
            _match_section_hungarian_post_threshold
            if use_post_threshold
            else _match_section_hungarian
        )(
            remaining_t1,
            remaining_t2,
            th=th,
            overlap_threshold=overlap_threshold,
            bank_code=effective_bank_code,
            margin_threshold=margin_threshold,
            borderline_score=borderline_score,
            embedding_service=embedding_service,
        )
        if remaining_t1 and remaining_t2
        else {
            "assignments": [],
            "unmatched_t1_indices": list(range(len(remaining_t1))),
            "unmatched_t2_indices": list(range(len(remaining_t2))),
            "ambiguous_t1_indices": set(),
            "ambiguous_t2_indices": set(),
            "diagnostics": {
                "score_matrix": [],
                "row_top_candidates": {},
                "column_top_candidates": {},
                "rejected_by_margin": [],
            },
        }
    )

    matcher_diagnostics["phases"].append(
        {
            "phase": "hungarian_residual",
            "section": "__residual__",
            "t1_uids": [_table_uid(item) for item in remaining_t1],
            "t2_uids": [_table_uid(item) for item in remaining_t2],
            **residual_result["diagnostics"],
        }
    )
    matcher_diagnostics["rejected_by_margin"].extend(
        residual_result["diagnostics"].get("rejected_by_margin", [])
    )

    for i, j, d in residual_result["assignments"]:
        t1_obj = remaining_t1[i]
        t2_obj = remaining_t2[j]
        pair = d.to_dict()
        pair["section"] = t1_obj.section
        pair["page_t1"] = t1_obj.page_pdf
        pair["title_t1"] = t1_obj.title
        pair["t1_uid"] = d.t1_uid
        pair["page_t2"] = t2_obj.page_pdf
        pair["title_t2"] = t2_obj.title
        pair["t2_uid"] = d.t2_uid
        pair["match_source"] = "hungarian_residual"
        if use_post_threshold and d.decision_level == "probable":
            probable_pairs.append(pair)
        else:
            pairs.append(pair)

    residual_ambiguous_t1 = set(residual_result["ambiguous_t1_indices"])
    residual_ambiguous_t2 = set(residual_result["ambiguous_t2_indices"])
    row_top_candidates = residual_result["diagnostics"].get("row_top_candidates", {})
    column_top_candidates = residual_result["diagnostics"].get(
        "column_top_candidates", {}
    )

    unmatched_t1: list[dict[str, Any]] = []
    for idx in residual_result["unmatched_t1_indices"]:
        table_t1 = remaining_t1[idx]
        uid = _table_uid(table_t1)
        top_candidates = list(row_top_candidates.get(uid, []))
        status = "ambiguous" if idx in residual_ambiguous_t1 else "confirmed"
        if not _is_known_section(table_t1.section):
            reason = "unknown_section"
        elif status == "ambiguous":
            reason = "ambiguous_candidate"
        elif not top_candidates:
            reason = "no_candidate_same_section"
        else:
            reason = str(top_candidates[0].get("reason", "weak_signals") or "weak_signals")
        unmatched_t1.append(
            {
                "t1_uid": uid,
                "t1_table_id": table_t1.table_id,
                "section": table_t1.section,
                "page_t1": table_t1.page_pdf,
                "title_t1": table_t1.title,
                "reason": reason,
                "unmatched_status": status,
                "best_score": float(top_candidates[0]["score"]) if top_candidates else 0.0,
                "best_decision_level": top_candidates[0]["decision_level"]
                if top_candidates
                else None,
                "best_indicator_overlap": float(top_candidates[0]["indicator_overlap"])
                if top_candidates
                else 0.0,
                "best_indicator_containment": float(
                    top_candidates[0]["indicator_containment"]
                )
                if top_candidates
                else 0.0,
            }
        )
        debug_unmatched_candidates.append({"t1_uid": uid, "candidates": top_candidates})

    unmatched_t2: list[dict[str, Any]] = []
    for idx in residual_result["unmatched_t2_indices"]:
        table_t2 = remaining_t2[idx]
        uid = _table_uid(table_t2)
        top_candidates = list(column_top_candidates.get(uid, []))
        status = "ambiguous" if idx in residual_ambiguous_t2 else "confirmed"
        if not _is_known_section(table_t2.section):
            reason = "unknown_section"
        elif status == "ambiguous":
            reason = "ambiguous_candidate"
        else:
            reason = "unmatched"
        unmatched_t2.append(
            {
                "t2_uid": uid,
                "t2_table_id": table_t2.table_id,
                "section": table_t2.section,
                "page_t2": table_t2.page_pdf,
                "title_t2": table_t2.title,
                "reason": reason,
                "unmatched_status": status,
            }
        )
        debug_unmatched_candidates_t2.append(
            {"t2_uid": uid, "candidates": top_candidates}
        )

    return {
        "pairs": pairs,
        "probable_pairs": probable_pairs,
        "unmatched_t1": unmatched_t1,
        "unmatched_t2": unmatched_t2,
        "debug_unmatched_candidates": debug_unmatched_candidates,
        "debug_unmatched_candidates_t2": debug_unmatched_candidates_t2,
        "matching_diagnostics": matcher_diagnostics,
    }


def match_tables_intra_section(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    overlap_threshold: float | None = None,
    bank_code: str | None = None,
    use_hungarian: bool | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Table matching with strict intra-section gating using the symmetric assignment matcher."""
    effective_bank_code = bank_code or _infer_bank_code(tables_t1, tables_t2)
    th = _load_compare_thresholds(bank_code=effective_bank_code)
    if overlap_threshold is None:
        overlap_threshold = th["overlap_threshold"]
    margin_threshold = th["margin_threshold"]
    borderline_score = th["borderline_score_threshold"]

    requested_mode = use_hungarian
    if use_hungarian is None:
        try:
            from vigilance.config import get_matching_thresholds

            get_matching_thresholds(bank_code=effective_bank_code)
        except Exception:
            pass

    logger.info(
        "Table matching: bank=%s matcher=symmetric_assignment requested_use_hungarian=%s",
        effective_bank_code,
        requested_mode,
    )
    if requested_mode is False:
        logger.warning(
            "use_hungarian=False is deprecated; the symmetric assignment matcher is used regardless."
        )

    return _match_tables_hungarian(
        tables_t1,
        tables_t2,
        th=th,
        overlap_threshold=overlap_threshold,
        effective_bank_code=effective_bank_code,
        margin_threshold=margin_threshold,
        borderline_score=borderline_score,
        embedding_service=embedding_service,
    )


def _allow_rbc_split_merge_rescue(
    *,
    bank_code: str | None,
    primary_table: TableArtifact,
    counterpart_table: TableArtifact,
    primary_decision: MatchDecision,
    union_containment: float,
    schema_score: float,
) -> bool:
    if not is_rbc_bank(bank_code):
        return True

    if _table_title_reliability(primary_table, bank_code) == "reliable":
        return True

    if union_containment < 0.90:
        return False
    if primary_decision.indicator_containment < 0.55:
        return False
    if primary_decision.structure_similarity < 0.55:
        return False
    if schema_score < 0.70:
        return False
    return True


def run_strict_intra_section_compare(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    overlap_threshold: float | None = None,
    bank_code: str | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Official comparison facade with strict section gating and explicit added/removed outputs."""
    effective_bank_code = bank_code or _infer_bank_code(tables_t1, tables_t2)
    th = _load_compare_thresholds(bank_code=effective_bank_code)
    if overlap_threshold is None:
        overlap_threshold = th["overlap_threshold"]

    base = match_tables_intra_section(
        tables_t1=tables_t1,
        tables_t2=tables_t2,
        overlap_threshold=overlap_threshold,
        bank_code=effective_bank_code,
        embedding_service=embedding_service,
    )
    table_by_t1_uid = {_table_uid(t): t for t in tables_t1}
    table_by_t2_uid = {_table_uid(t): t for t in tables_t2}

    pairs = list(base.get("pairs", []))
    probable_pairs = list(base.get("probable_pairs", []))
    debug_unmatched_candidates = list(base.get("debug_unmatched_candidates", []))
    debug_unmatched_candidates_t2 = list(base.get("debug_unmatched_candidates_t2", []))
    unmatched_t1 = list(base.get("unmatched_t1", []))
    unmatched_t2 = list(base.get("unmatched_t2", []))
    matching_diagnostics = dict(base.get("matching_diagnostics", {}))

    remaining_t1 = {
        str(item.get("t1_uid", "")): item for item in unmatched_t1 if item.get("t1_uid")
    }
    remaining_t2 = {
        str(item.get("t2_uid", "")): item for item in unmatched_t2 if item.get("t2_uid")
    }
    rescue_single_min_containment = th.get("rescue_single_min_containment", 0.70)
    rescue_single_min_title_similarity = th.get(
        "rescue_single_min_title_similarity", 0.45
    )
    rescue_split_merge_min_union_containment = th.get(
        "rescue_split_merge_min_union_containment", 0.80
    )
    rescue_split_merge_min_header_schema = th.get(
        "rescue_split_merge_min_header_schema", 0.65
    )
    rescued_matches_count = 0
    split_merge_rescues_count = 0

    # Rescue pass 1: bidirectional one-to-one rescue on remaining unmatched.
    rescue_candidates: dict[tuple[str, str], tuple[MatchDecision, str]] = {}
    for t1_uid in list(remaining_t1.keys()):
        table_t1 = table_by_t1_uid.get(t1_uid)
        if table_t1 is None:
            continue
        best_decision: MatchDecision | None = None
        best_t2_uid = ""
        for t2_uid in remaining_t2.keys():
            table_t2 = table_by_t2_uid.get(t2_uid)
            if table_t2 is None or not _sections_candidate_compatible(table_t1, table_t2):
                continue
            decision = match_decision(
                table_t1,
                table_t2,
                overlap_threshold=overlap_threshold,
                thresholds=th,
                bank_code=effective_bank_code,
                embedding_service=embedding_service,
            )
            if best_decision is None or decision.score > best_decision.score:
                best_decision = decision
                best_t2_uid = t2_uid
        if (
            best_decision is not None
            and best_t2_uid
            and best_decision.indicator_containment >= rescue_single_min_containment
            and best_decision.title_similarity >= rescue_single_min_title_similarity
        ):
            rescue_candidates[(t1_uid, best_t2_uid)] = (best_decision, "t1_to_t2")

    for t2_uid in list(remaining_t2.keys()):
        table_t2 = table_by_t2_uid.get(t2_uid)
        if table_t2 is None:
            continue
        best_decision: MatchDecision | None = None
        best_t1_uid = ""
        for t1_uid in remaining_t1.keys():
            table_t1 = table_by_t1_uid.get(t1_uid)
            if table_t1 is None or not _sections_candidate_compatible(table_t1, table_t2):
                continue
            decision = match_decision(
                table_t1,
                table_t2,
                overlap_threshold=overlap_threshold,
                thresholds=th,
                bank_code=effective_bank_code,
                embedding_service=embedding_service,
            )
            if best_decision is None or decision.score > best_decision.score:
                best_decision = decision
                best_t1_uid = t1_uid
        if (
            best_decision is not None
            and best_t1_uid
            and best_decision.indicator_containment >= rescue_single_min_containment
            and best_decision.title_similarity >= rescue_single_min_title_similarity
        ):
            current = rescue_candidates.get((best_t1_uid, t2_uid))
            if current is None or best_decision.score > current[0].score:
                rescue_candidates[(best_t1_uid, t2_uid)] = (best_decision, "t2_to_t1")

    for (t1_uid, t2_uid), (decision, rescue_origin) in sorted(
        rescue_candidates.items(),
        key=lambda item: item[1][0].score,
        reverse=True,
    ):
        if t1_uid not in remaining_t1 or t2_uid not in remaining_t2:
            continue
        table_t1 = table_by_t1_uid.get(t1_uid)
        table_t2 = table_by_t2_uid.get(t2_uid)
        if table_t1 is None:
            continue
        pair = decision.to_dict()
        pair["section"] = table_t1.section
        pair["page_t1"] = table_t1.page_pdf
        pair["title_t1"] = table_t1.title
        pair["t1_uid"] = t1_uid
        if table_t2 is not None:
            pair["page_t2"] = table_t2.page_pdf
            pair["title_t2"] = table_t2.title
        pair["t2_uid"] = t2_uid
        pair["reason"] = "single_rescue"
        pair["rescue_type"] = "single_rescue"
        pair["decision_level"] = "match"
        pair["match_source"] = "single_rescue"
        pair["rescue_origin"] = rescue_origin
        pairs.append(pair)
        rescued_matches_count += 1
        remaining_t1.pop(t1_uid, None)
        remaining_t2.pop(t2_uid, None)

    # Rescue pass 2a: one T1 matches union of two T2 fragments.
    for t1_uid in list(remaining_t1.keys()):
        table_t1 = table_by_t1_uid.get(t1_uid)
        if table_t1 is None:
            continue
        t1_set = _indicator_set(table_t1)
        if len(t1_set) < 2:
            continue
        best_combo: tuple[str, str, float, float, float] | None = None
        for left_uid, right_uid in combinations(list(remaining_t2.keys()), 2):
            left_table = table_by_t2_uid.get(left_uid)
            right_table = table_by_t2_uid.get(right_uid)
            if left_table is None or right_table is None:
                continue
            if not _sections_candidate_compatible(table_t1, left_table):
                continue
            if not _sections_candidate_compatible(table_t1, right_table):
                continue
            union_set = _indicator_set(left_table) | _indicator_set(right_table)
            if not union_set:
                continue
            union_containment = len(t1_set & union_set) / max(
                min(len(t1_set), len(union_set)), 1
            )
            schema_score = max(
                header_schema_similarity(
                    table_t1.headers or [], left_table.headers or []
                ),
                header_schema_similarity(
                    table_t1.headers or [], right_table.headers or []
                ),
            )
            if (
                union_containment >= rescue_split_merge_min_union_containment
                and schema_score >= rescue_split_merge_min_header_schema
            ):
                combo_score = (0.8 * union_containment) + (0.2 * schema_score)
                if best_combo is None or combo_score > best_combo[2]:
                    best_combo = (
                        left_uid,
                        right_uid,
                        combo_score,
                        union_containment,
                        schema_score,
                    )
        if not best_combo:
            continue
        left_uid, right_uid, _, union_containment, schema_score = best_combo
        left_table = table_by_t2_uid.get(left_uid)
        right_table = table_by_t2_uid.get(right_uid)
        if left_table is None or right_table is None:
            continue
        left_cont = _indicator_containment(table_t1, left_table)
        right_cont = _indicator_containment(table_t1, right_table)
        primary_uid = left_uid if left_cont >= right_cont else right_uid
        primary_table = left_table if primary_uid == left_uid else right_table
        decision = match_decision(
            table_t1,
            primary_table,
            overlap_threshold=overlap_threshold,
            thresholds=th,
            bank_code=effective_bank_code,
            embedding_service=embedding_service,
        )
        if decision.reason == "table_number_conflict":
            continue
        if not _allow_rbc_split_merge_rescue(
            bank_code=effective_bank_code,
            primary_table=primary_table,
            counterpart_table=table_t1,
            primary_decision=decision,
            union_containment=union_containment,
            schema_score=schema_score,
        ):
            continue
        pair = decision.to_dict()
        pair["section"] = table_t1.section
        pair["page_t1"] = table_t1.page_pdf
        pair["title_t1"] = table_t1.title
        pair["t1_uid"] = t1_uid
        pair["page_t2"] = primary_table.page_pdf
        pair["title_t2"] = primary_table.title
        pair["t2_uid"] = primary_uid
        pair["reason"] = "split_merge_rescue"
        pair["rescue_type"] = "split_merge_rescue"
        pair["decision_level"] = "match"
        pair["match_source"] = "split_merge_rescue"
        pair["split_members_t2"] = [left_uid, right_uid]
        pair["split_probable"] = True
        pairs.append(pair)
        rescued_matches_count += 1
        split_merge_rescues_count += 1
        remaining_t1.pop(t1_uid, None)
        remaining_t2.pop(left_uid, None)
        remaining_t2.pop(right_uid, None)

    # Rescue pass 2b: one T2 matches union of two T1 fragments.
    for t2_uid in list(remaining_t2.keys()):
        table_t2 = table_by_t2_uid.get(t2_uid)
        if table_t2 is None:
            continue
        t2_set = _indicator_set(table_t2)
        if len(t2_set) < 2:
            continue
        best_combo_t1: tuple[str, str, float, float, float] | None = None
        for left_uid, right_uid in combinations(list(remaining_t1.keys()), 2):
            left_table = table_by_t1_uid.get(left_uid)
            right_table = table_by_t1_uid.get(right_uid)
            if left_table is None or right_table is None:
                continue
            if not _sections_candidate_compatible(left_table, table_t2):
                continue
            if not _sections_candidate_compatible(right_table, table_t2):
                continue
            union_set = _indicator_set(left_table) | _indicator_set(right_table)
            if not union_set:
                continue
            union_containment = len(t2_set & union_set) / max(
                min(len(t2_set), len(union_set)), 1
            )
            schema_score = max(
                header_schema_similarity(
                    left_table.headers or [], table_t2.headers or []
                ),
                header_schema_similarity(
                    right_table.headers or [], table_t2.headers or []
                ),
            )
            if (
                union_containment >= rescue_split_merge_min_union_containment
                and schema_score >= rescue_split_merge_min_header_schema
            ):
                combo_score = (0.8 * union_containment) + (0.2 * schema_score)
                if best_combo_t1 is None or combo_score > best_combo_t1[2]:
                    best_combo_t1 = (
                        left_uid,
                        right_uid,
                        combo_score,
                        union_containment,
                        schema_score,
                    )
        if not best_combo_t1:
            continue
        left_uid, right_uid, _, union_containment, schema_score = best_combo_t1
        left_table = table_by_t1_uid.get(left_uid)
        right_table = table_by_t1_uid.get(right_uid)
        if left_table is None or right_table is None:
            continue
        left_cont = _indicator_containment(left_table, table_t2)
        right_cont = _indicator_containment(right_table, table_t2)
        primary_uid = left_uid if left_cont >= right_cont else right_uid
        primary_table = left_table if primary_uid == left_uid else right_table
        decision = match_decision(
            primary_table,
            table_t2,
            overlap_threshold=overlap_threshold,
            thresholds=th,
            bank_code=effective_bank_code,
            embedding_service=embedding_service,
        )
        if decision.reason == "table_number_conflict":
            continue
        if not _allow_rbc_split_merge_rescue(
            bank_code=effective_bank_code,
            primary_table=primary_table,
            counterpart_table=table_t2,
            primary_decision=decision,
            union_containment=union_containment,
            schema_score=schema_score,
        ):
            continue
        pair = decision.to_dict()
        pair["section"] = primary_table.section
        pair["page_t1"] = primary_table.page_pdf
        pair["title_t1"] = primary_table.title
        pair["t1_uid"] = primary_uid
        pair["page_t2"] = table_t2.page_pdf
        pair["title_t2"] = table_t2.title
        pair["t2_uid"] = t2_uid
        pair["reason"] = "split_merge_rescue"
        pair["rescue_type"] = "split_merge_rescue"
        pair["decision_level"] = "match"
        pair["match_source"] = "split_merge_rescue"
        pair["merge_members_t1"] = [left_uid, right_uid]
        pair["merge_probable"] = True
        pairs.append(pair)
        rescued_matches_count += 1
        split_merge_rescues_count += 1
        remaining_t1.pop(left_uid, None)
        remaining_t1.pop(right_uid, None)
        remaining_t2.pop(t2_uid, None)

    unmatched_t1 = [
        item for item in unmatched_t1 if str(item.get("t1_uid", "")) in remaining_t1
    ]
    unmatched_t2 = [
        item for item in unmatched_t2 if str(item.get("t2_uid", "")) in remaining_t2
    ]

    removed_tables_raw: list[dict[str, Any]] = []
    for item in unmatched_t1:
        if item.get("unmatched_status") != "confirmed":
            continue
        if item.get("reason") not in {
            "no_candidate_same_section",
            "low_indicator_overlap",
            "low_containment",
            "table_number_conflict",
            "unknown_section_penalized",
            "weak_signals",
        }:
            continue
        t1_uid = str(item.get("t1_uid", ""))
        table_t1 = table_by_t1_uid.get(t1_uid)
        removed_tables_raw.append(
            {
                "t1_uid": t1_uid,
                "t1_table_id": item["t1_table_id"],
                "section": item["section"],
                "page_t1": item.get("page_t1"),
                "title_t1": item.get("title_t1"),
                "reason": "removed_table",
                "source_reason": item["reason"],
                "first_column_indicators": list(
                    getattr(table_t1, "first_column_indicators", []) or []
                ),
                "first_column_indicators_raw": list(
                    getattr(table_t1, "first_column_indicators_raw", None) or []
                ),
            }
        )

    added_tables_raw: list[dict[str, Any]] = []
    for item in unmatched_t2:
        if item.get("unmatched_status") != "confirmed":
            continue
        if item.get("reason") != "unmatched":
            continue
        t2_uid = str(item.get("t2_uid", ""))
        table_t2 = table_by_t2_uid.get(t2_uid)
        added_tables_raw.append(
            {
                "t2_uid": t2_uid,
                "t2_table_id": item["t2_table_id"],
                "section": item["section"],
                "page_t2": item.get("page_t2"),
                "title_t2": item.get("title_t2"),
                "reason": "added_table",
                "source_reason": item["reason"],
                "first_column_indicators": list(
                    getattr(table_t2, "first_column_indicators", []) or []
                ),
                "first_column_indicators_raw": list(
                    getattr(table_t2, "first_column_indicators_raw", None) or []
                ),
            }
        )

    # Stable de-duplication for overlaps/noisy extraction.
    removed_seen: set[tuple[str, str, Any]] = set()
    removed_tables: list[dict[str, Any]] = []
    for item in removed_tables_raw:
        key = (
            str(item.get("section", "")),
            str(item.get("t1_table_id", "")),
            item.get("page_t1"),
        )
        if key in removed_seen:
            continue
        removed_seen.add(key)
        removed_tables.append(item)

    added_seen: set[tuple[str, str, Any]] = set()
    added_tables: list[dict[str, Any]] = []
    for item in added_tables_raw:
        key = (
            str(item.get("section", "")),
            str(item.get("t2_table_id", "")),
            item.get("page_t2"),
        )
        if key in added_seen:
            continue
        added_seen.add(key)
        added_tables.append(item)

    reason_counts: dict[str, int] = {}
    for pair in pairs:
        reason = str(pair.get("reason", "")).strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for item in unmatched_t1:
        reason = str(item.get("reason", "")).strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for item in unmatched_t2:
        reason = str(item.get("reason", "")).strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Add explain_match when include_explanation is enabled (plan Phase 5)
    include_explanation_flag = bool(th.get("include_explanation", False))
    if include_explanation_flag:
        for pair in pairs:
            t1_uid = str(pair.get("t1_uid", ""))
            t2_uid = str(pair.get("t2_uid", ""))
            tbl1 = table_by_t1_uid.get(t1_uid)
            tbl2 = table_by_t2_uid.get(t2_uid)
            if tbl1 is not None and tbl2 is not None:
                score_val = float(pair.get("score", 0))
                pair["explanation"] = explain_match(tbl1, tbl2, score_val)

    # Diagnostics: split_probable for unmatched T1 (plan Phase 6)
    diagnostics: list[dict[str, Any]] = []
    remaining_t1_set = set(remaining_t1.keys())
    for item in unmatched_t1:
        t1_uid = str(item.get("t1_uid", ""))
        if t1_uid not in remaining_t1_set:
            continue
        debug_entry = next(
            (d for d in debug_unmatched_candidates if d.get("t1_uid") == t1_uid),
            None,
        )
        if not debug_entry:
            continue
        table_t1 = table_by_t1_uid.get(t1_uid)
        if not table_t1:
            continue
        candidates_with_slabels: list[tuple[TableArtifact, float]] = []
        for c in debug_entry.get("candidates", []):
            t2_uid = str(c.get("t2_uid", ""))
            table_t2 = table_by_t2_uid.get(t2_uid)
            if table_t2 is not None:
                sl = _soft_label_overlap(table_t1, table_t2)
                candidates_with_slabels.append((table_t2, sl))
        split_candidates = _detect_split_diagnostic(table_t1, candidates_with_slabels)
        if split_candidates is not None:
            diagnostics.append(
                {
                    "t1_uid": t1_uid,
                    "reason": "split_probable",
                    "candidates_t2": [
                        {"t2_uid": _table_uid(t2), "t2_table_id": t2.table_id}
                        for t2, _ in split_candidates
                    ],
                }
            )

    # Validation 1-to-1: no duplicate T1/T2 in pairs, and every table appears in pairs or remaining (or in split/merge members).
    pair_t1_uids = {str(p.get("t1_uid", "")) for p in pairs}
    pair_t2_uids = {str(p.get("t2_uid", "")) for p in pairs}
    assert len(pair_t1_uids) == len(pairs), "duplicate t1_uid in pairs"
    assert len(pair_t2_uids) == len(pairs), "duplicate t2_uid in pairs"
    merge_members_t1_set: set[str] = set()
    for p in pairs:
        for uid in p.get("merge_members_t1") or []:
            merge_members_t1_set.add(str(uid))
    split_members_t2_set: set[str] = set()
    for p in pairs:
        for uid in p.get("split_members_t2") or []:
            split_members_t2_set.add(str(uid))
    covered_t1 = pair_t1_uids | merge_members_t1_set | set(remaining_t1.keys())
    covered_t2 = pair_t2_uids | split_members_t2_set | set(remaining_t2.keys())
    all_t1_uids = set(table_by_t1_uid.keys())
    all_t2_uids = set(table_by_t2_uid.keys())

    # Ensure any T1/T2 UID missing from coverage (e.g. empty t1_uid in unmatched item) is added to remaining and unmatched.
    missing_t1 = all_t1_uids - covered_t1
    for uid in missing_t1:
        tbl = table_by_t1_uid.get(uid)
        if tbl is None:
            continue
        remaining_t1[uid] = {
            "t1_uid": uid,
            "t1_table_id": getattr(tbl, "table_id", ""),
            "section": getattr(tbl, "section", ""),
            "page_t1": getattr(tbl, "page_pdf", None),
            "title_t1": getattr(tbl, "title", ""),
            "reason": "coverage_fixup",
            "unmatched_status": "confirmed",
        }
        unmatched_t1.append(remaining_t1[uid])
    missing_t2 = all_t2_uids - covered_t2
    for uid in missing_t2:
        tbl = table_by_t2_uid.get(uid)
        if tbl is None:
            continue
        remaining_t2[uid] = {
            "t2_uid": uid,
            "t2_table_id": getattr(tbl, "table_id", ""),
            "section": getattr(tbl, "section", ""),
            "page_t2": getattr(tbl, "page_pdf", None),
            "title_t2": getattr(tbl, "title", ""),
            "reason": "unmatched",
            "unmatched_status": "confirmed",
        }
        unmatched_t2.append(remaining_t2[uid])

    covered_t1 = pair_t1_uids | merge_members_t1_set | set(remaining_t1.keys())
    covered_t2 = pair_t2_uids | split_members_t2_set | set(remaining_t2.keys())
    assert all_t1_uids == covered_t1, (
        "T1 coverage: every t1_uid must be in pairs, merge_members_t1, or remaining_t1"
    )
    assert all_t2_uids == covered_t2, (
        "T2 coverage: every t2_uid must be in pairs, split_members_t2, or remaining_t2"
    )

    result = {
        "pairs": pairs,
        "probable_pairs": probable_pairs,
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "unmatched_t1": unmatched_t1,
        "unmatched_t2": unmatched_t2,
        "debug_unmatched_candidates": debug_unmatched_candidates,
        "debug_unmatched_candidates_t2": debug_unmatched_candidates_t2,
        "rescued_matches_count": rescued_matches_count,
        "split_merge_rescues_count": split_merge_rescues_count,
        "reasons": reason_counts,
        "diagnostics": diagnostics,
        "matching_diagnostics": matching_diagnostics,
        "unmatched_confirmed_t1": [
            item for item in unmatched_t1 if item.get("unmatched_status") == "confirmed"
        ],
        "unmatched_ambiguous_t1": [
            item for item in unmatched_t1 if item.get("unmatched_status") == "ambiguous"
        ],
        "unmatched_confirmed_t2": [
            item for item in unmatched_t2 if item.get("unmatched_status") == "confirmed"
        ],
        "unmatched_ambiguous_t2": [
            item for item in unmatched_t2 if item.get("unmatched_status") == "ambiguous"
        ],
    }
    return result
