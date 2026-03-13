"""Official table pairing engine for the active comparison pipeline.

This module implements a conservative pairing flow:

1. Build canonical comparison views from ``TableArtifact``.
2. Generate a small, high-recall shortlist per current-quarter table.
3. Route the shortlist through a final decision layer.
4. Emit explicit ``matched / ambiguous / unmatched`` states.

The default router is deterministic and conservative. An optional LLM router can
be enabled via configuration for final candidate routing, but the engine keeps
the shortlist deterministic and bounded.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from vigilance.config import get_matching_thresholds
from vigilance.models.table_models import (
    TableArtifact,
    derive_extraction_blockers,
    get_comparison_indicators,
    get_extraction_confidence,
    get_extraction_quality_flags,
    get_extraction_quality_profile,
    get_extraction_status,
    get_vision_raw_indicators,
    is_auto_compare_eligible,
)
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.matching_normalizer import (
    header_literal_fingerprint,
    header_schema_similarity,
    is_generic_title,
    normalize_for_matching,
    strip_temporal_expressions,
)

logger = logging.getLogger(__name__)

UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}
DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_ROUTER_MODEL = "gpt-4o-mini"
DEFAULT_ROUTER_TIMEOUT = 30.0

_TABLE_NUMBER_RE = re.compile(
    r"\b(?:tableau|table)\s*([a-z]?\d+[a-z]?|\d+[a-z]?)\b",
    re.IGNORECASE,
)
_TABLE_NUMBER_SHORT_RE = re.compile(r"\bT\s*([0-9]+[A-Za-z]?)\b")

# Page-local structure scoring (same-page / near-page multi-table matching)
PAGE_LOCAL_ORDER_MATCH_SAME_PAGE = 0.20
PAGE_LOCAL_ORDER_CONFLICT_SAME_PAGE = -0.15
PAGE_LOCAL_ORDER_MATCH_NEAR_PAGE = 0.12
PAGE_LOCAL_ORDER_CONFLICT_NEAR_PAGE = -0.08
PAGE_LOCAL_ROLE_MATCH_BONUS = 0.06
BBOX_Y_SIMILARITY_WEIGHT = 0.05

_GENERIC_INDICATOR_KEYS = frozenset(
    {
        "total",
        "autres",
        "other",
        "canada",
        "etats unis",
        "etatsunis",
        "united states",
        "europe",
        "royaume uni",
        "uk",
        "asie",
        "amerique latine",
        "amlat",
        "particuliers",
        "entreprises",
        "secteur public",
        "administrations publiques",
    }
)


MAX_SHORTLIST_SIZE = 15

_CONFIG_KEY_MAP = {
    "weight_label_overlap": "w_distinctive_overlap",
    "weight_containment": "w_containment",
    "weight_title": "w_title",
    "weight_structure": "w_size",
    "weight_position": "w_order_proximity",
    "weight_header_schema": "w_header_compatibility",
    "w_distinctive_overlap": "w_distinctive_overlap",
    "w_containment": "w_containment",
    "w_header_compatibility": "w_header_compatibility",
    "w_header_fingerprint": "w_header_fingerprint",
    "w_title": "w_title",
    "w_section": "w_section",
    "w_size": "w_size",
    "w_order_proximity": "w_order_proximity",
    "w_indicator_ordering": "w_indicator_ordering",
    "table_number_bonus_value": "table_number_bonus",
    "table_number_penalty_value": "table_number_penalty",
}


@dataclass(frozen=True)
class ScoringProfile:
    """Configurable scoring weights for candidate pair scoring.

    Built from ``get_matching_thresholds()`` with per-bank overrides.
    Supports adaptive per-pair adjustment via :func:`_adapt_weights`.
    """

    w_distinctive_overlap: float = 0.36
    w_containment: float = 0.23
    w_header_compatibility: float = 0.12
    w_header_fingerprint: float = 0.08
    w_title: float = 0.11
    w_section: float = 0.08
    w_size: float = 0.06
    w_order_proximity: float = 0.04
    w_indicator_ordering: float = 0.04
    table_number_bonus: float = 0.15
    table_number_penalty: float = -0.05
    adaptive_mode: str = "default"

    @classmethod
    def from_thresholds(cls, thresholds: dict[str, Any]) -> "ScoringProfile":
        kwargs: dict[str, Any] = {}
        for config_key, field_name in _CONFIG_KEY_MAP.items():
            if config_key in thresholds:
                try:
                    kwargs[field_name] = float(thresholds[config_key])
                except (TypeError, ValueError):
                    pass
        return cls(**kwargs)


def _adapt_weights(
    profile: ScoringProfile,
    *,
    title_reliability: float,
    title_sim: float,
    n_indicators: int,
    has_table_number: bool,
) -> ScoringProfile:
    """Adapt scoring weights based on signal availability for this specific pair.
    Table number is not used as a positive signal (zero-trust policy).
    """
    del has_table_number  # not used for adaptation
    if n_indicators <= 4:
        return replace(
            profile,
            w_distinctive_overlap=0.10,
            w_containment=0.10,
            w_title=0.30,
            w_header_compatibility=0.10,
            w_header_fingerprint=0.15,
            w_size=0.08,
            w_section=0.06,
            w_order_proximity=0.06,
            w_indicator_ordering=0.0,
            adaptive_mode="few_indicators",
        )

    if title_reliability >= 0.7 and title_sim >= 0.85:
        return replace(
            profile,
            w_title=0.22,
            w_distinctive_overlap=0.28,
            w_containment=0.20,
            adaptive_mode="strong_title",
        )

    return profile


def _indicator_order_similarity(keys1: list[str], keys2: list[str]) -> float:
    """Normalized concordance ratio on shared indicator positions.

    Returns 1.0 when shared indicators appear in the same relative order in both
    tables (strong evidence of same table), 0.0 when fewer than 3 indicators
    overlap (insufficient signal).
    """
    s2 = {k: i for i, k in enumerate(keys2)}
    common = [(i, s2[k]) for i, k in enumerate(keys1) if k in s2]
    if len(common) < 3:
        return 0.0
    concordant = 0
    total_pairs = 0
    for a in range(len(common)):
        for b in range(a + 1, len(common)):
            total_pairs += 1
            if (common[a][0] - common[b][0]) * (common[a][1] - common[b][1]) > 0:
                concordant += 1
    return concordant / total_pairs if total_pairs > 0 else 0.0


def _is_generic_indicator_key(value: str) -> bool:
    normalized = normalize_for_matching(str(value or ""), target="indicator")
    if not normalized:
        return True
    collapsed = normalized.replace(" ", "")
    if normalized in _GENERIC_INDICATOR_KEYS or collapsed in {
        item.replace(" ", "") for item in _GENERIC_INDICATOR_KEYS
    }:
        return True
    tokens = set(normalized.split())
    return tokens.issubset(
        {
            "total",
            "autres",
            "other",
            "canada",
            "etats",
            "unis",
            "united",
            "states",
            "europe",
            "royaume",
            "uni",
            "uk",
            "asie",
            "amerique",
            "latine",
            "particuliers",
            "entreprises",
            "secteur",
            "public",
            "administrations",
            "publiques",
        }
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_uid(table: TableArtifact) -> str:
    return f"{table.section}|{table.table_id}|p{table.page_pdf}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _normalize_title_value(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = strip_temporal_expressions(raw, target="title", aggressive=True)
    return normalize_for_matching(cleaned, target="title")


def _normalized_title(table: TableArtifact) -> str:
    candidates = [
        getattr(table, "title_clean", None),
        getattr(table, "title_raw", None),
        getattr(table, "title", None),
    ]
    for candidate in candidates:
        normalized = _normalize_title_value(candidate)
        if normalized:
            return normalized
    return ""


def _normalized_headers(table: TableArtifact) -> list[str]:
    headers = [str(h or "").strip() for h in (getattr(table, "headers", None) or [])]
    return [normalize_for_matching(item, target="header") for item in headers if item]


def _normalized_table_number(table: TableArtifact) -> str:
    explicit = normalize_for_matching(str(getattr(table, "table_number", "") or ""))
    if explicit:
        return explicit
    title_candidates = [
        getattr(table, "title_raw", None),
        getattr(table, "title_clean", None),
        getattr(table, "title", None),
    ]
    for title in title_candidates:
        text = str(title or "").strip()
        if not text:
            continue
        match = _TABLE_NUMBER_RE.search(text) or _TABLE_NUMBER_SHORT_RE.search(text)
        if match:
            return normalize_for_matching(match.group(1), target="generic")
    return ""


def _indicator_keys(table: TableArtifact) -> list[str]:
    """Return canonical indicator keys for pairing overlap (footnote markers normalized away)."""
    raw = get_comparison_indicators(table)
    keys: list[str] = []
    seen: set[str] = set()
    for s in raw:
        k = normalize_indicator_for_comparison(str(s or "").strip())
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _is_known_section(value: str | None) -> bool:
    return bool(value and str(value).strip().lower() not in UNKNOWN_SECTIONS)


def _section_value(table: TableArtifact) -> str:
    return str(getattr(table, "section", "") or "").strip().lower()


def _same_or_unknown_section(left: TableArtifact, right: TableArtifact) -> bool:
    left_section = _section_value(left)
    right_section = _section_value(right)
    if left_section == right_section:
        return True
    if not _is_known_section(left_section) and not _is_known_section(right_section):
        return True
    return False


def _row_count(table: TableArtifact) -> int:
    indicators = _indicator_keys(table)
    if indicators:
        return len(indicators)
    return len(getattr(table, "rows", None) or [])


def _size_compatibility(left: TableArtifact, right: TableArtifact) -> float:
    left_count = max(_row_count(left), 1)
    right_count = max(_row_count(right), 1)
    return min(left_count, right_count) / max(left_count, right_count)


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def _containment(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(min(len(left_set), len(right_set)), 1)


def _page_bonus(left: TableArtifact, right: TableArtifact) -> float:
    delta = abs(
        _safe_int(getattr(left, "page_pdf", 0))
        - _safe_int(getattr(right, "page_pdf", 0))
    )
    if delta <= 1:
        return 1.0
    if delta <= 3:
        return 0.7
    if delta <= 6:
        return 0.4
    return 0.0


def _bbox_y_center(table: TableArtifact) -> float | None:
    """Normalized vertical center of table bbox in [0, 1]. None if no valid bbox."""
    bbox = getattr(table, "bbox", None)
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        top = float(bbox[1])
        bottom = float(bbox[3])
        return (top + bottom) / 2.0
    if isinstance(bbox, dict):
        if "t" in bbox and "b" in bbox:
            return (float(bbox["t"]) + float(bbox["b"])) / 2.0
        if "y0" in bbox and "y1" in bbox:
            return (float(bbox["y0"]) + float(bbox["y1"])) / 2.0
        if "y" in bbox and "height" in bbox:
            return float(bbox["y"]) + float(bbox["height"]) / 2.0
    return None


def _page_local_order_bonus(left: TableArtifact, right: TableArtifact) -> float:
    """Bonus/penalty for same/different table index on (same or near) page."""
    idx_left = getattr(left, "table_index_on_page", None)
    idx_right = getattr(right, "table_index_on_page", None)
    if idx_left is None or idx_right is None:
        return 0.0
    page_left = _safe_int(getattr(left, "page_pdf", 0))
    page_right = _safe_int(getattr(right, "page_pdf", 0))
    delta_page = abs(page_left - page_right)
    if delta_page == 0:
        if idx_left == idx_right:
            return PAGE_LOCAL_ORDER_MATCH_SAME_PAGE
        return PAGE_LOCAL_ORDER_CONFLICT_SAME_PAGE
    if delta_page == 1:
        if idx_left == idx_right:
            return PAGE_LOCAL_ORDER_MATCH_NEAR_PAGE
        return PAGE_LOCAL_ORDER_CONFLICT_NEAR_PAGE
    return 0.0


def _bbox_y_similarity(left: TableArtifact, right: TableArtifact) -> float:
    """Similarity of vertical position when both on same or near page. [0, 1]."""
    page_left = _safe_int(getattr(left, "page_pdf", 0))
    page_right = _safe_int(getattr(right, "page_pdf", 0))
    if abs(page_left - page_right) > 1:
        return 0.0
    y_left = _bbox_y_center(left)
    y_right = _bbox_y_center(right)
    if y_left is None or y_right is None:
        return 0.0
    return 1.0 - min(1.0, abs(y_left - y_right))


def _page_local_role_bonus(left: TableArtifact, right: TableArtifact) -> float:
    """Bonus when page_local_role matches (first/first, last/last, etc.)."""
    role_left = getattr(left, "page_local_role", None) or ""
    role_right = getattr(right, "page_local_role", None) or ""
    if not role_left or not role_right:
        return 0.0
    if role_left == role_right:
        return PAGE_LOCAL_ROLE_MATCH_BONUS
    return 0.0


def _title_similarity(left: TableArtifact, right: TableArtifact) -> float:
    title_left = _normalized_title(left)
    title_right = _normalized_title(right)
    if not title_left or not title_right:
        return 0.0
    if is_generic_title(title_left) or is_generic_title(title_right):
        return 0.0
    return SequenceMatcher(None, title_left, title_right).ratio()


def _headers_similarity(left: TableArtifact, right: TableArtifact) -> float:
    headers_left = _normalized_headers(left)
    headers_right = _normalized_headers(right)
    if not headers_left or not headers_right:
        return 0.0
    return header_schema_similarity(headers_left, headers_right)


def _build_section_indicator_frequency(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
) -> dict[str, dict[str, int]]:
    """Build indicator frequency from certified tables only (same population as views)."""
    counts: dict[str, dict[str, int]] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_auto_compare_eligible(table):
            continue
        section = _section_value(table)
        section_counts = counts.setdefault(section, {})
        for key in set(_indicator_keys(table)):
            section_counts[key] = section_counts.get(key, 0) + 1
    return counts


def _distinctive_indicator_keys(
    table: TableArtifact,
    *,
    section_frequencies: dict[str, dict[str, int]],
    section_table_count: int,
) -> list[str]:
    keys = _indicator_keys(table)
    if not keys:
        return []
    section = _section_value(table)
    counts = section_frequencies.get(section, {})
    common_threshold = max(3, int(math.ceil(max(section_table_count, 1) * 0.45)))
    distinctive = [
        key
        for key in keys
        if not _is_generic_indicator_key(key) and counts.get(key, 0) < common_threshold
    ]
    if distinctive:
        return distinctive
    return [key for key in keys if not _is_generic_indicator_key(key)][:6]


def _quality_profile(table: TableArtifact) -> dict[str, Any]:
    """Normalized quality profile for pairing (avoid ad hoc debug_metrics)."""
    return get_extraction_quality_profile(table)


def _raw_indicator_stability(t1_view: TableView, t2_view: TableView) -> float:
    """
    Compare raw first-column indicators overlap vs normalized overlap.
    Returns a value in [0, 1]; low when normalization makes noisy extractions look similar.
    """
    raw_left = [
        normalize_for_matching(s, target="indicator")
        for s in get_vision_raw_indicators(t1_view.table)
        if normalize_for_matching(s, target="indicator")
    ]
    raw_right = [
        normalize_for_matching(s, target="indicator")
        for s in get_vision_raw_indicators(t2_view.table)
        if normalize_for_matching(s, target="indicator")
    ]
    raw_overlap = _jaccard(raw_left, raw_right)
    norm_overlap = _jaccard(t1_view.indicator_keys, t2_view.indicator_keys)
    if norm_overlap <= 0.2:
        return 1.0
    if norm_overlap <= 0:
        return 1.0
    return min(1.0, raw_overlap / max(norm_overlap, 0.01))


def _independent_anchor_count(score: "CandidateScore") -> int:
    """
    Return the precomputed anchor count from the candidate score.
    Used to require at least 2 anchors when extraction quality is low.
    """
    return score.anchor_count


@dataclass(slots=True)
class TableView:
    table: TableArtifact
    uid: str
    section: str
    normalized_title: str
    normalized_headers: list[str]
    normalized_table_number: str
    indicator_keys: list[str]
    indicator_distinctive_keys: list[str]
    table_size: int
    quality_profile: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_table(
        cls,
        table: TableArtifact,
        *,
        section_frequencies: dict[str, dict[str, int]],
        section_table_count: int,
    ) -> "TableView":
        return cls(
            table=table,
            uid=_table_uid(table),
            section=_section_value(table),
            normalized_title=_normalized_title(table),
            normalized_headers=_normalized_headers(table),
            normalized_table_number=_normalized_table_number(table),
            indicator_keys=_indicator_keys(table),
            indicator_distinctive_keys=_distinctive_indicator_keys(
                table,
                section_frequencies=section_frequencies,
                section_table_count=section_table_count,
            ),
            table_size=_row_count(table),
            quality_profile=_quality_profile(table),
        )


def _title_reliability_numeric(table: TableArtifact) -> float:
    """Map title_reliability string to a score in [0, 1] for pairing."""
    r = getattr(table, "title_reliability", None) or ""
    if str(r).strip().lower() == "reliable":
        return 1.0
    if str(r).strip().lower() == "weak":
        return 0.5
    return 0.3


@dataclass(slots=True)
class CandidateScore:
    t1_view: TableView
    t2_view: TableView
    total_score: float
    section_compatibility: float
    indicator_distinctive_overlap: float
    indicator_global_overlap: float
    indicator_containment: float
    header_compatibility: float
    title_similarity: float
    table_number_bonus: float
    order_proximity_bonus: float
    size_compatibility: float
    table_number_match: bool
    table_number_conflict: bool
    explanation: list[str] = field(default_factory=list)
    quality_penalty: float = 0.0
    raw_indicator_stability: float = 1.0
    anchor_count: int = 0
    confidence_cap_reason: str | None = None
    title_reliability_score: float = 0.0
    quality_suspect: bool = False
    page_local_order_bonus: float = 0.0
    page_local_role_bonus: float = 0.0
    bbox_y_similarity: float = 0.0
    header_fingerprint: float = 0.0
    indicator_ordering: float = 0.0
    adaptive_mode: str = "default"

    def as_feature_dict(self) -> dict[str, Any]:
        return {
            "t1_uid": self.t1_view.uid,
            "t2_uid": self.t2_view.uid,
            "score": round(self.total_score, 6),
            "section_compatibility": round(self.section_compatibility, 6),
            "indicator_distinctive_overlap": round(
                self.indicator_distinctive_overlap, 6
            ),
            "indicator_global_overlap": round(self.indicator_global_overlap, 6),
            "indicator_containment": round(self.indicator_containment, 6),
            "header_compatibility": round(self.header_compatibility, 6),
            "header_fingerprint": round(self.header_fingerprint, 6),
            "title_similarity": round(self.title_similarity, 6),
            "table_number_bonus": round(self.table_number_bonus, 6),
            "order_proximity_bonus": round(self.order_proximity_bonus, 6),
            "size_compatibility": round(self.size_compatibility, 6),
            "indicator_ordering": round(self.indicator_ordering, 6),
            "table_number_match": self.table_number_match,
            "table_number_conflict": self.table_number_conflict,
            "page_local_order_bonus": round(self.page_local_order_bonus, 6),
            "page_local_role_bonus": round(self.page_local_role_bonus, 6),
            "bbox_y_similarity": round(self.bbox_y_similarity, 6),
            "adaptive_mode": self.adaptive_mode,
            "reasons": list(self.explanation),
        }


def _candidate_score(
    t1_view: TableView,
    t2_view: TableView,
    profile: ScoringProfile | None = None,
    adapt: bool = True,
) -> CandidateScore:
    if profile is None:
        profile = ScoringProfile()

    section_compatibility = (
        1.0 if _same_or_unknown_section(t1_view.table, t2_view.table) else 0.0
    )
    indicator_distinctive_overlap = _jaccard(
        t1_view.indicator_distinctive_keys,
        t2_view.indicator_distinctive_keys,
    )
    indicator_global_overlap = _jaccard(t1_view.indicator_keys, t2_view.indicator_keys)
    indicator_containment = _containment(t1_view.indicator_keys, t2_view.indicator_keys)
    header_compatibility = _headers_similarity(t1_view.table, t2_view.table)
    hdr_fingerprint = header_literal_fingerprint(
        t1_view.normalized_headers,
        t2_view.normalized_headers,
    )
    title_similarity = _title_similarity(t1_view.table, t2_view.table)
    size_compatibility = _size_compatibility(t1_view.table, t2_view.table)
    order_proximity_bonus = _page_bonus(t1_view.table, t2_view.table)
    page_local_order_bonus = _page_local_order_bonus(t1_view.table, t2_view.table)
    page_local_role_bonus = _page_local_role_bonus(t1_view.table, t2_view.table)
    bbox_y_sim = _bbox_y_similarity(t1_view.table, t2_view.table)
    ind_ordering = _indicator_order_similarity(
        t1_view.indicator_keys, t2_view.indicator_keys
    )

    same_number = bool(
        t1_view.normalized_table_number
        and t1_view.normalized_table_number == t2_view.normalized_table_number
    )
    table_number_conflict = bool(
        t1_view.normalized_table_number
        and t2_view.normalized_table_number
        and t1_view.normalized_table_number != t2_view.normalized_table_number
    )

    raw_stability = _raw_indicator_stability(t1_view, t2_view)
    title_reliability_score = min(
        _title_reliability_numeric(t1_view.table),
        _title_reliability_numeric(t2_view.table),
    )

    n_indicators = min(len(t1_view.indicator_keys), len(t2_view.indicator_keys))
    if adapt:
        adapted = _adapt_weights(
            profile,
            title_reliability=title_reliability_score,
            title_sim=title_similarity,
            n_indicators=n_indicators,
            has_table_number=same_number,
        )
    else:
        adapted = profile

    # Zero-trust table number: no bonus/penalty in score (metadata only)
    tn_bonus = 0.0

    explanation: list[str] = []
    if same_number:
        explanation.append("same_table_number")
    if table_number_conflict:
        explanation.append("table_number_conflict")
    if indicator_distinctive_overlap >= 0.45:
        explanation.append("distinctive_overlap_strong")
    if indicator_containment >= 0.65:
        explanation.append("indicator_containment_strong")
    if title_similarity >= 0.75:
        explanation.append("title_similarity_strong")
    if header_compatibility >= 0.70:
        explanation.append("headers_compatible")
    if hdr_fingerprint >= 0.70:
        explanation.append("header_fingerprint_strong")
    if adapted.adaptive_mode != "default":
        explanation.append(f"adaptive_{adapted.adaptive_mode}")

    flags1 = get_extraction_quality_flags(t1_view.table)
    flags2 = get_extraction_quality_flags(t2_view.table)
    low_quality_left = (
        flags1.get("crop_rejected")
        or flags1.get("recrop_failed_incomplete")
        or not flags1.get("vision_extraction_applied", True)
    )
    low_quality_right = (
        flags2.get("crop_rejected")
        or flags2.get("recrop_failed_incomplete")
        or not flags2.get("vision_extraction_applied", True)
    )
    conf_left = get_extraction_confidence(t1_view.table)
    conf_right = get_extraction_confidence(t2_view.table)
    low_conf = conf_left < 0.5 or conf_right < 0.5
    quality_suspect = low_quality_left or low_quality_right or low_conf

    instability_penalty = 0.0
    if indicator_global_overlap >= 0.5 and raw_stability < 0.6:
        instability_penalty = 0.15 * (1.0 - raw_stability)

    title_contribution = adapted.w_title * title_similarity
    if title_reliability_score < 0.6:
        title_contribution *= 0.3
    header_contribution = adapted.w_header_compatibility * header_compatibility
    if quality_suspect:
        header_contribution *= 0.7

    total = (
        adapted.w_distinctive_overlap * indicator_distinctive_overlap
        + adapted.w_containment * indicator_containment
        + header_contribution
        + adapted.w_header_fingerprint * hdr_fingerprint
        + title_contribution
        + adapted.w_section * section_compatibility
        + adapted.w_size * size_compatibility
        + adapted.w_order_proximity * order_proximity_bonus
        + adapted.w_indicator_ordering * ind_ordering
        + tn_bonus
        + page_local_order_bonus
        + page_local_role_bonus
        + BBOX_Y_SIMILARITY_WEIGHT * bbox_y_sim
        - instability_penalty
    )
    quality_penalty = 0.08 if quality_suspect else 0.0
    total = max(0.0, min(1.0, total - quality_penalty))

    anchor_count = 0
    # Table number not used as anchor (zero-trust policy)
    if indicator_distinctive_overlap >= 0.40:
        anchor_count += 1
    if indicator_containment >= 0.52:
        anchor_count += 1
    if title_reliability_score >= 0.7 and title_similarity >= 0.72:
        anchor_count += 1
    if header_compatibility >= 0.72:
        anchor_count += 1
    if hdr_fingerprint >= 0.70:
        anchor_count += 1

    return CandidateScore(
        t1_view=t1_view,
        t2_view=t2_view,
        total_score=total,
        section_compatibility=section_compatibility,
        indicator_distinctive_overlap=indicator_distinctive_overlap,
        indicator_global_overlap=indicator_global_overlap,
        indicator_containment=indicator_containment,
        header_compatibility=header_compatibility,
        title_similarity=title_similarity,
        table_number_bonus=tn_bonus,
        order_proximity_bonus=order_proximity_bonus,
        size_compatibility=size_compatibility,
        table_number_match=same_number,
        table_number_conflict=table_number_conflict,
        explanation=explanation,
        quality_penalty=quality_penalty,
        raw_indicator_stability=raw_stability,
        anchor_count=anchor_count,
        confidence_cap_reason=None,
        title_reliability_score=title_reliability_score,
        quality_suspect=quality_suspect,
        page_local_order_bonus=page_local_order_bonus,
        page_local_role_bonus=page_local_role_bonus,
        bbox_y_similarity=bbox_y_sim,
        header_fingerprint=hdr_fingerprint,
        indicator_ordering=ind_ordering,
        adaptive_mode=adapted.adaptive_mode,
    )


def _eligible_table_views(
    tables: list[TableArtifact],
    *,
    section_frequencies: dict[str, dict[str, int]],
    section_counts: dict[str, int],
) -> tuple[list[TableView], list[dict[str, Any]]]:
    """Build views only for tables certified for auto-comparison. Uncertified tables are excluded with explicit reasons."""
    views: list[TableView] = []
    ineligible: list[dict[str, Any]] = []
    for table in tables:
        uid = _table_uid(table)
        if not is_auto_compare_eligible(table):
            extraction_blockers = derive_extraction_blockers(table)
            extraction_status = get_extraction_status(table)
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": list(
                        getattr(table, "comparison_blockers", []) or []
                    ),
                    "extraction_blockers": extraction_blockers,
                    "extraction_status": extraction_status,
                    "reason": "extraction_not_certified",
                }
            )
            continue
        if not bool(getattr(table, "comparison_eligible", False)):
            blockers = list(getattr(table, "comparison_blockers", []) or [])
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": blockers,
                    "reason": "comparison_ineligible",
                }
            )
            continue
        section = _section_value(table)
        views.append(
            TableView.from_table(
                table,
                section_frequencies=section_frequencies,
                section_table_count=section_counts.get(section, 1),
            )
        )
    return views, ineligible


def _hard_filter_candidate(t1_view: TableView, t2_view: TableView) -> bool:
    if not _same_or_unknown_section(t1_view.table, t2_view.table):
        return False
    if _size_compatibility(t1_view.table, t2_view.table) < 0.35:
        return False
    return True


def _shortlist_candidates(
    t2_view: TableView,
    t1_views: list[TableView],
    *,
    shortlist_size: int,
    profile: ScoringProfile | None = None,
    adapt: bool = True,
) -> list[CandidateScore]:
    candidates: list[CandidateScore] = []
    for t1_view in t1_views:
        if not _hard_filter_candidate(t1_view, t2_view):
            continue
        score = _candidate_score(t1_view, t2_view, profile=profile, adapt=adapt)
        if score.total_score < 0.12 and score.indicator_global_overlap < 0.15:
            continue
        candidates.append(score)
    if not candidates:
        return []
    candidates.sort(key=lambda item: item.total_score, reverse=True)

    selected: list[CandidateScore] = []
    seen: set[str] = set()

    def _add(candidate: CandidateScore) -> None:
        if candidate.t1_view.uid in seen:
            return
        seen.add(candidate.t1_view.uid)
        selected.append(candidate)

    _add(candidates[0])
    by_page_local_order = max(candidates, key=lambda item: item.page_local_order_bonus)
    by_role = max(candidates, key=lambda item: item.page_local_role_bonus)
    by_distinct = max(candidates, key=lambda item: item.indicator_distinctive_overlap)
    by_global = max(candidates, key=lambda item: item.indicator_global_overlap)
    by_title = max(candidates, key=lambda item: item.title_similarity)
    _add(by_page_local_order)
    if by_role.page_local_role_bonus > 0:
        _add(by_role)
    # Table number not used for shortlist injection (zero-trust policy)
    _add(by_distinct)
    _add(by_global)
    if by_title.title_similarity > 0:
        _add(by_title)
    for candidate in candidates:
        _add(candidate)
        if len(selected) >= shortlist_size:
            break
    return selected[:shortlist_size]


@dataclass(slots=True)
class PairingDecision:
    decision: str
    matched_t1_uid: str | None
    confidence: float
    reason_codes: list[str]
    requires_review: bool
    pairing_quality_flags: list[str] = field(default_factory=list)
    pairing_confidence_cap: str | None = None


class PairingRouter(Protocol):
    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision: ...


class ConservativePairingRouter:
    """Deterministic conservative router used by default."""

    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision:
        if not candidates:
            return PairingDecision(
                decision="no_match",
                matched_t1_uid=None,
                confidence=0.0,
                reason_codes=["no_candidates"],
                requires_review=False,
            )

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        reasons = list(top.explanation)

        low_quality = top.quality_suspect
        anchors = top.anchor_count

        if (
            top.indicator_global_overlap >= 0.60
            and top.indicator_distinctive_overlap < 0.25
        ):
            runner_up_fp = second.header_fingerprint if second else 0.0
            has_strong_disambiguator = (
                top.title_similarity >= 0.80
                or (top.header_fingerprint >= 0.75 and top.header_fingerprint - runner_up_fp >= 0.15)
            )
            if not has_strong_disambiguator:
                reasons = ["family_similarity_without_distinctive_anchor", *reasons]
                return PairingDecision(
                    decision="ambiguous",
                    matched_t1_uid=None,
                    confidence=min(0.75, top.total_score),
                    reason_codes=reasons,
                    requires_review=True,
                )

        if top.raw_indicator_stability < 0.6 and top.indicator_global_overlap >= 0.5:
            reasons.append("raw_normalized_instability")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.75, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if second is not None and second.total_score >= 0.55:
            if abs(top.total_score - second.total_score) < 0.06:
                reasons.append("close_competing_candidates")
                if low_quality:
                    return PairingDecision(
                        decision="ambiguous",
                        matched_t1_uid=None,
                        confidence=min(0.75, top.total_score),
                        reason_codes=reasons,
                        requires_review=True,
                    )
                return PairingDecision(
                    decision="ambiguous",
                    matched_t1_uid=None,
                    confidence=min(0.80, top.total_score),
                    reason_codes=reasons,
                    requires_review=True,
                )

        if top.table_number_match and top.indicator_containment < 0.35:
            reasons.append("table_number_only_is_not_enough")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.70, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if (
            low_quality
            and anchors < 2
            and (top.title_similarity >= 0.72 or top.header_compatibility >= 0.72)
        ):
            reasons.append("low_quality_title_or_header_only")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.70, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if (
            top.total_score >= 0.66
            and top.indicator_containment >= 0.52
            and (
                top.indicator_distinctive_overlap >= 0.40
                or (top.title_reliability_score >= 0.7 and top.title_similarity >= 0.72)
                or top.header_compatibility >= 0.72
                or top.header_fingerprint >= 0.72
            )
        ):
            if low_quality and anchors < 2:
                reasons.append("low_quality_insufficient_anchors")
                return PairingDecision(
                    decision="ambiguous",
                    matched_t1_uid=None,
                    confidence=min(0.75, top.total_score),
                    reason_codes=reasons,
                    requires_review=True,
                )
            reasons.append("deterministic_router_match")
            confidence = top.total_score
            quality_flags: list[str] = []
            cap_reason: str | None = None
            if low_quality:
                confidence = min(0.82, confidence)
                quality_flags.append("low_quality_match")
                cap_reason = "extraction_quality_cap"
            return PairingDecision(
                decision="match",
                matched_t1_uid=top.t1_view.uid,
                confidence=confidence,
                reason_codes=reasons,
                requires_review=False,
                pairing_quality_flags=quality_flags,
                pairing_confidence_cap=cap_reason,
            )

        if top.total_score >= 0.56 and top.indicator_containment >= 0.45:
            reasons.append("conservative_router_ambiguous")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=top.total_score,
                reason_codes=reasons,
                requires_review=True,
            )

        reasons.append("insufficient_pairing_signal")
        return PairingDecision(
            decision="no_match",
            matched_t1_uid=None,
            confidence=top.total_score,
            reason_codes=reasons,
            requires_review=False,
        )


def _table_summary_for_router(view: TableView) -> dict[str, Any]:
    return {
        "uid": view.uid,
        "section": view.section,
        "table_number_raw": getattr(view.table, "table_number", None),
        "title_raw": getattr(view.table, "title", None),
        "title_normalized": view.normalized_title,
        "headers_normalized": view.normalized_headers[:6],
        "top_distinctive_indicators": view.indicator_distinctive_keys[:8],
        "top_global_indicators": view.indicator_keys[:10],
        "table_size": view.table_size,
    }


class BatchLLMPairingRouter:
    """Optional final router using GPT-4o style batch routing.

    The shortlist is still deterministic; the LLM only chooses between provided
    candidates or returns ``no_match`` / ``ambiguous``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ROUTER_MODEL,
        timeout: float = DEFAULT_ROUTER_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _build_prompt(
        self, *, t2_view: TableView, candidates: list[CandidateScore]
    ) -> str:
        payload = {
            "instruction": (
                "Choisis le candidat qui represente le meme tableau conceptuel/reglementaire. "
                "N'utilise pas l'ordre des lignes comme preuve. Le numero et le titre "
                "sont secondaires. Une forte similarite de famille sans indicateurs "
                "distinctifs doit mener a 'ambiguous' ou 'no_match'."
            ),
            "allowed_answers": ["match:<t1_uid>", "no_match", "ambiguous"],
            "t2": _table_summary_for_router(t2_view),
            "candidates": [
                {
                    "candidate": _table_summary_for_router(score.t1_view),
                    "features": score.as_feature_dict(),
                }
                for score in candidates
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision:
        if not candidates:
            return PairingDecision(
                decision="no_match",
                matched_t1_uid=None,
                confidence=0.0,
                reason_codes=["no_candidates"],
                requires_review=False,
            )
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning(
                "openai package missing; falling back to deterministic router"
            )
            return ConservativePairingRouter().route(
                t2_view=t2_view, candidates=candidates
            )

        client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un routeur de tableaux bancaires. "
                        "Reponds uniquement en JSON avec decision, matched_t1_uid, "
                        "confidence et reason_codes."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(
                        t2_view=t2_view, candidates=candidates
                    ),
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = json.loads(content or "{}")
        except json.JSONDecodeError:
            logger.warning(
                "LLM router returned invalid JSON; using deterministic fallback"
            )
            return ConservativePairingRouter().route(
                t2_view=t2_view, candidates=candidates
            )

        decision = str(payload.get("decision", "") or "").strip()
        matched_t1_uid = str(payload.get("matched_t1_uid", "") or "").strip() or None
        confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence", 0.0))))
        reason_codes = [
            str(item).strip()
            for item in (payload.get("reason_codes", []) or [])
            if str(item).strip()
        ]
        if decision not in {"match", "no_match", "ambiguous"}:
            return ConservativePairingRouter().route(
                t2_view=t2_view, candidates=candidates
            )
        if decision == "match" and not matched_t1_uid:
            return ConservativePairingRouter().route(
                t2_view=t2_view, candidates=candidates
            )
        return PairingDecision(
            decision=decision,
            matched_t1_uid=matched_t1_uid,
            confidence=confidence,
            reason_codes=reason_codes or ["batch_llm_router"],
            requires_review=decision != "match",
        )


def _build_router(
    *,
    api_key: str | None,
    bank_code: str | None,
) -> PairingRouter:
    cfg = get_matching_thresholds(bank_code=bank_code)
    enabled = bool(cfg.get("batch_llm_pairing_enabled", False))
    if enabled and api_key:
        model = str(cfg.get("batch_llm_pairing_model", DEFAULT_ROUTER_MODEL))
        timeout = _safe_float(
            cfg.get("batch_llm_pairing_timeout", DEFAULT_ROUTER_TIMEOUT),
            DEFAULT_ROUTER_TIMEOUT,
        )
        return BatchLLMPairingRouter(api_key=api_key, model=model, timeout=timeout)
    return ConservativePairingRouter()


def _pair_dict(candidate: CandidateScore, decision: PairingDecision) -> dict[str, Any]:
    t1 = candidate.t1_view.table
    t2 = candidate.t2_view.table
    payload = candidate.as_feature_dict()
    payload.update(
        {
            "t1_uid": candidate.t1_view.uid,
            "t2_uid": candidate.t2_view.uid,
            "t1_table_id": t1.table_id,
            "t2_table_id": t2.table_id,
            "page_t1": t1.page_pdf,
            "page_t2": t2.page_pdf,
            "title_t1": t1.title or "",
            "title_t2": t2.title or "",
            "section": t1.section or t2.section or "",
            "reason": decision.reason_codes[0]
            if decision.reason_codes
            else "matched_pair",
            "reason_codes": list(decision.reason_codes),
            "decision_level": "match",
            "router_decision": decision.decision,
            "pairing_confidence": round(decision.confidence, 6),
        }
    )
    if getattr(decision, "pairing_quality_flags", None):
        payload["pairing_quality_flags"] = list(decision.pairing_quality_flags)
    if getattr(decision, "pairing_confidence_cap", None):
        payload["pairing_confidence_cap"] = decision.pairing_confidence_cap
    return payload


def _unmatched_previous_entry(
    view: TableView,
    *,
    ambiguous: bool,
    reason: str,
) -> dict[str, Any]:
    table = view.table
    return {
        "t1_uid": view.uid,
        "t1_table_id": table.table_id,
        "section": table.section,
        "page_t1": table.page_pdf,
        "title_t1": table.title or "",
        "reason": reason,
        "unmatched_status": "ambiguous" if ambiguous else "confirmed",
        "suspicion_flags": [reason] if ambiguous else [],
    }


def _unmatched_current_entry(
    view: TableView,
    *,
    ambiguous: bool,
    reason: str,
) -> dict[str, Any]:
    table = view.table
    return {
        "t2_uid": view.uid,
        "t2_table_id": table.table_id,
        "section": table.section,
        "page_t2": table.page_pdf,
        "title_t2": table.title or "",
        "reason": reason,
        "unmatched_status": "ambiguous" if ambiguous else "confirmed",
        "suspicion_flags": [reason] if ambiguous else [],
    }


def _added_table_entry(view: TableView) -> dict[str, Any]:
    table = view.table
    return {
        "uid": view.uid,
        "table_id": table.table_id,
        "t2_uid": view.uid,
        "t2_table_id": table.table_id,
        "section": table.section,
        "page": table.page_pdf,
        "page_t2": table.page_pdf,
        "title": table.title or "",
        "title_t2": table.title or "",
        "reason": "added_table",
        "source_reason": "pairing_unmatched",
        "first_column_indicators": list(view.indicator_keys),
        "first_column_indicators_raw": list(
            getattr(table, "first_column_indicators_raw", None) or []
        ),
    }


def _removed_table_entry(view: TableView) -> dict[str, Any]:
    table = view.table
    return {
        "uid": view.uid,
        "table_id": table.table_id,
        "t1_uid": view.uid,
        "t1_table_id": table.table_id,
        "section": table.section,
        "page": table.page_pdf,
        "page_t1": table.page_pdf,
        "title": table.title or "",
        "title_t1": table.title or "",
        "reason": "removed_table",
        "source_reason": "pairing_unmatched",
        "first_column_indicators": list(view.indicator_keys),
        "first_column_indicators_raw": list(
            getattr(table, "first_column_indicators_raw", None) or []
        ),
    }


def _candidate_debug_entry_t2(
    t2_uid: str, candidates: list[CandidateScore]
) -> dict[str, Any]:
    return {
        "t2_uid": t2_uid,
        "candidates": [candidate.as_feature_dict() for candidate in candidates],
    }


def _candidate_debug_entry_t1(
    t1_uid: str,
    candidates: list[CandidateScore],
) -> dict[str, Any]:
    normalized = []
    for candidate in candidates:
        item = candidate.as_feature_dict()
        item["t1_uid"] = t1_uid
        normalized.append(item)
    return {"t1_uid": t1_uid, "candidates": normalized}


def _resolve_collisions(
    provisional_matches: list[tuple[TableView, CandidateScore, PairingDecision]],
) -> tuple[
    list[tuple[TableView, CandidateScore, PairingDecision]], list[dict[str, Any]]
]:
    matches_by_t1: dict[
        str, list[tuple[TableView, CandidateScore, PairingDecision]]
    ] = {}
    for entry in provisional_matches:
        _, candidate, _ = entry
        matches_by_t1.setdefault(candidate.t1_view.uid, []).append(entry)

    accepted: list[tuple[TableView, CandidateScore, PairingDecision]] = []
    ambiguous: list[dict[str, Any]] = []

    for t1_uid, entries in matches_by_t1.items():
        if len(entries) == 1:
            accepted.append(entries[0])
            continue
        entries_sorted = sorted(
            entries, key=lambda item: item[2].confidence, reverse=True
        )
        top = entries_sorted[0]
        runner_up = entries_sorted[1]
        if top[2].confidence - runner_up[2].confidence >= 0.05:
            accepted.append(top)
            for t2_view, candidate, decision in entries_sorted[1:]:
                ambiguous.append(
                    {
                        "decision": "ambiguous",
                        "matched_t1_uid": None,
                        "confidence": round(decision.confidence, 6),
                        "reason_codes": ["collision_after_assignment"],
                        "t2_uid": t2_view.uid,
                        "candidate_t1_uids": [t1_uid],
                    }
                )
            continue
        candidate_ids = [candidate.t1_view.uid for _, candidate, _ in entries_sorted]
        for t2_view, candidate, decision in entries_sorted:
            ambiguous.append(
                {
                    "decision": "ambiguous",
                    "matched_t1_uid": None,
                    "confidence": round(decision.confidence, 6),
                    "reason_codes": ["collision_after_assignment"],
                    "t2_uid": t2_view.uid,
                    "candidate_t1_uids": candidate_ids,
                }
            )
    return accepted, ambiguous


_HUNGARIAN_MIN_SCORE = 0.56
_HUNGARIAN_AMBIGUITY_MARGIN = 0.06


def _hungarian_assignment(
    t2_views: list[TableView],
    t1_views: list[TableView],
    candidate_map_t2: dict[str, list[CandidateScore]],
    *,
    min_score: float = _HUNGARIAN_MIN_SCORE,
    ambiguity_margin: float = _HUNGARIAN_AMBIGUITY_MARGIN,
) -> tuple[
    list[tuple[TableView, CandidateScore, PairingDecision]],
    list[dict[str, Any]],
    set[str],
]:
    """Globally optimal 1-to-1 assignment using the Hungarian algorithm.

    Returns (accepted_matches, ambiguous_entries, explicit_no_match_t2_uids).
    """
    if not t2_views or not t1_views:
        return [], [], set()

    t2_idx = {v.uid: i for i, v in enumerate(t2_views)}
    t1_idx = {v.uid: i for i, v in enumerate(t1_views)}
    n_t2, n_t1 = len(t2_views), len(t1_views)

    score_matrix = np.full((n_t2, n_t1), -1e9, dtype=np.float64)
    best_candidate: dict[tuple[int, int], CandidateScore] = {}

    for t2_uid, candidates in candidate_map_t2.items():
        i = t2_idx.get(t2_uid)
        if i is None:
            continue
        for cand in candidates:
            j = t1_idx.get(cand.t1_view.uid)
            if j is None:
                continue
            if cand.total_score > score_matrix[i, j]:
                score_matrix[i, j] = cand.total_score
                best_candidate[(i, j)] = cand

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    accepted: list[tuple[TableView, CandidateScore, PairingDecision]] = []
    ambiguous: list[dict[str, Any]] = []
    no_match_t2: set[str] = set()
    matched_t2: set[int] = set()
    matched_t1: set[int] = set()

    assignments = sorted(
        zip(row_ind, col_ind),
        key=lambda pair: score_matrix[pair[0], pair[1]],
        reverse=True,
    )

    for i, j in assignments:
        s = score_matrix[i, j]
        if s < min_score:
            continue
        cand = best_candidate.get((i, j))
        if cand is None:
            continue

        if cand.table_number_match and cand.indicator_containment < 0.35:
            ambiguous.append(
                {
                    "decision": "ambiguous",
                    "matched_t1_uid": None,
                    "confidence": round(s, 6),
                    "reason_codes": ["table_number_only_insufficient_content"],
                    "t2_uid": t2_views[i].uid,
                    "candidate_t1_uids": [t1_views[j].uid],
                }
            )
            continue

        if cand.indicator_containment < 0.15 and cand.indicator_global_overlap < 0.10:
            no_match_t2.add(t2_views[i].uid)
            continue

        row_scores = sorted(
            [score_matrix[i, k] for k in range(n_t1) if score_matrix[i, k] > -1e8],
            reverse=True,
        )
        row_margin = (row_scores[0] - row_scores[1]) if len(row_scores) > 1 else 1.0

        if row_margin < ambiguity_margin:
            if s >= 0.85:
                pass
            else:
                runner_up_fp = 0.0
                runner_up_title = 0.0
                for k in range(n_t1):
                    if k == j:
                        continue
                    rival = best_candidate.get((i, k))
                    if rival is not None:
                        runner_up_fp = max(runner_up_fp, rival.header_fingerprint)
                        runner_up_title = max(runner_up_title, rival.title_similarity)

                fp_edge = cand.header_fingerprint - runner_up_fp
                title_edge = cand.title_similarity - runner_up_title
                has_strong_disambiguator = (
                    (cand.title_similarity >= 0.80 and title_edge >= 0.15)
                    or (cand.header_fingerprint >= 0.60 and fp_edge >= 0.20)
                    or (
                        title_edge >= 0.10
                        and fp_edge >= 0.10
                        and cand.indicator_distinctive_overlap >= 0.20
                    )
                    or cand.title_similarity >= 0.95
                    or cand.indicator_containment >= 0.75
                )
                if not has_strong_disambiguator:
                    ambiguous.append(
                        {
                            "decision": "ambiguous",
                            "matched_t1_uid": None,
                            "confidence": round(s, 6),
                            "reason_codes": ["hungarian_margin_ambiguous"],
                            "t2_uid": t2_views[i].uid,
                            "candidate_t1_uids": [
                                t1_views[k].uid
                                for k in range(n_t1)
                                if score_matrix[i, k] > min_score
                            ],
                        }
                    )
                    continue

        decision = PairingDecision(
            decision="match",
            matched_t1_uid=cand.t1_view.uid,
            confidence=s,
            reason_codes=list(cand.explanation) + ["hungarian_assignment"],
            requires_review=False,
        )
        accepted.append((t2_views[i], cand, decision))
        matched_t2.add(i)
        matched_t1.add(j)

    for i, view in enumerate(t2_views):
        if i not in matched_t2 and view.uid not in {a["t2_uid"] for a in ambiguous}:
            no_match_t2.add(view.uid)

    return accepted, ambiguous, no_match_t2


def _rescue_unmatched(
    *,
    unmatched_t1_views: list[TableView],
    unmatched_t2_views: list[TableView],
    profile: ScoringProfile | None = None,
    min_containment: float = 0.55,
    min_title_sim: float = 0.40,
    min_total_score: float = 0.56,
) -> list[tuple[TableView, CandidateScore, PairingDecision]]:
    """Single-rescue pass: attempt 1-to-1 matching of remaining unmatched tables."""
    if not unmatched_t1_views or not unmatched_t2_views:
        return []

    rescued: list[tuple[TableView, CandidateScore, PairingDecision]] = []
    used_t1: set[str] = set()
    used_t2: set[str] = set()

    rescue_candidates: list[tuple[float, TableView, TableView, CandidateScore]] = []
    for t2_view in unmatched_t2_views:
        for t1_view in unmatched_t1_views:
            if not _same_or_unknown_section(t1_view.table, t2_view.table):
                left_section = _section_value(t1_view.table)
                right_section = _section_value(t2_view.table)
                if _is_known_section(left_section) and _is_known_section(right_section):
                    continue
            score = _candidate_score(t1_view, t2_view, profile=profile)
            if (
                score.indicator_containment >= min_containment
                and score.title_similarity >= min_title_sim
                and score.total_score >= min_total_score
            ):
                rescue_candidates.append((score.total_score, t2_view, t1_view, score))

    rescue_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, t2_view, t1_view, score in rescue_candidates:
        if t1_view.uid in used_t1 or t2_view.uid in used_t2:
            continue
        decision = PairingDecision(
            decision="match",
            matched_t1_uid=t1_view.uid,
            confidence=score.total_score,
            reason_codes=list(score.explanation) + ["rescue_single"],
            requires_review=False,
        )
        rescued.append((t2_view, score, decision))
        used_t1.add(t1_view.uid)
        used_t2.add(t2_view.uid)

    return rescued


def run_strict_intra_section_compare(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    overlap_threshold: float | None = None,
    bank_code: str | None = None,
    embedding_service: Any | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Official conservative pairing facade used by the active pipeline."""
    del overlap_threshold, embedding_service

    thresholds = get_matching_thresholds(bank_code=bank_code)
    profile = ScoringProfile.from_thresholds(thresholds)

    section_frequencies = _build_section_indicator_frequency(tables_t1, tables_t2)
    section_counts: dict[str, int] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_auto_compare_eligible(table):
            continue
        section = _section_value(table)
        section_counts[section] = section_counts.get(section, 0) + 1

    t1_views, ineligible_t1_raw = _eligible_table_views(
        tables_t1,
        section_frequencies=section_frequencies,
        section_counts=section_counts,
    )
    t2_views, ineligible_t2_raw = _eligible_table_views(
        tables_t2,
        section_frequencies=section_frequencies,
        section_counts=section_counts,
    )

    router = _build_router(api_key=api_key, bank_code=bank_code)
    shortlist_size = _safe_int(
        thresholds.get("pairing_shortlist_size", DEFAULT_SHORTLIST_SIZE),
        DEFAULT_SHORTLIST_SIZE,
    )
    shortlist_size = max(1, min(shortlist_size, MAX_SHORTLIST_SIZE))

    use_hungarian = bool(thresholds.get("pairing_hungarian_enabled", False))
    hungarian_min = _safe_float(
        thresholds.get("pairing_hungarian_min_score", _HUNGARIAN_MIN_SCORE),
        _HUNGARIAN_MIN_SCORE,
    )
    hungarian_margin = _safe_float(
        thresholds.get("pairing_hungarian_margin", _HUNGARIAN_AMBIGUITY_MARGIN),
        _HUNGARIAN_AMBIGUITY_MARGIN,
    )

    rescue_min_containment = _safe_float(
        thresholds.get("rescue_single_min_containment", 0.55), 0.55
    )
    rescue_min_title = _safe_float(
        thresholds.get("rescue_single_min_title_similarity", 0.40), 0.40
    )

    candidate_map_t2: dict[str, list[CandidateScore]] = {}
    candidate_map_t1: dict[str, list[CandidateScore]] = {}
    ambiguous_pairs: list[dict[str, Any]] = []
    explicit_no_match_t2: set[str] = set()

    for t2_view in t2_views:
        shortlist = _shortlist_candidates(
            t2_view,
            t1_views,
            shortlist_size=shortlist_size,
            profile=profile,
            adapt=use_hungarian,
        )
        candidate_map_t2[t2_view.uid] = shortlist
        for candidate in shortlist:
            candidate_map_t1.setdefault(candidate.t1_view.uid, []).append(candidate)

    if use_hungarian and len(t2_views) > 0 and len(t1_views) > 0:
        accepted_matches, hungarian_ambiguous, explicit_no_match_t2 = (
            _hungarian_assignment(
                t2_views,
                t1_views,
                candidate_map_t2,
                min_score=hungarian_min,
                ambiguity_margin=hungarian_margin,
            )
        )
        ambiguous_pairs.extend(hungarian_ambiguous)
    else:
        provisional_matches: list[
            tuple[TableView, CandidateScore, PairingDecision]
        ] = []
        for t2_view in t2_views:
            shortlist = candidate_map_t2.get(t2_view.uid, [])
            decision = router.route(t2_view=t2_view, candidates=shortlist)
            if decision.decision == "match" and decision.matched_t1_uid:
                chosen = next(
                    (c for c in shortlist if c.t1_view.uid == decision.matched_t1_uid),
                    None,
                )
                if chosen is not None:
                    provisional_matches.append((t2_view, chosen, decision))
                    continue
            if decision.decision == "ambiguous":
                ambiguous_pairs.append(
                    {
                        "decision": "ambiguous",
                        "matched_t1_uid": None,
                        "confidence": round(decision.confidence, 6),
                        "reason_codes": list(decision.reason_codes),
                        "t2_uid": t2_view.uid,
                        "candidate_t1_uids": [c.t1_view.uid for c in shortlist],
                    }
                )
            else:
                explicit_no_match_t2.add(t2_view.uid)
        accepted_matches, collision_ambiguous = _resolve_collisions(provisional_matches)
        ambiguous_pairs.extend(collision_ambiguous)

    matched_t1_uids = {candidate.t1_view.uid for _, candidate, _ in accepted_matches}
    matched_t2_uids = {t2_view.uid for t2_view, _, _ in accepted_matches}

    ambiguous_t2_uids_pre = {
        str(item.get("t2_uid", "")).strip()
        for item in ambiguous_pairs
        if str(item.get("t2_uid", "")).strip()
    }

    remaining_t1 = [
        v
        for v in t1_views
        if v.uid not in matched_t1_uids and v.uid not in ambiguous_t2_uids_pre
    ]
    remaining_t2 = [
        v
        for v in t2_views
        if v.uid not in matched_t2_uids and v.uid not in ambiguous_t2_uids_pre
    ]
    rescued_matches = _rescue_unmatched(
        unmatched_t1_views=remaining_t1,
        unmatched_t2_views=remaining_t2,
        profile=profile,
        min_containment=rescue_min_containment,
        min_title_sim=rescue_min_title,
    )
    rescued_count = len(rescued_matches)
    for entry in rescued_matches:
        accepted_matches.append(entry)
        matched_t1_uids.add(entry[1].t1_view.uid)
        matched_t2_uids.add(entry[0].uid)

    ambiguous_t2_uids = {
        str(item.get("t2_uid", "")).strip()
        for item in ambiguous_pairs
        if str(item.get("t2_uid", "")).strip()
    }
    ambiguous_t1_uids: set[str] = set()
    for item in ambiguous_pairs:
        for candidate_uid in item.get("candidate_t1_uids", []) or []:
            candidate_uid = str(candidate_uid).strip()
            if candidate_uid:
                ambiguous_t1_uids.add(candidate_uid)

    pairs = [
        _pair_dict(candidate, decision) for _, candidate, decision in accepted_matches
    ]

    unmatched_t1: list[dict[str, Any]] = []
    unmatched_t2: list[dict[str, Any]] = []
    removed_tables: list[dict[str, Any]] = []
    added_tables: list[dict[str, Any]] = []

    for view in t1_views:
        if view.uid in matched_t1_uids:
            continue
        if view.uid in ambiguous_t1_uids:
            unmatched_t1.append(
                _unmatched_previous_entry(
                    view,
                    ambiguous=True,
                    reason="ambiguous_candidate",
                )
            )
            continue
        unmatched_t1.append(
            _unmatched_previous_entry(
                view,
                ambiguous=False,
                reason="removed_table",
            )
        )
        removed_tables.append(_removed_table_entry(view))

    for view in t2_views:
        if view.uid in matched_t2_uids:
            continue
        if view.uid in ambiguous_t2_uids:
            unmatched_t2.append(
                _unmatched_current_entry(
                    view,
                    ambiguous=True,
                    reason="ambiguous_candidate",
                )
            )
            continue
        reason = "no_match" if view.uid in explicit_no_match_t2 else "added_table"
        unmatched_t2.append(
            _unmatched_current_entry(
                view,
                ambiguous=False,
                reason=reason,
            )
        )
        added_tables.append(_added_table_entry(view))

    unmatched_confirmed_t1 = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "confirmed"
    ]
    unmatched_ambiguous_t1 = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "ambiguous"
    ]
    unmatched_confirmed_t2 = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "confirmed"
    ]
    unmatched_ambiguous_t2 = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "ambiguous"
    ]
    comparable_t1 = len(t1_views)
    comparable_t2 = len(t2_views)
    pairing_coverage = round(
        len(pairs) / max(min(comparable_t1, comparable_t2), 1),
        6,
    )

    def _ineligible_entry(item: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            f"{prefix}_table_id": item["table_id"],
            f"{prefix}_uid": item["uid"],
            "section": item["section"],
            f"page_{prefix}": item["page"],
            f"title_{prefix}": item["title"],
            "reason": item["reason"],
            "comparison_blockers": item.get("comparison_blockers", []),
        }
        if "extraction_blockers" in item:
            out["extraction_blockers"] = item["extraction_blockers"]
        if "extraction_status" in item:
            out["extraction_status"] = item["extraction_status"]
        return out

    ineligible_t1 = [_ineligible_entry(item, "t1") for item in ineligible_t1_raw]
    ineligible_t2 = [_ineligible_entry(item, "t2") for item in ineligible_t2_raw]

    return {
        "pairs": pairs,
        "matched_pairs": list(pairs),
        "probable_pairs": [],
        "suspicious_pairs": [],
        "ambiguous_pairs": ambiguous_pairs,
        "ambiguous_tables": [
            {
                "side": "previous",
                "uid": item.get("t1_uid"),
                "table_id": item.get("t1_table_id"),
                "title": item.get("title_t1"),
                "page": item.get("page_t1"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
            }
            for item in unmatched_ambiguous_t1
        ]
        + [
            {
                "side": "current",
                "uid": item.get("t2_uid"),
                "table_id": item.get("t2_table_id"),
                "title": item.get("title_t2"),
                "page": item.get("page_t2"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
            }
            for item in unmatched_ambiguous_t2
        ],
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "added_tables_confirmed": list(added_tables),
        "removed_tables_confirmed": list(removed_tables),
        "unmatched_t1": unmatched_t1,
        "unmatched_t2": unmatched_t2,
        "unmatched_confirmed_t1": unmatched_confirmed_t1,
        "unmatched_confirmed_t2": unmatched_confirmed_t2,
        "unmatched_ambiguous_t1": unmatched_ambiguous_t1,
        "unmatched_ambiguous_t2": unmatched_ambiguous_t2,
        "ambiguous_unmatched_previous": list(unmatched_ambiguous_t1),
        "ambiguous_unmatched_current": list(unmatched_ambiguous_t2),
        "ineligible_t1": ineligible_t1,
        "ineligible_t2": ineligible_t2,
        "debug_unmatched_candidates": [
            _candidate_debug_entry_t1(uid, candidates)
            for uid, candidates in sorted(candidate_map_t1.items())
            if candidates
        ],
        "debug_unmatched_candidates_t2": [
            _candidate_debug_entry_t2(uid, candidates)
            for uid, candidates in sorted(candidate_map_t2.items())
            if candidates
        ],
        "rescued_matches_count": rescued_count,
        "split_merge_rescues_count": 0,
        "vision_rescued_pairs": [],
        "reasons": [pair.get("reason", "") for pair in pairs if pair.get("reason")],
        "diagnostics": {
            "router": router.__class__.__name__,
            "shortlist_size": shortlist_size,
            "hungarian_enabled": use_hungarian,
            "scoring_profile": profile.adaptive_mode,
        },
        "matching_diagnostics": {
            "pairs_count": len(pairs),
            "ambiguous_pairs_count": len(ambiguous_pairs),
            "unmatched_t1_count": len(unmatched_t1),
            "unmatched_t2_count": len(unmatched_t2),
            "ineligible_t1_count": len(ineligible_t1),
            "ineligible_t2_count": len(ineligible_t2),
            "tables_comparable_t1": comparable_t1,
            "tables_comparable_t2": comparable_t2,
            "pairing_coverage": pairing_coverage,
        },
        "tables_comparable_t1": comparable_t1,
        "tables_comparable_t2": comparable_t2,
        "pairing_coverage": pairing_coverage,
    }
