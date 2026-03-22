"""Conservative merger for table fragments produced by imperfect segmentation.

When extracting tables from PDFs, layout-based segmentation often splits a single
logical table across multiple page boundaries or visual breaks. A table that
spans two pages may be emitted as two distinct fragments with partial headers,
overlapping indicators, and related footnotes. This module provides a merger that:

- Identifies candidate pairs of fragments belonging to the same logical table
- Scores pairs using header similarity, spatial proximity, title resemblance,
  indicator overlap, and continuation hints (e.g. "suite", "continued")
- Merges only pairs whose score exceeds a configurable threshold
- Optionally annotates near-threshold pairs with ``fragment_near_merge_hint``
  so downstream pairing logic can use this signal during split-merge rescue

The algorithm is conservative: it avoids merging unrelated tables (e.g. those
in different sections, on non-adjacent pages, or with low header similarity)
and penalizes near-duplicate indicators to prevent merging tables that are
merely similar rather than true continuations.

Usage:
    merged_tables, events = merge_table_fragments(tables, merge_score_min=0.85)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from vigilance.models.table_models import (
    TableArtifact,
    get_canonical_footnotes,
    get_comparison_indicators,
    get_vision_raw_indicators,
)
from vigilance.utils.footnotes_utils import normalize_footnotes_to_canonical
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.matching_normalizer import (
    header_schema_similarity,
    is_date_only_line,
    is_non_indicator_line,
    normalize_for_matching,
    strip_temporal_expressions,
)

UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}
_CONTINUATION_RE = re.compile(r"\b(?:suite|continued|cont[' ]?d)\b", re.IGNORECASE)
_TOTAL_ROW_RE = re.compile(r"^\s*(?:total|sous\s*total|net|solde)\b", re.IGNORECASE)


def _canonical_section(value: str | None) -> str:
    """Normalize a section string for comparison.

    Args:
        value: Raw section identifier (e.g. from table metadata).

    Returns:
        Lowercased, stripped section string; empty string if value is None.
    """
    return (value or "").strip().lower()


def _is_known_section(value: str | None) -> bool:
    """Check whether the section is a known (non-placeholder) section.

    Args:
        value: Raw section identifier.

    Returns:
        True if the canonical section is not empty, "unknown", or "unknown_section".
    """
    return _canonical_section(value) not in UNKNOWN_SECTIONS


def _normalize_title(value: str | None) -> str:
    """Normalize a table title for similarity comparison.

    Strips temporal expressions (years, quarters, dates) and applies
    matching normalization so titles can be compared consistently.

    Args:
        value: Raw table title.

    Returns:
        Normalized title string suitable for SequenceMatcher or similar.
    """
    cleaned = strip_temporal_expressions(value or "", target="title")
    return normalize_for_matching(cleaned, target="title")


def _title_similarity(left: str | None, right: str | None) -> float:
    """Compute similarity between two table titles (0.0 to 1.0).

    Uses SequenceMatcher on normalized titles. Returns fixed fallbacks when
    either or both titles are empty to avoid spurious scores.

    Args:
        left: First table title.
        right: Second table title.

    Returns:
        Similarity score: 0.60 if both empty, 0.35 if one empty, else ratio.
    """
    lnorm = _normalize_title(left)
    rnorm = _normalize_title(right)
    if not lnorm and not rnorm:
        return 0.60
    if not lnorm or not rnorm:
        return 0.35
    return SequenceMatcher(None, lnorm, rnorm).ratio()


def _extract_indicators(table: TableArtifact) -> set[str]:
    """Extract normalized indicator labels from a table's first column.

    Skips date-only lines and non-indicator lines, and normalizes each label
    for comparison.

    Args:
        table: Table artifact whose indicators to extract.

    Returns:
        Set of normalized indicator strings.
    """
    values: set[str] = set()
    for label in get_comparison_indicators(table):
        text = str(label or "").strip()
        if not text or is_date_only_line(text) or is_non_indicator_line(text):
            continue
        norm = normalize_indicator_for_comparison(text)
        if norm:
            values.add(norm)
    return values


def _indicator_jaccard(left: TableArtifact, right: TableArtifact) -> float:
    """Compute Jaccard similarity between two tables' indicator sets.

    High values indicate overlapping or near-duplicate indicator rows, which
    suggests the tables may be similar rather than continuations.

    Args:
        left: First table artifact.
        right: Second table artifact.

    Returns:
        Jaccard index: |intersection| / |union|; 0.0 if either set is empty.
    """
    lset = _extract_indicators(left)
    rset = _extract_indicators(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _indicator_non_overlap_score(left: TableArtifact, right: TableArtifact) -> float:
    """Score how little indicator overlap exists (1.0 = no overlap).

    Encourages merging when tables have distinct indicators (continuation);
    penalizes when indicators heavily overlap (likely duplicates).

    Args:
        left: First table artifact.
        right: Second table artifact.

    Returns:
        1.0 if Jaccard <= 0.05, else max(0, 1 - jaccard).
    """
    jaccard = _indicator_jaccard(left, right)
    if jaccard <= 0.05:
        return 1.0
    return max(0.0, 1.0 - jaccard)


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Parse a bounding box from various formats into (x0, y0, x1, y1).

    Supports: list/tuple [x0,y0,x1,y1], dict with x0/y0/x1/y1, dict with
    l/t/r/b, dict with x/y/width/height. Returns None for invalid input.

    Args:
        value: Bounding box in list, tuple, or dict format.

    Returns:
        Tuple (x0, y0, x1, y1) in normalized coordinates, or None if invalid.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x0, y0, x1, y1 = (
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    if isinstance(value, dict):
        # x0/y0/x1/y1
        if all(k in value for k in ("x0", "y0", "x1", "y1")):
            try:
                x0 = float(value["x0"])
                y0 = float(value["y0"])
                x1 = float(value["x1"])
                y1 = float(value["y1"])
            except (TypeError, ValueError):
                return None
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, y0, x1, y1
        # l/t/r/b
        if all(k in value for k in ("l", "t", "r", "b")):
            try:
                x0 = float(value["l"])
                y0 = float(value["t"])
                x1 = float(value["r"])
                y1 = float(value["b"])
            except (TypeError, ValueError):
                return None
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, y0, x1, y1
        # x/y/width/height
        if all(k in value for k in ("x", "y", "width", "height")):
            try:
                x0 = float(value["x"])
                y0 = float(value["y"])
                x1 = x0 + float(value["width"])
                y1 = y0 + float(value["height"])
            except (TypeError, ValueError):
                return None
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, y0, x1, y1
    return None


def _bbox_vertical_score(
    left: TableArtifact,
    right: TableArtifact,
) -> float:
    """Score vertical proximity of two tables for merge candidacy.

    Same-page: high score when right table sits just below left (small gap).
    Adjacent pages: high score when left is near bottom and right near top.
    Returns 0.55 as neutral fallback when bboxes are unavailable.

    Args:
        left: First (earlier) table artifact.
        right: Second (later) table artifact.

    Returns:
        Score in [0.0, 1.0]; higher means better vertical continuation.
    """
    bbox_left = _parse_bbox(left.bbox)
    bbox_right = _parse_bbox(right.bbox)
    if bbox_left is None or bbox_right is None:
        return 0.55

    _, y0_left, _, y1_left = bbox_left
    _, y0_right, _, y1_right = bbox_right
    if left.page_pdf == right.page_pdf:
        gap = y0_right - y1_left
        if gap < -0.05:
            return 0.0
        if gap <= 0.02:
            return 1.0
        if gap <= 0.25:
            return max(0.0, 1.0 - ((gap - 0.02) / 0.23))
        return 0.0

    if right.page_pdf == left.page_pdf + 1:
        near_bottom = y1_left >= 0.55
        near_top = y0_right <= 0.45
        if near_bottom and near_top:
            return 1.0
        if near_bottom or near_top:
            return 0.60
        return 0.30

    return 0.0


def _has_continuation_hint(table: TableArtifact) -> bool:
    """Detect if the table title or first indicator suggests a continuation.

    Looks for patterns like "suite", "continued", "cont'd" in the title or
    the first indicator row. Such hints boost merge score when present.

    Args:
        table: Table artifact to inspect.

    Returns:
        True if a continuation pattern is found.
    """
    text = str(table.title or "").strip()
    if text and _CONTINUATION_RE.search(text):
        return True
    first_indicators = get_vision_raw_indicators(table)
    if not first_indicators:
        first_indicators = get_comparison_indicators(table)
    if first_indicators:
        first = str(first_indicators[0] or "").strip()
        if first and _CONTINUATION_RE.search(first):
            return True
    return False


def _has_total_row(table: TableArtifact) -> bool:
    """Check if the table has a total/sous-total/net/solde row near the end.

    Examines the last few indicator rows. Total rows suggest the table is
    complete; merging with a continuation is penalized when no continuation
    hint is present.

    Args:
        table: Table artifact to inspect.

    Returns:
        True if a total-like row is found in the last 3 indicators.
    """
    labels = get_vision_raw_indicators(table)
    if not labels:
        labels = get_comparison_indicators(table)
    for raw_label in labels[-3:]:
        label = str(raw_label or "").strip()
        if _TOTAL_ROW_RE.search(label):
            return True
    return False


def _candidate_merge_score(left: TableArtifact, right: TableArtifact) -> float:
    """Compute merge score for a candidate (left, right) fragment pair.

    Returns 0.0 unless: same section, known section, right on same or next page,
    and header similarity >= 0.65. Then scores as weighted combination:
    35% header, 22% bbox vertical, 12% page adjacency, 10% title similarity,
    13% indicator non-overlap, 8% continuation hint. Penalties: -0.20 if left
    has total row but no continuation hint; -0.25 if indicator Jaccard >= 0.75
    (near-duplicates should not merge).

    Args:
        left: First (earlier) table artifact.
        right: Second (later) table artifact.

    Returns:
        Score in [0.0, 1.0]; >= merge_score_min indicates merge.
    """
    left_section = _canonical_section(left.section)
    right_section = _canonical_section(right.section)
    if left_section != right_section:
        return 0.0
    if not _is_known_section(left_section):
        return 0.0
    if right.page_pdf < left.page_pdf:
        return 0.0
    if right.page_pdf - left.page_pdf > 1:
        return 0.0

    header_score = header_schema_similarity(left.headers or [], right.headers or [])
    if header_score < 0.65:
        return 0.0

    page_score = 1.0 if right.page_pdf == left.page_pdf else 0.85
    bbox_score = _bbox_vertical_score(left, right)
    title_score = _title_similarity(left.title, right.title)
    indicator_non_overlap = _indicator_non_overlap_score(left, right)
    continuation_score = (
        1.0 if (_has_continuation_hint(left) or _has_continuation_hint(right)) else 0.0
    )

    score = (
        (0.35 * header_score)
        + (0.22 * bbox_score)
        + (0.12 * page_score)
        + (0.10 * title_score)
        + (0.13 * indicator_non_overlap)
        + (0.08 * continuation_score)
    )

    if _has_total_row(left) and continuation_score < 0.5:
        score -= 0.20
    if _indicator_jaccard(left, right) >= 0.75:
        # Near-duplicates should not be merged as fragments.
        score -= 0.25

    return max(0.0, min(1.0, score))


def _dedupe_preserve(values: list[str]) -> list[str]:
    """Deduplicate strings by normalized form while preserving original order.

    Uses normalize_indicator_for_comparison for deduplication keys. Skips
    empty strings and duplicates; returns first occurrence of each unique
    normalized value.

    Args:
        values: List of raw indicator strings.

    Returns:
        Deduplicated list in original order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        norm = normalize_indicator_for_comparison(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _merge_bbox(left: TableArtifact, right: TableArtifact) -> list[float] | None:
    """Compute the bounding box enclosing both tables.

    Same-page: returns the union (min x0,y0, max x1,y1). Cross-page: returns
    left's bbox only (merged bbox is not meaningful across pages).

    Args:
        left: First table artifact.
        right: Second table artifact.

    Returns:
        [x0, y0, x1, y1] or None if neither has a valid bbox.
    """
    bbox_left = _parse_bbox(left.bbox)
    bbox_right = _parse_bbox(right.bbox)
    if bbox_left is None:
        return list(bbox_right) if bbox_right is not None else None
    if bbox_right is None:
        return list(bbox_left)
    if left.page_pdf != right.page_pdf:
        return list(bbox_left)
    x0 = min(bbox_left[0], bbox_right[0])
    y0 = min(bbox_left[1], bbox_right[1])
    x1 = max(bbox_left[2], bbox_right[2])
    y1 = max(bbox_left[3], bbox_right[3])
    return [x0, y0, x1, y1]


def _merge_footnotes(
    left: TableArtifact, right: TableArtifact
) -> list[dict[str, str]] | None:
    """Merge footnotes from both fragments into a deduplicated list.

    Combines canonical footnotes from left and right, normalizes to canonical
    form, and deduplicates by (normalized_id, normalized_text) while preserving
    encounter order. Empty ids are replaced with positional numbers.

    Args:
        left: First table artifact.
        right: Second table artifact.

    Returns:
        List of {"id", "text"} dicts, or None if no footnotes.
    """
    combined = normalize_footnotes_to_canonical(
        get_canonical_footnotes(left) + get_canonical_footnotes(right)
    )
    if not combined:
        return None

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in combined:
        raw_id = str(item.get("id") or "").strip()
        raw_text = str(item.get("text") or "").strip()
        if not raw_text:
            continue
        norm_id = re.sub(r"\s+", "", raw_id).lower()
        norm_text = re.sub(r"\s+", " ", raw_text).strip().lower()
        dedupe_key = (norm_id, norm_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append({"id": raw_id or str(len(merged) + 1), "text": raw_text})
    return merged or None


def _merge_pair(left: TableArtifact, right: TableArtifact) -> TableArtifact:
    """Merge two table fragments into a single TableArtifact.

    Concatenates rows, deduplicates indicators and raw indicators, takes the
    longer header set, merges bbox and footnotes. Sets fragmentation_detected
    and table_id to "left__right". Metadata (bank_code, section, etc.) is
    inherited from left with right as fallback.

    Args:
        left: First (earlier) table fragment.
        right: Second (later) table fragment.

    Returns:
        New TableArtifact representing the merged table.
    """
    merged_rows = list(left.rows or []) + list(right.rows or [])
    merged_indicators = _dedupe_preserve(
        get_comparison_indicators(left) + get_comparison_indicators(right)
    )

    left_headers = list(left.headers or [])
    right_headers = list(right.headers or [])
    merged_headers = (
        right_headers if len(right_headers) > len(left_headers) else left_headers
    )
    merged_number: str | None = None
    if (
        left.table_number
        and right.table_number
        and left.table_number == right.table_number
    ):
        merged_number = left.table_number
    elif left.table_number:
        merged_number = left.table_number
    elif right.table_number:
        merged_number = right.table_number

    left_raw = get_vision_raw_indicators(left)
    right_raw = get_vision_raw_indicators(right)
    merged_raw: list[str] | None = None
    if left_raw or right_raw:
        merged_raw = _dedupe_preserve(list(left_raw) + list(right_raw))

    title_clean = getattr(left, "title_clean", None) or getattr(
        right, "title_clean", None
    )
    title_raw = getattr(left, "title_raw", None) or getattr(right, "title_raw", None)
    title_display = title_clean or (left.title or right.title)

    return TableArtifact(
        bank_code=left.bank_code or right.bank_code,
        section=left.section or right.section,
        page_pdf=min(int(left.page_pdf or 0), int(right.page_pdf or 0)),
        table_id=f"{left.table_id}__{right.table_id}",
        title=title_display,
        headers=merged_headers,
        rows=merged_rows,
        first_column_indicators=merged_indicators,
        extraction_method=left.extraction_method or right.extraction_method,
        table_number=merged_number,
        bbox=_merge_bbox(left, right),
        quarter=left.quarter or right.quarter,
        pdf_path=left.pdf_path or right.pdf_path,
        first_column_indicators_raw=merged_raw,
        footnotes=_merge_footnotes(left, right),
        fragmentation_detected=True,
        title_clean=title_clean,
        title_raw=title_raw,
        content_source=left.content_source or right.content_source,
        # comparison_blockers recomputed by TableArtifact.__post_init__
    )


def merge_table_sequence(tables: list[TableArtifact]) -> TableArtifact | None:
    """Merge an explicit ordered list of tables into one synthetic artifact.

    Iteratively merges tables from left to right using _merge_pair. Use when
    the caller has already determined the correct merge order.

    Args:
        tables: Ordered list of table artifacts to merge.

    Returns:
        Single merged TableArtifact, or None if the list is empty.
    """
    ordered = [table for table in tables if table is not None]
    if not ordered:
        return None
    merged = ordered[0]
    for table in ordered[1:]:
        merged = _merge_pair(merged, table)
    return merged


def merge_table_fragments(
    tables: list[TableArtifact],
    *,
    merge_score_min: float = 0.85,
) -> tuple[list[TableArtifact], list[dict[str, Any]]]:
    """Merge conservatively segmented fragments before logical table matching.

    Sorts tables by (section, page, table_id), then scans adjacent pairs.
    Pairs with score >= merge_score_min are merged; pairs with score between
    _NEAR_MERGE_THRESHOLD (0.60) and merge_score_min receive
    fragment_near_merge_hint annotations for downstream split-merge rescue.

    Args:
        tables: List of table artifacts from extraction.
        merge_score_min: Minimum _candidate_merge_score to perform a merge.
            Default 0.85.

    Returns:
        Tuple of (merged_table_list, events). Events describe merges and
        near-merge hints (merge_type, score, section, members, etc.).
    """
    if len(tables) < 2:
        return list(tables), []

    ordered = sorted(
        tables,
        key=lambda t: (
            _canonical_section(t.section),
            int(t.page_pdf or 0),
            str(t.table_id),
        ),
    )
    merged: list[TableArtifact] = []
    events: list[dict[str, Any]] = []

    idx = 0
    while idx < len(ordered):
        current = ordered[idx]
        if idx + 1 >= len(ordered):
            merged.append(current)
            idx += 1
            continue

        nxt = ordered[idx + 1]
        score = _candidate_merge_score(current, nxt)
        if score >= merge_score_min:
            combined = _merge_pair(current, nxt)
            merged.append(combined)
            events.append(
                {
                    "merge_type": "fragment_merge",
                    "score": round(score, 4),
                    "section": current.section,
                    "members": [
                        {
                            "table_id": current.table_id,
                            "page": current.page_pdf,
                            "title": current.title or "",
                        },
                        {
                            "table_id": nxt.table_id,
                            "page": nxt.page_pdf,
                            "title": nxt.title or "",
                        },
                    ],
                    "merged_table_id": combined.table_id,
                    "merged_page": combined.page_pdf,
                }
            )
            idx += 2
            continue

        if score >= _NEAR_MERGE_THRESHOLD:
            current.fragment_near_merge_hint = {
                "neighbor_table_id": nxt.table_id,
                "neighbor_page": nxt.page_pdf,
                "merge_score": round(score, 4),
            }
            nxt.fragment_near_merge_hint = {
                "neighbor_table_id": current.table_id,
                "neighbor_page": current.page_pdf,
                "merge_score": round(score, 4),
            }
            events.append(
                {
                    "merge_type": "fragment_near_merge_hint",
                    "score": round(score, 4),
                    "section": current.section,
                    "members": [
                        {
                            "table_id": current.table_id,
                            "page": current.page_pdf,
                            "title": current.title or "",
                        },
                        {
                            "table_id": nxt.table_id,
                            "page": nxt.page_pdf,
                            "title": nxt.title or "",
                        },
                    ],
                }
            )

        merged.append(current)
        idx += 1

    return merged, events


_NEAR_MERGE_THRESHOLD = 0.60


__all__ = ["merge_table_fragments", "merge_table_sequence"]
