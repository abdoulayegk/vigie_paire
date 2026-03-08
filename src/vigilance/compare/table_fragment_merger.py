"""Conservative merger for table fragments produced by imperfect segmentation."""

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
    return (value or "").strip().lower()


def _is_known_section(value: str | None) -> bool:
    return _canonical_section(value) not in UNKNOWN_SECTIONS


def _normalize_title(value: str | None) -> str:
    cleaned = strip_temporal_expressions(value or "", target="title")
    return normalize_for_matching(cleaned, target="title")


def _title_similarity(left: str | None, right: str | None) -> float:
    lnorm = _normalize_title(left)
    rnorm = _normalize_title(right)
    if not lnorm and not rnorm:
        return 0.60
    if not lnorm or not rnorm:
        return 0.35
    return SequenceMatcher(None, lnorm, rnorm).ratio()


def _extract_indicators(table: TableArtifact) -> set[str]:
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
    lset = _extract_indicators(left)
    rset = _extract_indicators(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _indicator_non_overlap_score(left: TableArtifact, right: TableArtifact) -> float:
    jaccard = _indicator_jaccard(left, right)
    if jaccard <= 0.05:
        return 1.0
    return max(0.0, 1.0 - jaccard)


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x0, y0, x1, y1 = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
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
    labels = get_vision_raw_indicators(table)
    if not labels:
        labels = get_comparison_indicators(table)
    for raw_label in labels[-3:]:
        label = str(raw_label or "").strip()
        if _TOTAL_ROW_RE.search(label):
            return True
    return False


def _candidate_merge_score(left: TableArtifact, right: TableArtifact) -> float:
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
    continuation_score = 1.0 if (_has_continuation_hint(left) or _has_continuation_hint(right)) else 0.0

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


def _merge_footnotes(left: TableArtifact, right: TableArtifact) -> list[dict[str, str]] | None:
    """Merge footnotes from both fragments, dedupe by normalized (id, text), preserve order."""
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
    merged_rows = list(left.rows or []) + list(right.rows or [])
    merged_indicators = _dedupe_preserve(
        get_comparison_indicators(left) + get_comparison_indicators(right)
    )

    left_headers = list(left.headers or [])
    right_headers = list(right.headers or [])
    merged_headers = right_headers if len(right_headers) > len(left_headers) else left_headers
    merged_number: str | None = None
    if left.table_number and right.table_number and left.table_number == right.table_number:
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

    title_clean = getattr(left, "title_clean", None) or getattr(right, "title_clean", None)
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
        comparison_blockers=list(
            dict.fromkeys(
                list(getattr(left, "comparison_blockers", None) or [])
                + list(getattr(right, "comparison_blockers", None) or [])
            )
        ),
    )


def merge_table_fragments(
    tables: list[TableArtifact],
    *,
    merge_score_min: float = 0.85,
) -> tuple[list[TableArtifact], list[dict[str, Any]]]:
    """Merge conservatively segmented fragments before logical table matching."""
    if len(tables) < 2:
        return list(tables), []

    ordered = sorted(
        tables,
        key=lambda t: (_canonical_section(t.section), int(t.page_pdf or 0), str(t.table_id)),
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

        merged.append(current)
        idx += 1

    return merged, events


__all__ = ["merge_table_fragments"]
