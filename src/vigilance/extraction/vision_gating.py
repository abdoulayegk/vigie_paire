"""Gating logic to decide when Vision fallback is warranted for table extraction."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

INDICATOR_COUNT_MIN = 3
DUPLICATE_RATIO_MAX = 0.20
HEADER_LIKE_RATIO_MAX = 0.20
MERGE_COUNT_MAX = 8
TABLE_QUALITY_MIN = 0.5


def is_table_extraction_suspect(table: Any) -> bool:
    """
    Return True if the table extraction is suspect and Vision fallback may help.
    Uses debug_metrics when available; conservative default thresholds.
    """
    dm = getattr(table, "debug_metrics", None) or {}

    indicator_count = dm.get("indicator_count")
    if indicator_count is not None and indicator_count < INDICATOR_COUNT_MIN:
        return True

    duplicate_ratio = dm.get("duplicate_ratio")
    if duplicate_ratio is not None and duplicate_ratio > DUPLICATE_RATIO_MAX:
        return True

    header_like_ratio = dm.get("header_like_ratio")
    if header_like_ratio is not None and header_like_ratio > HEADER_LIKE_RATIO_MAX:
        return True

    line_merges = dm.get("line_reconstruction_merges", dm.get("merge_count"))
    if line_merges is not None and line_merges > MERGE_COUNT_MAX:
        return True

    table_quality_score = dm.get("table_quality_score")
    if table_quality_score is not None and table_quality_score < TABLE_QUALITY_MIN:
        return True

    return False


def get_suspect_reasons(table: Any) -> list[str]:
    """Return human-readable reasons why extraction is suspect."""
    dm = getattr(table, "debug_metrics", None) or {}
    reasons: list[str] = []

    indicator_count = dm.get("indicator_count")
    if indicator_count is not None and indicator_count < INDICATOR_COUNT_MIN:
        reasons.append(f"low_indicator_count({indicator_count})")

    duplicate_ratio = dm.get("duplicate_ratio")
    if duplicate_ratio is not None and duplicate_ratio > DUPLICATE_RATIO_MAX:
        reasons.append(f"high_duplicate_ratio({duplicate_ratio:.2f})")

    header_like_ratio = dm.get("header_like_ratio")
    if header_like_ratio is not None and header_like_ratio > HEADER_LIKE_RATIO_MAX:
        reasons.append(f"high_header_like_ratio({header_like_ratio:.2f})")

    line_merges = dm.get("line_reconstruction_merges", dm.get("merge_count"))
    if line_merges is not None and line_merges > MERGE_COUNT_MAX:
        reasons.append(f"high_line_merges({line_merges})")

    table_quality_score = dm.get("table_quality_score")
    if table_quality_score is not None and table_quality_score < TABLE_QUALITY_MIN:
        reasons.append(f"low_quality_score({table_quality_score:.2f})")

    return reasons
