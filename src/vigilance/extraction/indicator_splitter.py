"""Strict splitter for fused first-column indicators."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_BULLET_SPLIT_RE = re.compile(r"(?:\n+|\s*[•▪◦·]\s+)")
_LEADING_BULLET_RE = re.compile(r"^\s*[•▪◦·\-]+\s*")
_SPACES_RE = re.compile(r"\s+")

_GROUP_HEADERS = [
    "actifs",
    "passifs",
    "capitaux propres",
    "fonds propres",
    "assets",
    "liabilities",
    "equity",
]

_FUSED_ANCHORS = [
    "actifs",
    "passifs",
    "depots",
    "dépôts",
    "prets",
    "prêts",
    "titres",
    "instruments financiers",
    "obligations",
    "autres",
]


@dataclass
class IndicatorSplitResult:
    rows: list[list[str]]
    split_status: str
    split_confidence: float
    needs_review: bool
    uncertainty_reasons: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return _SPACES_RE.sub(" ", (text or "").strip())


def _clean_segment(text: str) -> str:
    cleaned = _LEADING_BULLET_RE.sub("", text or "")
    return _normalize_text(cleaned)


def _looks_fused(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered or lowered.startswith("total "):
        return False
    words = re.findall(r"\w+", lowered)
    if len(lowered) < 90 or len(words) < 12:
        return False
    anchor_hits = sum(1 for anchor in _FUSED_ANCHORS if anchor in lowered)
    return anchor_hits >= 3


def _split_with_group_header(text: str) -> list[str] | None:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    for header in _GROUP_HEADERS:
        prefix = f"{header} "
        if lowered.startswith(prefix):
            head_len = len(prefix)
            remainder = normalized[head_len:].strip()
            if len(remainder) < 10:
                continue
            if remainder.lower().startswith(("total ", "dont ", "dont:")):
                continue
            return [normalized[: head_len - 1], remainder]
    return None


def _split_label(text: str) -> tuple[list[str], float, str]:
    raw_text = text or ""
    has_newline = "\n" in raw_text
    normalized = _normalize_text(text)
    if not normalized:
        return [], 0.0, "empty"

    # Strong signal: line breaks or bullets.
    if has_newline or any(marker in raw_text for marker in ("•", "▪", "◦", "·")):
        parts = [_clean_segment(part) for part in _BULLET_SPLIT_RE.split(raw_text)]
        parts = [part for part in parts if part]
        if len(parts) >= 2:
            return parts, 0.9, "deterministic_bullets_newlines"

    # Header + detail fused in a single cell.
    header_split = _split_with_group_header(normalized)
    if header_split and len(header_split) == 2:
        return header_split, 0.75, "header_detail_fused"

    return [normalized], 0.55, "no_split"


def _rows_from_labels(labels: Iterable[str], width: int) -> list[list[str]]:
    return [[label] + ([""] * max(0, width - 1)) for label in labels]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {_normalize_text(x).lower() for x in a if _normalize_text(x)}
    sb = {_normalize_text(x).lower() for x in b if _normalize_text(x)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_table_rows(
    rows: list[list[str]],
    *,
    strict: bool = True,
    vision_labels: list[str] | None = None,
) -> IndicatorSplitResult:
    """Split fused first-column cells into canonical indicator rows."""
    if not rows:
        return IndicatorSplitResult(
            rows=[],
            split_status="no_rows",
            split_confidence=1.0,
            needs_review=False,
            uncertainty_reasons=[],
        )

    max_width = max((len(row) for row in rows), default=1)
    split_rows: list[list[str]] = []
    unresolved_fused = False
    did_split = False
    min_confidence = 1.0
    uncertainty_reasons: list[str] = []

    for row in rows:
        if not row:
            continue

        base = list(row) + ([""] * max(0, max_width - len(row)))
        first_cell = base[0]
        labels, confidence, reason = _split_label(str(first_cell))
        min_confidence = min(min_confidence, confidence)

        if _looks_fused(str(first_cell)) and len(labels) == 1:
            unresolved_fused = True
            if "fused_cell_unresolved" not in uncertainty_reasons:
                uncertainty_reasons.append("fused_cell_unresolved")

        if len(labels) > 1:
            did_split = True
            # If any produced label still looks like a fused cell, keep manual review.
            if strict and any(_looks_fused(label) for label in labels):
                unresolved_fused = True
                if "fused_cell_unresolved" not in uncertainty_reasons:
                    uncertainty_reasons.append("fused_cell_unresolved")
            split_rows.extend(_rows_from_labels(labels, max_width))
            continue

        split_rows.append(base)

    split_status = "split_applied" if did_split else "unchanged"

    # Vision-assisted alignment in strict mode.
    if strict and vision_labels:
        deterministic_labels = [
            _normalize_text(str(row[0]))
            for row in split_rows
            if row and _normalize_text(str(row[0]))
        ]
        vision_clean = [_normalize_text(label) for label in vision_labels if _normalize_text(label)]
        overlap = _jaccard(deterministic_labels, vision_clean)

        if vision_clean and (unresolved_fused or overlap < 0.8):
            split_rows = _rows_from_labels(vision_clean, max_width)
            split_status = "vision_aligned"
            min_confidence = 0.95
            unresolved_fused = False
            uncertainty_reasons = [r for r in uncertainty_reasons if r != "fused_cell_unresolved"]

    needs_review = strict and unresolved_fused
    if needs_review and "split_ambiguous" not in uncertainty_reasons:
        uncertainty_reasons.append("split_ambiguous")

    return IndicatorSplitResult(
        rows=split_rows,
        split_status=split_status,
        split_confidence=max(0.0, min(1.0, min_confidence if min_confidence <= 1 else 1.0)),
        needs_review=needs_review,
        uncertainty_reasons=uncertainty_reasons,
    )
