"""Shared proof-rendering helpers used by Dash and comparison pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vigilance.extraction.pdf_preview import render_pdf_page
from vigilance.utils.pdf_crop import render_page_with_bbox_highlight_to_bytes


def normalize_proof_bbox(bbox: Any) -> list[float] | None:
    """Return [l, t, r, b] in 0..1 when usable for proof rendering."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        normalized = [
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        ]
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = normalized
    if not (
        0.0 <= left <= 1.0
        and 0.0 <= top <= 1.0
        and 0.0 <= right <= 1.0
        and 0.0 <= bottom <= 1.0
    ):
        return None
    if right <= left or bottom <= top:
        return None
    return normalized


def render_full_proof_bytes(
    pdf_path: str | Path | None,
    *,
    page: Any,
    bbox: Any,
    scale: float = 1.5,
    allow_full_page_fallback: bool = False,
) -> tuple[bytes | None, str, str]:
    """Render a full-page proof image with bbox highlight when available.

    Returns ``(image_bytes, status, mode_effective)`` where status is one of:
    ``ok``, ``pdf_missing``, ``page_missing``, ``bbox_missing``, or ``render_failed``.
    ``mode_effective`` is ``full`` or ``full_without_bbox`` when a page-only fallback
    is used because the bbox is unavailable.
    """

    pdf = str(pdf_path or "").strip()
    if not pdf:
        return None, "pdf_missing", "full"

    try:
        page_num = int(page)
    except (TypeError, ValueError):
        return None, "page_missing", "full"
    if page_num < 1:
        return None, "page_missing", "full"

    bbox_norm = normalize_proof_bbox(bbox)
    if bbox_norm is None:
        if not allow_full_page_fallback:
            return None, "bbox_missing", "full"
        raw = render_pdf_page(pdf, page_num, scale=scale, format="png")
        if raw:
            return raw, "ok", "full_without_bbox"
        return None, "render_failed", "full_without_bbox"

    raw = render_page_with_bbox_highlight_to_bytes(
        pdf,
        page_num,
        bbox_norm,
        scale=scale,
    )
    if raw:
        return raw, "ok", "full"
    if not allow_full_page_fallback:
        return None, "render_failed", "full"
    raw = render_pdf_page(pdf, page_num, scale=scale, format="png")
    if raw:
        return raw, "ok", "full"
    return None, "render_failed", "full"
