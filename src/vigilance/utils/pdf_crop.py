"""PDF table cropping utility using PyMuPDF. Renders a cropped region to PNG."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def crop_table_image(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    out_path: str,
    dpi: int = 300,
) -> bool:
    """
    Crop a table region from a PDF page and save as PNG.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        bbox_norm: Normalized bounding box [l, t, r, b] in 0..1 (left, top, right, bottom).
        out_path: Output path for the PNG file.
        dpi: Resolution for rendering (default 300).

    Returns:
        True on success, False on failure. Does not raise exceptions.
    """
    if not _validate_bbox(bbox_norm):
        return False

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available for crop_table_image")
        return False

    try:
        doc = fitz.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                return False
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            x0 = rect.x0 + l_norm * rect.width
            y0 = rect.y0 + t_norm * rect.height
            x1 = rect.x0 + r_norm * rect.width
            y1 = rect.y0 + b_norm * rect.height
            clip = fitz.Rect(x0, y0, x1, y1)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pix.save(out_path)
            return True
        finally:
            doc.close()
    except Exception:
        return False


def _validate_bbox(bbox_norm: list[float]) -> bool:
    """Validate normalized bbox: len=4, each in [0,1], x1>x0, y1>y0."""
    if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
        return False
    try:
        l_norm, t_norm, r_norm, b_norm = (
            float(bbox_norm[0]),
            float(bbox_norm[1]),
            float(bbox_norm[2]),
            float(bbox_norm[3]),
        )
    except (TypeError, ValueError):
        return False
    if not (0 <= l_norm <= 1 and 0 <= t_norm <= 1 and 0 <= r_norm <= 1 and 0 <= b_norm <= 1):
        return False
    if r_norm <= l_norm or b_norm <= t_norm:
        return False
    return True
