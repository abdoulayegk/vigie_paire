"""PDF table cropping utility using PyMuPDF. Renders a cropped region to PNG."""

from __future__ import annotations

import logging
from pathlib import Path

from .pymupdf_utils import configure_mupdf_runtime

logger = logging.getLogger(__name__)


def crop_table_image(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    out_path: str,
    dpi: int = 300,
    bottom_extension: float = 0.0,
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
    configure_mupdf_runtime(fitz)

    try:
        doc = fitz.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                return False
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            pad = 0.03
            b_norm_effective = min(1.0, b_norm + bottom_extension)
            x0 = rect.x0 + max(0.0, l_norm - pad) * rect.width
            y0 = rect.y0 + max(0.0, t_norm - pad) * rect.height
            x1 = rect.x0 + min(1.0, r_norm + pad) * rect.width
            y1 = rect.y0 + min(1.0, b_norm_effective + pad) * rect.height
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
    if not (
        0 <= l_norm <= 1 and 0 <= t_norm <= 1 and 0 <= r_norm <= 1 and 0 <= b_norm <= 1
    ):
        return False
    if r_norm <= l_norm or b_norm <= t_norm:
        return False
    return True


def bbox_sanity_profile(bbox_norm: list[float]) -> dict:
    """
    Compute a sanity profile for a normalized bbox [l, t, r, b].
    Used to gate Vision: reject too small, near-full-page, or extreme aspect ratio.
    """
    if not _validate_bbox(bbox_norm):
        return {
            "width_norm": 0.0,
            "height_norm": 0.0,
            "area_norm": 0.0,
            "is_near_full_page": False,
            "aspect_ratio": 0.0,
            "reject_reason": "invalid_bbox",
        }
    l_norm, t_norm, r_norm, b_norm = bbox_norm
    w = r_norm - l_norm
    h = b_norm - t_norm
    area = w * h
    aspect = w / h if h > 0 else 0.0
    # Near-full-page if area or any dimension is above threshold
    near_full = (
        area >= 0.90
        or w >= 0.95
        or h >= 0.95
    )
    profile: dict = {
        "width_norm": round(w, 6),
        "height_norm": round(h, 6),
        "area_norm": round(area, 6),
        "is_near_full_page": near_full,
        "aspect_ratio": round(aspect, 4),
    }
    return profile


def is_bbox_sane(bbox_norm: list[float], cfg: dict | None = None) -> tuple[bool, str | None, dict]:
    """
    Return (sane, reject_reason, profile). If sane is False, reject_reason is set.
    cfg may contain: bbox_min_width, bbox_min_height, bbox_min_area, bbox_max_area,
    bbox_near_full_page_threshold (all optional; defaults used if missing).
    """
    profile = bbox_sanity_profile(bbox_norm)
    if profile.get("reject_reason") == "invalid_bbox":
        return False, "invalid_bbox", profile

    cfg = cfg or {}
    min_w = float(cfg.get("bbox_min_width", 0.02))
    min_h = float(cfg.get("bbox_min_height", 0.02))
    min_area = float(cfg.get("bbox_min_area", 0.0005))
    max_area = float(cfg.get("bbox_max_area", 0.95))
    near_full_threshold = float(cfg.get("bbox_near_full_page_threshold", 0.90))

    w = profile["width_norm"]
    h = profile["height_norm"]
    area = profile["area_norm"]

    if w < min_w:
        profile["reject_reason"] = "bbox_too_narrow"
        return False, "bbox_too_narrow", profile
    if h < min_h:
        profile["reject_reason"] = "bbox_too_short"
        return False, "bbox_too_short", profile
    if area < min_area:
        profile["reject_reason"] = "bbox_area_too_small"
        return False, "bbox_area_too_small", profile
    if area > max_area:
        profile["reject_reason"] = "bbox_area_too_large"
        return False, "bbox_area_too_large", profile
    if area >= near_full_threshold or w >= near_full_threshold or h >= near_full_threshold:
        profile["reject_reason"] = "bbox_near_full_page"
        return False, "bbox_near_full_page", profile
    # Extreme aspect ratio inconsistent with tables (e.g. 20:1 or 1:20)
    if w / h > 20.0 or h / w > 20.0:
        profile["reject_reason"] = "extreme_aspect_ratio"
        return False, "extreme_aspect_ratio", profile

    return True, None, profile


def crop_table_region_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    top_extension: float = 0.0,
    horizontal_padding: float = 0.0,
    dpi: int | None = None,
) -> bytes:
    """
    Crop a table region from a PDF page and return PNG bytes.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number (matches render_pdf_page convention).
        bbox_norm: Normalized bounding box [l, t, r, b] in 0..1.
        scale: Render scale when dpi is not set (default 1.5, same as proof previews).
        bottom_extension: Extra height below bbox (e.g. for footnotes), in normalized 0..1.
        top_extension: Extra height above bbox (e.g. for title or missed rows), in normalized 0..1.
        horizontal_padding: Symmetric horizontal padding (normalized 0..1), clamped to page.
        dpi: If set, render at this resolution (72 * zoom); overrides scale. Use 300 for Vision/OCR.

    Returns:
        PNG bytes of the cropped region. Returns b"" on invalid bbox, page out of range,
        import failure, or any crop exception (no full-page fallback).
    """
    zoom = (dpi / 72.0) if dpi is not None else scale

    if not _validate_bbox(bbox_norm):
        return b""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available for crop_table_region_to_bytes")
        return b""
    configure_mupdf_runtime(fitz)

    try:
        doc = fitz.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                return b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            t_norm_effective = max(0.0, t_norm - top_extension)
            b_norm_effective = min(1.0, b_norm + bottom_extension)
            l_norm_effective = max(0.0, l_norm - horizontal_padding)
            r_norm_effective = min(1.0, r_norm + horizontal_padding)
            x0 = rect.x0 + l_norm_effective * rect.width
            y0 = rect.y0 + t_norm_effective * rect.height
            x1 = rect.x0 + r_norm_effective * rect.width
            y1 = rect.y0 + b_norm_effective * rect.height
            clip = fitz.Rect(x0, y0, x1, y1)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return b""


def render_page_with_bbox_highlight_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    dpi: int | None = None,
) -> bytes:
    """
    Render a full PDF page to PNG with a red bounding box around the table.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        bbox_norm: Normalized bounding box [l, t, r, b] in 0..1.
        scale: Render scale when dpi is not set.
        bottom_extension: Extra height included in the red box (e.g. for footnotes), in normalized 0..1.
        dpi: If set, render at this resolution (72 * zoom); overrides scale. Use 300 for Vision/OCR.

    Returns:
        PNG bytes of the full page with a red highlight box, or normal full page if bbox invalid.
    """
    from vigilance.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale

    if not _validate_bbox(bbox_norm):
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.debug(
            "PyMuPDF (fitz) not available for render_page_with_bbox_highlight_to_bytes"
        )
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""
    configure_mupdf_runtime(fitz)

    try:
        doc = fitz.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
                return full if full else b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            b_norm_effective = min(1.0, b_norm + bottom_extension)

            x0 = rect.x0 + l_norm * rect.width
            y0 = rect.y0 + t_norm * rect.height
            x1 = rect.x0 + r_norm * rect.width
            y1 = rect.y0 + b_norm_effective * rect.height

            # Draw a bright red rectangle with 3px width on the page
            page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 0, 0), width=3)

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""


def crop_footnote_region_to_bytes(
    pdf_path: str,
    page_number: int,
    table_bbox_norm: list[float],
    scale: float = 1.5,
    footnote_height: float = 0.25,
    dpi: int | None = None,
) -> bytes:
    """
    Crop only the footnote region below a table from a PDF page.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        table_bbox_norm: Normalized bounding box of the table [l, t, r, b] in 0..1.
        scale: Render scale when dpi is not set.
        footnote_height: Height of footnote region as fraction of page (default 0.25 = 25%).
        dpi: If set, render at this resolution; overrides scale.

    Returns:
        PNG bytes of the footnote region below the table.
    """
    from vigilance.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale

    if not _validate_bbox(table_bbox_norm):
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available for crop_footnote_region_to_bytes")
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""
    configure_mupdf_runtime(fitz)

    try:
        doc = fitz.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
                return full if full else b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = table_bbox_norm

            # Footnote region: from bottom of table to footnote_height below (or page bottom)
            footnote_top = b_norm
            footnote_bottom = min(1.0, b_norm + footnote_height)

            # Use full page width for footnotes (they often span the whole width)
            x0 = rect.x0
            y0 = rect.y0 + footnote_top * rect.height
            x1 = rect.x1
            y1 = rect.y0 + footnote_bottom * rect.height

            clip = fitz.Rect(x0, y0, x1, y1)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""
