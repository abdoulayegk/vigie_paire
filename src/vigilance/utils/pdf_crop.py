"""PDF table cropping utility using PyMuPDF. Renders a cropped region to PNG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _crop_cache_key(
    kind: str,
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    zoom: float,
    bottom_extension: float,
    top_extension: float = 0.0,
) -> tuple[Any, ...]:
    """Stable cache key for crop/render memoization within a run."""
    return (
        kind,
        pdf_path,
        page_number,
        tuple(bbox_norm),
        zoom,
        bottom_extension,
        top_extension,
    )


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


def crop_table_region_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    top_extension: float = 0.0,
    dpi: int | None = None,
    render_cache: dict[tuple[Any, ...], bytes] | None = None,
    bottom_stop_norm: float | None = None,
) -> bytes:
    """
    Crop a table region from a PDF page and return PNG bytes.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number (matches render_pdf_page convention).
        bbox_norm: Normalized bounding box [l, t, r, b] in 0..1.
        scale: Render scale when dpi is not set (default 1.5, same as proof previews).
        bottom_extension: Extra height below bbox (e.g. for footnotes), in normalized 0..1.
        top_extension: Extra height above bbox (e.g. for title), in normalized 0..1.
        dpi: If set, render at this resolution (72 * zoom); overrides scale. Use 300 for Vision/OCR.
        render_cache: Optional dict to memoize results within a run (keyed by path/page/bbox/zoom/ext).
        bottom_stop_norm: If set, cap bottom of crop at this y (0..1). Use to avoid including
            the next table on same page when extracting footnotes.

    Returns:
        PNG bytes of the cropped region, or full page bytes if bbox invalid or crop fails.
    """
    from vigilance.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale

    if render_cache is not None and _validate_bbox(bbox_norm):
        key = _crop_cache_key(
            "crop",
            pdf_path,
            page_number,
            bbox_norm,
            zoom,
            bottom_extension,
            top_extension,
        )
        if bottom_stop_norm is not None:
            key = (*key, bottom_stop_norm)
        if key in render_cache:
            return render_cache[key]

    if not _validate_bbox(bbox_norm):
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available for crop_table_region_to_bytes")
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

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
            t_norm_effective = max(0.0, t_norm - top_extension)
            b_norm_effective = min(1.0, b_norm + bottom_extension)
            if bottom_stop_norm is not None:
                b_norm_effective = min(b_norm_effective, bottom_stop_norm)
            x0 = rect.x0 + l_norm * rect.width
            y0 = rect.y0 + t_norm_effective * rect.height
            x1 = rect.x0 + r_norm * rect.width
            y1 = rect.y0 + b_norm_effective * rect.height
            clip = fitz.Rect(x0, y0, x1, y1)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            out = pix.tobytes("png")
            if render_cache is not None:
                key = _crop_cache_key(
                    "crop",
                    pdf_path,
                    page_number,
                    bbox_norm,
                    zoom,
                    bottom_extension,
                    top_extension,
                )
                if bottom_stop_norm is not None:
                    key = (*key, bottom_stop_norm)
                render_cache[key] = out
            return out
        finally:
            doc.close()
    except Exception:
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""


def render_page_with_bbox_highlight_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    dpi: int | None = None,
    render_cache: dict[tuple[Any, ...], bytes] | None = None,
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
        render_cache: Optional dict to memoize results within a run (keyed by path/page/bbox/zoom/ext).

    Returns:
        PNG bytes of the full page with a red highlight box, or normal full page if bbox invalid.
    """
    from vigilance.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale

    if render_cache is not None and _validate_bbox(bbox_norm):
        key = _crop_cache_key(
            "bbox", pdf_path, page_number, bbox_norm, zoom, bottom_extension
        )
        if key in render_cache:
            return render_cache[key]

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
            out = pix.tobytes("png")
            if render_cache is not None:
                key = _crop_cache_key(
                    "bbox", pdf_path, page_number, bbox_norm, zoom, bottom_extension
                )
                render_cache[key] = out
            return out
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
