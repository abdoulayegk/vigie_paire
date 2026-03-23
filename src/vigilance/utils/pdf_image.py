"""Convert a PDF page to a numpy RGB image for Vision fallback.

Used by docling_processor to render a page before calling GPT-4 Vision
on tables that Docling failed to extract. Returns None if PyMuPDF or
numpy is unavailable or on any error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def pdf_page_to_image(
    pdf_path: str,
    page_number: int,
    dpi: int = 300,
) -> "np.ndarray | None":
    """Render a single PDF page as a numpy RGB image.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page index (same as docling_processor page_num).
        dpi: Resolution for rendering (default 300, same as gpt4_vision_fallback).

    Returns:
        RGB image as numpy array (height, width, 3), or None on error or
        if PyMuPDF/numpy is unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        import fitz  # PyMuPDF
        from .pymupdf_utils import configure_mupdf_runtime
    except ImportError:
        return None
    configure_mupdf_runtime(fitz)

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = img[:, :, :3].copy()
        elif pix.n == 1:
            img = np.stack([img.squeeze()] * 3, axis=-1)
        else:
            img = img.copy()
        doc.close()
        return img
    except Exception:
        return None
