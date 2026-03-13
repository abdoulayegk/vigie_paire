"""Open PDF with PyMuPDF and neutralize malformed structure tree to avoid MuPDF stderr noise."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import fitz
else:
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        fitz = None  # type: ignore[assignment]


_STRUCTURE_TREE_MARKERS = ("structure tree", "No common ancestor", "StructTree")


def open_pdf_safely(pdf_path: Union[str, Path]) -> "fitz.Document":
    """
    Open a PDF with PyMuPDF, capture native stderr during open, and neutralize
    malformed StructTreeRoot to avoid "No common ancestor in structure tree" noise.

    Returns:
        Opened fitz.Document. Caller must close() when done.

    Raises:
        Same exceptions as fitz.open() on unrecoverable failure.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed")
    path_str = str(pdf_path)
    stderr_capture = io.StringIO()
    doc = None
    with redirect_stderr(stderr_capture):
        doc = fitz.open(path_str)
    stderr_text = stderr_capture.getvalue()
    if stderr_text and any(m in stderr_text for m in _STRUCTURE_TREE_MARKERS):
        logger.warning(
            "PDF has malformed structure tree; neutralized for safe use. path=%s",
            path_str[:200],
        )
    if doc is not None:
        try:
            catalog = doc.pdf_catalog()
            if catalog:
                doc.xref_set_key(catalog, "StructTreeRoot", "null")
        except Exception:
            pass
    return doc
