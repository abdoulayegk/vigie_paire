"""Conversion d'une page PDF en image numpy RGB pour le repli Vision.

Utilise par docling_processor pour rendre une page avant d'appeler GPT-4 Vision
sur les tableaux que Docling n'a pas reussi a extraire. Retourne None si PyMuPDF
ou numpy n'est pas disponible, ou en cas d'erreur.
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
    """Rend une page PDF unique sous forme d'image numpy RGB.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Index de page (base 1, meme convention que docling_processor).
        dpi: Resolution de rendu (defaut 300, identique a gpt4_vision_fallback).

    Returns:
        Image RGB sous forme de tableau numpy (hauteur, largeur, 3), ou ``None``
        en cas d'erreur ou si PyMuPDF/numpy n'est pas disponible.
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
