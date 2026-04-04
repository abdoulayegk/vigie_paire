"""Extraction de texte depuis des pages PDF via pdfplumber."""

from __future__ import annotations

import pdfplumber


def extract_page_text(pdf_path: str, page_index: int) -> str:
    """Extrait le contenu textuel d'une page PDF unique.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_index: Index de page 0-indexe.

    Returns:
        Texte extrait, ou ``""`` si pdfplumber retourne ``None``.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        text = page.extract_text()
        return text if text is not None else ""


def extract_pages_text(pdf_path: str, page_indices: list[int]) -> dict[int, str]:
    """Extrait le texte de plusieurs pages PDF en un seul appel d'ouverture.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_indices: Indices de pages 0-indexes a extraire.

    Returns:
        Dictionnaire ``{index_de_page: texte_extrait}``.
    """
    result: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            page = pdf.pages[idx]
            text = page.extract_text()
            result[idx] = text if text is not None else ""
    return result
