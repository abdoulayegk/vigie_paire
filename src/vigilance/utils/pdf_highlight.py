"""Recherche de texte et surlignage dans une region d'une page PDF via PyMuPDF."""

from __future__ import annotations

import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


def find_text_bboxes_in_region(
    pdf_path: str,
    page_number: int,
    text_to_find: str,
    region_bbox_norm: list[float],
) -> list[list[float]]:
    """Recherche un texte dans une region contrainte d'une page PDF et retourne les bboxes normalisees.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numero de page (base 1).
        text_to_find: Chaine exacte a localiser (ou partie de celle-ci).
        region_bbox_norm: Bbox normalisee [l, t, r, b] de la region de recherche.

    Returns:
        Liste de bounding boxes normalisees [l, t, r, b] ou le texte a ete trouve.
        Retourne une liste vide si non trouve.
    """
    if fitz is None:
        logger.warning(
            "PyMuPDF (fitz) is not installed. Text search for highlights will be disabled."
        )
        return []

    if not text_to_find or not text_to_find.strip():
        return []

    # Take only the first line (PDF text spans can break across lines).
    # Do NOT truncate further — truncation causes false positives when multiple
    # rows share a common prefix (e.g. "Billets avec remboursement de capital à
    # recours limité – Série 5" vs "– Série 6").
    search_text = text_to_find.strip().split("\n")[0].strip()

    try:
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            return []

        page = doc[page_number - 1]
        rect = page.rect

        # Convert normalized region bbox to absolute coordinates
        rx0 = rect.x0 + region_bbox_norm[0] * rect.width
        ry0 = rect.y0 + region_bbox_norm[1] * rect.height
        rx1 = rect.x0 + region_bbox_norm[2] * rect.width
        ry1 = rect.y0 + region_bbox_norm[3] * rect.height
        clip_rect = fitz.Rect(rx0, ry0, rx1, ry1)

        # Search for the text strictly inside the region
        matches = page.search_for(search_text, clip=clip_rect)

        # Convert absolute matches back to normalized coords
        normalized_matches = []
        for match_rect in matches:
            nl = (match_rect.x0 - rect.x0) / rect.width
            nt = (match_rect.y0 - rect.y0) / rect.height
            nr = (match_rect.x1 - rect.x0) / rect.width
            nb = (match_rect.y1 - rect.y0) / rect.height
            normalized_matches.append(
                [max(0.0, nl), max(0.0, nt), min(1.0, nr), min(1.0, nb)]
            )

        return normalized_matches

    except Exception as e:
        logger.error(f"Error searching text in PDF: {e}")
        return []
