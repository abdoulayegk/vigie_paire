"""Conversion des bbox PyMuPDF Layout vers le format pipeline Vision."""

from __future__ import annotations

from typing import Sequence


def normalize_pymupdf_bbox(
    bbox: Sequence[float],
    page_width: float,
    page_height: float,
) -> list[float]:
    """Convertir une bbox absolue PyMuPDF en ``[l, t, r, b]`` normalise 0..1.

    PyMuPDF utilise une origine en haut a gauche et des coordonnees en points.
    Le pipeline Vision / ``pdf_crop`` attend le meme repere, normalise par la
    largeur et la hauteur de page.

    Args:
        bbox: ``[x0, y0, x1, y1]`` en points page.
        page_width: Largeur de page en points (> 0).
        page_height: Hauteur de page en points (> 0).

    Returns:
        ``[l, t, r, b]`` dans ``[0, 1]``.

    Raises:
        ValueError: Dimensions ou bbox invalides.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Invalid page size: width={page_width}, height={page_height}")
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain exactly 4 values, got {len(bbox)}")
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    left = max(0.0, min(1.0, x0 / page_width))
    top = max(0.0, min(1.0, y0 / page_height))
    right = max(0.0, min(1.0, x1 / page_width))
    bottom = max(0.0, min(1.0, y1 / page_height))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return [left, top, right, bottom]


def page_number_from_layout(page_number: int, *, zero_based_input: bool = False) -> int:
    """Normaliser un numero de page vers le convention pipeline (1-indexe).

    Args:
        page_number: Valeur brute (JSON Layout est deja 1-indexe ; l'API
            ``pages=`` de pymupdf4llm est 0-indexee).
        zero_based_input: Si True, convertit ``n`` -> ``n + 1``.

    Returns:
        Page 1-indexee (>= 1).
    """
    value = int(page_number)
    if zero_based_input:
        value += 1
    if value < 1:
        raise ValueError(f"page_number must be >= 1 after normalization, got {value}")
    return value
