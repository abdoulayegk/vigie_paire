"""Utilitaires de géométrie BBox et helpers pour l'extraction Docling.

Ce module centralise la manipulation des chemins PDF, le calcul des intersections
de Bounding Boxes (BBox) et le traitement des fragments d'en-tête.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}


def _coerce_pdf_path(pdf_path: str | Path | os.PathLike[str] | None) -> Path:
    """Valider et convertir un chemin PDF en objet Path."""
    if pdf_path is None:
        raise ValueError("Chemin PDF requis pour l'extraction.")
    try:
        path = Path(pdf_path)
    except TypeError as exc:
        raise ValueError(f"Chemin PDF invalide: {pdf_path!r}") from exc
    if not str(path).strip():
        raise ValueError("Chemin PDF vide pour l'extraction.")
    return path


def _looks_short_textual_header_fragment(value: str) -> bool:
    """Déterminer si une valeur ressemble à un fragment d'en-tête textuel court."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return False
    if any(char.isdigit() for char in text):
        return False
    words = text.split()
    if len(words) == 0 or len(words) > 4:
        return False
    if len(text) > 28:
        return False
    return any(char.isalpha() for char in text)


_REFERENCE_TEXT_SPLIT_RE = re.compile(r"\s{2,}|\t+|\s+\|\s+")


def _build_indicator_reference_text(
    raw_text: str | None,
    *,
    max_chars: int,
) -> str | None:
    """Retourner un dictionnaire OCR filtré pour l'extraction d'indicateurs."""
    text = str(raw_text or "").strip()
    if len(text) <= 20 or max_chars <= 0:
        return None

    raw_lines = [str(line).strip() for line in text.splitlines() if str(line).strip()]
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_lines]
    if not lines:
        return None

    for line in raw_lines[:3]:
        parts = [
            re.sub(r"\s+", " ", part).strip()
            for part in _REFERENCE_TEXT_SPLIT_RE.split(line)
            if re.sub(r"\s+", " ", part).strip()
        ]
        if len(parts) >= 3 and all(_looks_short_textual_header_fragment(part) for part in parts[:3]):
            return None

    return text[:max_chars]
