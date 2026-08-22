"""Orchestration de la passe de localisation tables_layout."""

from __future__ import annotations

import logging
from pathlib import Path

from vigie.extraction.table_locator.models import TableLocationResult
from vigie.extraction.tables_layout.table_locator import detect_table_anchors

logger = logging.getLogger("vigie.extraction.tables_layout")


def run_tables_layout_pass(
    pdf_path: Path,
    page_ranges: list[tuple[int, int]] | None = None,
    *,
    reference_text_max_chars: int = 6000,
) -> TableLocationResult:
    """Orchestrer la detection PyMuPDF Layout vers des ancres Vision.

    Args:
        pdf_path: Chemin du PDF natif.
        page_ranges: Plages 1-indexees optionnelles.
        reference_text_max_chars: Plafond du texte de reference.

    Returns:
        Resultat d'ancrage pret pour le pipeline Vision.

    Raises:
        RuntimeError: Echec de detection (aucun fallback Docling).
    """
    logger.info("Running tables_layout structural pass (use_ocr=False)")
    return detect_table_anchors(
        Path(pdf_path),
        page_ranges=page_ranges,
        reference_text_max_chars=reference_text_max_chars,
    )
