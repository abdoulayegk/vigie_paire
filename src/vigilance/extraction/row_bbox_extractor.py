"""Extraction des bounding boxes par ligne depuis les tableaux PDF.

Utilise pdfplumber pour obtenir les positions des cellules et les
associer aux textes d'indicateurs, en vue du recadrage visuel de preuve.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vigilance.utils.text_normalize_base import normalize_text_base

logger = logging.getLogger(__name__)

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.debug("pdfplumber not available for row bbox extraction")


def _bbox_overlap_ratio(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Calculer le ratio de chevauchement (intersection / aire minimale)."""
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)
    if xi1 <= xi0 or yi1 <= yi0:
        return 0.0
    area_i = (xi1 - xi0) * (yi1 - yi0)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    if min(area1, area2) <= 0:
        return 0.0
    return area_i / min(area1, area2)


def _normalize_for_match(text: str) -> str:
    """Normaliser le texte pour la correspondance (accents, apostrophes, espaces, minuscules)."""
    return normalize_text_base(text or "")


def extract_row_bboxes_from_pdf(
    pdf_path: str | Path,
    page_num: int,
    table_bbox: tuple[float, float, float, float] | None,
    indicator_texts: list[str],
) -> list[tuple[str, float, float]]:
    """Extraire les bboxes par ligne (indicateur, y0_pdf, y1_pdf) pour les cellules de premiere colonne.

    Args:
        pdf_path: Chemin vers le PDF.
        page_num: Numero de page (1-indexed).
        table_bbox: Bbox optionnel du tableau ``(x0, y0, x1, y1)`` pour
            identifier le tableau correspondant.
        indicator_texts: Liste des textes d'indicateurs a associer.

    Returns:
        Liste de tuples ``(texte_indicateur, y0_pdf, y1_pdf)``.
    """
    if not PDFPLUMBER_AVAILABLE:
        return []

    raw_pdf_path = str(pdf_path or "").strip()
    if not raw_pdf_path:
        return []
    pdf_path = Path(raw_pdf_path)
    if not pdf_path.exists():
        return []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            tables = page.find_tables()

            if not tables:
                return []

            # Find best matching table by bbox overlap
            best_table = None
            best_overlap = 0.0
            for t in tables:
                tb = t.bbox
                if table_bbox:
                    overlap = _bbox_overlap_ratio(table_bbox, tb)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_table = t
                else:
                    best_table = t
                    break

            if best_table is None:
                return []

            cells = getattr(best_table, "cells", [])
            extracted = best_table.extract()
            if not extracted:
                return []

            # Build row -> cell mapping (by y0)
            rows_cells: dict[float, list] = {}
            for cell in cells:
                x0, y0, x1, y1 = cell[:4]
                y_key = round(y0, 1)
                if y_key not in rows_cells:
                    rows_cells[y_key] = []
                rows_cells[y_key].append((x0, y0, x1, y1))

            min_x0 = min(c[0] for c in cells) if cells else 0
            result: list[tuple[str, float, float]] = []
            seen_indicators: set[str] = set()

            # Match extracted rows to indicators by order and text similarity
            norm_indicators = [_normalize_for_match(t) for t in indicator_texts]
            row_idx = 0

            for y_key in sorted(rows_cells.keys()):
                row_cells = rows_cells[y_key]
                if row_idx >= len(extracted):
                    break

                row = extracted[row_idx]
                row_idx += 1
                if not row:
                    continue

                first_cell_text = str(row[0]).strip() if row and row[0] else ""
                if not first_cell_text:
                    continue

                # Find leftmost cell for this row
                left_cells = [c for c in row_cells if abs(c[0] - min_x0) < 30]
                if not left_cells:
                    continue
                _, y0, _, y1 = left_cells[0]

                norm_cell = _normalize_for_match(first_cell_text)

                # Match to indicator (by order or text)
                best_ind = None
                for i, ind_text in enumerate(indicator_texts):
                    if ind_text in seen_indicators:
                        continue
                    norm_ind = norm_indicators[i]
                    # Match if similar or by position
                    if norm_cell and norm_ind:
                        if norm_cell in norm_ind or norm_ind in norm_cell:
                            best_ind = ind_text
                            break
                        if len(norm_cell) > 10 and len(norm_ind) > 10:
                            # Jaccard-like: words overlap
                            wc = set(norm_cell.split())
                            wi = set(norm_ind.split())
                            if wc & wi and len(wc & wi) / max(len(wc), len(wi)) > 0.5:
                                best_ind = ind_text
                                break
                    # Fallback: use by order
                    if best_ind is None and i == len(result):
                        best_ind = ind_text
                        break

                if best_ind is None:
                    for ind_text in indicator_texts:
                        if ind_text not in seen_indicators:
                            best_ind = ind_text
                            break

                if best_ind:
                    seen_indicators.add(best_ind)
                    result.append((best_ind, y0, y1))

            return result

    except Exception as e:
        logger.warning(f"Row bbox extraction failed for {pdf_path} p{page_num}: {e}")
        return []
