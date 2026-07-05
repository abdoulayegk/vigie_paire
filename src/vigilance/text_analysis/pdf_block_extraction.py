"""Extraction Docling des blocs PDF et des regions de tableaux."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

from .constants import _MULTISPACE_RE
from .models import PDFBlock
from .pdf_block_classification import _infer_table_footnote_bboxes

def _docling_bbox_to_norm(docling_doc: Any, prov: Any) -> list[float] | None:
    """Convertit une bounding box Docling en coordonnées normalisées [0, 1].

    Applique ``to_top_left_origin`` (système Docling en bas-gauche) puis
    ``normalized`` pour obtenir [x0, y0, x1, y1] dans l'espace page normalisé.
    Retourne None si la conversion échoue (page manquante, bbox invalide).
    """
    try:
        page_obj = docling_doc.pages[prov.page_no]
        page_height = page_obj.size.height
        norm = prov.bbox.to_top_left_origin(page_height=page_height).normalized(page_obj.size)
        return [
            max(0.0, min(1.0, float(norm.l))),
            max(0.0, min(1.0, float(norm.t))),
            max(0.0, min(1.0, float(norm.r))),
            max(0.0, min(1.0, float(norm.b))),
        ]
    except Exception:
        return None


def _extract_docling_page_blocks(
    pdf_path: Path,
    page_numbers: list[int],
) -> tuple[dict[int, list[PDFBlock]], dict[int, list[list[float]]], dict[int, list[list[float]]]]:
    """Extrait tous les blocs de texte d'un PDF via Docling pour les pages demandées.

    Lance Docling sans OCR sur la plage ``[min(pages), max(pages)]``, puis filtre
    les blocs par page. Retourne trois structures indexées par numéro de page :

    - ``page_blocks`` : liste de PDFBlock triés par position (y, x)
    - ``table_bboxes_by_page`` : bounding boxes des tableaux détectés
    - ``footnote_bboxes_by_page`` : zones de notes inférées sous les tableaux

    Args:
        pdf_path: Chemin vers le fichier PDF source.
        page_numbers: Pages à extraire (numérotation 1-based Docling).

    Returns:
        Tuple ``(page_blocks, table_bboxes_by_page, footnote_bboxes_by_page)``.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if not page_numbers:
        return {}, {}, {}

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    start_page = min(page_numbers)
    end_page = max(page_numbers)
    result = converter.convert(str(pdf_path), page_range=(start_page, end_page))
    docling_doc = result.document

    table_bboxes_by_page: dict[int, list[list[float]]] = {}
    for table in getattr(docling_doc, "tables", []) or []:
        try:
            if not getattr(table, "prov", None):
                continue
            page = int(getattr(table.prov[0], "page_no", 0) or 0)
            if page not in page_numbers:
                continue
            bbox = _docling_bbox_to_norm(docling_doc, table.prov[0])
            if not bbox:
                continue
            table_bboxes_by_page.setdefault(page, []).append(bbox)
        except Exception:
            continue
    footnote_bboxes_by_page = _infer_table_footnote_bboxes(table_bboxes_by_page)

    page_blocks: dict[int, list[PDFBlock]] = {page: [] for page in page_numbers}
    line_numbers: dict[int, int] = {page: 0 for page in page_numbers}
    for page in page_numbers:
        for item, _level in docling_doc.iterate_items(page_no=page, with_groups=False):
            text = _MULTISPACE_RE.sub(" ", str(getattr(item, "text", "") or "").replace("\n", " ").strip()).strip()
            if not text:
                continue
            label = str(getattr(item, "label", "") or "").lower()
            item_provs = [
                prov for prov in list(getattr(item, "prov", []) or []) if int(getattr(prov, "page_no", 0) or 0) == page
            ]
            if not item_provs:
                continue
            bbox_norm = _docling_bbox_to_norm(docling_doc, item_provs[0])
            if not bbox_norm:
                continue
            line_numbers[page] += 1
            initial_block_type = {
                "footnote": "footnote",
                "page_header": "header_footer",
                "page_footer": "header_footer",
                "caption": "table",
            }.get(label, "other")
            page_blocks[page].append(
                PDFBlock(
                    block_id=f"p{page:03d}_d{line_numbers[page]:03d}",
                    page=page,
                    bbox_norm=bbox_norm,
                    text=text,
                    line_number=line_numbers[page],
                    block_type=initial_block_type,
                    source_label=label,
                )
            )
        page_blocks[page].sort(key=lambda block: (round(block.y0, 4), round(block.bbox_norm[0], 4)))
    return page_blocks, table_bboxes_by_page, footnote_bboxes_by_page
