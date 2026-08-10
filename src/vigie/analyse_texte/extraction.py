"""Extraction et classification des blocs narratifs provenant des PDF.

Le module coordonne les moteurs d'extraction, applique les filtres de mise en
page et construit les audits de section consommés par la suite du pipeline. Il
ne porte pas la logique de comparaison sémantique.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from pathlib import Path
from typing import Any

from vigie.analyse_texte.constants import _MULTISPACE_RE
from vigie.analyse_texte.models import PDFBlock, ResolvedSection, SectionAudit
from vigie.analyse_texte.normalization import (
    _bbox_overlap_ratio,
    _block_overlaps_table,
    _infer_table_footnote_bboxes,
    _is_chart_axis_label_row,
    _is_not_applicable_marker,
    _is_running_report_chrome,
    _is_table_unit_label,
    _looks_like_footnote,
    _looks_like_narrative_paragraph,
    _looks_like_table_caption_title,
    _looks_like_table_footnote_text,
    _looks_like_table_or_financial_grid,
    _normalized_block_text,
)
from vigie.analyse_texte.sections import (
    _next_section_by_key,
    _section_window_for_page,
    _sorted_sections,
)

_DOCLING_TEXT_PAGE_BATCH_SIZE = 2
logger = logging.getLogger(__name__)

_COMPOSITE_GRID_ROW_LABELS = frozenset({"crédit", "marché", "opérationnel", "total"})
_COMPOSITE_GRID_CAPTION_RE = re.compile(
    r"\br[eé]partition\s+des?\s+risques?\s+par\s+secteur",
    flags=re.IGNORECASE,
)
_COMPOSITE_GRID_NUMERIC_CELL_RE = re.compile(r"^\s*\(?\s*-?\d[\d\s.,]*\s*\)?\s*%?\s*$")


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


def _export_docling_markdown(docling_doc: Any) -> str:
    """Retourne le markdown natif Docling sans post-traitement maison."""
    export_to_markdown = getattr(docling_doc, "export_to_markdown", None)
    if callable(export_to_markdown):
        value = export_to_markdown()
        return str(value or "")
    return ""


def _text_docling_ocr_enabled() -> bool:
    """Indique si l'OCR Docling est actif pour l'extraction narrative."""
    raw = os.environ.get("VIGIE_TEXT_OCR_ENABLED")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _extract_pymupdf_fallback_blocks(
    pdf_path: Path,
    page_numbers: list[int],
) -> dict[int, list[PDFBlock]]:
    """Retourne des blocs PyMuPDF pour récupérer le texte absent de Docling.

    Certains PDF bancaires contiennent du texte que Docling classe comme image ou
    ne remonte pas dans ``iterate_items``. PyMuPDF lit la couche texte native du
    PDF et sert ici uniquement de filet de sécurité : les blocs qui chevauchent
    déjà un bloc Docling ne seront pas ajoutés.
    """
    try:
        import pymupdf
    except Exception:
        return {}

    fallback: dict[int, list[PDFBlock]] = {page: [] for page in page_numbers}
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            for page in page_numbers:
                if page < 1 or page > doc.page_count:
                    continue
                page_obj = doc[page - 1]
                width = float(page_obj.rect.width) or 1.0
                height = float(page_obj.rect.height) or 1.0
                line_no = 0
                for block in page_obj.get_text("blocks"):
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    clean_text = _MULTISPACE_RE.sub(" ", str(text or "").replace("\n", " ").strip()).strip()
                    if not clean_text:
                        continue
                    line_no += 1
                    fallback[page].append(
                        PDFBlock(
                            block_id=f"p{page:03d}_m{line_no:03d}",
                            page=page,
                            bbox_norm=[
                                max(0.0, min(1.0, float(x0) / width)),
                                max(0.0, min(1.0, float(y0) / height)),
                                max(0.0, min(1.0, float(x1) / width)),
                                max(0.0, min(1.0, float(y1) / height)),
                            ],
                            text=clean_text,
                            line_number=10_000 + line_no,
                            block_type="other",
                            source_label="pymupdf_fallback",
                        )
                    )
    except Exception:
        return {}
    return fallback


def _merge_missing_fallback_blocks(
    page_blocks: dict[int, list[PDFBlock]],
    fallback_blocks: dict[int, list[PDFBlock]],
) -> None:
    """Ajoute les blocs PyMuPDF qui ne sont pas déjà couverts par Docling."""
    for page, candidates in fallback_blocks.items():
        existing = page_blocks.setdefault(page, [])
        existing_norms = [_normalized_block_text(block.text) for block in existing]
        for candidate in candidates:
            cand_norm = _normalized_block_text(candidate.text)
            if len(cand_norm) < 20:
                continue
            if any(cand_norm == norm for norm in existing_norms if norm):
                continue
            if any(
                _bbox_overlap_ratio(candidate.bbox_norm, block.bbox_norm) >= 0.60
                or _bbox_overlap_ratio(block.bbox_norm, candidate.bbox_norm) >= 0.60
                for block in existing
            ):
                continue
            existing.append(candidate)
            existing_norms.append(cand_norm)


def _distinct_x_clusters(blocks: list[PDFBlock], *, tolerance: float = 0.06) -> int:
    """Compte les colonnes visuelles distinctes d'une série de blocs."""
    centers = sorted((float(block.bbox_norm[0]) + float(block.bbox_norm[2])) / 2.0 for block in blocks)
    clusters: list[float] = []
    for center in centers:
        if not clusters or center - clusters[-1] > tolerance:
            clusters.append(center)
            continue
        clusters[-1] = (clusters[-1] + center) / 2.0
    return len(clusters)


def _augment_table_regions_with_composite_grids(
    page_blocks: dict[int, list[PDFBlock]],
    table_bboxes_by_page: dict[int, list[list[float]]],
) -> None:
    """Détecte les tableaux éclatés en cellules de texte indépendantes.

    Certains rapports annuels représentent un grand tableau sous forme de
    diagramme. Docling n'en reconnaît alors que quelques sous-tableaux et
    étiquette le reste comme texte ou titre. Une répétition des mêmes libellés
    de lignes dans au moins trois colonnes, accompagnée de nombreuses cellules
    numériques et d'une légende de répartition, permet de reconstruire une zone
    tabulaire englobante avec une forte confiance.

    Les blocs contenus dans cette zone sont marqués ``table`` immédiatement :
    cette décision spatiale doit précéder la confiance accordée aux faux
    ``section_header`` produits par Docling.
    """
    for page, blocks in page_blocks.items():
        label_blocks_by_text: dict[str, list[PDFBlock]] = {label: [] for label in _COMPOSITE_GRID_ROW_LABELS}
        for block in blocks:
            normalized = _normalized_block_text(block.text)
            if normalized in label_blocks_by_text:
                label_blocks_by_text[normalized].append(block)

        if any(len(label_blocks_by_text[label]) < 3 for label in _COMPOSITE_GRID_ROW_LABELS):
            continue
        if _distinct_x_clusters(label_blocks_by_text["crédit"]) < 3:
            continue

        row_labels = [block for matches in label_blocks_by_text.values() for block in matches]
        label_top = min(block.y0 for block in row_labels)
        label_bottom = max(block.y1 for block in row_labels)
        numeric_cells = [
            block
            for block in blocks
            if block.y0 >= label_top - 0.02
            and block.y1 <= label_bottom + 0.02
            and _COMPOSITE_GRID_NUMERIC_CELL_RE.fullmatch(str(block.text or ""))
        ]
        if len(numeric_cells) < 6:
            continue

        captions = [
            block
            for block in blocks
            if block.y1 <= label_top
            and label_top - block.y1 <= 0.35
            and _COMPOSITE_GRID_CAPTION_RE.search(str(block.text or ""))
        ]
        if not captions:
            continue
        caption = max(captions, key=lambda block: block.y1)

        overlapping_detected_tables = [
            bbox
            for bbox in table_bboxes_by_page.get(page, [])
            if len(bbox) == 4 and float(bbox[1]) <= label_bottom + 0.05 and float(bbox[3]) >= label_top - 0.05
        ]
        region_bottom = max(
            [
                label_bottom,
                *(block.y1 for block in numeric_cells),
                *(float(bbox[3]) for bbox in overlapping_detected_tables),
            ]
        )
        composite_bbox = [0.0, max(0.0, caption.y0), 1.0, min(1.0, region_bottom + 0.005)]
        table_bboxes_by_page.setdefault(page, []).append(composite_bbox)

        for block in blocks:
            midpoint = (block.y0 + block.y1) / 2.0
            if composite_bbox[1] <= midpoint <= composite_bbox[3]:
                block.block_type = "table"


def _docling_page_batches(page_numbers: list[int]) -> list[tuple[int, int, list[int]]]:
    """Découpe les pages demandées en plages Docling de taille bornée.

    Les rapports annuels peuvent contenir plus d'une centaine de pages dans
    les sections ciblées. Les convertir en une seule fois fait exploser la
    mémoire sur certaines banques. Les plages discontinues sont également
    séparées afin de ne jamais analyser de pages hors périmètre.
    """
    pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not pages:
        return []

    batches: list[tuple[int, int, list[int]]] = []
    batch: list[int] = [pages[0]]
    for page in pages[1:]:
        is_contiguous = page == batch[-1] + 1
        if not is_contiguous or len(batch) >= _DOCLING_TEXT_PAGE_BATCH_SIZE:
            batches.append((batch[0], batch[-1], batch))
            batch = [page]
            continue
        batch.append(page)
    batches.append((batch[0], batch[-1], batch))
    return batches


def _extract_docling_page_blocks(
    pdf_path: Path,
    page_numbers: list[int],
) -> tuple[
    dict[int, list[PDFBlock]],
    dict[int, list[list[float]]],
    dict[int, list[list[float]]],
    str,
]:
    """Extrait tous les blocs de texte d'un PDF via Docling pour les pages demandées.

    Lance Docling par plages bornées de pages contiguës, avec OCR narratif
    opt-in via ``VIGIE_TEXT_OCR_ENABLED=1``, puis filtre les blocs par page.
    Ce découpage évite l'épuisement de mémoire sur les sections annuelles très
    longues, sans inclure de pages hors périmètre.
    Retourne quatre valeurs :

    - ``page_blocks`` : liste de PDFBlock triés par position (y, x)
    - ``table_bboxes_by_page`` : bounding boxes des tableaux détectés
    - ``footnote_bboxes_by_page`` : zones de notes inférées sous les tableaux
    - ``raw_docling_markdown`` : markdown natif Docling sans marqueurs maison

    Args:
        pdf_path: Chemin vers le fichier PDF source.
        page_numbers: Pages à extraire (numérotation 1-based Docling).

    Returns:
        Tuple ``(page_blocks, table_bboxes_by_page, footnote_bboxes_by_page, raw_docling_markdown)``.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if not page_numbers:
        return {}, {}, {}, ""

    opts = PdfPipelineOptions()
    opts.do_ocr = _text_docling_ocr_enabled()
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    requested_pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    table_bboxes_by_page: dict[int, list[list[float]]] = {}
    page_blocks: dict[int, list[PDFBlock]] = {page: [] for page in requested_pages}
    line_numbers: dict[int, int] = {page: 0 for page in requested_pages}
    raw_markdown_parts: list[str] = []

    for start_page, end_page, batch_pages in _docling_page_batches(requested_pages):
        logger.info("Docling extraction texte: pages %d-%d (%s)", start_page, end_page, pdf_path.name)
        result = converter.convert(str(pdf_path), page_range=(start_page, end_page))
        docling_doc = result.document
        raw_markdown = _export_docling_markdown(docling_doc)
        if raw_markdown.strip():
            raw_markdown_parts.append(raw_markdown.strip())

        for table in getattr(docling_doc, "tables", []) or []:
            try:
                if not getattr(table, "prov", None):
                    continue
                page = int(getattr(table.prov[0], "page_no", 0) or 0)
                if page not in batch_pages:
                    continue
                bbox = _docling_bbox_to_norm(docling_doc, table.prov[0])
                if not bbox:
                    continue
                table_bboxes_by_page.setdefault(page, []).append(bbox)
            except Exception:
                continue

        for page in batch_pages:
            for item, _level in docling_doc.iterate_items(page_no=page, with_groups=False):
                text = _MULTISPACE_RE.sub(" ", str(getattr(item, "text", "") or "").replace("\n", " ").strip()).strip()
                if not text:
                    continue
                label = str(getattr(item, "label", "") or "").lower()
                item_provs = [
                    prov
                    for prov in list(getattr(item, "prov", []) or [])
                    if int(getattr(prov, "page_no", 0) or 0) == page
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
                        heading_level=int(_level) if isinstance(_level, int) else None,
                    )
                )
            page_blocks[page].sort(key=lambda block: (round(block.y0, 4), round(block.bbox_norm[0], 4)))

        # Le document Docling d'un lot peut être volumineux. Le libérer avant
        # de convertir la plage suivante limite le pic mémoire du processus.
        del docling_doc
        del result
        gc.collect()

    _merge_missing_fallback_blocks(page_blocks, _extract_pymupdf_fallback_blocks(pdf_path, requested_pages))
    for page in requested_pages:
        page_blocks[page].sort(key=lambda block: (round(block.y0, 4), round(block.bbox_norm[0], 4)))
    _augment_table_regions_with_composite_grids(page_blocks, table_bboxes_by_page)
    footnote_bboxes_by_page = _infer_table_footnote_bboxes(table_bboxes_by_page)
    return page_blocks, table_bboxes_by_page, footnote_bboxes_by_page, "\n\n".join(raw_markdown_parts)


def _classify_block_type(
    block: PDFBlock,
    repeated_text_counts: dict[str, int],
    table_bboxes: list[list[float]] | None = None,
    footnote_bboxes: list[list[float]] | None = None,
) -> str:
    """Classifie un bloc PDF en contenu, table ou note de table.

    Priorités d'application :
    1. Type ``table`` ou chevauchement géométrique avec un tableau → table,
       sauf paragraphe narratif long (grilles titre|Description Docling).
    2. Chevauchement avec une zone sous un tableau + indice de note → note de table.
    3. Le marqueur autonome « s.o. » → non applicable.
    4. Tout le reste est conservé pour le markdown et la comparaison. Les
       heuristiques lexicales peuvent encore reconnaître les grilles sans bbox,
       mais aucun symbole, texte court ou marqueur de note isolé ne suffit à
       exclure un bloc.

    Args:
        block: Bloc PDF à classifier.
        repeated_text_counts: Nombre d'occurrences normalisées de chaque texte (toutes pages).
        table_bboxes: Bounding boxes des tableaux sur la même page.
        footnote_bboxes: Zones de notes inférées sur la même page.

    Returns:
        Chaîne parmi ``"narrative"``, ``"table"``, ``"table_footnote"``,
        ``"footnote"``, ``"header_footer"``, ``"not_applicable"`` ou ``"other"``.
    """
    if block.block_type == "table":
        return "table"

    from vigie.analyse_texte.markdown import _is_docling_heading_block

    text = block.text.strip()
    table_bboxes = table_bboxes or []
    footnote_bboxes = footnote_bboxes or []
    # Ces éléments de présentation doivent être retirés avant de faire
    # confiance à l'étiquette ``section_header`` de Docling, qui peut aussi
    # être attribuée à une cellule de tableau ou à un en-tête courant.
    if _is_running_report_chrome(text):
        return "header_footer"
    if _is_table_unit_label(text):
        return "table"
    if _looks_like_table_caption_title(text):
        return "table"
    if _is_chart_axis_label_row(text):
        return "table"
    if _is_not_applicable_marker(text):
        return "not_applicable"
    if re.fullmatch(r"\d{1,3}", text) and (block.y1 <= 0.15 or block.y0 >= 0.85):
        return "header_footer"
    if _block_overlaps_table(block, footnote_bboxes) and _looks_like_table_footnote_text(text):
        return "table_footnote"

    if _is_docling_heading_block(block):
        if _block_overlaps_table(block, table_bboxes):
            words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", block.text.strip())
            if not (block.text.strip().isupper() and len(words) >= 4):
                return "table"
        return "other"

    norm = _normalized_block_text(block.text)
    if not norm:
        return "other"
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    digits = re.findall(r"\d", text)
    numeric_tokens = re.findall(r"\b\S*\d\S*\b", text)
    rating_tokens = re.findall(r"\b(?:[A-Z]{1,4}[+-]?|[A-Z][a-z]\d|[A-Z]{1,3}\s*\(hyb\)|FPUNV)\b", text)
    short_word_ratio = sum(1 for word in words if len(word) <= 4) / max(1, len(words)) if words else 0.0
    digit_ratio = len(digits) / max(1, len(text))
    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    repeated = repeated_text_counts.get(norm, 0)
    if repeated >= 2 and (block.y1 <= 0.12 or block.y0 >= 0.88):
        return "header_footer"
    if _block_overlaps_table(block, table_bboxes):
        # Grilles narratives 2 colonnes (ex. Risques connus | Description) :
        # Docling pose une bbox TABLE, mais les cellules Description sont du prose.
        if _looks_like_narrative_paragraph(text):
            return "narrative"
        return "table"
    if _block_overlaps_table(block, footnote_bboxes) and _looks_like_table_footnote_text(text):
        return "table_footnote"
    if _looks_like_footnote(text):
        return "footnote"
    if _looks_like_narrative_paragraph(text):
        return "narrative"
    if (
        _looks_like_table_or_financial_grid(text)
        or (digit_ratio >= 0.12 and len(words) <= 16)
        or (len(numeric_tokens) >= 10 and len(words) <= 20)
        or len(rating_tokens) >= 6
        or (len(numeric_tokens) >= 4 and short_word_ratio >= 0.45)
        or (short_word_ratio >= 0.62 and upper_ratio >= 0.18 and len(words) >= 18)
        or "\t" in text
        or "  " in text
        or ("|" in text)
        or (len(words) <= 8 and len(digits) >= 4)
        or (upper_ratio >= 0.7 and len(words) <= 12)
    ):
        return "table"
    if len(words) >= 8 and len(text) >= 45:
        return "narrative"
    return "other"


def _exclusion_reason_for_block(block_type: str, in_window: bool) -> str:
    """Retourne la raison d'exclusion d'un bloc hors périmètre ou tabulaire."""
    if not in_window:
        return "outside_target_section"
    return {
        "table": "table_like_block",
        "table_footnote": "table_footnote",
        "not_applicable": "not_applicable",
        "header_footer": "running_header_footer",
    }.get(block_type, "")


def _table_regions_for_pages(
    section_key: str,
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
    pages: list[int],
) -> list[dict[str, Any]]:
    """Construit la liste des régions de tableaux et de notes pour un ensemble de pages.

    Chaque région est un dictionnaire avec ``table_id``, ``page``, ``region_type``
    (``"table"`` ou ``"footnote"``) et ``bbox``. Stockée dans ``SectionAudit.table_regions``
    pour la traçabilité et l'audit de l'extraction.
    """
    regions: list[dict[str, Any]] = []
    for page in pages:
        for idx, bbox in enumerate(table_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_tbl_{idx:02d}",
                    "page": page,
                    "region_type": "table",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
        for idx, bbox in enumerate(footnote_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_ftn_{idx:02d}",
                    "page": page,
                    "region_type": "footnote",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
    return regions


def _repeated_text_counts(page_blocks: dict[int, list[PDFBlock]]) -> dict[str, int]:
    """Compte le nombre de pages distinctes où chaque texte normalisé apparaît.

    Un texte qui apparaît sur ≥ 2 pages est un candidat en-tête/pied de page.
    Utilisé par ``_classify_block_type`` pour détecter et exclure ces répétitions.
    """
    counts: dict[str, int] = {}
    for blocks in page_blocks.values():
        seen_on_page: set[str] = set()
        for block in blocks:
            norm = _normalized_block_text(block.text)
            if not norm or norm in seen_on_page:
                continue
            seen_on_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1
    return counts


def _build_section_audit(
    *,
    section: ResolvedSection,
    next_section: ResolvedSection | None,
    page_blocks: dict[int, list[PDFBlock]],
    repeated_text_counts: dict[str, int],
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
) -> SectionAudit:
    """Construit l'audit complet d'une section : blocs inclus, exclus et régions de tableaux.

    Pour chaque bloc de la section, applique la fenêtre verticale de la section,
    classifie le type de bloc et décide de son inclusion. Tout contenu dans la
    fenêtre est conservé, sauf les tables confirmées, leurs notes associées et
    le marqueur autonome « s.o. ».
    Les autres blocs restent comparables même s'ils sont courts, chiffrés ou
    ressemblent lexicalement à une note.

    Args:
        section: Section résolue avec ses pages et son ancre.
        next_section: Section suivante dans le PDF (pour délimiter la fenêtre basse).
        page_blocks: Blocs extraits par Docling, indexés par page.
        repeated_text_counts: Comptages de textes répétés pour détection en-têtes/pieds.
        table_bboxes_by_page: Bounding boxes des tableaux, par page.
        footnote_bboxes_by_page: Zones de notes inférées, par page.

    Returns:
        ``SectionAudit`` avec ``included_blocks``, ``excluded_blocks`` et ``table_regions``.
    """
    included_blocks: list[PDFBlock] = []
    excluded_blocks: list[PDFBlock] = []
    for page in section.pages:
        blocks = page_blocks.get(page, [])
        page_tables = table_bboxes_by_page.get(page, [])
        page_footnotes = footnote_bboxes_by_page.get(page, [])
        top_cutoff, bottom_cutoff = _section_window_for_page(section, page, next_section)
        for block in blocks:
            section_block = PDFBlock(
                block_id=block.block_id,
                page=block.page,
                bbox_norm=list(block.bbox_norm),
                text=block.text,
                line_number=block.line_number,
                block_type=block.block_type,
                source_label=block.source_label,
                heading_level=block.heading_level,
            )
            midpoint = (block.y0 + block.y1) / 2.0
            in_window = top_cutoff <= midpoint < bottom_cutoff
            block_type = _classify_block_type(section_block, repeated_text_counts, page_tables, page_footnotes)
            section_block.block_type = block_type
            section_block.included = in_window and block_type not in {
                "table",
                "table_footnote",
                "not_applicable",
                "header_footer",
            }
            section_block.exclusion_reason = (
                "" if section_block.included else _exclusion_reason_for_block(block_type, in_window)
            )
            if section_block.included:
                included_blocks.append(section_block)
            else:
                excluded_blocks.append(section_block)
    return SectionAudit(
        section_key=section.section_key,
        section_title=section.title,
        start_page=section.start_page,
        end_page=section.end_page,
        anchor_page=section.anchor_page,
        anchor_text=section.anchor_text,
        anchor_bbox_norm=section.anchor_bbox_norm,
        end_anchor_page=section.end_anchor_page,
        end_anchor_text=section.end_anchor_text,
        end_anchor_bbox_norm=section.end_anchor_bbox_norm,
        included_blocks=included_blocks,
        excluded_blocks=excluded_blocks,
        table_regions=_table_regions_for_pages(
            section.section_key,
            table_bboxes_by_page,
            footnote_bboxes_by_page,
            section.pages,
        ),
    )


def _extract_audits_for_pdf(
    *,
    pdf_path: Path,
    sections: dict[str, ResolvedSection],
    raw_docling_markdown_path: Path | None = None,
) -> tuple[list[SectionAudit], str]:
    """Extrait les blocs narratifs de chaque section via Docling + heuristiques.

    Cette etape construit les audits qui seront convertis en markdown
    ``source of truth``. Les appels GPT de comparaison/triage relisent ensuite
    exclusivement ce markdown, pas les PDFs directement.
    """
    if not sections:
        return [], ""
    section_order = _next_section_by_key(sections)
    unique_pages = sorted({page for section in sections.values() for page in section.pages})
    page_blocks, table_bboxes_by_page, footnote_bboxes_by_page, raw_docling_markdown = _extract_docling_page_blocks(
        pdf_path,
        unique_pages,
    )
    if raw_docling_markdown_path is not None:
        raw_docling_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        raw_docling_markdown_path.write_text(raw_docling_markdown, encoding="utf-8")
    repeated_counts = _repeated_text_counts(page_blocks)
    audits: list[SectionAudit] = []
    for section in _sorted_sections(sections):
        audit = _build_section_audit(
            section=section,
            next_section=section_order.get(section.section_key),
            page_blocks=page_blocks,
            repeated_text_counts=repeated_counts,
            table_bboxes_by_page=table_bboxes_by_page,
            footnote_bboxes_by_page=footnote_bboxes_by_page,
        )
        audits.append(audit)
    return audits, raw_docling_markdown
