"""Moteur d'extraction docling du pipeline.

L'implementation est repartie dans les mixins du sous-package ``docling`` ;
ce module assemble ``DoclingProcessor`` et les entrees utilitaires.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf

from vigie.support.utils.pymupdf_utils import configure_mupdf_runtime
from vigie.support.utils.table_page_structure import derive_page_local_structure
from .docling_pass import DoclingPassMixin
from .document import DocumentExtractionMixin
from .models import (
    ExtractedDocument,
    ExtractedSection,
    ExtractedTable,
)
from .page_ranges import PageRangeMixin
from .runtime import (
    CACHE_AVAILABLE,
    PDFCacheManager,
)
from .sections import SectionAssociationMixin
from .titles import TableTitleMixin
from .vision_pass import VisionPassMixin

logger = logging.getLogger("vigie.extraction.docling_processor")
configure_mupdf_runtime(pymupdf)


class DoclingProcessor(
    DocumentExtractionMixin,
    DoclingPassMixin,
    VisionPassMixin,
    TableTitleMixin,
    SectionAssociationMixin,
    PageRangeMixin,
):
    """Processeur d'extraction docling.

    L'implementation est repartie dans les mixins du sous-package ``docling`` ;
    seule l'initialisation reste ici, avec les points d'entree module-level.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        enhance_images: bool = True,
        openai_api_key: str | None = None,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ):
        """Initialiser le processeur Docling avec les options d'extraction.

        Args:
            use_ocr: Active l'OCR Docling pour les PDF numerises.
            enhance_images: Applique une amelioration d'image avant traitement.
            openai_api_key: Cle API OpenAI pour l'extraction Vision GPT-4o.
            use_cache: Active le cache des extractions.
            cache_dir: Repertoire du cache (defaut : repertoire systeme).
        """
        self.use_ocr = use_ocr
        self.enhance_images = enhance_images
        self.openai_api_key = openai_api_key
        self._converter = None
        self._initialized = False

        # Charger les patterns configurables
        self.bank_code_for_patterns: str | None = None  # Sera set lors de extract_document
        self.extraction_patterns = None
        try:
            from vigie.support.utils.pattern_loader import get_patterns

            self.extraction_patterns = get_patterns()  # Chargement par defaut
            logger.debug("Patterns d'extraction charges")
        except Exception as e:
            logger.warning(f"Impossible de charger les patterns configurables: {e}")

        # Initialiser le cache
        self.use_cache = use_cache and CACHE_AVAILABLE
        self._cache: PDFCacheManager | None = None
        if self.use_cache:
            self._cache = PDFCacheManager(cache_dir) if cache_dir else PDFCacheManager()
            logger.info("Cache PDF active")


def extract_pdf(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    use_ocr: bool = False,
    enhance_images: bool = True,
    page_ranges: list[tuple[int, int]] | None = None,
    use_vision_extraction: bool | None = None,
) -> ExtractedDocument:
    """Extraire un PDF complet en combinant structure Docling et contenu Vision par tableau.

    Fonction utilitaire de niveau module qui instancie un ``DoclingProcessor``
    et appelle ``extract_document``. C'est le point d'entrée le plus simple
    pour une extraction complète.

    Paramètres
    ----------
    pdf_path:
        Chemin vers le PDF à extraire.
    bank_code:
        Code banque (ex. ``"rbc"``).
    quarter:
        Libellé du trimestre (ex. ``"Q1-2025"``).
    year:
        Année numérique.
    use_ocr:
        Active l'OCR Docling pour les PDF numérisés.
    enhance_images:
        Applique une amélioration d'image avant le traitement.
    page_ranges:
        Liste de tuples ``(start, end)`` pour limiter l'extraction à des pages
        spécifiques. ``None`` = tout le document.
    use_vision_extraction:
        Force l'activation/désactivation de Vision GPT-4o. ``None`` = résolution
        automatique depuis la config et l'environnement.
    """
    processor = DoclingProcessor(
        use_ocr=use_ocr,
        enhance_images=enhance_images,
    )
    return processor.extract_document(
        pdf_path,
        bank_code,
        quarter,
        year,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )


def extract_pdf_targeted(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]],
    use_ocr: bool = False,
    use_vision_extraction: bool | None = None,
) -> ExtractedDocument:
    """Extraire uniquement des plages de pages ciblées d'un PDF.

    Wrapper autour de ``extract_pdf`` avec ``page_ranges`` obligatoire.
    Utile pour extraire une section spécifique sans traiter tout le document,
    ce qui réduit le temps d'extraction et la consommation mémoire.
    """
    return extract_pdf(
        pdf_path,
        bank_code,
        quarter,
        year,
        use_ocr=use_ocr,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )


def extract_section_content(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    section_name: str,
    start_page: int,
    end_page: int,
) -> ExtractedSection:
    """Extraire le contenu complet d'une section d'un PDF (texte + tableaux).

    Extrait les pages ``start_page`` à ``end_page`` du PDF, puis combine tout
    le contenu textuel et tous les tableaux détectés en un seul ``ExtractedSection``.

    Paramètres
    ----------
    section_name:
        Nom de la section (utilisé comme identifiant et titre dans le résultat).
    start_page:
        Première page de la section (1-indexé).
    end_page:
        Dernière page de la section (incluse).
    """
    doc = extract_pdf_targeted(pdf_path, bank_code, quarter, year, page_ranges=[(start_page, end_page)])

    # Combiner tout le contenu
    text = ""
    tables = []
    if doc.sections:
        text = "\n\n".join(s.text_content for s in doc.sections)
        for s in doc.sections:
            tables.extend(s.tables)
    elif doc.all_tables:
        tables = doc.all_tables

    return ExtractedSection(
        section_id=f"{section_name.lower().replace(' ', '_')}",
        title=section_name,
        start_page=doc.sections[0].start_page if doc.sections else start_page,
        end_page=doc.sections[-1].end_page if doc.sections else end_page,
        text_content=text,
        tables=tables,
    )


def extract_tables_docling_by_sections(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    section_ranges: list[dict[str, Any]] | None = None,
    use_vision_extraction: bool | None = None,
) -> list[ExtractedTable]:
    """Extraire les tableaux sur des plages de sections et les annoter avec leur section.

    C'est le **point d'entrée principal** pour le pipeline CLI (``run_tables.py``)
    et pour ``comparison_runner.py``. Il :

    1. Normalise les plages de sections (format ``{section, start, end}``).
    2. Appelle ``extract_tables_docling_priority`` pour extraire tous les tableaux
       sur les pages couvertes par les plages.
    3. Assigne la section canonique à chaque tableau selon sa page.
    4. Écrit un fichier de débogage JSON (``extraction_debug_writer``) si disponible.
    5. Journalise les statistiques d'extraction.

    Paramètres
    ----------
    section_ranges:
        Liste de dictionnaires ``{section: str, start: int, end: int}`` définissant
        les plages de pages à extraire. Si ``None`` ou vide, extrait tout le document.
    use_vision_extraction:
        Force l'activation/désactivation de Vision GPT-4o. ``None`` = résolution
        automatique.

    Retourne
    --------
    Liste de ``ExtractedTable`` avec le champ ``section`` renseigné pour chaque
    tableau selon sa page d'appartenance.
    """
    normalized_ranges: list[tuple[str, int, int]] = []
    page_ranges: list[tuple[int, int]] = []

    for item in section_ranges or []:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start", 0) or 0)
        end = int(item.get("end", start) or start)
        if start <= 0 or end < start:
            continue
        section_name = str(item.get("section", "")).strip() or "unknown_section"
        normalized_ranges.append((section_name, start, end))
        page_ranges.append((start, end))

    tables = extract_tables_docling_priority(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        page_ranges=page_ranges or None,
        use_vision_extraction=use_vision_extraction,
    )
    _assign_canonical_table_ids(tables)

    if not normalized_ranges:
        pass
    else:
        for table in tables:
            page = int(getattr(table, "page_number", 0) or 0)
            for section_name, start, end in normalized_ranges:
                if start <= page <= end:
                    table.section = section_name
                    break

    try:
        from .extraction_debug_writer import write_extraction_debug

        write_extraction_debug(bank=bank_code, quarter=quarter, year=year, tables=tables)
    except Exception:
        pass

    logger.info(
        "extraction_docling_tables pdf=%s bank=%s quarter=%s year=%s tables_count=%d page_ranges=%s",
        str(pdf_path),
        bank_code,
        quarter,
        year,
        len(tables),
        page_ranges,
    )
    return tables


def _page_role_from_index(index_on_page: int, tables_on_page: int) -> str:
    """Determiner le role positionnel d'un tableau sur sa page (single/first/middle/last)."""
    if tables_on_page <= 1:
        return "single"
    if index_on_page <= 1:
        return "first"
    if index_on_page >= tables_on_page:
        return "last"
    return "middle"


def _assign_canonical_table_ids(tables: list[ExtractedTable]) -> None:
    """Assigner des identifiants deterministes bases sur la page et l'ordre local des tableaux."""
    if not tables:
        return

    page_counts: dict[int, int] = {}
    for table in tables:
        page = int(getattr(table, "page_number", 0) or 0)
        page_counts[page] = page_counts.get(page, 0) + 1

    structure = derive_page_local_structure(tables)
    fallback_index_by_page: dict[int, int] = {}
    used_ids: set[str] = set()

    for table in tables:
        page = int(getattr(table, "page_number", 0) or 0)
        prior_id = str(getattr(table, "table_id", "") or "")
        info = structure.get((prior_id, page))

        if info is not None:
            index_on_page = int(info.get("table_index_on_page", 0) or 0)
            tables_on_page = int(info.get("tables_on_page", page_counts.get(page, 1)) or 1)
            bbox_top = info.get("bbox_top")
            page_local_role = str(
                info.get(
                    "page_local_role",
                    _page_role_from_index(index_on_page, tables_on_page),
                )
            )
        else:
            fallback_index_by_page[page] = fallback_index_by_page.get(page, 0) + 1
            index_on_page = fallback_index_by_page[page]
            tables_on_page = int(page_counts.get(page, 1) or 1)
            bbox_top = None
            page_local_role = _page_role_from_index(index_on_page, tables_on_page)

        table_id = f"tbl_p{page:03d}_i{index_on_page:02d}"
        if table_id in used_ids:
            raise ValueError(f"Duplicate canonical table_id generated: {table_id!r} on page {page}")

        table.table_index_on_page = index_on_page
        table.tables_on_page = tables_on_page
        table.bbox_top = float(bbox_top) if bbox_top is not None else None
        table.page_local_role = page_local_role
        table.table_id = table_id
        used_ids.add(table_id)


def extract_tables_docling_priority(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]] | None = None,
    use_vision_extraction: bool | None = None,
) -> list[ExtractedTable]:
    """Extraire uniquement les tableaux d'un PDF (structure Docling + contenu Vision).

    Wrapper simplifié autour de ``extract_pdf`` qui retourne directement la liste
    plate de tous les tableaux (``doc.all_tables``) sans les sections ni le
    contenu textuel. Utilisé en interne par ``extract_tables_docling_by_sections``.
    """
    doc = extract_pdf(
        pdf_path,
        bank_code,
        quarter,
        year,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )
    return doc.all_tables
