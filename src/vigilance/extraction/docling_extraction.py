"""Fonctions d'extraction Docling/Vision (hors classe DoclingProcessor)."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from ..utils.feature_flags import extraction_cache_mode_tag
from ..utils.genai import get_openai_api_key
from .docling_conversion import _docling_table_to_dict
from .docling_normalization import (
    _extract_table_context,
    _find_table_title_in_text,
    _reconstruct_table_rows,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .docling_processor import ExtractedDocument

# Import de la gestion memoire (optionnel)
try:
    from ..utils.memory import (
        ChunkedProcessor,
        check_memory_threshold,
        cleanup_memory,
        get_memory_usage_mb,
        with_memory_check,
    )

    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    logger.debug("Utilitaires memoire non disponibles")

# Import du gestionnaire de cache (optionnel)
try:
    from ..utils.pdf_cache import PDFCacheManager

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.debug("PDFCacheManager non disponible")


def extract_pdf(pdf_path: str, bank_code: str, quarter: str, year: int = 2025) -> ExtractedDocument:
    """
    Fonction utilitaire pour extraire un document PDF.
    """
    from .docling_processor import DoclingProcessor

    processor = DoclingProcessor()
    return processor.extract_document(pdf_path, bank_code, quarter, year)


def extract_pdf_with_fallback(
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int = 2025,
    openai_api_key: str | None = None,
) -> ExtractedDocument:
    """
    Fonction utilitaire pour extraire un PDF avec tous les fallbacks activés.
    """
    from .docling_processor import DoclingProcessor

    processor = DoclingProcessor(use_vision_fallback=True, openai_api_key=openai_api_key)
    return processor.extract_document(pdf_path, bank_code, quarter, year)


def extract_pdf_targeted(
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int = 2025,
    page_ranges: list[tuple[int, int]] | None = None,
) -> ExtractedDocument:
    """
    Fonction utilitaire pour extraire des pages specifiques d'un PDF.
    """
    from .docling_processor import DoclingProcessor

    processor = DoclingProcessor()
    return processor.extract_document(pdf_path, bank_code, quarter, year, page_ranges)


def extract_section_content(pdf_path: str, start_page: int, end_page: int) -> dict:
    """
    Extraire le contenu d'une plage de pages specifique.

    Retourne le texte brut et les tableaux extraits pour une section.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber non installe")
        return {"text_content": "", "tables": [], "page_count": 0}

    text_parts = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # Ajuster les bornes
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        for page_num in range(start_idx, end_idx):
            page = pdf.pages[page_num]
            actual_page = page_num + 1

            # Extraire le texte
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(f"--- Page {actual_page} ---\\n{text}")

            # Extraire les tableaux
            page_tables = page.extract_tables() or []
            for table_idx, table_data in enumerate(page_tables):
                if not table_data or len(table_data) < 2:
                    continue

                # Trouver le titre du tableau dans le texte
                title = _find_table_title_in_text(text, table_idx)

                # Reconstruire les lignes en fusionnant les cellules fragmentees
                reconstructed_rows = _reconstruct_table_rows(table_data)

                if reconstructed_rows:
                    tables.append(
                        {
                            "table_id": f"table_p{actual_page}_{table_idx}",
                            "page_number": actual_page,
                            "title": title,
                            "headers": reconstructed_rows[0] if reconstructed_rows else [],
                            "rows": reconstructed_rows[1:] if len(reconstructed_rows) > 1 else [],
                            "raw_text_context": _extract_table_context(text, title),
                        }
                    )

    return {
        "text_content": "\\n\\n".join(text_parts),
        "tables": tables,
        "page_count": end_idx - start_idx,
    }


def extract_section_content_docling(
    pdf_path: str, start_page: int, end_page: int, bank_code: str = "unknown"
) -> dict:
    """
    Extraire le contenu d'une section en utilisant Docling (meilleure qualite).
    """
    from .docling_processor import DoclingProcessor

    processor = DoclingProcessor()

    try:
        # Extraire avec Docling
        page_ranges = [(start_page, end_page)]
        result = processor.extract_document(pdf_path, bank_code, "unknown", 2025, page_ranges)

        # Convertir les tableaux en format dict
        tables = []
        for t in result.all_tables:
            # Construire les lignes avec la premiere colonne comme indicateur
            formatted_rows = []
            for row in t.rows:
                if row and len(row) > 0:
                    # Nettoyer les cellules
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    # Ne garder que les lignes avec du contenu
                    if any(cleaned_row):
                        formatted_rows.append(cleaned_row)

            tables.append(
                {
                    "table_id": t.table_id,
                    "page_number": t.page_number,
                    "title": t.title or t.title_clean,
                    "table_number": t.table_number,
                    "headers": t.headers,
                    "rows": formatted_rows,
                    "first_column_indicators": t.first_column_indicators,
                    "footnotes": t.footnotes,
                    "section": t.section,
                }
            )

        return {
            "text_content": result.metadata.get("text_content", ""),
            "tables": tables,
            "page_count": end_page - start_page + 1,
            "extraction_method": "docling",
        }

    except Exception as e:
        logger.warning(f"Docling extraction failed in docling-only mode: {e}")
        return {
            "text_content": "",
            "tables": [],
            "page_count": end_page - start_page + 1,
            "extraction_method": "docling_only_warning",
            "warning": "Docling extraction failed in docling-only mode",
        }


def extract_tables_with_context(pdf_path: str, start_page: int, end_page: int) -> list[dict]:
    """
    Extraire les tableaux avec leur contexte textuel (lignes completes).

    Combine le texte de la page avec les tableaux pour avoir
    des lignes plus completes et significatives.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    tables_with_context = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(max(0, start_page - 1), min(len(pdf.pages), end_page)):
            page = pdf.pages[page_num]
            actual_page = page_num + 1

            # Extraire le texte complet de la page
            page_text = page.extract_text() or ""

            # Extraire les tableaux
            page_tables = page.extract_tables() or []

            for table_idx, table_data in enumerate(page_tables):
                if not table_data or len(table_data) < 2:
                    continue

                # Trouver le titre du tableau dans le texte
                title = _find_table_title_in_text(page_text, table_idx)

                # Reconstruire les lignes en fusionnant les cellules fragmentees
                reconstructed = _reconstruct_table_rows(table_data)

                if reconstructed:
                    # Extraire les indicateurs (première colonne)
                    indicators = [
                        row[0].strip()
                        for row in reconstructed
                        if row and len(row) > 0 and row[0].strip()
                    ]

                    tables_with_context.append(
                        {
                            "table_id": f"table_p{actual_page}_{table_idx}",
                            "page_number": actual_page,
                            "title": title,
                            "headers": reconstructed[0] if reconstructed else [],
                            "rows": reconstructed[1:] if len(reconstructed) > 1 else [],
                            "raw_text_context": _extract_table_context(page_text, title),
                            "first_column_indicators": indicators,
                        }
                    )

    return tables_with_context


def extract_tables_docling_priority(
    pdf_path: str,
    start_page: int,
    end_page: int,
    use_ocr: bool = False,
    fast_mode: bool = False,
    openai_api_key: str | None = None,
    use_cache: bool = True,
    section_name: str | None = None,
) -> list[dict]:
    # Appliquer le decorateur de gestion memoire si disponible
    if MEMORY_UTILS_AVAILABLE:
        return _extract_tables_docling_priority_impl(
            pdf_path,
            start_page,
            end_page,
            use_ocr,
            fast_mode,
            openai_api_key,
            use_cache,
            section_name,
        )
    else:
        return _extract_tables_docling_priority_impl(
            pdf_path,
            start_page,
            end_page,
            use_ocr,
            fast_mode,
            openai_api_key,
            use_cache,
            section_name,
        )


def _extract_tables_docling_priority_impl(
    pdf_path: str,
    start_page: int,
    end_page: int,
    use_ocr: bool = False,
    fast_mode: bool = False,
    openai_api_key: str | None = None,
    use_cache: bool = True,
    section_name: str | None = None,
) -> list[dict]:
    """
    Extraire les tableaux en utilisant DOCLING en priorité.
    La validation GPT-4o Vision a été retirée (coût/latence).
    """
    cache_section = f"{start_page}-{end_page}_{extraction_cache_mode_tag()}"
    if section_name:
        cache_section = f"{section_name}_{cache_section}"

    cache_manager = None

    # Verifier le cache
    if use_cache and CACHE_AVAILABLE:
        cache_manager = PDFCacheManager()
        cached_data = cache_manager.load_from_cache(pdf_path, cache_section)
        if cached_data and "data" in cached_data:
            tables = cached_data["data"]
            if isinstance(tables, list):
                logger.info(f"Cache hit: {len(tables)} tableaux charges depuis cache")
                return tables

    allow_vision_fallback = False  # Fallback Vision désactivé pour le moment

    if fast_mode:
        logger.info(
            "fast_mode demande, mais fallback extraction pdfplumber retire; Docling conserve."
        )
    logger.info(f"Extraction mode qualite (Docling): pages {start_page}-{end_page}")

    tables_result = []

    try:
        tables_result = _extract_with_docling_targeted(
            pdf_path, start_page, end_page, use_ocr=use_ocr
        )

        if tables_result:
            logger.info(f"Docling: {len(tables_result)} tableaux extraits")
        elif allow_vision_fallback:
            logger.info("Docling n'a extrait aucun tableau, fallback Vision active.")
            tables_result = _extract_with_vision_targeted(
                pdf_path=pdf_path,
                start_page=start_page,
                end_page=end_page,
                section_name=section_name,
                api_key=openai_api_key,
            )

    except Exception as e:
        logger.warning(f"Docling a échoué: {e}")
        if allow_vision_fallback:
            logger.info("Fallback vers GPT-4o Vision...")
            tables_result = _extract_with_vision_targeted(
                pdf_path=pdf_path,
                start_page=start_page,
                end_page=end_page,
                section_name=section_name,
                api_key=openai_api_key,
            )
        else:
            logger.warning("Aucun fallback d'extraction actif. Retour liste vide.")
            tables_result = []

    # Sauvegarder en cache
    if use_cache and cache_manager and tables_result:
        cache_manager.save_to_cache(
            pdf_path,
            cache_section,
            tables_result,
            metadata={
                "start_page": start_page,
                "end_page": end_page,
                "fast_mode": fast_mode,
                "tables_count": len(tables_result),
            },
        )

    return tables_result


def _extract_with_vision_targeted(
    pdf_path: str,
    start_page: int,
    end_page: int,
    section_name: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Extraction Vision ciblée pour remplacer le fallback pdfplumber."""
    try:
        from .vision_table_extractor import VisionTableExtractor
    except ImportError as e:
        logger.warning(f"VisionTableExtractor non disponible: {e}")
        return []

    effective_key = api_key or get_openai_api_key()
    if not effective_key:
        logger.warning("OPENAI_API_KEY absente: fallback Vision indisponible")
        return []

    try:
        extractor = VisionTableExtractor(
            api_key=effective_key, model="gpt-4o", save_proof_images=False
        )
        result = extractor.extract_from_section(
            pdf_path=pdf_path,
            start_page=start_page,
            end_page=end_page,
            section_name=section_name or "unknown",
            bank_code="unknown",
        )
    except Exception as e:
        logger.warning(f"Extraction Vision echouee: {e}")
        return []

    tables: list[dict] = []
    for extracted in result.extracted_tables:
        table_data = {
            "table_id": extracted.table_id,
            "page_number": extracted.page_number,
            "title": extracted.title,
            "headers": extracted.headers,
            "rows": extracted.rows,
            "first_column_indicators": extracted.first_column_indicators,
            "footnotes": extracted.footnotes,
            "extraction_method": "gpt4o_vision",
            "confidence": extracted.confidence,
        }
        tables.append(table_data)

    logger.info(f"Vision fallback: {len(tables)} tableaux extraits")
    return tables


def _extract_with_docling_targeted(
    pdf_path: str,
    start_page: int,
    end_page: int,
    use_ocr: bool = False,
) -> list[dict]:
    """
    Extraction ciblée avec Docling sur des pages spécifiques.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        logger.error(f"Docling non installé: {e}")
        raise

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.do_picture_description = False

    _raw_threads = os.environ.get("DOCLING_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    num_threads = int(_raw_threads) if _raw_threads else 4
    default_device = "mps" if sys.platform == "darwin" else "auto"
    device_str = os.environ.get("DOCLING_DEVICE", default_device)
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
        }
        device = device_map.get(device_str.lower(), AcceleratorDevice.AUTO)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=device
        )
    except ImportError:
        pass

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    docling_page_range = None
    page_range_str = f"{start_page}-{end_page}"
    for module_path, name in [
        ("docling.document_converter", "PageRange"),
        ("pypdf", "PageRange"),
    ]:
        try:
            mod = __import__(module_path, fromlist=[name])
            PageRangeCls = getattr(mod, name)
            docling_page_range = PageRangeCls(page_range_str)
            break
        except (ImportError, AttributeError, TypeError, ValueError):
            continue

    convert_kwargs: dict = {}
    if docling_page_range is not None:
        convert_kwargs["page_range"] = docling_page_range

    logger.info("Conversion Docling en cours...")
    result = converter.convert(pdf_path, **convert_kwargs)
    doc = result.document

    # Extraire les tableaux des pages cibles
    tables = []
    tables_filtered_empty_rows = 0
    tables_by_page: dict[int, int] = {}

    for table in doc.tables:
        # Obtenir le numéro de page
        page_num = table.prov[0].page_no if table.prov else 0

        # Filtrer par plage de pages
        if not (start_page <= page_num <= end_page):
            continue

        # Extraire les données du tableau
        table_data = _docling_table_to_dict(table, page_num)

        if table_data and table_data.get("rows"):
            tables.append(table_data)
            tables_by_page[page_num] = tables_by_page.get(page_num, 0) + 1
        else:
            tables_filtered_empty_rows += 1
            title = (table_data or {}).get("title") or "(sans titre)"
            logger.warning(f"Docling page {page_num}: tableau ignore (rows vides) - titre: {title}")

    # Log par page pour analyse de completude
    if tables_by_page:
        counts_str = ", ".join(f"p{k}:{v}" for k, v in sorted(tables_by_page.items()))
        logger.info(f"Docling tableaux par page: {counts_str}")
    logger.info(
        f"Docling: {len(tables)} tableaux extraits des pages {start_page}-{end_page}"
        + (
            f" ({tables_filtered_empty_rows} ignores car rows vides)"
            if tables_filtered_empty_rows
            else ""
        )
    )
    return tables


def _extract_with_pdfplumber_enhanced(
    pdf_path: str, start_page: int, end_page: int, rotated_pages: list[dict] = None
) -> list[dict]:
    """
    Legacy no-op: fallback extraction pdfplumber retiré du pipeline principal.
    """
    _ = (pdf_path, start_page, end_page, rotated_pages)
    logger.warning("Fallback extraction pdfplumber retire: retour liste vide.")
    return []
