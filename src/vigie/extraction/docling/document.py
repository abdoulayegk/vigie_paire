"""Initialisation de docling et extraction d'un document complet.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pymupdf

from vigie.support.utils.footnotes_utils import normalize_footnotes_to_canonical
from vigie.support.utils.pattern_loader import get_patterns

from ..docling_bbox_helpers import _coerce_pdf_path
from .config import _resolve_vision_extraction_enabled
from .models import ExtractedDocument, ExtractedSection, ExtractedTable
from .runtime import (
    CHUNK_SIZE_PAGES,
    MEMORY_UTILS_AVAILABLE,
    ChunkedProcessor,
    check_memory_threshold,
    cleanup_memory,
    get_memory_usage_mb,
)

logger = logging.getLogger("vigie.extraction.docling_processor")


class DocumentExtractionMixin:
    """Initialisation de docling et extraction d'un document complet."""

    def _initialize_docling(self):
        """Initialisation différée (lazy) du convertisseur Docling.

        Docling est initialisé à la première utilisation pour éviter le coût
        de chargement des modèles ML si l'extraction n'est pas nécessaire.
        Configure le device d'accélération (MPS sur macOS, AUTO ailleurs) et
        le nombre de threads depuis les variables d'environnement.

        Lève une exception si Docling n'est pas installé.
        """
        if self._initialized:
            return

        try:
            from docling.datamodel.base_models import InputFormat  # noqa: PLC0415 - initialisation Docling couteuse
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415 - chargement differe
            from docling.document_converter import (  # noqa: PLC0415 - chargement differe
                DocumentConverter,
                PdfFormatOption,
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.use_ocr
            pipeline_options.do_table_structure = True
            pipeline_options.do_picture_description = False  # Desactiver pour acceleration

            _raw_threads = os.environ.get("DOCLING_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
            num_threads = int(_raw_threads) if _raw_threads else (os.cpu_count() or 8)
            default_device = "mps" if sys.platform == "darwin" else "auto"
            device_str = os.environ.get("DOCLING_DEVICE", default_device)

            try:
                from docling.datamodel.accelerator_options import (  # noqa: PLC0415 - API optionnelle selon version Docling
                    AcceleratorDevice,
                    AcceleratorOptions,
                )

                device_map = {
                    "auto": AcceleratorDevice.AUTO,
                    "cpu": AcceleratorDevice.CPU,
                    "cuda": AcceleratorDevice.CUDA,
                    "mps": AcceleratorDevice.MPS,
                }
                device = device_map.get(device_str.lower(), AcceleratorDevice.AUTO)
                pipeline_options.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
            except ImportError:
                pass  # Utiliser les options par defaut de Docling

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            self._initialized = True
            logger.info(
                "Convertisseur Docling initialise (OCR=%s, threads=%s, device=%s)",
                self.use_ocr,
                num_threads,
                device_str,
            )

        except ImportError as e:
            logger.warning(f"Docling non disponible: {e}. Utilisation de l'extraction de secours.")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Erreur initialisation Docling: {e}. Utilisation de l'extraction de secours.")
            self._initialized = True

    def extract_document(
        self,
        pdf_path: str | Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        section: str | None = None,
        labels_only: bool = False,
        use_vision_extraction: bool | None = None,
    ) -> ExtractedDocument:
        """Extraire tout le contenu d'un document PDF.

        Args:
            labels_only: Si True, ne stocker que la premiere colonne (pas de montants)

        Gestion memoire integree:
        - Verification du seuil memoire avant extraction
        - Traitement par chunks pour les gros documents
        - Garbage collection entre les chunks

        Args:
            pdf_path: Chemin vers le fichier PDF
            bank_code: Identifiant de la banque (ex: 'cibc', 'bnc')
            quarter: Identifiant du trimestre (ex: 't1', 't2', 't3')
            year: Annee du rapport
            page_ranges: Liste optionnelle de tuples (start_page, end_page) pour extraction ciblee
                         Si None, extrait tout le document
            section: Nom de la section pour le cache (optionnel)
            use_vision_extraction: Si True, Vision (GPT-4o) comme source contenu pour tous les tableaux.
                Si None, lu depuis config vision_extraction.enabled.

        Returns:
            ExtractedDocument avec tout le contenu extrait
        """
        use_vision_extraction = _resolve_vision_extraction_enabled(bank_code, use_vision_extraction)

        pdf_path = _coerce_pdf_path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trouve: {pdf_path}")

        # Charger les patterns spécifiques à la banque si nécessaire
        if bank_code != self.bank_code_for_patterns and self.extraction_patterns:
            try:
                self.extraction_patterns = get_patterns(bank_code=bank_code)
                self.bank_code_for_patterns = bank_code
                logger.debug(f"Patterns rechargés pour banque: {bank_code}")
            except Exception as e:
                logger.warning(f"Impossible de recharger les patterns pour {bank_code}: {e}")

        # Verifier et nettoyer la memoire si necessaire
        if MEMORY_UTILS_AVAILABLE:
            if check_memory_threshold():
                logger.warning(
                    f"Haute utilisation memoire ({get_memory_usage_mb():.0f}MB), nettoyage avant extraction..."
                )
                cleanup_memory(force=True)

        logger.info(f"Extraction du document: {pdf_path}")
        if MEMORY_UTILS_AVAILABLE:
            logger.info(f"Memoire actuelle: {get_memory_usage_mb():.0f}MB")

        if page_ranges:
            ranges_str = ", ".join(f"{s}-{e}" for s, e in page_ranges)
            logger.info(f"Extraction ciblee sur les pages: {ranges_str}")

        # Charger depuis le cache si disponible
        if self.use_cache and self._cache and section:
            cached = self._cache.load_from_cache(str(pdf_path), section)
            if cached and isinstance(cached.get("data"), dict):
                doc_data = cached["data"]
                doc_data.setdefault("file_path", str(pdf_path))
                doc_data.setdefault("bank_code", bank_code)
                doc_data.setdefault("quarter", quarter)
                doc_data.setdefault("year", year)
                return self._dict_to_extracted_document(doc_data)

        self._initialize_docling()

        # Determiner le nombre total de pages pour le traitement par chunks
        total_pages = self._get_page_count(pdf_path)
        use_chunked = total_pages > CHUNK_SIZE_PAGES and page_ranges is None

        if use_chunked and MEMORY_UTILS_AVAILABLE:
            logger.info(f"Document volumineux ({total_pages} pages), traitement par chunks de {CHUNK_SIZE_PAGES} pages")
            result = self._extract_chunked(
                pdf_path,
                bank_code,
                quarter,
                year,
                total_pages,
                labels_only=labels_only,
                use_vision_extraction=use_vision_extraction,
            )
        elif self._converter is not None:
            result = self._extract_with_docling(
                pdf_path,
                bank_code,
                quarter,
                year,
                page_ranges,
                labels_only=labels_only,
                use_vision_extraction=use_vision_extraction,
            )
        else:
            result = self._docling_unavailable_document(pdf_path, bank_code, quarter, year, page_ranges)

        # Sauvegarder dans le cache si active
        if self.use_cache and self._cache and section:
            try:
                doc_dict = result.to_dict()
                doc_dict["file_path"] = str(pdf_path)
                doc_dict["bank_code"] = bank_code
                doc_dict["quarter"] = quarter
                doc_dict["year"] = year
                doc_dict["total_pages"] = result.total_pages
                self._cache.save_to_cache(
                    str(pdf_path),
                    section,
                    doc_dict,
                    metadata={"bank_code": bank_code, "quarter": quarter, "year": year},
                )
            except Exception as e:
                logger.warning(f"Erreur sauvegarde cache: {e}")

        # Cleanup apres extraction
        if MEMORY_UTILS_AVAILABLE:
            cleanup_memory()

        return result

    def _get_page_count(self, pdf_path: Path) -> int:
        """Obtenir le nombre total de pages d'un PDF."""
        try:
            doc = pymupdf.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Impossible d'obtenir le nombre de pages avec PyMuPDF: {e}")
            return 0

    def _docling_unavailable_document(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None,
        *,
        error: str | None = None,
    ) -> ExtractedDocument:
        """Retourne un document vide quand Docling est indisponible (pas de fallback)."""
        total_pages = self._get_page_count(pdf_path)
        metadata: dict[str, Any] = {
            "extraction_method": "docling_unavailable",
            "page_ranges": page_ranges,
            "warning": "Docling indisponible; extraction non effectuee.",
        }
        if error:
            metadata["error"] = error
        return ExtractedDocument(
            file_path=str(pdf_path),
            bank_code=bank_code,
            quarter=quarter,
            year=year,
            total_pages=total_pages,
            all_tables=[],
            metadata=metadata,
        )

    def _extract_chunked(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        total_pages: int,
        *,
        labels_only: bool = False,
        use_vision_extraction: bool = False,
    ) -> ExtractedDocument:
        """Extraction par chunks pour gros documents.

        Traite le document par groupes de pages pour eviter les depassements memoire.

        Args:
            pdf_path: Chemin vers le PDF
            bank_code: Code de la banque
            quarter: Trimestre
            year: Annee
            total_pages: Nombre total de pages
            labels_only: Extraire uniquement les etiquettes sans valeurs.
            use_vision_extraction: Utiliser l'extraction par vision (OCR avance).

        Returns:
            ExtractedDocument avec tout le contenu
        """
        all_tables = []
        text_parts = []

        # Utiliser le ChunkedProcessor pour gerer la memoire
        processor = ChunkedProcessor(chunk_size=CHUNK_SIZE_PAGES)

        for start_page, end_page in processor.iterate_pages(total_pages):
            logger.info(
                f"Traitement pages {start_page}-{end_page}/{total_pages} (memoire: {get_memory_usage_mb():.0f}MB)"
            )

            # Extraire ce chunk
            page_ranges = [(start_page, end_page)]

            if self._converter is not None:
                chunk_result = self._extract_with_docling(
                    pdf_path,
                    bank_code,
                    quarter,
                    year,
                    page_ranges,
                    labels_only=labels_only,
                    use_vision_extraction=use_vision_extraction,
                )
            else:
                chunk_result = self._docling_unavailable_document(pdf_path, bank_code, quarter, year, page_ranges)

            # Accumuler les resultats
            all_tables.extend(chunk_result.all_tables)
            if chunk_result.metadata.get("text_content"):
                text_parts.append(chunk_result.metadata["text_content"])

            # Enregistrer le progres
            processor.record_progress(len(chunk_result.all_tables))

            # Liberer les references pour GC
            del chunk_result

        # Statistiques finales
        stats = processor.get_stats()
        logger.info(
            f"Extraction terminee: {stats['items_processed']} tableaux, "
            f"{stats['chunks_processed']} chunks, "
            f"{stats['memory_cleanups']} cleanups memoire"
        )

        return ExtractedDocument(
            file_path=str(pdf_path),
            bank_code=bank_code,
            quarter=quarter,
            year=year,
            total_pages=total_pages,
            all_tables=all_tables,
            metadata={
                "extraction_method": "chunked",
                "chunk_size": CHUNK_SIZE_PAGES,
                "chunks_processed": stats["chunks_processed"],
                "peak_memory_mb": stats["peak_memory_mb"],
                "text_content": "\n\n".join(text_parts)[:50000],
            },
        )

    def _dict_to_extracted_document(self, data: dict) -> ExtractedDocument:
        """Convertir un dict en ExtractedDocument."""
        # Reconstruire les tables
        all_tables = []
        for t in data.get("all_tables", []):
            all_tables.append(
                ExtractedTable(
                    table_id=t.get("table_id", ""),
                    page_number=t.get("page_number", 0),
                    title=t.get("title"),
                    headers=t.get("headers", []),
                    rows=t.get("rows", []),
                    first_column_indicators=t.get("first_column_indicators", []),
                    first_column_indicators_raw=t.get("first_column_indicators_raw"),
                    footnotes=normalize_footnotes_to_canonical(t.get("footnotes", [])),
                    section=t.get("section"),
                    section_phase=t.get("section_phase"),
                    table_number=t.get("table_number"),
                    title_clean=t.get("title_clean"),
                    title_raw=t.get("title_raw"),
                    unit_context=t.get("unit_context"),
                    title_resolution_method=t.get("title_resolution_method"),
                    context_before=t.get("context_before", ""),
                    context_after=t.get("context_after", ""),
                    bbox=t.get("bbox"),
                    table_index_on_page=t.get("table_index_on_page"),
                    tables_on_page=t.get("tables_on_page"),
                    bbox_top=t.get("bbox_top"),
                    page_local_role=t.get("page_local_role"),
                    first_column_groups=t.get("first_column_groups"),
                    hierarchical_indicator_signature=t.get("hierarchical_indicator_signature"),
                    title_reliability=t.get("title_reliability"),
                    debug_metrics=t.get("debug_metrics", {}),
                    extraction_method=t.get("extraction_method"),
                    extraction_status=t.get("extraction_status", "ok"),
                )
            )

        # Reconstruire les sections
        sections = []
        for s in data.get("sections", []):
            section_tables = []
            for t in s.get("tables", []):
                section_tables.append(
                    ExtractedTable(
                        table_id=t.get("table_id", ""),
                        page_number=t.get("page_number", 0),
                        title=t.get("title"),
                        headers=t.get("headers", []),
                        rows=t.get("rows", []),
                        first_column_indicators=t.get("first_column_indicators", []),
                        first_column_indicators_raw=t.get("first_column_indicators_raw"),
                        footnotes=normalize_footnotes_to_canonical(t.get("footnotes", [])),
                        section=t.get("section"),
                        section_phase=t.get("section_phase"),
                        table_number=t.get("table_number"),
                        title_clean=t.get("title_clean"),
                        title_raw=t.get("title_raw"),
                        unit_context=t.get("unit_context"),
                        title_resolution_method=t.get("title_resolution_method"),
                        context_before=t.get("context_before", ""),
                        context_after=t.get("context_after", ""),
                        bbox=t.get("bbox"),
                        table_index_on_page=t.get("table_index_on_page"),
                        tables_on_page=t.get("tables_on_page"),
                        bbox_top=t.get("bbox_top"),
                        page_local_role=t.get("page_local_role"),
                        first_column_groups=t.get("first_column_groups"),
                        hierarchical_indicator_signature=t.get("hierarchical_indicator_signature"),
                        title_reliability=t.get("title_reliability"),
                        debug_metrics=t.get("debug_metrics", {}),
                        extraction_method=t.get("extraction_method"),
                        extraction_status=t.get("extraction_status", "ok"),
                    )
                )

            sections.append(
                ExtractedSection(
                    section_id=s.get("section_id", ""),
                    title=s.get("title", ""),
                    start_page=s.get("start_page", 0),
                    end_page=s.get("end_page", 0),
                    text_content=s.get("text_content", ""),
                    tables=section_tables,
                    phase=s.get("phase"),
                )
            )

        return ExtractedDocument(
            file_path=data.get("file_path", ""),
            bank_code=data.get("bank_code", ""),
            quarter=data.get("quarter", ""),
            year=data.get("year", 0),
            total_pages=data.get("total_pages", 0),
            sections=sections,
            all_tables=all_tables,
            metadata=data.get("metadata", {}),
        )
