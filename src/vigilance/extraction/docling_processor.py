"""
Processeur PDF base sur Docling pour l'extraction de contenu structure des rapports bancaires.
Outil principal d'extraction pour les tableaux, le texte et la structure des documents.

Pipeline d'extraction:
1. Docling (extraction structurée)
2. GPT-4 Vision (fallback principal)
3. Docling-only warning si Vision indisponible

Fonctionnalités:
- Cache des extractions pour éviter re-traitement (PDFCacheManager)
- Validation avec GPT-4 Vision (optionnel)
- Gestion memoire pour gros documents (traitement par chunks)
"""

import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz  # Import PyMuPDF (fitz)

from ..utils.feature_flags import is_vision_fallback_enabled
from ..utils.genai import get_openai_api_key
from ..utils.indicator_cleaner import normalize_indicator_for_comparison
from ..utils.matching_normalizer import is_date_only_line, is_non_indicator_line, strip_temporal_expressions
from .docling_normalization import (
    _extract_table_context_split,
    _is_footnote_row,
    _merge_fragmented_cells,
)
from .table_title_resolver import (
    extract_table_number_and_inline_title,
    is_table_number_line,
    is_unit_context_line,
    resolve_title_from_lines,
)

logger = logging.getLogger(__name__)

# Import de la gestion memoire
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

# Import des constantes
try:
    from ..config.constants import EXTRACTION, MEMORY

    CHUNK_SIZE_PAGES = EXTRACTION.CHUNK_SIZE_PAGES
    DPI = EXTRACTION.DPI
    DPI_FAST = EXTRACTION.DPI_FAST
except ImportError:
    CHUNK_SIZE_PAGES = 15
    DPI = 300
    DPI_FAST = 150

# Import du gestionnaire de cache
try:
    from ..utils.pdf_cache import PDFCacheManager

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.debug("PDFCacheManager non disponible")

# Patterns pour détecter les titres de sections principales
# Focus: Gestion du capital et Gestion des risques
SECTION_TITLE_PATTERNS = [
    # Sections de capital
    (r"^gestion\s+du\s+capital$", "Gestion du capital", 1),
    (r"^capital\s+management$", "Capital Management", 1),
    (r"^situation\s+des\s+fonds\s+propres$", "Situation des fonds propres", 1),
    (r"^gestion\s+des\s+fonds\s+propres$", "Gestion des fonds propres", 1),
    (r"^fonds\s+propres\s+r[eé]glementaires$", "Fonds propres réglementaires", 1),
    # Sections de risque
    (r"^gestion\s+des?\s+risques?$", "Gestion des risques", 1),
    (r"^gestion\s+du\s+risque$", "Gestion du risque", 1),
    (r"^risk\s+management$", "Risk Management", 1),
    (r"^facteurs?\s+de\s+risque", "Facteurs de risque", 1),
    (r"^risque\s+de\s+cr[eé]dit$", "Risque de crédit", 1),
    (r"^risque\s+de\s+march[eé]$", "Risque de marché", 1),
    (r"^risque\s+de\s+liquidit[eé]$", "Risque de liquidité", 1),
    (r"^risque\s+op[eé]rationnel$", "Risque opérationnel", 1),
    (r"^credit\s+risk$", "Credit Risk", 1),
    (r"^market\s+risk$", "Market Risk", 1),
    (r"^liquidity\s+risk$", "Liquidity Risk", 1),
]

# Compiler les patterns
COMPILED_SECTION_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), name, phase)
    for pattern, name, phase in SECTION_TITLE_PATTERNS
]


@dataclass
class ExtractedTable:
    """Represente un tableau extrait d'un PDF."""

    table_id: str
    page_number: int
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str] = field(default_factory=list)  # Fingerprint
    footnotes: list[str] = field(default_factory=list)
    section: str | None = None
    section_phase: int | None = None  # Phase de la section (1, 2, 3)
    table_number: str | None = None  # Numéro extrait du titre (ex: "28", "31", "5a")
    title_clean: str | None = None  # Titre sans le numéro
    title_raw: str | None = None  # Titre brut avant nettoyage/matching
    unit_context: str | None = None  # Ligne d'unité (ex: en millions de dollars)
    title_resolution_method: str | None = None  # caption/layout_anchor/text_fallback...
    context_before: str = ""  # 1-2 lignes au-dessus (pour table_type_classifier)
    context_after: str = ""  # 1-2 lignes en-dessous
    bbox: list[float] | None = None  # [l, t, r, b] normalisées 0–1 depuis Docling prov
    first_column_indicators_raw: list[str] | None = None  # Brut avant normalisation
    debug_metrics: dict[str, Any] = field(default_factory=dict)  # row_count, merge_count, etc.
    extraction_method: str | None = None  # docling | vision_fallback_gpt4o

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedSection:
    """Represente une section extraite d'un PDF."""

    section_id: str
    title: str
    start_page: int
    end_page: int
    text_content: str
    tables: list[ExtractedTable] = field(default_factory=list)
    phase: int | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tables"] = [t.to_dict() for t in self.tables]
        return result


@dataclass
class ExtractedDocument:
    """Represente un document PDF entierement extrait."""

    file_path: str
    bank_code: str
    quarter: str
    year: int
    total_pages: int
    sections: list[ExtractedSection] = field(default_factory=list)
    all_tables: list[ExtractedTable] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "bank_code": self.bank_code,
            "quarter": self.quarter,
            "year": self.year,
            "total_pages": self.total_pages,
            "sections": [s.to_dict() for s in self.sections],
            "all_tables": [t.to_dict() for t in self.all_tables],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DoclingProcessor:
    """
    Processeur PDF principal utilisant IBM Docling pour l'extraction structuree.
    Gere les mises en page de tableaux complexes, les cellules fusionnees et le contenu pivote.

    Pipeline de fallback:
    1. Docling (extraction native)
    2. GPT-4 Vision (fallback principal)
    3. Docling-only warning si Vision indisponible
    """

    def __init__(
        self,
        use_ocr: bool = False,
        enhance_images: bool = True,
        use_vision_fallback: bool = False,  # Désactivé pour le moment (coût/latence)
        openai_api_key: str | None = None,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ):
        """
        Initialiser le processeur Docling.

        Args:
            use_ocr: Activer l'OCR pour les documents numerises
            enhance_images: Appliquer l'amelioration d'image avant le traitement
            use_vision_fallback: Activer le fallback GPT-4 Vision pour tableaux complexes
            openai_api_key: Clé API OpenAI pour le fallback Vision
            use_cache: Activer le cache des extractions (defaut False, desactive pour eviter de conserver des extractions de mauvaise qualite)
            cache_dir: Repertoire du cache (optionnel)
        """
        self.use_ocr = use_ocr
        self.enhance_images = enhance_images

        # Securisation de l'usage de Vision
        self.openai_api_key = openai_api_key
        if use_vision_fallback:
            from ..utils.genai import is_genai_configured

            if not is_genai_configured() and not openai_api_key:
                logger.warning("OPENAI_API_KEY non disponible. Fallback Vision désactivé.")
                self.use_vision_fallback = False
            else:
                self.use_vision_fallback = True
        else:
            self.use_vision_fallback = False

        self._converter = None
        self._initialized = False
        self._vision_fallback = None

        # Charger les patterns configurables
        self.bank_code_for_patterns: str | None = None  # Sera set lors de extract_document
        self.extraction_patterns = None
        try:
            from ..utils.pattern_loader import get_patterns

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

    def _initialize_docling(self):
        """Initialisation differee du convertisseur Docling."""
        if self._initialized:
            return

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.use_ocr
            pipeline_options.do_table_structure = True
            pipeline_options.do_picture_description = False  # Desactiver pour acceleration

            _raw_threads = os.environ.get("DOCLING_NUM_THREADS") or os.environ.get(
                "OMP_NUM_THREADS"
            )
            num_threads = int(_raw_threads) if _raw_threads else (os.cpu_count() or 8)
            default_device = "mps" if sys.platform == "darwin" else "auto"
            device_str = os.environ.get("DOCLING_DEVICE", default_device)

            try:
                from docling.datamodel.accelerator_options import (
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
                pipeline_options.accelerator_options = AcceleratorOptions(
                    num_threads=num_threads, device=device
                )
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
            logger.warning(
                f"Erreur initialisation Docling: {e}. Utilisation de l'extraction de secours."
            )
            self._initialized = True

        # Initialiser le fallback Vision si activé
        if self.use_vision_fallback:
            try:
                from .gpt4_vision_fallback import GPT4VisionFallback

                self._vision_fallback = GPT4VisionFallback(api_key=self.openai_api_key)
                logger.info("Fallback GPT-4 Vision initialisé")
            except ImportError as e:
                logger.warning(f"Fallback Vision non disponible: {e}")
                self._vision_fallback = None

    def extract_document(
        self,
        pdf_path: str | Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        section: str | None = None,
        labels_only: bool = False,
    ) -> ExtractedDocument:
        """
        Extraire tout le contenu d'un document PDF.

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

        Returns:
            ExtractedDocument avec tout le contenu extrait
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trouve: {pdf_path}")

        # Charger les patterns spécifiques à la banque si nécessaire
        if bank_code != self.bank_code_for_patterns and self.extraction_patterns:
            try:
                from ..utils.pattern_loader import get_patterns

                self.extraction_patterns = get_patterns(bank_code=bank_code)
                self.bank_code_for_patterns = bank_code
                logger.debug(f"Patterns rechargés pour banque: {bank_code}")
            except Exception as e:
                logger.warning(f"Impossible de recharger les patterns pour {bank_code}: {e}")

        # Verifier et nettoyer la memoire si necessaire
        if MEMORY_UTILS_AVAILABLE:
            if check_memory_threshold():
                logger.warning(
                    f"Haute utilisation memoire ({get_memory_usage_mb():.0f}MB), "
                    "nettoyage avant extraction..."
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
            logger.info(
                f"Document volumineux ({total_pages} pages), "
                f"traitement par chunks de {CHUNK_SIZE_PAGES} pages"
            )
            result = self._extract_chunked(
                pdf_path, bank_code, quarter, year, total_pages, labels_only=labels_only
            )
        elif self._converter is not None:
            result = self._extract_with_docling(
                pdf_path, bank_code, quarter, year, page_ranges, labels_only=labels_only
            )
        else:
            result = self._extract_with_fallback(
                pdf_path, bank_code, quarter, year, page_ranges, labels_only=labels_only
            )

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
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Impossible d'obtenir le nombre de pages avec PyMuPDF: {e}")
            return 0

    def _extract_chunked(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        total_pages: int,
        *,
        labels_only: bool = False,
    ) -> ExtractedDocument:
        """
        Extraction par chunks pour gros documents.

        Traite le document par groupes de pages pour eviter les depassements memoire.

        Args:
            pdf_path: Chemin vers le PDF
            bank_code: Code de la banque
            quarter: Trimestre
            year: Annee
            total_pages: Nombre total de pages

        Returns:
            ExtractedDocument avec tout le contenu
        """
        all_tables = []
        text_parts = []

        # Utiliser le ChunkedProcessor pour gerer la memoire
        processor = ChunkedProcessor(chunk_size=CHUNK_SIZE_PAGES)

        for start_page, end_page in processor.iterate_pages(total_pages):
            logger.info(
                f"Traitement pages {start_page}-{end_page}/{total_pages} "
                f"(memoire: {get_memory_usage_mb():.0f}MB)"
            )

            # Extraire ce chunk
            page_ranges = [(start_page, end_page)]

            if self._converter is not None:
                chunk_result = self._extract_with_docling(
                    pdf_path, bank_code, quarter, year, page_ranges, labels_only=labels_only
                )
            else:
                chunk_result = self._extract_with_fallback(
                    pdf_path, bank_code, quarter, year, page_ranges, labels_only=labels_only
                )

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
                    footnotes=t.get("footnotes", []),
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
                    debug_metrics=t.get("debug_metrics", {}),
                    extraction_method=t.get("extraction_method"),
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
                    footnotes=t.get("footnotes", []),
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
                    debug_metrics=t.get("debug_metrics", {}),
                    extraction_method=t.get("extraction_method"),
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

    def _is_page_in_ranges(self, page_num: int, page_ranges: list[tuple[int, int]] | None) -> bool:
        """Verifier si une page est dans les plages cibles."""
        if page_ranges is None:
            return True
        return any(start <= page_num <= end for start, end in page_ranges)

    def _normalize_page_ranges(
        self, page_ranges: list[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        """Normaliser les plages pour garantir start >= 1 et end >= start."""
        if not page_ranges:
            return []

        normalized: list[tuple[int, int]] = []
        for idx, page_range in enumerate(page_ranges):
            try:
                start_raw, end_raw = page_range
                start = int(start_raw)
                end = int(end_raw)
            except (TypeError, ValueError):
                logger.warning("Plage de pages invalide ignoree a l'index %s: %s", idx, page_range)
                continue

            start_norm = max(1, start)
            if start_norm != start:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: start=%s -> %s", idx, start, start_norm
                )

            end_norm = max(start_norm, end)
            if end_norm != end:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: end=%s -> %s", idx, end, end_norm
                )

            normalized.append((start_norm, end_norm))

        normalized.sort(key=lambda r: (r[0], r[1]))
        return normalized

    def _build_docling_page_range(
        self, page_ranges: list[tuple[int, int]] | None
    ) -> tuple[int, int] | None:
        """Construire une plage Docling compatible (start, end), indexee a 1."""
        normalized = self._normalize_page_ranges(page_ranges)
        if not normalized:
            return None

        if len(normalized) == 1:
            return normalized[0]

        range_start = min(start for start, _ in normalized)
        range_end = max(end for _, end in normalized)
        logger.warning(
            "Plusieurs plages demandees (%s) compressees en plage englobante (%s-%s) "
            "pour compatibilite Docling; filtrage applicatif conserve.",
            normalized,
            range_start,
            range_end,
        )
        return (range_start, range_end)

    def _extract_with_docling(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        labels_only: bool = False,
    ) -> ExtractedDocument:
        """Extraire en utilisant la bibliotheque Docling avec pipeline de prétraitement."""
        try:
            normalized_page_ranges = self._normalize_page_ranges(page_ranges)
            effective_page_ranges = normalized_page_ranges or None
            docling_page_range = self._build_docling_page_range(effective_page_ranges)
            convert_kwargs: dict = {}
            if docling_page_range is not None:
                convert_kwargs["page_range"] = docling_page_range

            result = self._converter.convert(str(pdf_path), **convert_kwargs)
            doc = result.document

            # Extraire les tableaux
            all_tables = []
            failed_tables = []  # Tables qui ont échoué et nécessitent fallback
            tables_by_page: dict[int, int] = {}

            for idx, table in enumerate(doc.tables):
                page_num = table.prov[0].page_no if table.prov else 0

                table_bbox: list[float] | None = None
                try:
                    if table.prov and hasattr(table.prov[0], "bbox") and table.prov[0].bbox is not None:
                        raw_bbox = table.prov[0].bbox
                        page_obj = doc.pages.get(page_num) if hasattr(doc, "pages") else None
                        if page_obj and hasattr(page_obj, "size") and page_obj.size:
                            norm = raw_bbox.to_top_left_origin(page_height=page_obj.size.height)
                            norm = norm.normalized(page_obj.size)
                            table_bbox = [norm.l, norm.t, norm.r, norm.b]
                        elif hasattr(raw_bbox, "as_tuple"):
                            table_bbox = list(raw_bbox.as_tuple())
                except Exception:
                    table_bbox = None

                # Filtrer par plage de pages si specifie
                if not self._is_page_in_ranges(page_num, effective_page_ranges):
                    continue

                tables_by_page[page_num] = tables_by_page.get(page_num, 0) + 1

                headers = self._extract_table_headers(table)
                rows_raw = self._extract_table_rows(table)
                row_count_before_merge = len(rows_raw)
                rows = _merge_fragmented_cells(rows_raw)
                row_count_after_merge = len(rows)

                # Vérifier si l'extraction a réussi
                extraction_quality = self._assess_table_quality(headers, rows)

                if extraction_quality < 0.5:
                    # Marquer pour fallback
                    failed_tables.append(
                        {"idx": idx, "page_number": page_num, "reason": "low_quality_extraction"}
                    )
                    logger.warning(
                        f"Table {idx} page {page_num}: qualité faible, fallback nécessaire"
                    )

                # Extraire le titre et le numero avec resolution robuste (caption en premier signal)
                caption_title = self._find_table_title(table)
                title_lines = [caption_title] if caption_title else []
                first_row_cells: list[str] | None = None
                if (bank_code or "").lower() == "cibc" and rows:
                    first_row_cells = [
                        str(row[0]).strip() for row in rows if row and len(row) > 0
                    ]
                title_resolution = self._resolve_title_metadata_from_lines(
                    title_lines, first_row_cells=first_row_cells
                )

                resolved_title = title_resolution.get("title") or ""
                title_raw = title_resolution.get("title_raw") or caption_title
                table_number = title_resolution.get("table_number") or None
                unit_context = title_resolution.get("unit_context") or None
                title_resolution_method = title_resolution.get("resolution_method") or None

                if not table_number:
                    parsed_number, parsed_inline = self._extract_table_number(title_raw)
                    table_number = parsed_number
                    if not resolved_title and parsed_inline:
                        resolved_title = parsed_inline

                if not resolved_title and title_raw:
                    resolved_title = title_raw

                title_clean = resolved_title or None

                # Exclure les lignes de notes du grid avant indicateurs et sortie
                data_rows = [r for r in rows if not _is_footnote_row(r)]
                footnote_row_filtered_count = len(rows) - len(data_rows)

                # Extraire les indicateurs (première colonne) - comportement inchangé
                first_column_indicators_raw_list: list[str] = []
                indicators: list[str] = []
                for row in data_rows:
                    if not row or len(row) == 0:
                        continue
                    cell = str(row[0]).strip() if row[0] else ""
                    if cell and not is_date_only_line(cell):
                        first_column_indicators_raw_list.append(cell)
                        indicators.append(normalize_indicator_for_comparison(cell))

                # Métriques en lecture seule (observabilité, pas de changement de comportement)
                merge_count = row_count_before_merge - row_count_after_merge
                unique_normalized = len(set(indicators))
                duplicate_ratio = (
                    0 if len(indicators) == 0 else 1 - (unique_normalized / len(indicators))
                )
                header_like_count = sum(
                    1 for r in first_column_indicators_raw_list if is_non_indicator_line(r)
                )
                header_like_ratio = (
                    header_like_count / len(first_column_indicators_raw_list)
                    if first_column_indicators_raw_list
                    else 0.0
                )
                debug_metrics = {
                    "row_count_before_merge": row_count_before_merge,
                    "row_count_after_merge": row_count_after_merge,
                    "merge_count": merge_count,
                    "footnote_row_filtered_count": footnote_row_filtered_count,
                    "indicator_count": len(indicators),
                    "duplicate_ratio": duplicate_ratio,
                    "header_like_ratio": header_like_ratio,
                    "table_quality_score": extraction_quality,
                }

                out_headers = [] if labels_only else headers
                out_rows = [] if labels_only else data_rows

                extracted_table = ExtractedTable(
                    table_id=f"tableau_{idx}",
                    page_number=page_num,
                    title=resolved_title or title_raw,
                    headers=out_headers,
                    rows=out_rows,
                    first_column_indicators=indicators,
                    first_column_indicators_raw=first_column_indicators_raw_list,
                    footnotes=self._extract_table_footnotes(table),
                    table_number=table_number,
                    title_clean=title_clean,
                    title_raw=title_raw,
                    unit_context=unit_context,
                    title_resolution_method=title_resolution_method or ("caption" if caption_title else None),
                    bbox=table_bbox,
                    debug_metrics=debug_metrics,
                )
                all_tables.append(extracted_table)

                if (
                    os.environ.get("ENABLE_VISION_FALLBACK") == "1"
                    and table_bbox
                    and len(table_bbox) == 4
                ):
                    try:
                        from .vision_fallback_integration import _try_vision_first_column_fallback

                        _try_vision_first_column_fallback(
                            extracted_table=extracted_table,
                            pdf_path=str(pdf_path),
                            bank_code=bank_code,
                            quarter=quarter,
                            year=year,
                            page_num=page_num,
                            table_bbox=table_bbox,
                        )
                    except Exception:
                        pass

                if os.environ.get("ENABLE_TABLE_CROP_DUMP") == "1" and table_bbox and len(table_bbox) == 4:
                    try:
                        from ..utils.pdf_crop import crop_table_image

                        crop_dir = Path("outputs/debug_crops") / f"{bank_code}_{quarter}_{year}"
                        crop_path = crop_dir / f"{extracted_table.table_id}_p{page_num}.png"
                        crop_table_image(
                            str(pdf_path),
                            page_num,
                            table_bbox,
                            str(crop_path),
                            dpi=300,
                        )
                    except Exception:
                        pass

            # Log par page pour analyse de completude
            if tables_by_page:
                counts_str = ", ".join(f"p{k}:{v}" for k, v in sorted(tables_by_page.items()))
                logger.info(f"Docling tableaux par page: {counts_str}")

            # Appliquer fallback pour les tables échouées
            if failed_tables and self._vision_fallback:
                all_tables = self._apply_fallback_extraction(
                    pdf_path, all_tables, failed_tables, labels_only=labels_only
                )

            # Enrichir les tableaux sans titre en cherchant dans le texte PDF
            all_tables = self._enrich_tables_with_titles(all_tables, pdf_path)
            all_tables = self._enrich_tables_with_context(all_tables, pdf_path)

            # Compter les tableaux avec numéros
            tables_with_numbers = sum(1 for t in all_tables if t.table_number)
            if tables_with_numbers > 0:
                logger.info(
                    f"Tableaux avec numéros détectés: {tables_with_numbers}/{len(all_tables)}"
                )

            # Extraire le contenu textuel pour les sections
            text_content = doc.export_to_markdown()

            # Associer les tableaux à leurs sections parentes
            all_tables = self._associate_tables_with_sections(all_tables, text_content)

            # Compter les sections détectées
            sections_found = set(t.section for t in all_tables if t.section)
            if sections_found:
                logger.info(f"Sections détectées: {', '.join(sections_found)}")

            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=len(doc.pages) if hasattr(doc, "pages") else 0,
                all_tables=all_tables,
                metadata={
                    "extraction_method": "docling",
                    "failed_tables_count": len(failed_tables),
                    "sections_detected": list(sections_found),
                    "page_ranges": page_ranges,
                    "text_content": text_content[:50000],  # Limite pour la memoire
                },
            )

        except Exception as e:
            logger.error(
                "Echec de l'extraction Docling (%s): %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return self._extract_with_fallback(
                pdf_path, bank_code, quarter, year, page_ranges, labels_only=labels_only
            )

    def _extract_with_fallback(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        labels_only: bool = False,
    ) -> ExtractedDocument:
        """Extraction de secours via GPT-4o Vision si Docling indisponible."""
        total_pages = self._get_page_count(pdf_path)
        allow_vision = is_vision_fallback_enabled()

        if not allow_vision:
            logger.warning("Fallback Vision desactive; retour Docling-only avec warning.")
            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=total_pages,
                metadata={
                    "extraction_method": "docling_only_warning",
                    "vision_status": "disabled_by_flag",
                    "page_ranges": page_ranges,
                    "warning": "Docling indisponible et fallback Vision desactive",
                },
            )

        try:
            from .vision_table_extractor import VisionTableExtractor
        except ImportError as e:
            logger.warning(f"VisionTableExtractor non disponible: {e}")
            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=total_pages,
                metadata={
                    "extraction_method": "docling_only_warning",
                    "vision_status": "import_error",
                    "page_ranges": page_ranges,
                    "warning": "VisionTableExtractor non disponible",
                },
            )

        api_key = self.openai_api_key or get_openai_api_key()
        if not api_key:
            logger.warning("OPENAI_API_KEY absente: fallback Vision ignore.")
            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=total_pages,
                metadata={
                    "extraction_method": "docling_only_warning",
                    "vision_status": "skipped_no_api_key",
                    "page_ranges": page_ranges,
                    "warning": "OPENAI_API_KEY absente, impossible d'utiliser Vision",
                },
            )

        ranges = page_ranges or [(1, total_pages)]
        vision = VisionTableExtractor(
            api_key=api_key,
            model="gpt-4o",
            save_proof_images=False,
            labels_only=labels_only,
        )

        all_tables: list[ExtractedTable] = []
        vision_errors: list[str] = []
        for start_page, end_page in ranges:
            try:
                result = vision.extract_from_section(
                    pdf_path=str(pdf_path),
                    start_page=start_page,
                    end_page=end_page,
                    section_name="fallback_vision",
                    bank_code=bank_code,
                )
                extracted_tables = getattr(result, "extracted_tables", None)
                if not isinstance(extracted_tables, list):
                    raise ValueError("Vision payload invalide: extracted_tables manquant")
            except Exception as e:
                msg = f"Vision fallback error {start_page}-{end_page}: {e}"
                logger.warning(msg)
                vision_errors.append(msg)
                continue

            for extracted in extracted_tables:
                try:
                    table_number, title_clean = self._extract_table_number(extracted.title)
                    out_headers = [] if labels_only else extracted.headers
                    out_rows = [] if labels_only else extracted.rows
                    out_footnotes = [] if labels_only else extracted.footnotes
                    title_raw = getattr(extracted, "title_raw", None) or extracted.title
                    unit_context = getattr(extracted, "unit_context", None)
                    resolution_method = getattr(extracted, "title_resolution_method", None)
                    all_tables.append(
                        ExtractedTable(
                            table_id=extracted.table_id,
                            page_number=extracted.page_number,
                            title=extracted.title,
                            headers=out_headers,
                            rows=out_rows,
                            first_column_indicators=extracted.first_column_indicators,
                            footnotes=out_footnotes,
                            table_number=table_number,
                            title_clean=title_clean,
                            title_raw=title_raw,
                            unit_context=unit_context,
                            title_resolution_method=resolution_method,
                            bbox=getattr(extracted, "bbox", None),
                        )
                    )
                except Exception as e:
                    msg = (
                        f"Vision fallback parse error {start_page}-{end_page}: "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.warning(msg)
                    vision_errors.append(msg)
                    continue

        if all_tables:
            metadata = {
                "extraction_method": "gpt4o_vision",
                "vision_status": "applied",
                "page_ranges": page_ranges,
                "warning": "Docling indisponible: fallback Vision utilise",
            }
        else:
            metadata = {
                "extraction_method": "docling_only_warning",
                "vision_status": "failed",
                "page_ranges": page_ranges,
                "warning": "Docling indisponible et fallback Vision en echec",
            }
        if vision_errors:
            metadata["vision_errors"] = vision_errors

        all_tables = self._enrich_tables_with_context(all_tables, pdf_path)

        return ExtractedDocument(
            file_path=str(pdf_path),
            bank_code=bank_code,
            quarter=quarter,
            year=year,
            total_pages=total_pages,
            all_tables=all_tables,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_text_lines(text: str) -> list[str]:
        """Nettoyer un bloc de texte en liste de lignes non vides."""
        if not text:
            return []
        lines = []
        for line in str(text).split("\n"):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _resolve_title_metadata_from_lines(
        self,
        lines: list[str],
        first_row_cells: list[str] | None = None,
    ) -> dict[str, str]:
        """Resoudre le titre semantique via le resolver central."""
        return resolve_title_from_lines(
            lines,
            bank_code=self.bank_code_for_patterns,
            first_row_cells=first_row_cells,
        )

    def _title_quality_score(self, title: str | None) -> int:
        """
        Evaluer la qualite d'un titre.

        Les lignes purement meta (TABLEAU N seul, unite, date) ont un score faible.
        """
        if not title or not str(title).strip():
            return 0

        value = str(title).strip()
        score = 1

        if not is_unit_context_line(value):
            score += 2
        if not is_table_number_line(value):
            score += 2

        number, inline = extract_table_number_and_inline_title(value)
        if number and inline:
            inline_temporal = bool(
                strip_temporal_expressions(inline, target="title", aggressive=True).strip()
            )
            if inline_temporal:
                score += 1

        temporal_free = strip_temporal_expressions(value, target="title", aggressive=True)
        if temporal_free.strip():
            score += 1

        if len(value) >= 12:
            score += 1

        return score

    def _resolve_page_title_candidates(self, page_text: str) -> list[dict[str, str]]:
        """Construire les candidats titre sur une page (1 candidat par ligne TABLEAU quand possible)."""
        lines = self._normalize_text_lines(page_text)
        if not lines:
            return []

        number_indices = [idx for idx, line in enumerate(lines) if is_table_number_line(line)]
        candidates: list[dict[str, str]] = []

        if not number_indices:
            candidate = self._resolve_title_metadata_from_lines(lines)
            if candidate.get("title") or candidate.get("table_number") or candidate.get("title_raw"):
                candidates.append(candidate)
            return candidates

        for idx in number_indices:
            start = max(0, idx - 4)
            end = min(len(lines), idx + 4)
            window_lines = lines[start:end]
            candidate = self._resolve_title_metadata_from_lines(window_lines)

            line_number, _ = extract_table_number_and_inline_title(lines[idx])
            if line_number and not candidate.get("table_number"):
                candidate["table_number"] = line_number
            if not candidate.get("title_raw"):
                candidate["title_raw"] = lines[idx]
            if not candidate.get("resolution_method"):
                candidate["resolution_method"] = "layout_anchor"
            candidates.append(candidate)

        # Dedup simple en preservant l'ordre
        deduped: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.get("table_number", ""), candidate.get("title", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)

        return deduped

    def _find_table_title(self, table) -> str | None:
        """Extraire le titre du tableau a partir de l'objet tableau Docling."""
        try:
            if hasattr(table, "caption") and table.caption:
                return str(table.caption)
        except Exception:
            pass
        return None

    def _find_table_titles_in_text(self, text_content: str) -> dict[int, list[str]]:
        """
        Cherche les titres de tableaux (TABLEAU N, TABLE N, TN) dans le texte par page.

        Args:
            text_content: Contenu textuel complet avec marqueurs de page

        Returns:
            Dict {page_num: [liste des titres trouvés sur cette page]}
        """
        titles_by_page = {}

        # Patterns pour tous les formats de tableaux
        patterns = [
            # TD/BMO: TABLEAU 28 : Titre
            re.compile(r"TABLEAU\s+(\d+)\s*[:\-–—]?\s*([^\n]+)", re.IGNORECASE),
            # Anglais: TABLE 28 : Title
            re.compile(r"TABLE\s+(\d+)\s*[:\-–—]?\s*([^\n]+)", re.IGNORECASE),
            # BNS: T5 Titre
            re.compile(r"^T(\d+[A-Za-z]?)\s+([^\n]+)", re.MULTILINE),
        ]

        # Parser le texte page par page
        current_page = 1
        for line in text_content.split("\n"):
            # Détecter les marqueurs de page
            if line.startswith("--- Page ") or line.startswith("## Page "):
                try:
                    current_page = int(re.search(r"Page\s+(\d+)", line).group(1))
                except:
                    pass
                continue

            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    full_title = line.strip()
                    if current_page not in titles_by_page:
                        titles_by_page[current_page] = []
                    titles_by_page[current_page].append(full_title)
                    break

        return titles_by_page

    def _enrich_tables_with_titles(
        self, tables: list[ExtractedTable], pdf_path: Path
    ) -> list[ExtractedTable]:
        """
        Enrichit les tableaux sans titre en cherchant dans le texte PDF.

        Args:
            tables: Liste des tableaux extraits
            pdf_path: Chemin vers le PDF source

        Returns:
            Liste des tableaux avec titres enrichis
        """
        try:
            import pdfplumber
        except ImportError:
            return tables

        tables_by_page: dict[int, list[ExtractedTable]] = {}
        for table in tables:
            tables_by_page.setdefault(table.page_number, []).append(table)

        bank_code = (self.bank_code_for_patterns or "").lower()

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num not in tables_by_page:
                    continue

                page_text = page.extract_text() or ""
                candidates = self._resolve_page_title_candidates(page_text)
                page_tables = tables_by_page[page_num]

                # CIBC: un candidat par tableau (lignes page + first_column de chaque tableau)
                if bank_code == "cibc" and len(page_tables) >= 1 and len(candidates) <= 1:
                    lines = self._normalize_text_lines(page_text)
                    per_table_candidates: list[dict[str, str]] = []
                    for table in page_tables:
                        first_row_cells = list(table.first_column_indicators or [])
                        if not first_row_cells and getattr(table, "rows", None):
                            first_row_cells = [
                                str(row[0]).strip()
                                for row in table.rows
                                if row and len(row) > 0
                            ]
                        cand = resolve_title_from_lines(
                            lines,
                            bank_code="cibc",
                            first_row_cells=first_row_cells or None,
                        )
                        per_table_candidates.append(cand)
                    candidates = per_table_candidates

                if not candidates:
                    continue

                by_number: dict[str, list[int]] = {}
                for idx, candidate in enumerate(candidates):
                    number = (candidate.get("table_number") or "").strip()
                    if number:
                        by_number.setdefault(number, []).append(idx)

                available = set(range(len(candidates)))

                for table in page_tables:
                    selected_idx: int | None = None
                    current_number = str(table.table_number or "").strip()

                    # 1) Priorite numero si disponible
                    if current_number and current_number in by_number:
                        for idx in by_number[current_number]:
                            if idx in available:
                                selected_idx = idx
                                break

                    # 2) Fallback positionnel (ordre des tableaux sur la page)
                    if selected_idx is None and available:
                        selected_idx = min(available)

                    if selected_idx is None:
                        continue

                    available.remove(selected_idx)
                    candidate = candidates[selected_idx]

                    candidate_title = (candidate.get("title") or "").strip()
                    candidate_title_raw = (candidate.get("title_raw") or "").strip()
                    candidate_number = (candidate.get("table_number") or "").strip()
                    candidate_unit = (candidate.get("unit_context") or "").strip()
                    candidate_method = (candidate.get("resolution_method") or "").strip()

                    # On remplace si le candidat est clairement meilleur semantiquement.
                    current_title = (table.title or "").strip()
                    if self._title_quality_score(candidate_title) > self._title_quality_score(
                        current_title
                    ):
                        table.title = candidate_title or current_title or None
                        table.title_clean = candidate_title or table.title_clean
                    if candidate_method and not table.title_resolution_method:
                        table.title_resolution_method = candidate_method

                    if candidate_title_raw:
                        table.title_raw = candidate_title_raw
                    elif not table.title_raw:
                        table.title_raw = current_title or None

                    if candidate_number and not table.table_number:
                        table.table_number = candidate_number

                    if candidate_unit:
                        table.unit_context = candidate_unit

                    # Si le titre reste vide, fallback explicite sur title_raw.
                    if not table.title and table.title_raw:
                        table.title = table.title_raw
                    if not table.title_clean and table.title:
                        table.title_clean = table.title

        return tables

    def _enrich_tables_with_context(
        self, tables: list[ExtractedTable], pdf_path: Path
    ) -> list[ExtractedTable]:
        """
        Enrichit les tableaux avec context_before et context_after (pour table_type_classifier).
        """
        try:
            import pdfplumber
        except ImportError:
            return tables

        if not pdf_path or not str(pdf_path) or not Path(pdf_path).exists():
            return tables

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for table in tables:
                    page_num = table.page_number
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text() or ""
                    cb, ca = _extract_table_context_split(text, table.title or table.title_clean)
                    table.context_before = cb
                    table.context_after = ca
        except (FileNotFoundError, OSError) as e:
            logger.debug("Skip context enrichment (file unavailable): %s", e)

        return tables

    def _extract_table_number(self, title: str | None) -> tuple[str | None, str | None]:
        """
        Extrait le numéro du tableau depuis le titre.

        Formats supportés:
        - TD/BMO: TABLEAU 28 : Titre..., TABLE 31 - Title..., TABLEAU 1
        - BNS: T5 Titre..., T11A Titre..., T14A Titre...

        Args:
            title: Titre complet du tableau

        Returns:
            Tuple (numéro, titre_nettoyé) ou (None, titre_original)
        """
        if not title:
            return None, None

        table_number, inline_title = extract_table_number_and_inline_title(title)
        if inline_title:
            return table_number, inline_title
        return table_number, (None if table_number else title)

    def _detect_sections_in_text(self, text_content: str) -> list[dict]:
        """
        Détecte les titres de sections dans le texte du document.

        Args:
            text_content: Contenu textuel du document

        Returns:
            Liste de sections détectées avec leur position, nom et phase
        """
        sections = []
        lines = text_content.split("\n")

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # Ignorer les lignes trop longues (pas un titre) ou trop courtes
            if len(line_stripped) < 5 or len(line_stripped) > 80:
                continue

            for pattern, section_name, phase in COMPILED_SECTION_PATTERNS:
                if pattern.match(line_stripped):
                    sections.append(
                        {
                            "line_num": line_num,
                            "name": section_name,
                            "phase": phase,
                            "original_text": line_stripped,
                        }
                    )
                    break

        return sections

    def _associate_tables_with_sections(
        self, tables: list[ExtractedTable], text_content: str
    ) -> list[ExtractedTable]:
        """
        Associe chaque tableau à sa section parente basée sur la position dans le document.

        Args:
            tables: Liste des tableaux extraits
            text_content: Contenu textuel du document

        Returns:
            Liste des tableaux avec section associée
        """
        sections = self._detect_sections_in_text(text_content)

        if not sections:
            return tables

        # Pour chaque tableau, trouver la section précédente la plus proche
        for table in tables:
            # Chercher dans le texte autour du titre du tableau
            table_title = table.title or ""

            # Trouver la section la plus proche avant ce tableau (basé sur page)
            best_section = None
            for section in sections:
                # On ne peut pas directement mapper ligne -> page sans page_breaks
                # Donc on utilise une heuristique : sections apparaissent avant leurs tableaux
                # Pour l'instant, on assigne la dernière section vue
                if section["line_num"] < (table.page_number * 50):  # Estimation ~50 lignes/page
                    best_section = section

            if best_section:
                table.section = best_section["name"]
                table.section_phase = best_section["phase"]

        return tables

    def _extract_table_headers(self, table) -> list[str]:
        """Extraire les en-tetes du tableau Docling."""
        try:
            if hasattr(table, "data") and table.data:
                grid = table.data.grid
                if grid and len(grid) > 0:
                    return [
                        str(cell.text) if hasattr(cell, "text") else str(cell) for cell in grid[0]
                    ]
        except Exception as e:
            logger.debug(f"Erreur lors de l'extraction des en-tetes: {e}")
        return []

    def _extract_table_rows(self, table) -> list[list[str]]:
        """Extraire les lignes de donnees du tableau Docling."""
        try:
            if hasattr(table, "data") and table.data:
                grid = table.data.grid
                if grid and len(grid) > 1:
                    rows = []
                    for row in grid[1:]:
                        rows.append(
                            [str(cell.text) if hasattr(cell, "text") else str(cell) for cell in row]
                        )
                    return rows
        except Exception as e:
            logger.debug(f"Erreur lors de l'extraction des lignes: {e}")
        return []

    def _extract_table_footnotes(self, table) -> list[str]:
        """Extraire les notes de bas de page associees a un tableau."""
        footnotes = []
        try:
            if hasattr(table, "footnotes"):
                for fn in table.footnotes:
                    footnotes.append(str(fn))
        except Exception:
            pass
        return footnotes

    def _infer_table_title(self, page_text: str, table_idx: int) -> str | None:
        """Inferer le titre du tableau a partir du texte environnant."""
        # Rechercher les patterns de titres courants dans les rapports bancaires francais
        title_patterns = [
            "Tableau",
            "Table",
            "Sommaire",
            "Resume",
            "Ratio",
            "Indicateur",
            "Analyse",
        ]

        lines = page_text.split("\n")
        for i, line in enumerate(lines):
            for pattern in title_patterns:
                if pattern.lower() in line.lower() and len(line) < 200:
                    return line.strip()

        return f"Tableau {table_idx + 1}"

    def _assess_table_quality(self, headers: list[str], rows: list[list[str]]) -> float:
        """
        Évaluer la qualité d'une extraction de tableau.

        Returns:
            Score entre 0.0 (mauvais) et 1.0 (excellent)
        """
        if not headers and not rows:
            return 0.0

        score = 0.0

        # Vérifier la présence d'en-têtes
        if headers:
            score += 0.3
            # Vérifier que les en-têtes ne sont pas vides
            non_empty_headers = [h for h in headers if h and h.strip()]
            if non_empty_headers:
                score += 0.1 * (len(non_empty_headers) / len(headers))

        # Vérifier la présence de lignes
        if rows:
            score += 0.3
            # Vérifier la cohérence du nombre de colonnes
            col_counts = [len(row) for row in rows]
            if col_counts:
                most_common = max(set(col_counts), key=col_counts.count)
                consistency = col_counts.count(most_common) / len(col_counts)
                score += 0.2 * consistency

        # Vérifier que le nombre de colonnes correspond aux en-têtes
        if headers and rows:
            header_count = len(headers)
            matching_rows = sum(1 for row in rows if len(row) == header_count)
            if matching_rows > 0:
                score += 0.1 * (matching_rows / len(rows))

        return min(1.0, score)

    def _apply_fallback_extraction(
        self,
        pdf_path: Path,
        tables: list[ExtractedTable],
        failed_tables: list[dict],
        *,
        labels_only: bool = False,
    ) -> list[ExtractedTable]:
        """
        Appliquer le fallback GPT-4 Vision pour les tables échouées.
        """
        for failed in failed_tables:
            idx = failed["idx"]
            page_num = failed["page_number"]

            logger.info(f"Fallback pour table {idx} page {page_num}")

            try:
                page_image = None
                try:
                    from ..utils.pdf_image import pdf_page_to_image

                    page_image = pdf_page_to_image(str(pdf_path), page_num)
                except Exception as e:
                    logger.warning(f"Impossible de charger la page en image: {e}")

                if self._vision_fallback and page_image is not None:
                    try:
                        vision_result = self._vision_fallback.extract_table_from_image(
                            page_image,
                            context=f"Tableau de rapport bancaire, page {page_num}",
                        )

                        if vision_result.success:
                            # SMART MERGE: Fusionner avec les données Docling existantes
                            original_table = tables[idx]

                            docling_data_dict = {
                                "title": original_table.title,
                                "headers": original_table.headers,
                                "rows": original_table.rows,
                                "footnotes": original_table.footnotes,
                            }

                            # Appel Fusion (Docling + Vision)
                            merged_result = self._vision_fallback.smart_merge_results(
                                docling_data_dict, vision_result, context=f"Tableau page {page_num}"
                            )

                            # Recalculer les indicateurs (comportement inchangé: cell + not is_date_only_line)
                            indicators_fb: list[str] = []
                            first_column_raw_fb: list[str] = []
                            for row in merged_result.rows or []:
                                if not row or len(row) == 0:
                                    continue
                                cell = str(row[0]).strip() if row[0] else ""
                                if cell and not is_date_only_line(cell):
                                    first_column_raw_fb.append(cell)
                                    indicators_fb.append(normalize_indicator_for_comparison(cell))
                            n_rows = len(merged_result.rows or [])
                            unique_fb = len(set(indicators_fb))
                            dup_ratio_fb = 0 if not indicators_fb else 1 - (unique_fb / len(indicators_fb))
                            hdr_like_fb = (
                                sum(1 for r in first_column_raw_fb if is_non_indicator_line(r))
                                / len(first_column_raw_fb)
                                if first_column_raw_fb
                                else 0.0
                            )
                            debug_metrics_fb = {
                                "row_count_before_merge": n_rows,
                                "row_count_after_merge": n_rows,
                                "merge_count": 0,
                                "footnote_row_filtered_count": 0,
                                "indicator_count": len(indicators_fb),
                                "duplicate_ratio": dup_ratio_fb,
                                "header_like_ratio": hdr_like_fb,
                            }

                            out_headers = [] if labels_only else merged_result.headers
                            out_rows = [] if labels_only else merged_result.rows

                            # Remplacer les données de la table
                            tables[idx] = ExtractedTable(
                                table_id=original_table.table_id,
                                page_number=page_num,
                                title=merged_result.table_title or original_table.title,
                                headers=out_headers,
                                rows=out_rows,
                                footnotes=merged_result.footnotes,
                                section=original_table.section,
                                table_number=original_table.table_number,
                                title_clean=original_table.title_clean,
                                title_raw=original_table.title_raw or original_table.title,
                                unit_context=original_table.unit_context,
                                title_resolution_method=original_table.title_resolution_method,
                                first_column_indicators=indicators_fb,
                                first_column_indicators_raw=first_column_raw_fb,
                                debug_metrics=debug_metrics_fb,
                                context_before=original_table.context_before,
                                context_after=original_table.context_after,
                                bbox=getattr(original_table, "bbox", None),
                            )

                            logger.info(
                                f"Table {idx} : Smart Merge Docling+Vision appliqué "
                                f"({len(merged_result.rows)} lignes, confiance {merged_result.confidence:.2f})"
                            )
                    except Exception as e:
                        logger.warning(f"Fallback Vision échoué: {e}")

            except Exception as e:
                logger.error(f"Fallback complet échoué pour table {idx}: {e}")

        return tables


# -----------------------------------------------------------------------------
# Fonctions utilitaires d'extraction (API publique)
# -----------------------------------------------------------------------------


def extract_pdf(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    use_ocr: bool = False,
    enhance_images: bool = True,
    use_vision_fallback: bool = False,  # Désactivé pour le moment
    page_ranges: list[tuple[int, int]] | None = None,
) -> ExtractedDocument:
    """Extraire tout le contenu d'un PDF."""
    processor = DoclingProcessor(
        use_ocr=use_ocr,
        enhance_images=enhance_images,
        use_vision_fallback=use_vision_fallback,
    )
    return processor.extract_document(pdf_path, bank_code, quarter, year, page_ranges)


def extract_pdf_targeted(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]],
    use_ocr: bool = False,
) -> ExtractedDocument:
    """Extraire des pages specifiques d'un PDF."""
    return extract_pdf(
        pdf_path,
        bank_code,
        quarter,
        year,
        use_ocr=use_ocr,
        use_vision_fallback=False,
        page_ranges=page_ranges,
    )


def extract_pdf_with_fallback(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    use_ocr: bool = False,
) -> ExtractedDocument:
    """Alias pour extract_pdf (fallback Vision désactivé pour le moment)."""
    return extract_pdf(
        pdf_path, bank_code, quarter, year, use_ocr=use_ocr, use_vision_fallback=False
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
    """Extraire le contenu d'une section specifique."""
    doc = extract_pdf_targeted(
        pdf_path, bank_code, quarter, year, page_ranges=[(start_page, end_page)]
    )

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
) -> list[ExtractedTable]:
    """Extract tables on selected section ranges and tag them with section names."""
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
    )

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

    return tables


def extract_tables_docling_priority(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]] | None = None,
) -> list[ExtractedTable]:
    """Extraire uniquement les tableaux via Docling."""
    doc = extract_pdf(
        pdf_path, bank_code, quarter, year, page_ranges=page_ranges, use_vision_fallback=False
    )
    return doc.all_tables


def extract_tables_with_context(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]] | None = None,
) -> list[ExtractedTable]:
    """Extraire les tableaux avec contexte enrichi."""
    return extract_tables_docling_priority(pdf_path, bank_code, quarter, year, page_ranges)
