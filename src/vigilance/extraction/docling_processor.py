"""
Processeur PDF base sur Docling pour l'extraction de contenu structure des rapports bancaires.
Outil principal d'extraction pour les tableaux, le texte et la structure des documents.

Pipeline d'extraction:
1. Docling (structure: detection tableaux + bbox + page)
2. Pour chaque tableau: crop (bbox + extension) -> GPT-4o Vision (contenu)
3. Si Docling indisponible: document vide (pas de fallback Vision full-page)

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz  # Import PyMuPDF (fitz)

from ..utils.footnotes_utils import normalize_footnotes_to_canonical
from ..utils.genai import get_openai_api_key
from ..utils.indicator_cleaner import (
    normalize_indicator_for_comparison,
)
from ..utils.matching_normalizer import (
    strip_temporal_expressions,
)
from ..utils.rbc_table_signals import (
    classify_rbc_title_reliability,
    is_rbc_bank,
    is_unreliable_rbc_title,
)
from .docling_normalization import (
    _extract_table_context_split,
    # _is_footnote_row and _merge_fragmented_cells removed: Docling content no longer extracted.
)
from .page_title_assistant import PageTitleAssistant, PageTitleResult
from .table_title_resolver import (
    extract_table_number_and_inline_title,
    is_table_number_line,
    is_unit_context_line,
    resolve_title_from_lines,
)

logger = logging.getLogger(__name__)

_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}


def _env_bool(*names: str) -> bool | None:
    """Parse bool-like env var from the first provided var that exists."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in _ENV_TRUE:
            return True
        if value in _ENV_FALSE:
            return False
    return None


def _resolve_vision_extraction_enabled(bank_code: str, explicit: bool | None) -> bool:
    """Resolution order: explicit arg > env > bank config."""
    if explicit is not None:
        return bool(explicit)

    env_choice = _env_bool("VIGILANCE_VISION_EXTRACTION_ENABLED")
    if env_choice is not None:
        return env_choice

    try:
        from ..config import get_vision_extraction_config

        cfg = get_vision_extraction_config(bank_code=bank_code) or {}
        if "enabled" in cfg:
            return bool(cfg.get("enabled"))
    except Exception:
        pass
    return False


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

# Nombre de workers pour parallelliser les appels Vision (1 = sequentiel)
VISION_EXTRACTION_MAX_WORKERS = 4

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
    footnotes: list[dict[str, str]] = field(default_factory=list)
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
    first_column_indicators_spatial: list[dict[str, Any]] | None = (
        None  # Liste d'objets avec texte brut et bbox: [{"text": str, "bbox": [l, t, r, b]}]
    )
    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    title_reliability: str | None = None
    debug_metrics: dict[str, Any] = field(
        default_factory=dict
    )  # row_count, merge_count, etc.
    extraction_method: str | None = None  # docling | vision_fallback_gpt4o
    fragmentation_detected: bool = False

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


def _compute_vision_quality_summary(tables: list[Any]) -> dict[str, Any]:
    """Aggregate per-table debug_metrics into a single quality summary dict."""
    total = len(tables)
    attempted = 0
    ok = 0
    partial = 0
    failed = 0
    truncated = 0
    low_confidence = 0
    no_reference_text = 0
    recrop_used_count = 0
    bbox_rejected = 0

    for t in tables:
        dm = getattr(t, "debug_metrics", None)
        if not isinstance(dm, dict):
            continue
        if dm.get("vision_extraction_attempted"):
            attempted += 1
        status = dm.get("vision_status", "")
        if status == "ok":
            ok += 1
        elif status == "partial":
            partial += 1
        elif status == "failed":
            failed += 1
        if dm.get("appears_truncated"):
            truncated += 1
        conf = dm.get("vision_extraction_confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < 0.85 and status in ("ok", "partial"):
            low_confidence += 1
        if dm.get("crop_reject_reason"):
            bbox_rejected += 1
        if dm.get("recrop_used"):
            recrop_used_count += 1
        if not dm.get("has_reference_text") and dm.get("vision_extraction_attempted"):
            no_reference_text += 1

    return {
        "total_tables": total,
        "attempted": attempted,
        "ok": ok,
        "partial": partial,
        "failed": failed,
        "truncated": truncated,
        "low_confidence": low_confidence,
        "no_reference_text": no_reference_text,
        "recrop_used": recrop_used_count,
        "bbox_rejected": bbox_rejected,
    }


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
        openai_api_key: str | None = None,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ):
        """
        Initialiser le processeur Docling.

        Args:
            use_ocr: Activer l'OCR pour les documents numerises
            enhance_images: Appliquer l'amelioration d'image avant le traitement
            openai_api_key: Cle API OpenAI pour Vision (contenu par tableau)
            use_cache: Activer le cache des extractions (defaut False)
            cache_dir: Repertoire du cache (optionnel)
        """
        self.use_ocr = use_ocr
        self.enhance_images = enhance_images
        self.openai_api_key = openai_api_key
        self._converter = None
        self._initialized = False

        # Charger les patterns configurables
        self.bank_code_for_patterns: str | None = (
            None  # Sera set lors de extract_document
        )
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
            pipeline_options.do_picture_description = (
                False  # Desactiver pour acceleration
            )

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
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self._initialized = True
            logger.info(
                "Convertisseur Docling initialise (OCR=%s, threads=%s, device=%s)",
                self.use_ocr,
                num_threads,
                device_str,
            )

        except ImportError as e:
            logger.warning(
                f"Docling non disponible: {e}. Utilisation de l'extraction de secours."
            )
            self._initialized = True
        except Exception as e:
            logger.warning(
                f"Erreur initialisation Docling: {e}. Utilisation de l'extraction de secours."
            )
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
            use_vision_extraction: Si True, Vision (GPT-4o) comme source contenu pour tous les tableaux.
                Si None, lu depuis config vision_extraction.enabled.

        Returns:
            ExtractedDocument avec tout le contenu extrait
        """
        use_vision_extraction = _resolve_vision_extraction_enabled(
            bank_code, use_vision_extraction
        )

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
                logger.warning(
                    f"Impossible de recharger les patterns pour {bank_code}: {e}"
                )

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
            result = self._docling_unavailable_document(
                pdf_path, bank_code, quarter, year, page_ranges
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
                    pdf_path,
                    bank_code,
                    quarter,
                    year,
                    page_ranges,
                    labels_only=labels_only,
                    use_vision_extraction=use_vision_extraction,
                )
            else:
                chunk_result = self._docling_unavailable_document(
                    pdf_path, bank_code, quarter, year, page_ranges
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
                    first_column_groups=t.get("first_column_groups"),
                    hierarchical_indicator_signature=t.get(
                        "hierarchical_indicator_signature"
                    ),
                    title_reliability=t.get("title_reliability"),
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
                        first_column_indicators_raw=t.get(
                            "first_column_indicators_raw"
                        ),
                        footnotes=normalize_footnotes_to_canonical(
                            t.get("footnotes", [])
                        ),
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
                        first_column_groups=t.get("first_column_groups"),
                        hierarchical_indicator_signature=t.get(
                            "hierarchical_indicator_signature"
                        ),
                        title_reliability=t.get("title_reliability"),
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

    def _is_page_in_ranges(
        self, page_num: int, page_ranges: list[tuple[int, int]] | None
    ) -> bool:
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
                logger.warning(
                    "Plage de pages invalide ignoree a l'index %s: %s", idx, page_range
                )
                continue

            start_norm = max(1, start)
            if start_norm != start:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: start=%s -> %s",
                    idx,
                    start,
                    start_norm,
                )

            end_norm = max(start_norm, end)
            if end_norm != end:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: end=%s -> %s",
                    idx,
                    end,
                    end_norm,
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

    def _vision_extract_one_table(
        self,
        item: tuple[int, int, list[float] | None, str, str | None],
        shared: dict[str, Any],
    ) -> tuple[int, ExtractedTable, int]:
        """Extrait un tableau via Vision (crop + API). Retourne (idx, ExtractedTable, page_num)."""
        idx, page_num, table_bbox, table_id, reference_text = item
        pdf_path = shared["pdf_path"]
        bank_code = shared["bank_code"]
        quarter = shared["quarter"]
        year = shared["year"]
        pdf_sha = shared["pdf_sha"]
        vision_extraction_cfg = shared["vision_extraction_cfg"]
        bottom_extension_footnotes = shared["bottom_extension_footnotes"]
        top_extension_title = shared["top_extension_title"]
        horizontal_padding = shared["horizontal_padding"]
        adaptive_bottom_enabled = shared["adaptive_bottom_extension_enabled"]
        adaptive_bottom_increment = shared["adaptive_bottom_extension_increment"]
        vision_extractor = shared["vision_extractor"]
        schema_failure_flag = shared["schema_failure_flag"]
        vision_schema_error_cls = shared["vision_schema_error_cls"]
        schema_failure_policy = shared["schema_failure_policy"]
        labels_only = shared["labels_only"]
        vision_crop_dpi: int = shared.get("vision_crop_dpi", 300)
        vision_preprocess: bool | None = shared.get("vision_preprocess")
        vision_model_name: str | None = shared.get("vision_model_name")

        vision_status_str = "failed"
        warnings_list: list[str] = []
        title = ""
        table_number: str | None = None
        title_clean: str | None = None
        title_raw: str | None = None
        out_headers: list[str] = []
        out_rows: list[list[str]] = []
        indicators_raw_text: list[str] = []
        indicators: list[str] = []
        indicators_spatial_raw: list[Any] = []
        footnotes: list[dict] = []
        vision_confidence = 0.0
        vision_extraction_attempted = False
        vision_schema_contract_failed = False
        vision_extraction_disabled_reason: str | None = None
        crop_reject_reason: str | None = None
        bbox_sanity_profile: dict[str, Any] | None = None
        vision_result: Any = None

        if vision_extractor and table_bbox and len(table_bbox) == 4:
            vision_extraction_attempted = True
            try:
                from ..utils.pdf_crop import crop_table_region_to_bytes

                if schema_failure_flag[0]:
                    vision_extraction_attempted = False
                    vision_schema_contract_failed = True
                    warnings_list = ["Vision disabled after schema contract failure"]
                    vision_extraction_disabled_reason = shared.get(
                        "vision_extraction_disabled_reason"
                    )
                else:
                    from ..utils.pdf_crop import is_bbox_sane

                    sane, crop_reject_reason, bbox_sanity_profile = is_bbox_sane(
                        table_bbox, vision_extraction_cfg
                    )
                    if not sane:
                        vision_extraction_attempted = True
                        vision_status_str = "failed"
                        warnings_list = [
                            f"bbox sanity gate: {crop_reject_reason or 'rejected'}; Vision skipped"
                        ]
                    else:
                        # Adaptive bottom: larger extension when table bbox is far from page bottom (footnotes likely)
                        initial_bottom_ext = bottom_extension_footnotes
                        if (
                            adaptive_bottom_enabled
                            and table_bbox
                            and len(table_bbox) >= 4
                            and table_bbox[3] < 0.85
                        ):
                            initial_bottom_ext = min(
                                1.0 - table_bbox[3],
                                bottom_extension_footnotes + adaptive_bottom_increment,
                            )

                        def _recrop(ext: float) -> bytes:
                            return crop_table_region_to_bytes(
                                str(pdf_path),
                                page_num,
                                table_bbox,
                                bottom_extension=ext,
                                top_extension=top_extension_title,
                                horizontal_padding=horizontal_padding,
                                dpi=vision_crop_dpi,
                            )

                        crop_bytes = _recrop(initial_bottom_ext)
                        if not crop_bytes:
                            vision_extraction_attempted = True
                            vision_status_str = "failed"
                            warnings_list = [
                                "crop rejected or empty; Vision skipped (invalid bbox, page, or crop failure)"
                            ]
                        else:
                            vision_result = vision_extractor.extract_with_quality_pass(
                                crop_bytes=crop_bytes,
                                bank_code=bank_code,
                                pdf_sha=pdf_sha,
                                page_number=page_num,
                                bbox_norm=table_bbox,
                                vision_cfg=vision_extraction_cfg,
                                initial_bottom_extension=initial_bottom_ext,
                                get_recrop_fn=_recrop,
                                reference_text=reference_text,
                            )
                            if vision_result is not None:
                                title = vision_result.table_title or ""
                                table_number, title_clean = self._extract_table_number(
                                    title or None
                                )
                                title_raw = title or None
                                out_headers = (
                                    [] if labels_only else (vision_result.headers or [])
                                )
                                out_rows = (
                                    [] if labels_only else (vision_result.rows or [])
                                )
                                indicators_spatial_raw = list(
                                    vision_result.indicators or []
                                )
                                indicators_raw_text = [
                                    item.get("text", "")
                                    if isinstance(item, dict)
                                    else str(item)
                                    for item in indicators_spatial_raw
                                ]
                                indicators = [
                                    normalize_indicator_for_comparison(text)
                                    for text in indicators_raw_text
                                ]
                                footnotes = (
                                    []
                                    if labels_only
                                    else vision_result.to_footnotes_list()
                                )
                                vision_confidence = vision_result.confidence
                                vision_status_str = vision_result.vision_status or "ok"
                                warnings_list = list(vision_result.warnings or [])
                            else:
                                vision_status_str = "failed"
                                warnings_list = ["VisionFullExtractor returned None"]
            except BaseException as e:
                if type(e) is vision_schema_error_cls:
                    reason = f"Vision schema contract invalid: {e}"
                    if schema_failure_policy == "degrade_to_docling":
                        schema_failure_flag[0] = True
                        shared["vision_extraction_disabled_reason"] = reason
                        vision_status_str = "failed"
                        warnings_list = [reason[:300]]
                        vision_schema_contract_failed = True
                        vision_extraction_disabled_reason = reason
                    else:
                        raise RuntimeError(reason) from e
                else:
                    vision_status_str = "failed"
                    warnings_list = [str(e)[:300]]
        else:
            if not vision_extractor:
                warnings_list = ["no vision extractor (API key missing or init failed)"]
            elif not table_bbox:
                warnings_list = ["no bbox from Docling"]
            else:
                warnings_list = ["bbox invalid"]

        requested_max_completion_tokens = int(
            vision_extraction_cfg.get("vision_max_completion_tokens", 65536)
        )
        debug_metrics: dict[str, Any] = {
            "vision_status": vision_status_str,
            "vision_extraction_attempted": vision_extraction_attempted,
            "vision_extraction_applied": vision_status_str in ("ok", "partial"),
            "vision_extraction_confidence": vision_confidence,
            "vision_schema_contract_failed": vision_schema_contract_failed,
            "has_reference_text": bool(reference_text and len(reference_text.strip()) > 20),
            "warnings": warnings_list,
            "vision_max_completion_tokens_requested": requested_max_completion_tokens,
            "vision_max_completion_tokens_rescue_used": False,
        }
        if vision_model_name:
            debug_metrics["vision_model"] = vision_model_name
            debug_metrics["vision_role"] = "extraction_primary"
        if warnings_list:
            debug_metrics["vision_warning_codes"] = list(warnings_list)
            known_failure_codes = {
                "vision_truncated",
                "vision_invalid_json",
                "vision_schema_validation_failed",
                "vision_retry_exhausted",
                "vision_transport_error",
                "vision_structured_output_fallback",
                "vision_lean_mode",
                "vision_rows_missing_from_fallback",
            }
            failure_causes = [
                code for code in warnings_list if code in known_failure_codes
            ]
            if failure_causes:
                debug_metrics["vision_failure_causes"] = failure_causes
        if vision_extraction_disabled_reason:
            debug_metrics["vision_extraction_disabled_reason"] = (
                vision_extraction_disabled_reason
            )
        if crop_reject_reason:
            debug_metrics["crop_reject_reason"] = crop_reject_reason
        if bbox_sanity_profile is not None:
            debug_metrics["bbox_sanity_profile"] = bbox_sanity_profile
        # Recrop and completeness (from vision result when available)
        if vision_result is not None:
            if hasattr(vision_result, "appears_truncated"):
                debug_metrics["appears_truncated"] = vision_result.appears_truncated
            if hasattr(vision_result, "recrop_attempted"):
                debug_metrics["recrop_attempted"] = vision_result.recrop_attempted
            if hasattr(vision_result, "recrop_used"):
                debug_metrics["recrop_used"] = vision_result.recrop_used
            if hasattr(vision_result, "recrop_failed_incomplete"):
                debug_metrics["recrop_failed_incomplete"] = (
                    vision_result.recrop_failed_incomplete
                )
            if getattr(vision_result, "requested_max_completion_tokens", None) is not None:
                debug_metrics["vision_max_completion_tokens_requested"] = (
                    vision_result.requested_max_completion_tokens
                )
            debug_metrics["vision_max_completion_tokens_rescue_used"] = bool(
                getattr(vision_result, "rescue_used", False)
            )
            if getattr(vision_result, "finish_reason", None):
                debug_metrics["vision_finish_reason"] = vision_result.finish_reason
            if getattr(vision_result, "prompt_tokens", None) is not None:
                debug_metrics["vision_prompt_tokens"] = vision_result.prompt_tokens
            if getattr(vision_result, "completion_tokens", None) is not None:
                debug_metrics["vision_completion_tokens"] = (
                    vision_result.completion_tokens
                )
            if getattr(vision_result, "total_tokens", None) is not None:
                debug_metrics["vision_total_tokens"] = vision_result.total_tokens

        extracted_table = ExtractedTable(
            table_id=table_id,
            page_number=page_num,
            title=title or None,
            headers=out_headers,
            rows=out_rows,
            first_column_indicators=indicators,
            first_column_indicators_raw=indicators_raw_text,
            first_column_indicators_spatial=indicators_spatial_raw
            if indicators_spatial_raw
            else None,
            footnotes=footnotes,
            bbox=table_bbox,
            table_number=table_number,
            title_clean=title_clean,
            title_raw=title_raw,
            title_reliability=classify_rbc_title_reliability(
                title_clean or title or title_raw,
                bank_code=bank_code,
            ),
            extraction_method=(
                "vision_full_gpt4o"
                if vision_status_str in ("ok", "partial")
                else "vision_failed"
            ),
            debug_metrics=debug_metrics,
        )
        return (idx, extracted_table, page_num)

    def _extract_with_docling(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        labels_only: bool = False,
        use_vision_extraction: bool = False,
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

            def _get_vision_extraction_config(bank: str) -> dict:
                try:
                    from ..config import get_vision_extraction_config as _gvec

                    return _gvec(bank_code=bank) or {}
                except Exception:
                    return {}

            # Vision extraction: OpenAI Vision as content source (indicators + footnotes) for all tables
            vision_extraction_cfg: dict = {}
            bottom_extension_footnotes = 0.0
            top_extension_title = 0.03
            horizontal_padding = 0.02
            adaptive_bottom_extension_enabled = False
            adaptive_bottom_extension_increment = 0.06
            # fallback_to_docling removed: Vision is the sole content source (Rules 1+5)
            schema_failure_policy = "fail_fast"
            vision_extractor = None
            vision_model_name: str | None = None
            pdf_sha = ""
            vision_extraction_disabled_reason: str | None = None
            vision_schema_contract_failed = False
            vision_schema_error_cls: type[Exception] = Exception
            if use_vision_extraction:
                try:
                    vision_extraction_cfg = _get_vision_extraction_config(bank_code)
                    bottom_extension_footnotes = float(
                        vision_extraction_cfg.get("bottom_extension_footnotes", 0.12)
                    )
                    top_extension_title = float(
                        vision_extraction_cfg.get("top_extension_title", 0.03)
                    )
                    horizontal_padding = float(
                        vision_extraction_cfg.get("horizontal_padding", 0.02)
                    )
                    adaptive_bottom_extension_enabled = bool(
                        vision_extraction_cfg.get(
                            "adaptive_bottom_extension_enabled", False
                        )
                    )
                    adaptive_bottom_extension_increment = float(
                        vision_extraction_cfg.get(
                            "adaptive_bottom_extension_increment", 0.06
                        )
                    )
                    # fallback_to_docling removed: Vision is the sole content source (Rules 1+5)
                    schema_failure_policy = (
                        str(
                            vision_extraction_cfg.get(
                                "schema_failure_policy", "fail_fast"
                            )
                        )
                        .strip()
                        .lower()
                    )
                    if schema_failure_policy not in {
                        "fail_fast",
                        "degrade_to_docling",
                    }:
                        schema_failure_policy = "fail_fast"
                    from ..config import resolve_openai_model
                    from ..utils.genai import get_openai_api_key
                    from .vision_cache import compute_pdf_sha256
                    from .vision_full_extractor import (
                        VisionFullExtractor,
                        VisionSchemaContractError,
                    )

                    pdf_sha = compute_pdf_sha256(str(pdf_path))
                    api_key = self.openai_api_key or get_openai_api_key()
                    vision_model_name = resolve_openai_model("extraction_primary")
                    vision_cache_enabled = bool(
                        vision_extraction_cfg.get("vision_cache_enabled", True)
                    )
                    if api_key:
                        vision_extractor = VisionFullExtractor(
                            api_key=api_key,
                            model=vision_model_name,
                            use_cache=vision_cache_enabled,
                        )
                    else:
                        logger.warning(
                            "Vision extraction: OPENAI_API_KEY absente, desactivation"
                        )
                        use_vision_extraction = False
                    vision_schema_error_cls = VisionSchemaContractError
                except Exception as e:
                    logger.warning("Vision extraction init failed: %s", e)
                    use_vision_extraction = False

            # ---------------------------------------------------------------------------
            # Steps 2+3: Docling = structure only. Vision = single content source.
            # ---------------------------------------------------------------------------
            # Construire la liste des tableaux a traiter (dans les plages de pages).
            vision_items: list[
                tuple[int, int, list[float] | None, str, str | None]
            ] = []
            for idx, table in enumerate(doc.tables):
                page_num = table.prov[0].page_no if table.prov else 0
                table_bbox: list[float] | None = None
                try:
                    if (
                        table.prov
                        and hasattr(table.prov[0], "bbox")
                        and table.prov[0].bbox is not None
                    ):
                        raw_bbox = table.prov[0].bbox
                        page_obj = (
                            doc.pages.get(page_num) if hasattr(doc, "pages") else None
                        )
                        if page_obj and hasattr(page_obj, "size") and page_obj.size:
                            norm = raw_bbox.to_top_left_origin(
                                page_height=page_obj.size.height
                            )
                            norm = norm.normalized(page_obj.size)
                            table_bbox = [norm.l, norm.t, norm.r, norm.b]
                        elif hasattr(raw_bbox, "as_tuple"):
                            table_bbox = list(raw_bbox.as_tuple())
                except Exception:
                    table_bbox = None
                if not self._is_page_in_ranges(page_num, effective_page_ranges):
                    continue
                table_id = f"tableau_{idx}"
                reference_text: str | None = None
                try:
                    if hasattr(table, "text") and table.text:
                        _ref_raw = str(table.text).strip()
                        if len(_ref_raw) > 20:
                            ref_max_chars = int(
                                vision_extraction_cfg.get(
                                    "vision_reference_text_max_chars", 6000
                                )
                            )
                            if ref_max_chars > 0:
                                reference_text = _ref_raw[:ref_max_chars]
                except Exception:
                    pass
                vision_items.append(
                    (idx, page_num, table_bbox, table_id, reference_text)
                )

            def _bbox_area(b: list[float]) -> float:
                if not b or len(b) < 4:
                    return 0.0
                w = max(0.0, float(b[2]) - float(b[0]))
                h = max(0.0, float(b[3]) - float(b[1]))
                return w * h

            def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
                if len(a) < 4 or len(b) < 4:
                    return 0.0
                x0 = max(a[0], b[0])
                y0 = max(a[1], b[1])
                x1 = min(a[2], b[2])
                y1 = min(a[3], b[3])
                if x1 <= x0 or y1 <= y0:
                    return 0.0
                inter = (x1 - x0) * (y1 - y0)
                area_a = _bbox_area(a)
                area_b = _bbox_area(b)
                if area_a <= 0 or area_b <= 0:
                    return 0.0
                return inter / min(area_a, area_b)

            def _detect_overlapping_bboxes(
                items: list[tuple[int, int, list[float] | None, str, str | None]],
            ) -> list[tuple[int, int, int, float]]:
                by_page: dict[int, list[tuple[int, list[float]]]] = {}
                for idx, page_num, bbox, table_id, _ in items:
                    if bbox and len(bbox) >= 4:
                        by_page.setdefault(page_num, []).append((idx, bbox))
                overlaps: list[tuple[int, int, int, float]] = []
                for page_num, boxes in by_page.items():
                    if len(boxes) < 2:
                        continue
                    for i in range(len(boxes)):
                        for j in range(i + 1, len(boxes)):
                            idx_a, bbox_a = boxes[i]
                            idx_b, bbox_b = boxes[j]
                            ratio = _bbox_overlap_ratio(bbox_a, bbox_b)
                            if ratio > 0.01:
                                overlaps.append((page_num, idx_a, idx_b, ratio))
                return overlaps

            overlap_pairs = _detect_overlapping_bboxes(vision_items)
            if overlap_pairs:
                for page_num, idx_a, idx_b, ratio in overlap_pairs:
                    logger.warning(
                        "vision_extraction bbox_overlap page=%s idx=%s/%s ratio=%.3f",
                        page_num,
                        idx_a,
                        idx_b,
                        ratio,
                    )

            tables_per_page: dict[int, int] = {}
            for _idx, page_num, _bbox, _tid, _ref in vision_items:
                tables_per_page[page_num] = tables_per_page.get(page_num, 0) + 1
            if tables_per_page:
                logger.info(
                    "vision_extraction tables_detected_per_page %s",
                    dict(sorted(tables_per_page.items())),
                )

            all_tables = []
            tables_by_page: dict[int, int] = {}
            if vision_items:
                schema_failure_flag: list[bool] = [False]
                shared: dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "bank_code": bank_code,
                    "quarter": quarter,
                    "year": year,
                    "pdf_sha": pdf_sha,
                    "vision_extraction_cfg": vision_extraction_cfg,
                    "bottom_extension_footnotes": bottom_extension_footnotes,
                    "top_extension_title": top_extension_title,
                    "horizontal_padding": horizontal_padding,
                    "adaptive_bottom_extension_enabled": adaptive_bottom_extension_enabled,
                    "adaptive_bottom_extension_increment": adaptive_bottom_extension_increment,
                    "vision_extractor": vision_extractor,
                    "schema_failure_flag": schema_failure_flag,
                    "vision_schema_error_cls": vision_schema_error_cls,
                    "schema_failure_policy": schema_failure_policy,
                    "labels_only": labels_only,
                    "vision_crop_dpi": int(vision_extraction_cfg.get("vision_crop_dpi", 300)),
                    "vision_preprocess": vision_extraction_cfg.get("vision_preprocess", True),
                    "vision_model_name": vision_model_name,
                }
                if vision_extractor:
                    try:
                        vision_extractor.validate_schema()
                    except vision_schema_error_cls as e:
                        reason = str(e) or "Vision schema contract invalid"
                        if schema_failure_policy == "fail_fast":
                            raise
                        schema_failure_flag[0] = True
                        shared["vision_extraction_disabled_reason"] = reason
                vision_max_workers = int(
                    vision_extraction_cfg.get("vision_extraction_max_workers", 4)
                )
                max_workers = min(
                    max(1, vision_max_workers),
                    len(vision_items),
                )
                if max_workers <= 1:
                    for item in vision_items:
                        _idx, extracted_table, pnum = self._vision_extract_one_table(
                            item, shared
                        )
                        all_tables.append(extracted_table)
                        tables_by_page[pnum] = tables_by_page.get(pnum, 0) + 1
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                self._vision_extract_one_table,
                                item,
                                shared,
                            )
                            for item in vision_items
                        ]
                        results: list[tuple[int, ExtractedTable, int]] = []
                        for fut in futures:
                            try:
                                results.append(fut.result())
                            except Exception as exc:
                                if type(exc) is vision_schema_error_cls:
                                    raise
                                raise
                        results.sort(key=lambda x: x[0])
                        for _idx, extracted_table, pnum in results:
                            all_tables.append(extracted_table)
                            tables_by_page[pnum] = tables_by_page.get(pnum, 0) + 1
                    if max_workers > 1:
                        logger.info(
                            "Vision extraction parallele: %d tableaux, %d workers",
                            len(vision_items),
                            max_workers,
                        )

            if tables_by_page:
                counts_str = ", ".join(
                    f"p{k}:{v}" for k, v in sorted(tables_by_page.items())
                )
                logger.info("Docling tableaux par page: %s", counts_str)

            rejected_bbox_sanity = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("crop_reject_reason")
            )
            if rejected_bbox_sanity:
                logger.info(
                    "vision_extraction tables_rejected_bbox_sanity count=%s",
                    rejected_bbox_sanity,
                )
            recrop_attempted = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("recrop_attempted")
            )
            recrop_used = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("recrop_used")
            )
            if recrop_attempted or recrop_used:
                logger.info(
                    "vision_extraction recrop attempted=%s used=%s",
                    recrop_attempted,
                    recrop_used,
                )

            # --- Vision extraction quality summary (one log line per run) ---
            if all_tables:
                _qsum = _compute_vision_quality_summary(all_tables)
                logger.info("vision_extraction_quality_summary %s", _qsum)

            # Extraire le contenu textuel pour les sections
            text_content = doc.export_to_markdown()

            # Page-Level Title Assist: lightweight Vision pre-pass for missing titles
            all_tables = self._page_level_title_assist(
                all_tables,
                pdf_path,
                bank_code,
                vision_extraction_cfg,
            )

            # Enrichir les titres manquants depuis le texte de la page (pdfplumber)
            # sans melanger contenu Docling/Vision : seul le champ titre est complete.
            all_tables = self._enrich_tables_with_titles(all_tables, pdf_path)

            # Associer les tableaux à leurs sections parentes
            all_tables = self._associate_tables_with_sections(all_tables, text_content)

            # Compter les sections détectées
            sections_found = set(t.section for t in all_tables if t.section)
            if sections_found:
                logger.info("Sections détectées: %s", ", ".join(sections_found))

            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=len(doc.pages) if hasattr(doc, "pages") else 0,
                all_tables=all_tables,
                metadata={
                    "extraction_method": "vision_full_gpt4o",
                    "sections_detected": list(sections_found),
                    "page_ranges": page_ranges,
                    "text_content": text_content[:50000],
                },
            )

        except Exception as e:
            if "Vision schema contract invalid" in str(e):
                raise
            logger.error(
                "Echec de l'extraction Docling (%s): %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return self._docling_unavailable_document(
                pdf_path, bank_code, quarter, year, page_ranges, error=str(e)
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

    def _page_level_title_assist(
        self,
        tables: list[ExtractedTable],
        pdf_path: Path,
        bank_code: str,
        vision_extraction_cfg: dict[str, Any],
    ) -> list[ExtractedTable]:
        """Apply page-level title assist to fill missing/weak titles.

        A lightweight GPT-4o pass on the full page image extracts candidate titles.
        Only replaces a table's title if:
        - The current title is empty or has a low quality score
        - The candidate confidence meets the threshold
        """
        assist_cfg = vision_extraction_cfg.get("page_level_title_assist", {})
        if not isinstance(assist_cfg, dict):
            assist_cfg = {}
        if not assist_cfg.get("enabled", False):
            return tables

        min_confidence = float(assist_cfg.get("min_confidence", 0.7))
        max_candidates = int(assist_cfg.get("max_candidates", 10))
        weak_title_threshold = int(assist_cfg.get("weak_title_threshold", 3))
        allow_positional_fallback = bool(
            assist_cfg.get("allow_positional_fallback", False)
        )
        max_pages_per_run = int(assist_cfg.get("max_pages_per_run", 50))

        bank_is_rbc = is_rbc_bank(bank_code)

        # Identify pages with at least one weak/missing title
        pages_needing_assist: dict[int, list[ExtractedTable]] = {}
        for table in tables:
            score = self._title_quality_score(table.title)
            needs_assist = score < weak_title_threshold
            if bank_is_rbc and is_unreliable_rbc_title(
                table.title, bank_code=bank_code
            ):
                needs_assist = True
            if needs_assist:
                pages_needing_assist.setdefault(table.page_number, []).append(table)

        if not pages_needing_assist:
            return tables

        logger.info(
            "page_level_title_assist starting pages_count=%s",
            len(pages_needing_assist),
        )

        # Limit pages to process (budget)
        page_nums_sorted = sorted(pages_needing_assist.keys())[:max_pages_per_run]
        if len(page_nums_sorted) < len(pages_needing_assist):
            logger.debug(
                "Page-level title assist: limiting to %s pages (budget %s)",
                max_pages_per_run,
                len(pages_needing_assist),
            )

        try:
            api_key = self.openai_api_key or get_openai_api_key()
            if not api_key:
                logger.debug("Page-level title assist: no API key, skipping")
                return tables
            api_retry_max = int(vision_extraction_cfg.get("api_retry_max", 3))
            api_retry_backoff_ms = float(
                vision_extraction_cfg.get("api_retry_backoff_ms", 1000)
            )
            assistant = PageTitleAssistant(
                api_key=api_key,
                min_confidence=min_confidence,
                max_candidates=max_candidates,
                api_retry_max=api_retry_max,
                api_retry_backoff_ms=api_retry_backoff_ms,
            )
        except Exception as e:
            logger.debug("Page-level title assist init failed: %s", e)
            return tables

        for page_num in page_nums_sorted:
            page_tables = pages_needing_assist[page_num]
            try:
                from .pdf_preview import render_pdf_page

                page_bytes = render_pdf_page(
                    str(pdf_path), page_num, scale=2.0, format="png"
                )
                if not page_bytes:
                    continue

                result = assistant.extract_page_titles(page_bytes, page_num)
                if not result or not result.candidates:
                    continue

                self._apply_page_title_candidates(
                    page_tables,
                    result,
                    weak_title_threshold,
                    allow_positional_fallback,
                )
            except Exception as e:
                logger.debug(
                    "Page-level title assist failed for page %s: %s", page_num, e
                )

        return tables

    def _apply_page_title_candidates(
        self,
        page_tables: list[ExtractedTable],
        result: PageTitleResult,
        weak_title_threshold: int,
        allow_positional_fallback: bool = False,
    ) -> None:
        """Map page-level title candidates to tables and apply when appropriate.

        Only applies title when matched by exact table_number or bbox proximity,
        unless allow_positional_fallback is True (not recommended for multi-table pages).
        """
        used_candidates: set[int] = set()
        bank_code = self.bank_code_for_patterns

        for table in page_tables:
            current_score = self._title_quality_score(table.title)
            current_reliability = classify_rbc_title_reliability(
                table.title,
                bank_code=bank_code,
            )
            if (
                current_score >= weak_title_threshold
                and current_reliability == "reliable"
            ):
                continue

            candidate: dict[str, Any] | None = None
            candidate_idx: int | None = None
            match_method: str = ""

            # 1) Match by table_number
            table_num = str(table.table_number or "").strip()
            if table_num:
                for idx, c in enumerate(result.candidates):
                    if idx in used_candidates:
                        continue
                    if str(c.get("table_number", "")).strip() == table_num:
                        candidate = c
                        candidate_idx = idx
                        match_method = "table_number"
                        break

            # 2) Match by bbox proximity (title above table)
            if candidate is None and table.bbox:
                other_bboxes = [
                    t.bbox for t in page_tables
                    if t is not table and getattr(t, "bbox", None) and len(getattr(t, "bbox", [])) >= 4
                ]
                candidate = result.get_candidate_by_bbox_proximity(
                    table.bbox,
                    other_table_bboxes=other_bboxes if len(page_tables) > 1 else None,
                )
                if candidate is not None:
                    for idx, c in enumerate(result.candidates):
                        if c is candidate and idx not in used_candidates:
                            candidate_idx = idx
                            match_method = "bbox_proximity"
                            break
                    else:
                        candidate = None
                        candidate_idx = None

            # 3) Positional fallback (first unused candidate) only when allowed
            if candidate is None and allow_positional_fallback:
                for idx, c in enumerate(result.candidates):
                    if idx not in used_candidates:
                        candidate = c
                        candidate_idx = idx
                        match_method = "fallback"
                        break

            if candidate is None or candidate_idx is None:
                continue

            # Verify candidate quality is better than current
            candidate_title = str(
                candidate.get("title_semantic") or candidate.get("title_full") or ""
            ).strip()
            candidate_score = self._title_quality_score(candidate_title)
            candidate_reliability = classify_rbc_title_reliability(
                candidate_title,
                bank_code=bank_code,
            )
            better_candidate = candidate_score > current_score
            if is_rbc_bank(bank_code):
                better_candidate = better_candidate or (
                    current_reliability != "reliable"
                    and candidate_reliability == "reliable"
                )
            if not better_candidate:
                continue

            # Apply the candidate
            used_candidates.add(candidate_idx)
            table.title = candidate_title or table.title
            table.title_clean = candidate_title or table.title_clean
            table.title_raw = (
                str(candidate.get("title_full") or candidate_title).strip()
                or table.title_raw
            )

            candidate_number = str(candidate.get("table_number", "")).strip()
            if candidate_number and not table.table_number:
                table.table_number = candidate_number

            table.title_reliability = candidate_reliability

            table.title_resolution_method = (
                f"page_level_assist (conf={candidate.get('confidence', 0):.2f})"
            )

            if not table.debug_metrics:
                table.debug_metrics = {}
            table.debug_metrics["page_title_assist_used"] = True
            table.debug_metrics["page_title_assist_match_method"] = match_method
            if match_method == "bbox_proximity" and len(page_tables) > 1:
                table.debug_metrics["page_title_assist_multi_table_guard"] = True

            logger.info(
                "Page-level title assist: table %s p%s -> '%s' (conf=%.2f)",
                table.table_id,
                table.page_number,
                candidate_title[:60],
                candidate.get("confidence", 0),
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
                strip_temporal_expressions(
                    inline, target="title", aggressive=True
                ).strip()
            )
            if inline_temporal:
                score += 1

        temporal_free = strip_temporal_expressions(
            value, target="title", aggressive=True
        )
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

        number_indices = [
            idx for idx, line in enumerate(lines) if is_table_number_line(line)
        ]
        candidates: list[dict[str, str]] = []

        if not number_indices:
            candidate = self._resolve_title_metadata_from_lines(lines)
            if (
                candidate.get("title")
                or candidate.get("table_number")
                or candidate.get("title_raw")
            ):
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
                if (
                    bank_code == "cibc"
                    and len(page_tables) >= 1
                    and len(candidates) <= 1
                ):
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
                    # Ne pas ecraser le contenu fourni par Vision (seule source de verite).
                    if getattr(table, "extraction_method", None) == "vision_full_gpt4o":
                        continue
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
                    candidate_method = (
                        candidate.get("resolution_method") or ""
                    ).strip()

                    # On remplace si le candidat est clairement meilleur semantiquement.
                    current_title = (table.title or "").strip()
                    if self._title_quality_score(
                        candidate_title
                    ) > self._title_quality_score(current_title):
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
                    cb, ca = _extract_table_context_split(
                        text, table.title or table.title_clean
                    )
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
                if section["line_num"] < (
                    table.page_number * 50
                ):  # Estimation ~50 lignes/page
                    best_section = section

            if best_section:
                table.section = best_section["name"]
                table.section_phase = best_section["phase"]

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
    page_ranges: list[tuple[int, int]] | None = None,
    use_vision_extraction: bool | None = None,
) -> ExtractedDocument:
    """Extraire tout le contenu d'un PDF (Docling structure + Vision par tableau)."""
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
    """Extraire des pages specifiques d'un PDF."""
    return extract_pdf(
        pdf_path,
        bank_code,
        quarter,
        year,
        use_ocr=use_ocr,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )


def extract_pdf_with_fallback(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    use_ocr: bool = False,
) -> ExtractedDocument:
    """Alias pour extract_pdf (compatibilite API)."""
    return extract_pdf(pdf_path, bank_code, quarter, year, use_ocr=use_ocr)


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
    use_vision_extraction: bool | None = None,
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
        use_vision_extraction=use_vision_extraction,
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

        write_extraction_debug(
            bank=bank_code, quarter=quarter, year=year, tables=tables
        )
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


def extract_tables_docling_priority(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]] | None = None,
    use_vision_extraction: bool | None = None,
) -> list[ExtractedTable]:
    """Extraire uniquement les tableaux (Docling structure + Vision par tableau)."""
    doc = extract_pdf(
        pdf_path,
        bank_code,
        quarter,
        year,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )
    return doc.all_tables


def extract_tables_with_context(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    page_ranges: list[tuple[int, int]] | None = None,
    use_vision_extraction: bool | None = None,
) -> list[ExtractedTable]:
    """Extraire les tableaux avec contexte enrichi."""
    return extract_tables_docling_priority(
        pdf_path,
        bank_code,
        quarter,
        year,
        page_ranges=page_ranges,
        use_vision_extraction=use_vision_extraction,
    )
