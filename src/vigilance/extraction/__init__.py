"""
Module d'extraction pour le traitement des PDF avec Docling.
Gere l'analyse des documents, la detection des sections et l'extraction des tableaux.

Focus sur les sections cibles:
- Gestion du capital / Fonds propres
- Gestion des risques

Includes:
- DoclingProcessor: Extraction principale avec fallback
- SectionLocator: Localisation automatique des sections cibles
- GPT4VisionFallback: Extraction via GPT-4 Vision pour cas complexes
"""

try:
    from .docling_processor import (
        DoclingProcessor,
        extract_pdf,
        extract_pdf_targeted,
        extract_pdf_with_fallback,
        extract_section_content,
        extract_tables_docling_by_sections,
        extract_tables_docling_priority,
        extract_tables_with_context,
    )
except ImportError:
    DoclingProcessor = None
    extract_pdf = None
    extract_pdf_targeted = None
    extract_pdf_with_fallback = None
    extract_section_content = None
    extract_tables_docling_by_sections = None
    extract_tables_docling_priority = None
    extract_tables_with_context = None
from .indicator_splitter import IndicatorSplitResult, split_table_rows
from .section_detector import SectionDetector
from .section_locator import SectionLocator, locate_sections_in_pdf
from .table_extractor import TableExtractor

# Imports optionnels (peuvent échouer si dépendances manquantes)
try:
    from .gpt4_vision_fallback import (
        GPT4VisionFallback,
        get_vision_fallback,
    )
except ImportError:
    GPT4VisionFallback = None
    get_vision_fallback = None

# Import TableImageExtractor pour extraction images
try:
    from .table_image_extractor import (
        ExtractionResult,
        TableImage,
        TableImageExtractor,
        extract_tables_as_images,
        get_table_images_for_comparison,
    )
except ImportError:
    TableImageExtractor = None
    TableImage = None
    ExtractionResult = None

# Import VisionTableExtractor (Vision-Native pipeline)
try:
    from .vision_table_extractor import (
        DetectedTable,
        ExtractedTableData,
        ExtractionPipelineResult,
        VisionTableExtractor,
        extract_tables_vision_native,
    )
except ImportError:
    VisionTableExtractor = None
    extract_tables_vision_native = None
    DetectedTable = None
    ExtractedTableData = None
    ExtractionPipelineResult = None

__all__ = [
    "DoclingProcessor",
    "extract_pdf",
    "extract_pdf_with_fallback",
    "extract_pdf_targeted",
    "extract_section_content",
    "extract_tables_docling_by_sections",
    "extract_tables_with_context",
    "extract_tables_docling_priority",
    "SectionDetector",
    "SectionLocator",
    "locate_sections_in_pdf",
    "TableExtractor",
    "IndicatorSplitResult",
    "split_table_rows",
    "GPT4VisionFallback",
    "get_vision_fallback",
    # TableImageExtractor
    "TableImageExtractor",
    "TableImage",
    "ExtractionResult",
    "extract_tables_as_images",
    "get_table_images_for_comparison",
]
