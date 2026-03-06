"""Public extraction surface for the canonical Docling + Vision pipeline."""

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
    "IndicatorSplitResult",
    "split_table_rows",
    # TableImageExtractor
    "TableImageExtractor",
    "TableImage",
    "ExtractionResult",
    "extract_tables_as_images",
    "get_table_images_for_comparison",
]
