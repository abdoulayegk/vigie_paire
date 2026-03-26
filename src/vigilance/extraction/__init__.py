"""Surface publique minimale du module d'extraction actif.

Expose uniquement les points d'entrée encore utilisés par le pipeline
canonique Vision full extraction et par la localisation des sections.
"""

try:
    from .docling_processor import (
        DoclingProcessor,
        extract_pdf,
        extract_pdf_targeted,
        extract_section_content,
        extract_tables_docling_by_sections,
        extract_tables_docling_priority,
    )
except ImportError:
    DoclingProcessor = None
    extract_pdf = None
    extract_pdf_targeted = None
    extract_section_content = None
    extract_tables_docling_by_sections = None
    extract_tables_docling_priority = None
from .section_locator import SectionLocator, locate_sections_in_pdf

__all__ = [
    "DoclingProcessor",
    "extract_pdf",
    "extract_pdf_targeted",
    "extract_section_content",
    "extract_tables_docling_by_sections",
    "extract_tables_docling_priority",
    "SectionLocator",
    "locate_sections_in_pdf",
]
