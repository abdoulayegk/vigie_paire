"""Package des sous-composants spécialisés d'extraction (footnotes, sections, tables)."""

from __future__ import annotations

from vigilance.extraction.components.footnote_extractor import extract_clean_footnotes
from vigilance.extraction.components.section_detector import detect_section_key
from vigilance.extraction.components.table_extractor import validate_table_structure

__all__ = [
    "extract_clean_footnotes",
    "detect_section_key",
    "validate_table_structure",
]
