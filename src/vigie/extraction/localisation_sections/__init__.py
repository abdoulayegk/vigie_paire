"""Localisation des sections cibles dans les rapports bancaires."""

from vigie.extraction.localisation_sections.bank_config import (
    BANK_SECTION_NAMES,
    _get_bank_section_names,
    _load_bank_config,
)
from vigie.extraction.localisation_sections.models import (
    SHARED_PAGE_TOP_THRESHOLD,
    LocatedSection,
    SectionMapping,
    TocEntry,
    VisualTextElement,
    normalize_text,
)
from vigie.extraction.localisation_sections.patterns import (
    FOLLOWING_SECTION_PATTERNS,
    RISK_SUBSECTIONS,
    SECTION_PATTERNS,
    SECTION_TITLE_ALIASES,
    T4_SECTION_TITLE_PROFILES,
    TOC_PATTERNS,
)
from vigie.extraction.localisation_sections.section_locator import (
    SectionLocator,
    locate_sections_in_pdf,
)

__all__ = [
    "BANK_SECTION_NAMES",
    "FOLLOWING_SECTION_PATTERNS",
    "RISK_SUBSECTIONS",
    "SECTION_PATTERNS",
    "SECTION_TITLE_ALIASES",
    "SHARED_PAGE_TOP_THRESHOLD",
    "T4_SECTION_TITLE_PROFILES",
    "TOC_PATTERNS",
    "LocatedSection",
    "SectionLocator",
    "SectionMapping",
    "TocEntry",
    "VisualTextElement",
    "_get_bank_section_names",
    "_load_bank_config",
    "locate_sections_in_pdf",
    "normalize_text",
]
