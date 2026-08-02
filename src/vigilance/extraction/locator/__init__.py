"""Modules issus du decoupage de ``section_locator.py``.

Le decoupage est mene par etapes : chaque module regroupe une responsabilite
extraite du monolithe, sans changement de comportement. ``section_locator``
reste la facade publique et re-exporte tout ce qui etait accessible avant.
"""

from .models import (
    SHARED_PAGE_TOP_THRESHOLD,
    LocatedSection,
    SectionMapping,
    TocEntry,
    VisualTextElement,
    normalize_text,
)
from .patterns import (
    FOLLOWING_SECTION_PATTERNS,
    RISK_SUBSECTIONS,
    SECTION_PATTERNS,
    SECTION_TITLE_ALIASES,
    T4_SECTION_TITLE_PROFILES,
    TOC_PATTERNS,
)

__all__ = [
    "FOLLOWING_SECTION_PATTERNS",
    "RISK_SUBSECTIONS",
    "SECTION_PATTERNS",
    "SECTION_TITLE_ALIASES",
    "SHARED_PAGE_TOP_THRESHOLD",
    "T4_SECTION_TITLE_PROFILES",
    "TOC_PATTERNS",
    "LocatedSection",
    "SectionMapping",
    "TocEntry",
    "VisualTextElement",
    "normalize_text",
]
