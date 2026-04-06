"""Package d'extraction de blocs de texte par section."""

from __future__ import annotations

from .text_extractor import TextBlock, TextExtractor, extract_text_blocks_by_sections

__all__ = [
    "TextBlock",
    "TextExtractor",
    "extract_text_blocks_by_sections",
]
