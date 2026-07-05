"""Package du pipeline texte canonique."""

from __future__ import annotations

from .text_comparison_excel import generate_text_comparison_excel, generate_text_vigie_excel
from .text_comparison_writer import get_text_comparison_path, load_text_comparison, write_text_comparison

__all__ = [
    "generate_text_comparison_excel",
    "generate_text_vigie_excel",
    "get_text_comparison_path",
    "load_text_comparison",
    "write_text_comparison",
]
