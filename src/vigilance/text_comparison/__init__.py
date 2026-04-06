"""Package de comparaison de blocs de texte entre trimestres."""

from __future__ import annotations

from .text_comparison_builder import build_text_comparison
from .text_comparison_writer import get_text_comparison_path, load_text_comparison, write_text_comparison
from .text_diff_gpt import run_section_diff
from .text_triage import run_text_triage

__all__ = [
    "build_text_comparison",
    "get_text_comparison_path",
    "load_text_comparison",
    "write_text_comparison",
    "run_section_diff",
    "run_text_triage",
]
