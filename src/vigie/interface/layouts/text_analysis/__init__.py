"""Layouts de l'onglet Analyse Textuelle."""

from vigie.interface.layouts.text_analysis.change_card import (
    _build_change_card,
    _build_side_by_side,
)
from vigie.interface.layouts.text_analysis.highlight import (
    _HIGHLIGHT_ADDED_STYLE,
    _HIGHLIGHT_REMOVED_STYLE,
    _highlight_text,
)
from vigie.interface.layouts.text_analysis.page import (
    build_filtered_text_cards,
    build_text_analysis_tab,
)

__all__ = [
    "_HIGHLIGHT_ADDED_STYLE",
    "_HIGHLIGHT_REMOVED_STYLE",
    "_build_change_card",
    "_build_side_by_side",
    "_highlight_text",
    "build_filtered_text_cards",
    "build_text_analysis_tab",
]
