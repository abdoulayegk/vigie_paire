"""Package des sous-composants de mise en page pour l'analyse textuelle (text_filters, text_cards, text_panels)."""

from __future__ import annotations

from vigilance.dash_app.layouts.text_analysis.text_cards import build_text_section_card
from vigilance.dash_app.layouts.text_analysis.text_filters import build_text_analysis_filters
from vigilance.dash_app.layouts.text_analysis.text_panels import build_text_analysis_detail_panel

__all__ = [
    "build_text_analysis_filters",
    "build_text_section_card",
    "build_text_analysis_detail_panel",
]
