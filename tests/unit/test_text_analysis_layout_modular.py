"""Tests unitaires pour les composants de mise en page d'analyse textuelle modulaires."""

from __future__ import annotations

from vigilance.dash_app.layouts.text_analysis import (
    build_text_analysis_detail_panel,
    build_text_analysis_filters,
    build_text_section_card,
)


def test_build_text_analysis_filters() -> None:
    filters = build_text_analysis_filters(["rbc", "td"])
    assert filters is not None


def test_build_text_section_card() -> None:
    card = build_text_section_card(
        "sec-1",
        "Gestion des risques de liquidité",
        {"impact_level": "MAJEUR", "explanation": "Mise à jour BSIF."},
    )
    assert card is not None


def test_build_text_analysis_detail_panel() -> None:
    panel = build_text_analysis_detail_panel(
        {"posture_change": "Prudente", "themes_amf": ["EXIGENCES_REGLEMENTAIRES"], "explanation": "Explication complète."}
    )
    assert panel is not None
