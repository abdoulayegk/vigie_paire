"""Tests unitaires pour les composants modulaires Dash (badges et cartes de preuve)."""

from __future__ import annotations

from vigilance.dash_app.components.detail_widgets import (
    build_comparison_proof_card,
    build_proof_badges,
)


def test_build_proof_badges_added() -> None:
    widget = build_proof_badges({"change_type": "footnote_added"}, year_t2="2025")
    assert widget is not None


def test_build_comparison_proof_card() -> None:
    card = build_comparison_proof_card("item-1", {"change_type": "indicator_added", "description": "Ratio 12.5%"}, title="T4 2025")
    assert card is not None
