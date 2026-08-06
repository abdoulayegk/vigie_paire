"""Unit tests for UI detection fallback logic."""

from __future__ import annotations

from vigie.interface.ui_detection import _fallback_sections


def test_fallback_sections_splits_capital_and_risks() -> None:
    """Fallback reparti entre gestion du capital et gestion des risques."""
    sections = _fallback_sections(12)

    assert sections == [
        {
            "type": "gestion_capital",
            "label": "Gestion du capital",
            "start_page": 1,
            "end_page": 6,
        },
        {
            "type": "gestion_risques",
            "label": "Gestion des risques",
            "start_page": 7,
            "end_page": 12,
        },
    ]
