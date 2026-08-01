"""Tests unitaires pour l'intégration de la normalisation d'extraction avec les adaptateurs de banques."""

from __future__ import annotations

from vigilance.extraction.docling_normalization import normalize_table_card_with_bank_adapter


def test_normalize_table_card_with_rbc_adapter() -> None:
    raw_card = {
        "title": "TABLEAU  54   Risques de liquidité",
        "footnotes": [{"id": "1", "text": "  Note explicative  "}],
    }
    normalized = normalize_table_card_with_bank_adapter(raw_card, bank_code="rbc")
    assert normalized["title"] == "Tableau 54 Risques de liquidité"
    assert len(normalized["footnotes"]) == 1
    assert normalized["footnotes"][0]["text"] == "Note explicative"


def test_normalize_table_card_with_td_adapter() -> None:
    raw_card = {
        "title": "RATIO DE LIQUIDITÉ¹",
        "footnotes": [{"id": "1", "text": "¹ Texte note TD"}],
    }
    normalized = normalize_table_card_with_bank_adapter(raw_card, bank_code="td")
    assert normalized["title"] == "RATIO DE LIQUIDITÉ"
