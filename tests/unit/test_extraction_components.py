"""Tests unitaires pour les sous-composants d'extraction (footnotes, sections, tables)."""

from __future__ import annotations

from vigilance.extraction.components import (
    detect_section_key,
    extract_clean_footnotes,
    validate_table_structure,
)


def test_detect_section_key() -> None:
    assert detect_section_key("Gestion du capital et fonds propres") == "gestion_capital"
    assert detect_section_key("Ratio de liquidité à long terme - Risques") == "gestion_risques"
    assert detect_section_key("Exigences de la réglementation BSIF") == "gestion_reglementation"
    assert detect_section_key("Section inconnue") == "unknown_section"


def test_extract_clean_footnotes() -> None:
    raw = [
        {"id": "1", "text": "  Le NSFR est calculé selon les normes BSIF.  "},
        {"symbol": "2", "content": "Seuil minimum imposé."},
    ]
    cleaned = extract_clean_footnotes(raw)
    assert len(cleaned) == 2
    assert cleaned[0]["id"] == "1"
    assert cleaned[0]["text"] == "Le NSFR est calculé selon les normes BSIF."


def test_validate_table_structure() -> None:
    assert validate_table_structure({"title": "TABLEAU 54 RATIO DE LIQUIDITÉ"}) is True
    assert validate_table_structure({}) is False
