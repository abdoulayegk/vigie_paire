"""Unit tests for canonical section taxonomy."""

from __future__ import annotations

from vigilance.extraction.section_taxonomy import canonicalize_section


def test_canonicalize_gestion_capital() -> None:
    assert canonicalize_section("gestion_capital") == "capital_management"


def test_canonicalize_gestion_des_risques() -> None:
    assert canonicalize_section("Gestion des risques") == "risk_management"
