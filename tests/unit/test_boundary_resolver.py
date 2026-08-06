"""Tests du resolveur de bornes T4 base sur la TDM structurelle."""

from __future__ import annotations

from vigie.extraction.localisation_sections.boundary_resolver import (
    map_toc_title_to_concept,
    resolve_t4_section_bounds,
)
from vigie.extraction.localisation_sections.toc_locator import TocStructure, TocStructureEntry


def test_map_toc_title_to_concept_capital_and_risk() -> None:
    assert map_toc_title_to_concept("Gestion des fonds propres", "rbc") == "gestion_capital"
    assert map_toc_title_to_concept("Situation des fonds propres", "td") == "gestion_capital"
    assert map_toc_title_to_concept("des fonds propres", "rbc") == "gestion_capital"
    assert map_toc_title_to_concept("Gestion du risque", "rbc") == "gestion_risques"
    assert map_toc_title_to_concept("Cadre de gestion des fonds propres", "rbc") is None
    assert map_toc_title_to_concept("Risque de credit", "rbc") is None
    assert map_toc_title_to_concept("Situation financière", "rbc") is None
    assert map_toc_title_to_concept("SITUATION FINANCIÈRE DU GROUPE", "td") is None
    assert map_toc_title_to_concept("PROGRAMME DE LCBA Situation des fonds propres", "td") == "gestion_capital"


def test_resolve_bounds_prefers_later_body_hit_for_duplicates() -> None:
    toc = TocStructure(
        rg_page=20,
        confidence=0.9,
        offset=2,
        entries=[
            TocStructureEntry("FACTEURS DE RISQUE ET GESTION", 25, physical_page=27),
            TocStructureEntry("Situation des fonds propres", 73, physical_page=75),
            TocStructureEntry("Gestion des risques", 92, physical_page=94),
            TocStructureEntry("Facteurs de risque et gestion des risques", 82, physical_page=84),
            TocStructureEntry("Normes comptables", 127, physical_page=129),
        ],
    )
    outcome = resolve_t4_section_bounds(toc, bank_code="td")
    by_key = {s.section_type: s for s in outcome.sections}
    assert by_key["gestion_capital"].start_page == 75
    assert by_key["gestion_risques"].start_page == 84
    assert by_key["gestion_risques"].end_page == 128
    assert map_toc_title_to_concept("Gestion des risques", "td") is None
    assert map_toc_title_to_concept("Gestion des risques", "bnc") == "gestion_risques"


def test_resolve_bounds_uses_toc_successor_order() -> None:
    toc = TocStructure(
        rg_page=17,
        confidence=0.9,
        offset=2,
        entries=[
            TocStructureEntry("Gestion du risque", 74, physical_page=76),
            TocStructureEntry("Gestion des fonds propres", 125, physical_page=127),
            TocStructureEntry("Questions comptables", 138, physical_page=140),
        ],
    )
    outcome = resolve_t4_section_bounds(toc, bank_code="rbc")
    assert outcome.used_toc is True
    by_key = {s.section_type: s for s in outcome.sections}
    assert by_key["gestion_risques"].start_page == 76
    assert by_key["gestion_risques"].end_page == 126
    assert by_key["gestion_capital"].start_page == 127
    assert by_key["gestion_capital"].end_page == 139


def test_resolve_bounds_respects_bnc_capital_before_risk() -> None:
    toc = TocStructure(
        rg_page=17,
        confidence=0.9,
        offset=0,
        entries=[
            TocStructureEntry("Gestion du capital", 62, physical_page=62),
            TocStructureEntry("Gestion des risques", 72, physical_page=72),
            TocStructureEntry("Methodes comptables", 119, physical_page=119),
        ],
    )
    outcome = resolve_t4_section_bounds(toc, bank_code="bnc")
    by_key = {s.section_type: s for s in outcome.sections}
    assert by_key["gestion_capital"].start_page == 62
    assert by_key["gestion_capital"].end_page == 71
    assert by_key["gestion_risques"].start_page == 72
    assert by_key["gestion_risques"].end_page == 118


def test_resolve_bounds_fallback_title_when_toc_incomplete() -> None:
    toc = TocStructure(rg_page=None, confidence=0.1, entries=[], anomalies=["rg_opening_not_found"])

    def find_start(key: str):
        if key == "gestion_risques":
            return 76, "Gestion du risque"
        if key == "gestion_capital":
            return 127, "Gestion des fonds propres"
        return None

    def find_end(key: str, start: int):
        if key == "gestion_risques":
            return 126, "annual_t4_physical_successor"
        return None, "annual_t4_unresolved_no_successor"

    outcome = resolve_t4_section_bounds(
        toc,
        bank_code="rbc",
        find_start=find_start,
        find_end=find_end,
        total_pages=200,
    )
    assert outcome.used_toc is False
    by_key = {s.section_type: s for s in outcome.sections}
    assert by_key["gestion_risques"].detection_method == "annual_t4_toc_fallback_title"
    assert by_key["gestion_risques"].end_page == 126
    assert by_key["gestion_capital"].end_page is None
    assert by_key["gestion_capital"].end_detection_method == "annual_t4_unresolved_no_successor"
    assert any("boundary_unresolved:gestion_capital" in a for a in outcome.anomalies)


def test_no_silent_plus_119_cap() -> None:
    toc = TocStructure(
        rg_page=17,
        confidence=0.9,
        entries=[
            TocStructureEntry("Gestion du risque", 74, physical_page=76),
        ],
    )
    outcome = resolve_t4_section_bounds(
        toc,
        bank_code="rbc",
        find_start=lambda key: (127, "Gestion des fonds propres") if key == "gestion_capital" else None,
        find_end=lambda key, start: (None, "annual_t4_unresolved_no_successor"),
        total_pages=300,
    )
    capital = next(s for s in outcome.sections if s.section_type == "gestion_capital")
    # Pas de start+119: end reste None ou borne par next target, jamais 127+119.
    assert capital.end_page is None or capital.end_page < 200
    assert "annual_t4_safety_cap_no_successor" not in (capital.end_detection_method or "")
