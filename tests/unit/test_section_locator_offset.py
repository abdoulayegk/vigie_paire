"""Unit tests for SectionLocator page_number_offset (document vs physical)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import vigie.extraction.localisation_sections.section_locator as section_locator_module
from vigie.extraction.localisation_sections import (
    LocatedSection,
    SectionLocator,
    TocEntry,
)


def test_uses_document_page_numbers_toc() -> None:
    """toc et variantes (toc_*) sont en numerotation document -> offset applique."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("toc") is True
    assert locator._uses_document_page_numbers("toc_corrected") is True
    assert locator._uses_document_page_numbers("toc_cibc_recalibrated") is True


def test_uses_document_page_numbers_manual_override() -> None:
    """manual_override et manual_override_guardrail sont en numerotation document."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("manual_override") is True
    assert locator._uses_document_page_numbers("manual_override_guardrail") is True


def test_uses_document_page_numbers_scan_visual_genai_false() -> None:
    """scan, genai_fallback, visual donnent deja des numeros physiques -> pas d'offset."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("scan") is False
    assert locator._uses_document_page_numbers("scan_subsection") is False
    assert locator._uses_document_page_numbers("genai_fallback") is False
    assert locator._uses_document_page_numbers("visual") is False
    assert locator._uses_document_page_numbers("visual_temp") is False


def test_uses_document_page_numbers_empty_false() -> None:
    """detection_method vide -> pas d'offset."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("") is False


def test_cibc_page_number_offset_from_config() -> None:
    """CIBC a page_number_offset=3 dans bank_profiles."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._get_page_number_offset() == 3


def test_period_page_number_offset_overrides_default() -> None:
    """Un offset par periode prend priorite sur l'offset par defaut."""
    assert SectionLocator(bank_code="cibc", quarter="t4", year=2025)._get_page_number_offset() == 13
    assert SectionLocator(bank_code="cibc", quarter="t4", year=2024)._get_page_number_offset() == 16
    assert SectionLocator(bank_code="cibc", quarter="t1", year=2025)._get_page_number_offset() == 3
    assert SectionLocator(bank_code="bnc", quarter="t4", year=2025)._get_page_number_offset() == 0
    assert SectionLocator(bank_code="bnc", quarter="t4", year=2024)._get_page_number_offset() == 2


def test_offset_applied_only_to_document_sections() -> None:
    """Apres etape 4.5: section toc -> start_page augmente de l'offset; section scan -> inchange."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    offset = locator._get_page_number_offset()
    assert offset == 3

    section_toc = LocatedSection(
        section_type="gestion_capital",
        title_found="Gestion des fonds propres",
        start_page=20,
        end_page=24,
        detection_method="toc",
    )
    section_scan = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion du risque",
        start_page=25,
        end_page=45,
        detection_method="scan",
    )

    # Logique identique a l'etape 4.5: appliquer offset uniquement si document
    adjusted = []
    for s in [section_toc, section_scan]:
        if locator._uses_document_page_numbers(s.detection_method):
            new_start = s.start_page + offset
            new_end = (s.end_page + offset) if s.end_page is not None else None
            adjusted.append(
                LocatedSection(
                    section_type=s.section_type,
                    title_found=s.title_found,
                    start_page=new_start,
                    end_page=new_end,
                    confidence=s.confidence,
                    detection_method=s.detection_method,
                    end_detection_method=s.end_detection_method,
                    detected_span=s.detected_span,
                    final_span=s.final_span,
                    constraint_applied=s.constraint_applied,
                    constraint_reason=s.constraint_reason,
                )
            )
        else:
            adjusted.append(s)

    toc_result = next(s for s in adjusted if s.detection_method == "toc")
    scan_result = next(s for s in adjusted if s.detection_method == "scan")

    assert toc_result.start_page == 23, "toc: document 20 + offset 3 = physique 23"
    assert toc_result.end_page == 27, "toc: document 24 + offset 3 = physique 27"
    assert scan_result.start_page == 25, "scan: deja physique, pas d'offset"
    assert scan_result.end_page == 45, "scan: deja physique, pas d'offset"


def test_content_validation_reads_physical_pages_for_toc_section() -> None:
    """La validation TDM lit la page physique après application de l'offset."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    section = LocatedSection(
        section_type="gestion_capital",
        title_found="Gestion des fonds propres",
        start_page=20,
        end_page=21,
        detection_method="toc",
    )
    text_by_page = {
        20: "mauvaise page imprimée",
        23: "bonne page physique",
        24: "suite physique",
    }

    assert locator._extract_section_text(section, text_by_page) == "bonne page physique\nsuite physique"


def test_consensus_single_source_is_not_full_consensus() -> None:
    locator = SectionLocator(bank_code="td", quarter="t2", year=2025)
    section = LocatedSection("gestion_capital", "Situation des fonds propres", 20)

    assert locator._calculate_consensus(section, [], []) == 0.55


def test_consensus_strong_page_disagreement_can_trigger_correction() -> None:
    locator = SectionLocator(bank_code="td", quarter="t2", year=2025)
    section = LocatedSection("gestion_capital", "Situation des fonds propres", 20)
    toc = [TocEntry("Situation des fonds propres", 20)]
    scans = [LocatedSection("gestion_capital", "Situation des fonds propres", 40)]

    assert locator._calculate_consensus(section, toc, scans) < 0.5


def test_t4_annual_offset_diagnostic_does_not_overwrite_document_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Le page_offset entier historique T4 reste séparé du diagnostic global."""
    pdf_path = tmp_path / "RBC_2024_T4.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    locator = SectionLocator(bank_code="rbc", quarter="t4", year=2024)
    locator.bank_config["section_semantic_localization"]["enabled"] = False
    text_by_page = {page: "Contenu suffisamment long du rapport annuel" for page in range(1, 41)}
    sections = [
        LocatedSection("gestion_risques", "Gestion du risque", 10, 19, 0.9, "scan_exact"),
        LocatedSection("gestion_capital", "Gestion des fonds propres", 20, 30, 0.9, "scan_exact"),
    ]

    monkeypatch.setattr(locator, "_extract_text_by_page", lambda _path: text_by_page)
    monkeypatch.setattr(locator, "_parse_full_toc", lambda _pages: [])
    monkeypatch.setattr(locator, "_scan_section_titles", lambda _pages: sections)
    monkeypatch.setattr(locator, "_extract_visual_elements", lambda _path: {})
    monkeypatch.setattr(locator, "_rebase_annual_t4_section_starts", lambda current, *_args: current)
    monkeypatch.setattr(locator, "_determine_end_pages", lambda current, *_args: current)
    monkeypatch.setattr(locator, "_validate_with_cross_reference", lambda current, *_args: current)
    monkeypatch.setattr(locator, "_apply_section_length_constraints", lambda section, *_args, **_kwargs: section)
    monkeypatch.setattr(locator, "_refine_cibc_target_sections", lambda current, *_args: current)
    monkeypatch.setattr(
        section_locator_module,
        "locate_toc_structure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture sans TDM structurelle")),
    )

    class FakeAnnualValidator:
        def __init__(self, **_kwargs):
            pass

        def validate(self, _pdf_path, current, _pages, _candidates):
            return types.SimpleNamespace(
                sections=current,
                toc_entries=[],
                diagnostics={
                    "enabled": True,
                    "status": "verified",
                    "page_offset": 2,
                    "warnings": [],
                },
            )

    fake_module = types.ModuleType("vigie.extraction.annual_section_boundary_validator")
    fake_module.AnnualSectionBoundaryValidator = FakeAnnualValidator
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    mapping = locator.locate_sections(pdf_path)

    assert isinstance(mapping.boundary_validation["page_offset"], dict)
    assert mapping.boundary_validation["annual_page_offset"] == 2
    assert mapping.boundary_validation["annual_t4"]["page_offset"] == 2


def test_t4_toc_parser_scans_annual_report_front_matter() -> None:
    """T4 cherche la TDM dans les pages annuelles 1-25, pas seulement 1-6."""
    locator = SectionLocator(bank_code="td", quarter="t4", year=2024)
    text_by_page = {page: "" for page in range(1, 131)}
    text_by_page[18] = (
        "Table des matières\n"
        "75 Situation des fonds propres\n"
        "84 Facteurs de risque et gestion des risques\n"
        "128 Informations complémentaires"
    )

    entries = locator._parse_full_toc(text_by_page)

    assert any(e.title == "Situation des fonds propres" and e.page == 75 for e in entries)
    assert any(e.title == "Facteurs de risque et gestion des risques" and e.page == 84 for e in entries)


def test_non_t4_toc_parser_keeps_existing_front_matter_window() -> None:
    """T1-T3 gardent la fenetre historique des premieres pages."""
    locator = SectionLocator(bank_code="td", quarter="t3", year=2025)
    text_by_page = {page: "" for page in range(1, 131)}
    text_by_page[18] = (
        "Table des matières\n75 Situation des fonds propres\n84 Facteurs de risque et gestion des risques"
    )

    assert locator._parse_full_toc(text_by_page) == []


def test_t4_toc_parser_prefers_strong_late_toc_over_early_soft_marker() -> None:
    """Un vrai marqueur TDM page 15-20 bat un simple sommaire preliminaire."""
    locator = SectionLocator(bank_code="td", quarter="t4", year=2024)
    text_by_page = {page: "" for page in range(1, 131)}
    text_by_page[2] = "Sommaire\n4 Faits saillants\n5 Message aux actionnaires"
    text_by_page[18] = (
        "Table des matières\n"
        "75 Situation des fonds propres\n"
        "84 Facteurs de risque et gestion des risques\n"
        "128 Informations complémentaires"
    )

    entries = locator._parse_full_toc(text_by_page)

    assert all(e.page != 4 for e in entries)
    assert any(e.title == "Situation des fonds propres" and e.page == 75 for e in entries)


def test_toc_end_does_not_stop_on_same_section_family() -> None:
    """Un sous-titre de meme famille ne doit pas couper la section cible."""
    locator = SectionLocator(bank_code="td", quarter="t4", year=2024)
    sections = locator._detect_sections_from_full_toc(
        [
            TocEntry(
                title="Facteurs de risque qui pourraient avoir une incidence sur les résultats futurs",
                page=84,
                level=0,
            ),
            TocEntry(title="Gestion des risques", page=93, level=0),
            TocEntry(
                title="Méthodes et estimations comptables critiques",
                page=128,
                level=0,
            ),
        ]
    )

    risk = next(s for s in sections if s.section_type == "gestion_risques")

    assert risk.start_page == 84
    assert risk.end_page == 127
