from __future__ import annotations

import re

from vigilance.extraction.genai_toc_detector import GenAITOCDetector
from vigilance.extraction.section_locator import (
    RISK_SUBSECTIONS,
    SECTION_PATTERNS,
    LocatedSection,
    SectionLocator,
    TocEntry,
    VisualTextElement,
)


def _matches_risk_pattern(title: str) -> bool:
    return any(
        re.search(pattern, title, flags=re.IGNORECASE)
        for pattern in SECTION_PATTERNS["gestion_risques"]["patterns"]
    )


def test_data_cloud_third_party_and_resilience_titles_are_in_risk_scope() -> None:
    titles = [
        "Risque lié aux données",
        "Risque technologique",
        "Risque lié aux tiers",
        "Services infonuagiques",
        "Résilience opérationnelle",
        "Protection des données et vie privée",
    ]

    assert all(_matches_risk_pattern(title) for title in titles)


def test_data_cloud_and_resilience_are_known_risk_subsections() -> None:
    expected = {
        "Risque lié aux données",
        "Risque technologique",
        "Risque lié aux tiers",
        "Services infonuagiques",
        "Résilience opérationnelle",
        "Protection des données",
    }

    assert expected.issubset(set(RISK_SUBSECTIONS))


def test_toc_prompt_explicitly_covers_extended_risk_scope() -> None:
    prompt = " ".join(GenAITOCDetector.SECTION_DETECTION_PROMPT.split())

    assert "données" in prompt
    assert "services infonuagiques" in prompt
    assert "fournisseurs et tiers" in prompt
    assert "résilience opérationnelle" in prompt


def test_data_only_risk_section_passes_context_validation() -> None:
    locator = SectionLocator()
    section = LocatedSection(
        section_type="gestion_risques",
        title_found="Risques liés aux données",
        start_page=12,
        end_page=12,
        confidence=0.8,
        detection_method="test",
    )
    text_by_page = {
        12: (
            "Risques liés aux données\n"
            "La gouvernance des données couvre la qualité des données, leur "
            "protection, leur localisation, leur traçabilité et la résilience "
            "opérationnelle."
        )
    }

    is_valid, score = locator._validate_section_content(section, text_by_page)

    assert is_valid is True
    assert score > 0.4


def test_historical_risk_vocabulary_keeps_validation_threshold() -> None:
    locator = SectionLocator()
    section = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion des risques",
        start_page=12,
        end_page=12,
        confidence=0.8,
        detection_method="test",
    )
    text_by_page = {
        12: (
            "Gestion des risques\n"
            "Le risque de crédit et l'exposition du portefeuille sont suivis "
            "avec des scénarios de stress."
        )
    }

    is_valid, score = locator._validate_section_content(section, text_by_page)

    assert is_valid is True
    assert score > 0.4


def test_risk_subsection_fallback_is_accent_insensitive() -> None:
    locator = SectionLocator()

    section = locator._find_first_risk_subsection(
        {12: "RISQUE LIE AUX DONNEES\nContenu de la section"}
    )

    assert section is not None
    assert section.section_type == "gestion_risques"


def _visual_title(text: str, page: int) -> VisualTextElement:
    return VisualTextElement(
        text=text,
        page=page,
        x0=40.0,
        y0=50.0,
        x1=240.0,
        y1=70.0,
        font_size=14.0,
        is_bold=True,
        line_number=1,
        page_width=600.0,
        page_height=800.0,
    )


def test_anchor_resolution_matches_bmo_titles_without_spaces() -> None:
    locator = SectionLocator(bank_code="bmo", quarter="t4", year=2025)
    sections = [
        LocatedSection(
            section_type="capital_management",
            title_found="Gestion globale du capital",
            start_page=60,
            end_page=66,
            confidence=0.95,
            detection_method="scan_exact",
        ),
        LocatedSection(
            section_type="risk_management",
            title_found="Gestion globale des risques",
            start_page=69,
            end_page=109,
            confidence=0.95,
            detection_method="scan_exact",
        ),
    ]
    visual_elements = {
        60: [_visual_title("Gestionglobaleducapital", 60)],
        69: [_visual_title("Gestionglobaledesrisques", 69)],
    }

    resolved = locator._resolve_section_anchors(sections, visual_elements)

    assert [section.anchor_found for section in resolved] == [True, True]
    assert [section.anchor_text for section in resolved] == [
        "Gestionglobaleducapital",
        "Gestionglobaledesrisques",
    ]


def test_anchor_resolution_matches_bnc_titles_with_doubled_characters() -> None:
    locator = SectionLocator(bank_code="bnc", quarter="t4", year=2025)
    sections = [
        LocatedSection(
            section_type="capital_management",
            title_found="Gestion du capital",
            start_page=62,
            end_page=71,
            confidence=0.95,
            detection_method="scan_exact",
        ),
        LocatedSection(
            section_type="risk_management",
            title_found="Gestion des risques",
            start_page=72,
            end_page=118,
            confidence=0.95,
            detection_method="scan_exact",
        ),
    ]
    visual_elements = {
        62: [_visual_title("GGeessttiioonn  dduu  ccaappiittaall", 62)],
        72: [_visual_title("GGeessttiioonn  ddeess  rriissqquueess", 72)],
    }

    resolved = locator._resolve_section_anchors(sections, visual_elements)

    assert [section.anchor_found for section in resolved] == [True, True]
    assert [section.anchor_text for section in resolved] == [
        "GGeessttiioonn  dduu  ccaappiittaall",
        "GGeessttiioonn  ddeess  rriissqquueess",
    ]


def test_refine_shared_page_extends_end_when_boundary_not_at_top() -> None:
    locator = SectionLocator(bank_code="td", quarter="t4", year=2025)
    sections = [
        LocatedSection(
            section_type="gestion_risques",
            title_found="Gestion des risques",
            start_page=84,
            end_page=128,
            confidence=0.95,
            detection_method="toc",
            end_detection_method="toc_next_section",
        )
    ]
    toc_entries = [
        TocEntry(
            title="Méthodes et estimations comptables critiques",
            page=127,
            level=0,
        )
    ]
    visual_elements = {
        129: [
            VisualTextElement(
                text="NORMES ET MÉTHODES COMPTABLES",
                page=129,
                x0=40.0,
                y0=520.0,
                x1=560.0,
                y1=540.0,
                font_size=14.0,
                is_bold=True,
                is_uppercase=True,
                line_number=1,
                page_width=600.0,
                page_height=800.0,
            )
        ]
    }

    refined = locator._refine_shared_page_boundaries(sections, toc_entries, visual_elements)

    assert refined[0].end_page == 129
    assert refined[0].end_anchor_page == 129
    assert refined[0].end_anchor_text == "NORMES ET MÉTHODES COMPTABLES"
    assert refined[0].end_anchor_bbox_norm == [40.0 / 600.0, 520.0 / 800.0, 560.0 / 600.0, 540.0 / 800.0]
    assert refined[0].end_detection_method == "toc_next_section+shared_page"
