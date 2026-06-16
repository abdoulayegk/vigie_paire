from __future__ import annotations

import re

from vigilance.extraction.genai_toc_detector import GenAITOCDetector
from vigilance.extraction.section_locator import (
    RISK_SUBSECTIONS,
    SECTION_PATTERNS,
    LocatedSection,
    SectionLocator,
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
