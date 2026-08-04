from __future__ import annotations

import re

from vigie.extraction.genai_toc_detector import GenAITOCDetector
from vigie.extraction.localisation_sections import (
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


def test_annual_t4_risk_boundary_uses_physical_successor_not_flat_toc() -> None:
    """Un sous-thème risque T4 ne peut pas raccourcir la section.

    Le scénario reproduit le problème BNC: une TDM plate donne une fin à p.82,
    mais les pages 83-118 restent dans les risques et la comptabilité commence
    réellement à p.119.
    """
    locator = SectionLocator(bank_code="bnc", quarter="t4", year=2025)
    text_by_page = {
        page: "Gestion des risques\nRisque opérationnel et conformité"
        for page in range(72, 119)
    }
    text_by_page[83] = "Propriété intellectuelle\nLe risque est suivi par la Banque."
    text_by_page[119] = (
        "Méthodes comptables significatives et estimations comptables\n"
        "Les états financiers consolidés suivent."
    )

    section = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion des risques",
        start_page=72,
        end_page=82,
        confidence=0.95,
        detection_method="toc",
        end_detection_method="toc_next_section",
    )

    determined = locator._determine_end_pages([section], text_by_page, [], total_pages=130)

    assert determined[0].end_page == 118
    assert determined[0].end_detection_method == "annual_t4_physical_successor"
    assert determined[0].constraint_applied is False


def test_annual_t4_risk_start_is_a_physical_root_title() -> None:
    locator = SectionLocator(bank_code="bnc", quarter="t4", year=2025)
    text_by_page = {
        18: "Table des matières\nGestion des risques 72",
        30: "Le comité discute de la gestion des risques.",
        72: "Gestion des risques\nNotre cadre de gestion des risques.",
    }

    assert locator._find_annual_t4_risk_start(text_by_page, total_pages=130) == (72, "Gestion des risques")


def test_non_t4_keeps_predefined_toc_boundary() -> None:
    locator = SectionLocator(bank_code="bnc", quarter="t3", year=2025)
    section = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion des risques",
        start_page=20,
        end_page=28,
        confidence=0.95,
        detection_method="toc",
        end_detection_method="toc_next_section",
    )

    determined = locator._determine_end_pages([section], {20: "Gestion des risques"}, [], total_pages=60)

    assert determined[0].end_page == 28
    assert determined[0].end_detection_method == "toc_next_section"


def test_annual_t4_profile_creates_capital_and_risk_sections_from_physical_titles() -> None:
    """Un T4 peut être localisé sans TDM, sans pages configurées ni GenAI."""
    locator = SectionLocator(bank_code="bmo", quarter="t4", year=2025)
    text_by_page = {
        20: "Gestion globale du capital\nLe capital est géré de façon prudente.",
        27: "Gestion globale des risques\nLe cadre de risque est présenté ci-dessous.",
        40: "Questions comptables\nLes méthodes comptables suivent.",
    }

    sections = locator._rebase_annual_t4_section_starts([], text_by_page, total_pages=50)
    determined = locator._determine_end_pages(sections, text_by_page, [], total_pages=50)
    by_type = {section.section_type: section for section in determined}

    assert by_type["gestion_capital"].start_page == 20
    assert by_type["gestion_capital"].end_page == 26
    assert by_type["gestion_risques"].start_page == 27
    assert by_type["gestion_risques"].end_page == 39
    assert all(section.detection_method == "annual_t4_physical_title" for section in determined)


def test_annual_t4_prefers_bank_profile_title_over_earlier_generic_alias() -> None:
    """Un alias narratif précoce ne doit pas supplanter le vrai chapitre RBC."""
    locator = SectionLocator(bank_code="rbc", quarter="t4", year=2025)
    text_by_page = {
        28: "Gestion des risques\nBref sommaire des fonctions de contrôle.",
        75: "\n".join(
            [
                *(f"Ligne de mise en page {index}" for index in range(40)),
                "Gestion du risque",
                "Nous avons à gérer les risques inhérents au secteur des services financiers.",
            ]
        ),
    }

    root = locator._find_annual_t4_section_start(
        "gestion_risques",
        text_by_page,
        total_pages=100,
    )

    assert root == (75, "Gestion du risque")


def test_annual_t4_prefers_top_level_root_over_later_narrative_mention() -> None:
    """Une mention au bas d'une page ne doit pas voler la racine BNC."""
    locator = SectionLocator(bank_code="bnc", quarter="t4", year=2024)
    text_by_page = {
        31: "\n".join(
            [
                *(f"Ligne narrative {index}" for index in range(44)),
                "Gestion des risques.",
            ]
        ),
        67: "Gestion des risques\nLe cadre de gestion des risques est présenté ci-dessous.",
    }

    root = locator._find_annual_t4_section_start(
        "gestion_risques",
        text_by_page,
        total_pages=130,
    )

    assert root == (67, "Gestion des risques")


def test_annual_t4_keeps_earlier_root_when_later_risk_subtopic_repeats_title() -> None:
    """Le titre racine RBC précède une répétition dans un sous-thème ESG."""
    locator = SectionLocator(bank_code="rbc", quarter="t4", year=2024)
    text_by_page = {
        72: "\n".join(
            [
                *(f"Ligne de mise en page {index}" for index in range(55)),
                "Gestion du risque",
                "Le cadre de gestion du risque est présenté ci-dessous.",
            ]
        ),
        122: "\n".join(
            [
                *(f"Ligne narrative {index}" for index in range(33)),
                "Gestion du risque",
                "Les risques environnementaux et sociaux sont traités ici.",
            ]
        ),
    }

    root = locator._find_annual_t4_section_start(
        "gestion_risques",
        text_by_page,
        total_pages=140,
    )

    assert root == (72, "Gestion du risque")


def test_annual_t4_bmo_capital_stops_at_intermediate_securitization_chapter() -> None:
    """Le capital BMO ne doit pas absorber le chapitre de titrisation."""
    locator = SectionLocator(bank_code="bmo", quarter="t4", year=2025)
    text_by_page = {
        60: "Gestion globale du capital\nLe capital est géré de façon prudente.",
        68: "Entités de titrisation soutenues par BMO\nContenu hors gestion du capital.",
        69: "Gestion globale des risques\nLe cadre de risque est présenté ci-dessous.",
        90: "Questions comptables\nLes méthodes comptables suivent.",
    }

    sections = locator._rebase_annual_t4_section_starts([], text_by_page, total_pages=100)
    determined = locator._determine_end_pages(sections, text_by_page, [], total_pages=100)
    by_type = {section.section_type: section for section in determined}

    assert by_type["gestion_capital"].end_page == 67


def test_annual_t4_bmo_accepts_exact_successor_even_when_extracted_late() -> None:
    """Un titre BMO configuré reste une borne même après une table longue."""
    locator = SectionLocator(bank_code="bmo", quarter="t4", year=2024)
    text_by_page = {
        61: "Gestion globale du capital\nLe capital est géré de façon prudente.",
        68: "\n".join(
            [
                *(f"Ligne de tableau {index}" for index in range(39)),
                "Entitésstructuréesettitrisation",
                "Contenu hors gestion du capital.",
            ]
        ),
        70: "Gestion globale des risques\nLe cadre de risque est présenté ci-dessous.",
        90: "Questions comptables\nLes méthodes comptables suivent.",
    }

    sections = locator._rebase_annual_t4_section_starts([], text_by_page, total_pages=100)
    determined = locator._determine_end_pages(sections, text_by_page, [], total_pages=100)
    by_type = {section.section_type: section for section in determined}

    assert by_type["gestion_capital"].end_page == 67


def test_annual_t4_td_uses_financial_group_boundary_not_objectives_sentence() -> None:
    locator = SectionLocator(bank_code="td", quarter="t4", year=2025)
    text_by_page = {
        20: "Situation des fonds propres\nPrésentation de la gestion du capital.",
        21: "Les objectifs de la Banque en matière de gestion des fonds propres sont les suivants.",
        24: "Situation financière du Groupe\nTitrisation et arrangements hors bilan.",
        28: "Facteurs de risque et gestion des risques\nPrincipaux risques.",
        40: "Normes et méthodes comptables\nMéthodes comptables significatives.",
    }

    sections = locator._rebase_annual_t4_section_starts([], text_by_page, total_pages=50)
    determined = locator._determine_end_pages(sections, text_by_page, [], total_pages=50)
    by_type = {section.section_type: section for section in determined}

    assert by_type["gestion_capital"].start_page == 20
    assert by_type["gestion_capital"].end_page == 23
    assert by_type["gestion_risques"].start_page == 28
    assert by_type["gestion_risques"].end_page == 39
