from __future__ import annotations

from pathlib import Path

from vigie.extraction.annual_section_boundary_validator import (
    AnnualSectionBoundaryValidator,
    StructuredTOCEntry,
    parse_docling_toc_markdown,
    reconcile_boundary_roles,
)
from vigie.extraction.genai_toc_detector import (
    AnnualTOCAnalysis,
    PageTransitionValidation,
    TOCBoundaryRole,
)
from vigie.extraction.localisation_sections import LocatedSection

BNC_TOC_MARKDOWN = """
| Analyse financière | 25 | Gestion du capital | 53 |
|---|---|---|---|
| Analyse sectorielle | 28 | Gestion des risques | 62 |
| Particuliers et Entreprises | 29 | Principales méthodes et estimations comptables | 107 |
| Données financières supplémentaires | 114 | Glossaire | 124 |
"""

TD_TOC_MARKDOWN = """
| SITUATION FINANCIÈRE DU GROUPE | |
|---|---|
| Qualité du portefeuille de crédit Situation des fonds propres | 62 75 |
| Titrisation et arrangements hors bilan | 81 |
| Transactions entre parties liées | 82 |
| FACTEURS DE RISQUE ET GESTION DES RISQUES | |
| Facteurs de risque qui pourraient avoir une incidence sur les résultats | 84 |
| Gestion des risques | 93 |
| NORMES ET MÉTHODES COMPTABLES | |
| Méthodes et estimations comptables critiques | 128 |
"""


def test_parse_docling_toc_markdown_preserves_bnc_columns() -> None:
    entries = parse_docling_toc_markdown(BNC_TOC_MARKDOWN, max_pages=240)

    assert ("Gestion du capital", 53) in {(entry.title, entry.page) for entry in entries}
    assert ("Gestion des risques", 62) in {(entry.title, entry.page) for entry in entries}
    assert (
        "Principales méthodes et estimations comptables",
        107,
    ) in {(entry.title, entry.page) for entry in entries}


def test_docling_reconciles_incorrect_vision_page_numbers() -> None:
    roles = [
        TOCBoundaryRole(
            section_type="risk_management",
            title_found="Gestion des risques",
            start_page=56,
            successor_title="Principales méthodes comptables",
            successor_page=107,
            confidence=0.95,
        )
    ]
    docling_entries = parse_docling_toc_markdown(BNC_TOC_MARKDOWN, max_pages=240)

    resolved, warnings = reconcile_boundary_roles(roles, docling_entries, [])

    assert resolved[0].start_page == 62
    assert resolved[0].successor_page == 107
    assert resolved[0].successor_title == "Principales méthodes et estimations comptables"
    assert "risk_management:vision_start_page_56_reconciled_to_62" in warnings


def test_docling_infers_group_start_from_first_numbered_child() -> None:
    entries = parse_docling_toc_markdown(TD_TOC_MARKDOWN, max_pages=150)

    assert (
        "FACTEURS DE RISQUE ET GESTION DES RISQUES",
        84,
        "docling_inferred_group_start",
    ) in {(entry.title, entry.page, entry.source) for entry in entries}
    assert (
        "NORMES ET MÉTHODES COMPTABLES",
        128,
        "docling_inferred_group_start",
    ) in {(entry.title, entry.page, entry.source) for entry in entries}


def test_reconciliation_falls_back_to_vision_for_docling_merged_cell() -> None:
    roles = [
        TOCBoundaryRole(
            "capital_management",
            "Situation des fonds propres",
            75,
            "FACTEURS DE RISQUE ET GESTION DES RISQUES",
            83,
            0.95,
        ),
        TOCBoundaryRole(
            "risk_management",
            "FACTEURS DE RISQUE ET GESTION DES RISQUES",
            83,
            "NORMES ET MÉTHODES COMPTABLES",
            128,
            0.95,
        ),
    ]
    docling_entries = parse_docling_toc_markdown(TD_TOC_MARKDOWN, max_pages=150)
    vision_entries = [
        StructuredTOCEntry("Situation des fonds propres", 75, source="vision"),
        StructuredTOCEntry(
            "FACTEURS DE RISQUE ET GESTION DES RISQUES",
            83,
            source="vision",
        ),
        StructuredTOCEntry("NORMES ET MÉTHODES COMPTABLES", 128, source="vision"),
    ]

    resolved, _warnings = reconcile_boundary_roles(
        roles,
        docling_entries,
        vision_entries,
    )
    by_type = {role.section_type: role for role in resolved}

    assert by_type["capital_management"].start_page == 75
    assert by_type["capital_management"].successor_page == 81
    assert by_type["risk_management"].start_page == 84
    assert by_type["risk_management"].successor_page == 128


class FakeDetector:
    api_key = "test-key"

    def __init__(self) -> None:
        self.transition_calls: list[tuple[int, str]] = []

    def analyze_annual_toc_page(self, _pdf_path: Path, page: int) -> AnnualTOCAnalysis:
        if page != 15:
            return AnnualTOCAnalysis(False, 0.0, page, [], [], [])
        return AnnualTOCAnalysis(
            is_master_toc=True,
            confidence=0.98,
            page_number=15,
            entries=[
                {"title": "Gestion du capital", "page": 53, "level": 0},
                {"title": "Gestion des risques", "page": 56, "level": 0},
                {
                    "title": "Principales méthodes comptables",
                    "page": 107,
                    "level": 0,
                },
            ],
            boundaries=[
                TOCBoundaryRole(
                    "capital_management",
                    "Gestion du capital",
                    53,
                    "Gestion des risques",
                    56,
                    0.97,
                ),
                TOCBoundaryRole(
                    "risk_management",
                    "Gestion des risques",
                    56,
                    "Principales méthodes comptables",
                    107,
                    0.96,
                ),
            ],
            warnings=[],
        )

    def validate_section_transition(
        self,
        _pdf_path: Path,
        _previous_page: int,
        candidate_page: int,
        *,
        section_type: str,
        expected_title: str,
    ) -> PageTransitionValidation:
        self.transition_calls.append((candidate_page, expected_title))
        return PageTransitionValidation(
            confirmed=True,
            confidence=0.97,
            observed_title=expected_title,
            previous_page_belongs_to_prior_section=True,
            candidate_page_starts_expected_section=True,
            reason=section_type,
        )


def _docling_reader(
    _pdf_path: Path,
    _toc_page: int,
    _total_pages: int,
) -> list[StructuredTOCEntry]:
    return parse_docling_toc_markdown(BNC_TOC_MARKDOWN, max_pages=240)


def test_validator_corrects_bnc_2023_risk_boundary_to_page_108() -> None:
    detector = FakeDetector()
    validator = AnnualSectionBoundaryValidator(
        "bnc",
        2023,
        detector=detector,
        docling_reader=_docling_reader,
    )
    sections = [
        LocatedSection(
            section_type="capital_management",
            title_found="Gestion du capital",
            start_page=55,
            end_page=63,
            confidence=1.0,
            detection_method="annual_t4_physical_title",
            end_detection_method="annual_t4_physical_successor",
        ),
        LocatedSection(
            section_type="risk_management",
            title_found="Gestion des risques",
            start_page=64,
            end_page=183,
            confidence=1.0,
            detection_method="annual_t4_physical_title",
            end_detection_method="annual_t4_safety_cap_no_successor",
        ),
    ]
    text_by_page = {page: "" for page in range(1, 241)}
    text_by_page.update(
        {
            55: "Gestion du capital\nRatios de fonds propres",
            64: "Gestion des risques\nCadre de gestion des risques",
            109: "Principales méthodes et estimations comptables\nNouvelle section",
        }
    )

    outcome = validator.validate(
        Path("BNC_2023_T4.pdf"),
        sections,
        text_by_page,
        candidate_pages=[15],
    )
    by_type = {section.section_type: section for section in outcome.sections}

    assert by_type["capital_management"].start_page == 55
    assert by_type["capital_management"].end_page == 63
    assert by_type["risk_management"].start_page == 64
    assert by_type["risk_management"].end_page == 108
    assert by_type["risk_management"].end_detection_method == "annual_t4_vision_verified_successor"
    assert outcome.diagnostics["page_offset"] == 2
    assert outcome.diagnostics["offset_votes"] == [2, 2]
    assert outcome.diagnostics["status"] == "verified"
    assert len(detector.transition_calls) == 3


def test_validator_preserves_existing_bounds_without_openai() -> None:
    detector = FakeDetector()
    detector.api_key = ""
    validator = AnnualSectionBoundaryValidator(
        "bnc",
        2023,
        detector=detector,
        docling_reader=_docling_reader,
    )
    section = LocatedSection(
        section_type="risk_management",
        title_found="Gestion des risques",
        start_page=64,
        end_page=183,
        confidence=1.0,
        detection_method="annual_t4_physical_title",
        end_detection_method="annual_t4_safety_cap_no_successor",
    )

    outcome = validator.validate(
        Path("BNC_2023_T4.pdf"),
        [section],
        {64: "Gestion des risques"},
        candidate_pages=[15],
    )

    assert outcome.sections[0].end_page == 183
    assert outcome.diagnostics["status"] == "not_validated"
    assert "openai_vision_unavailable" in outcome.diagnostics["warnings"]
