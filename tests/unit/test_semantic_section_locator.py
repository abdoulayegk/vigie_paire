"""Tests du renforcement sémantique de la localisation des sections."""

from __future__ import annotations

from vigie.extraction.localisation_sections.models import LocatedSection, TocEntry
from vigie.extraction.localisation_sections.page_offsets import infer_page_offset
from vigie.extraction.localisation_sections.semantic_locator import (
    SemanticDecisionBatch,
    SemanticEntryDecision,
    merge_semantic_sections,
    resolve_semantic_toc_sections,
)


def _semantic_config() -> dict:
    return {
        "section_semantic_localization": {
            "enabled": True,
            "embedding_model": "embedding-test",
            "llm_model": "llm-test",
            "shortlist_per_concept": 5,
            "max_candidates": 40,
            "min_llm_candidate_confidence": 0.7,
            "min_llm_confidence": 0.8,
            "ambiguous_margin": 0.04,
            "allow_regulatory_discovery": True,
        }
    }


def _fake_embeddings(texts: list[str], _model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        normalized = text.lower()
        if any(token in normalized for token in ("capital", "fonds propres", "solidité", "absorption")):
            vectors.append([1.0, 0.0, 0.0, 0.0])
        elif any(token in normalized for token in ("risque", "menace", "résilience", "quantique")):
            vectors.append([0.0, 1.0, 0.0, 0.0])
        elif any(token in normalized for token in ("réglement", "regulatory")):
            vectors.append([0.0, 0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 0.0, 1.0])
    return vectors


def test_new_subsections_do_not_cut_major_sections() -> None:
    entries = [
        TocEntry("Solidité financière et capacité d'absorption", 20, level=0),
        TocEntry("Ratios prudentiels et coussins internes", 21, level=1),
        TocEntry("Gouvernance des modèles génératifs", 24, level=1),
        TocEntry("Gestion intégrée des menaces", 30, level=0),
        TocEntry("Risque quantique", 33, level=1),
        TocEntry("Questions comptables", 40, level=0),
    ]

    roles = {
        "Solidité financière et capacité d'absorption": ("capital_management", "main_section", None),
        "Ratios prudentiels et coussins internes": ("capital_management", "subsection", "toc_000"),
        "Gouvernance des modèles génératifs": ("risk_management", "subsection", "toc_003"),
        "Gestion intégrée des menaces": ("risk_management", "main_section", None),
        "Risque quantique": ("risk_management", "subsection", "toc_003"),
        "Questions comptables": ("other", "main_section", None),
    }

    def decide(candidates: list[dict], _model: str) -> SemanticDecisionBatch:
        decisions = []
        for candidate in candidates:
            concept, role, parent = roles[candidate["title"]]
            decisions.append(
                SemanticEntryDecision(
                    candidate_id=candidate["candidate_id"],
                    concept=concept,
                    role=role,
                    parent_candidate_id=parent,
                    confidence=0.95,
                    reason="fixture",
                )
            )
        return SemanticDecisionBatch(decisions=decisions, warnings=[])

    outcome = resolve_semantic_toc_sections(
        entries,
        bank_code="future_bank",
        config=_semantic_config(),
        embedding_provider=_fake_embeddings,
        decision_provider=decide,
    )

    by_type = {section.section_type: section for section in outcome.sections}
    assert by_type["gestion_capital"].start_page == 20
    assert by_type["gestion_capital"].end_page == 29
    assert by_type["gestion_risques"].start_page == 30
    assert by_type["gestion_risques"].end_page == 39
    assert entries[1].semantic_role == "subsection"
    assert entries[4].semantic_parent_title == "Gestion intégrée des menaces"


def test_regulatory_subsection_is_not_promoted_to_independent_section() -> None:
    entries = [
        TocEntry("Gestion des risques", 20, level=0),
        TocEntry("Faits nouveaux en matière de réglementation", 28, level=1),
        TocEntry("Questions comptables", 40, level=0),
    ]

    def decide(candidates: list[dict], _model: str) -> SemanticDecisionBatch:
        decisions = []
        for candidate in candidates:
            if candidate["title"] == "Gestion des risques":
                concept, role, parent = "risk_management", "main_section", None
            elif "réglementation" in candidate["title"]:
                concept, role, parent = "regulatory_updates", "subsection", "toc_000"
            else:
                concept, role, parent = "other", "main_section", None
            decisions.append(
                SemanticEntryDecision(
                    candidate_id=candidate["candidate_id"],
                    concept=concept,
                    role=role,
                    parent_candidate_id=parent,
                    confidence=0.96,
                    reason="fixture",
                )
            )
        return SemanticDecisionBatch(decisions=decisions, warnings=[])

    outcome = resolve_semantic_toc_sections(
        entries,
        bank_code="cibc",
        config=_semantic_config(),
        embedding_provider=_fake_embeddings,
        decision_provider=decide,
    )

    assert {section.section_type for section in outcome.sections} == {"gestion_risques"}
    assert outcome.diagnostics["concept_status"]["regulatory_updates"] == "not_found"


def test_close_competing_major_titles_return_ambiguous() -> None:
    entries = [
        TocEntry("Solidité financière", 20, level=0),
        TocEntry("Capacité financière", 45, level=0),
    ]

    def decide(candidates: list[dict], _model: str) -> SemanticDecisionBatch:
        confidences = {"Solidité financière": 0.91, "Capacité financière": 0.89}
        return SemanticDecisionBatch(
            decisions=[
                SemanticEntryDecision(
                    candidate_id=candidate["candidate_id"],
                    concept="capital_management",
                    role="main_section",
                    confidence=confidences[candidate["title"]],
                    reason="fixture",
                )
                for candidate in candidates
            ],
            warnings=[],
        )

    outcome = resolve_semantic_toc_sections(
        entries,
        bank_code="future_bank",
        config=_semantic_config(),
        embedding_provider=_fake_embeddings,
        decision_provider=decide,
    )

    assert outcome.status == "ambiguous"
    assert outcome.sections == []
    assert outcome.diagnostics["concept_status"]["capital_management"] == "ambiguous"


def test_incomplete_llm_batch_fails_closed_as_ambiguous() -> None:
    entries = [
        TocEntry("Solidité financière", 20, level=0),
        TocEntry("Gestion intégrée des risques", 30, level=0),
    ]

    def decide(candidates: list[dict], _model: str) -> SemanticDecisionBatch:
        first = candidates[0]
        return SemanticDecisionBatch(
            decisions=[
                SemanticEntryDecision(
                    candidate_id=first["candidate_id"],
                    concept="capital_management",
                    role="main_section",
                    confidence=0.96,
                    reason="fixture volontairement incomplète",
                )
            ],
            warnings=[],
        )

    outcome = resolve_semantic_toc_sections(
        entries,
        bank_code="future_bank",
        config=_semantic_config(),
        embedding_provider=_fake_embeddings,
        decision_provider=decide,
    )

    assert outcome.status == "ambiguous"
    assert outcome.sections == []
    assert "missing_decisions:1" in outcome.diagnostics["warnings"]


def test_novel_title_below_strong_threshold_is_forwarded_to_vision() -> None:
    entries = [
        TocEntry("Gestion intégrée des menaces", 30, level=0),
        TocEntry("Risque quantique", 34, level=1),
        TocEntry("Questions comptables", 40, level=0),
    ]

    def decide(candidates: list[dict], _model: str) -> SemanticDecisionBatch:
        decisions = []
        for candidate in candidates:
            title = candidate["title"]
            if title == "Gestion intégrée des menaces":
                concept, role, confidence = "risk_management", "main_section", 0.72
            elif title == "Risque quantique":
                concept, role, confidence = "risk_management", "subsection", 0.91
            else:
                concept, role, confidence = "other", "main_section", 0.96
            decisions.append(
                SemanticEntryDecision(
                    candidate_id=candidate["candidate_id"],
                    concept=concept,
                    role=role,
                    confidence=confidence,
                    reason="fixture",
                )
            )
        return SemanticDecisionBatch(decisions=decisions, warnings=[])

    outcome = resolve_semantic_toc_sections(
        entries,
        bank_code="future_bank",
        config=_semantic_config(),
        embedding_provider=_fake_embeddings,
        decision_provider=decide,
    )

    risk = next(section for section in outcome.sections if section.section_type == "gestion_risques")
    assert risk.semantic_status == "vision_required"
    assert outcome.diagnostics["concept_status"]["risk_management"] == "vision_required"


def test_multi_anchor_offset_overrides_stale_configuration() -> None:
    entries = [
        TocEntry("Gestion des fonds propres", 26),
        TocEntry("Gestion du risque", 31),
        TocEntry("Questions comptables", 50),
    ]
    text_by_page = {
        30: "Gestion des fonds propres\nRatios CET1",
        35: "Gestion du risque\nVue d'ensemble",
        54: "Questions comptables\nMéthodes",
    }

    outcome = infer_page_offset(text_by_page, entries, configured_offset=3)

    assert outcome.offset == 4
    assert outcome.status == "inferred_override"
    assert outcome.votes == {4: 3}


def test_single_anchor_cannot_override_configured_offset() -> None:
    entries = [TocEntry("Gestion du capital", 62)]
    text_by_page = {66: "Gestion du capital\nRatios"}

    outcome = infer_page_offset(text_by_page, entries, configured_offset=0)

    assert outcome.offset == 0
    assert outcome.status == "ambiguous"
    assert outcome.votes == {4: 1}


def test_semantic_merge_preserves_strong_detection_and_marks_conflict() -> None:
    current = LocatedSection("gestion_risques", "Gestion des risques", 20, confidence=0.95)
    semantic = LocatedSection(
        "gestion_risques",
        "Facteurs émergents",
        40,
        confidence=0.94,
        detection_method="toc_semantic",
    )

    merged, warnings = merge_semantic_sections([current], [semantic])

    assert merged == [current]
    assert current.semantic_status == "ambiguous"
    assert warnings == ["semantic_page_conflict:risk_management:deterministic=20:semantic=40"]
