"""Tests ciblés pour les écarts de comparaison textuelle (RBC T4 et cas génériques)."""

from __future__ import annotations

from vigie.analyse_texte.docling_markdown import DoclingSegment, _is_end_boundary_heading
from vigie.analyse_texte.extraction import (
    _classify_block_type,
    _inferred_end_anchor_y,
    _merge_missing_fallback_blocks,
)
from vigie.analyse_texte.global_reconciliation import (
    _Node,
    _reconcile_component,
    _ReconciliationResponse,
)
from vigie.analyse_texte.models import PDFBlock, SectionAudit
from vigie.analyse_texte.normalization import _sanitize_semantic_text
from vigie.analyse_texte.semantic_chunking import (
    _deterministic_ranges,
    _requires_semantic_partition,
    _split_sentences,
)
from vigie.analyse_texte.summary import _is_non_cosmetic_change
from vigie.analyse_texte.text_comparison.text_comparison_writer import (
    deduplicate_and_group_section_changes,
)
from vigie.analyse_texte.triage_parts.exclusions import _deterministic_bank_specific_exclusion
from vigie.extraction.localisation_sections.boundary_resolver import resolve_t4_section_bounds
from vigie.extraction.localisation_sections.toc_locator import TocStructure, TocStructureEntry


def test_sanitize_semantic_text_keeps_indefinite_deferral_skeleton() -> None:
    raw = (
        "Le 12 février 2025, le BSIF a annoncé un report indéfini des augmentations "
        "du plancher de fonds propres, avec un préavis d'au moins deux ans."
    )

    cleaned = _sanitize_semantic_text(raw)

    assert "12" not in cleaned
    assert "2025" not in cleaned
    assert "BSIF" not in cleaned
    assert "annoncé un report indéfini" in cleaned
    assert "préavis" in cleaned
    assert cleaned != "Le a annoncé , ."


def test_definition_sentence_is_split_from_continuation() -> None:
    paragraph = (
        "L'appétit pour le risque se définit comme étant le niveau et le type de risque "
        "que la banque est prête à accepter. "
        "Il reflète les objectifs stratégiques de l'entreprise. "
        "Il est généralement fixé par le conseil d'administration. "
        "L'appétit pour le risque est étayé par les pouvoirs délégués."
    )
    sentences = _split_sentences(paragraph)
    scores = [0.90] * (len(sentences) - 1)

    assert _requires_semantic_partition(paragraph)
    ranges = _deterministic_ranges(sentences, scores)

    assert ranges[0] == (0, 1)
    assert "se définit comme étant" in sentences[0]
    assert not any("se définit" in " ".join(sentences[start:end]) for start, end in ranges[1:])


def test_calendar_exclusion_keeps_indefinite_deferral_with_notice() -> None:
    date_only = {
        "diff_type": "modified",
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2027."
        ),
        "source_text_t2": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2026."
        ),
    }
    indefinite = {
        "diff_type": "modified",
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2027."
        ),
        "source_text_t2": (
            "Le 12 février 2025, le BSIF a annoncé un report indéfini des augmentations "
            "du plancher, avec un préavis d'au moins deux ans."
        ),
    }

    assert _deterministic_bank_specific_exclusion(date_only) == "mise_a_jour_calendrier"
    assert _deterministic_bank_specific_exclusion(indefinite) is None


def test_esg_to_durability_same_disclosure_modified_is_kept() -> None:
    text_t1 = (
        "Le Comité de gouvernance examine les questions ESG et supervise "
        "la divulgation des enjeux environnementaux et sociaux de la banque."
    )
    text_t2 = (
        "Le Comité de gouvernance examine les questions liées à la durabilité et supervise "
        "la divulgation des enjeux de durabilité de la banque."
    )
    previous = _Node(
        node_id="n0000",
        order=0,
        change={
            "change_id": "esg_removed",
            "section_key": "gestion_risques",
            "diff_type": "removed",
            "source_text_t1": text_t1,
            "pages_t1": [12],
        },
        side="t1",
        text=text_t1,
    )
    current = _Node(
        node_id="n0001",
        order=1,
        change={
            "change_id": "durabilite_added",
            "section_key": "gestion_risques",
            "diff_type": "added",
            "source_text_t2": text_t2,
            "pages_t2": [10],
        },
        side="t2",
        text=text_t2,
    )
    response = _ReconciliationResponse.model_validate(
        {
            "decision": "same_disclosure_modified",
            "confidence": "high",
            "rationale": "Même divulgation de gouvernance, terminologie ESG remplacée par durabilité.",
            "matches": [
                {
                    "t1_node_id": "n0000",
                    "t2_node_id": "n0001",
                    "text_t1": text_t1,
                    "text_t2": text_t2,
                }
            ],
        }
    )

    replacements, audit = _reconcile_component(component=[previous, current], response=response)

    kept = [item for item in replacements.values() if item is not None]
    assert len(kept) == 1
    assert kept[0]["diff_type"] == "modified"
    assert "ESG" in kept[0]["source_text_t1"]
    assert "durabilité" in kept[0]["source_text_t2"]
    assert audit["applied"] is True


def test_removed_basel_paragraph_is_retained_even_if_triage_is_irrelevant() -> None:
    change = {
        "diff_type": "removed",
        "source_text_t1": (
            "Les réformes définitives de Bâle III publiées par le CBCB établissent "
            "le cadre prudentiel applicable aux fonds propres réglementaires."
        ),
        "genai_triage": {"is_relevant": False, "themes_amf": []},
    }

    assert _is_non_cosmetic_change(change["genai_triage"]) is False
    assert _is_non_cosmetic_change(change["genai_triage"], change) is True


def test_modified_indefinite_deferral_is_retained_even_if_triage_is_irrelevant() -> None:
    date_only = {
        "diff_type": "modified",
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2027."
        ),
        "source_text_t2": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2026."
        ),
        "genai_triage": {"is_relevant": False, "themes_amf": []},
    }
    indefinite = {
        "diff_type": "modified",
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé un report d'un an de "
            "l'augmentation du plancher de fonds propres jusqu'à l'exercice 2026."
        ),
        "source_text_t2": (
            "Le 12 février 2025, le BSIF a annoncé un report indéfini des "
            "augmentations du plancher de fonds propres, avec un préavis "
            "d'au moins deux ans."
        ),
        "genai_triage": {
            "is_relevant": False,
            "themes_amf": [],
            "coherence_review_required": True,
            "source": "triage_coherence_review_required",
        },
    }

    assert _is_non_cosmetic_change(date_only["genai_triage"], date_only) is False
    assert _is_non_cosmetic_change(indefinite["genai_triage"], indefinite) is True


def test_distinct_removed_paragraphs_in_same_subsection_are_not_collapsed() -> None:
    changes = [
        {
            "diff_type": "removed",
            "subsection_heading": "Accord de Bâle III",
            "source_text_t1": "Les réformes définitives de Bâle III sont appliquées progressivement.",
            "pages_t1": [128],
        },
        {
            "diff_type": "removed",
            "subsection_heading": "Accord de Bâle III",
            "source_text_t1": "Le CBCB a publié les exigences finales de fonds propres.",
            "pages_t1": [129],
        },
    ]

    grouped = deduplicate_and_group_section_changes(changes)

    assert len(grouped) == 2


def test_visual_fallback_cells_are_not_classified_as_tables() -> None:
    block = PDFBlock(
        block_id="p076_m001",
        page=76,
        bbox_norm=[0.12, 0.40, 0.35, 0.48],
        text="Facteurs macroéconomiques",
        line_number=1,
        source_label="pymupdf_visual",
    )

    assert _classify_block_type(block, {}) == "other"


def test_visual_fallback_recovers_short_labels() -> None:
    page_blocks = {76: []}
    fallback = {
        76: [
            PDFBlock(
                "p076_m001",
                76,
                [0.20, 0.42, 0.40, 0.48],
                "stratégiques",
                1,
                source_label="pymupdf_fallback",
            )
        ]
    }

    _merge_missing_fallback_blocks(page_blocks, fallback, {76: [[0.10, 0.35, 0.90, 0.70]]})

    assert len(page_blocks[76]) == 1
    assert page_blocks[76][0].source_label == "pymupdf_visual"
    assert page_blocks[76][0].text == "stratégiques"


def test_accounting_questions_heading_closes_capital_section() -> None:
    audits = [
        SectionAudit(
            section_key="gestion_capital",
            section_title="Gestion du capital",
            start_page=127,
            end_page=140,
            anchor_page=127,
            anchor_text="Gestion des fonds propres",
            anchor_bbox_norm=[0.1, 0.1, 0.8, 0.14],
            included_blocks=[],
            excluded_blocks=[],
            end_anchor_page=140,
            end_anchor_text="Questions en matière de comptabilité et de contrôle",
        )
    ]

    assert _is_end_boundary_heading(
        DoclingSegment(
            kind="heading",
            text="Questions en matière de comptabilité et de contrôle",
            heading_level=2,
        ),
        audits,
    )
    assert not _is_end_boundary_heading(
        DoclingSegment(
            kind="heading",
            text="Faits nouveaux en matière de réglementation",
            heading_level=3,
        ),
        audits,
    )


def test_inferred_end_anchor_clips_accounting_heading() -> None:
    blocks = [
        PDFBlock("p140_d001", 140, [0.1, 0.20, 0.9, 0.28], "Faits nouveaux en matière de réglementation", 1),
        PDFBlock("p140_d002", 140, [0.1, 0.30, 0.9, 0.40], "Le 11 septembre 2025, le BSIF a publié des révisions.", 2),
        PDFBlock("p140_d003", 140, [0.1, 0.55, 0.9, 0.62], "Questions en matière de comptabilité et de contrôle", 3),
        PDFBlock("p140_d004", 140, [0.1, 0.65, 0.9, 0.80], "Les méthodes comptables sont décrites ci-dessous.", 4),
    ]

    assert _inferred_end_anchor_y(blocks, "en matière de comptabilité etde contrôle") == 0.55


def test_resolve_bounds_skips_nested_regulatory_successor() -> None:
    toc = TocStructure(
        rg_page=17,
        confidence=0.9,
        offset=2,
        entries=[
            TocStructureEntry("Gestion des fonds propres", 125, physical_page=127),
            TocStructureEntry("Faits nouveaux en matière de réglementation", 138, physical_page=140),
            TocStructureEntry("Questions comptables", 138, physical_page=140),
        ],
    )

    outcome = resolve_t4_section_bounds(toc, bank_code="rbc")
    capital = next(section for section in outcome.sections if section.section_type == "gestion_capital")

    assert capital.end_page == 139
    assert capital.end_anchor_text == "Questions comptables"
