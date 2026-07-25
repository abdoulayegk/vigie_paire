from __future__ import annotations

import pytest

from vigilance.text_analysis.boundary_repair import (
    BoundaryDisposition,
    RepairableBlock,
    classify_boundary,
    repair_block_boundaries,
)
from vigilance.text_analysis.canonical_cleanup import (
    adjacent_duplicate_key,
    canonicalize_surface_text,
    cleanup_canonical_fragment,
    is_quarterly_running_chrome,
    is_standalone_table_marker_definition,
)
from vigilance.text_analysis.chunking import _chunk_subsection_text
from vigilance.text_analysis.docling_markdown import (
    DoclingSegment,
    _filter_reinserted_section_segments,
    _parse_docling_markdown,
    _repair_nonadjacent_dangling_boundaries,
    _should_keep_docling_segment,
)
from vigilance.text_analysis.models import SectionAudit


def test_surface_cleanup_decodes_html_and_normalizes_spaces() -> None:
    assert canonicalize_surface_text("  M&amp;I\u00a0 : contrôle  ") == "M&I: contrôle"
    assert adjacent_duplicate_key("M&amp;I") == adjacent_duplicate_key("M&I")


@pytest.mark.parametrize(
    "text",
    [
        "28 Banque Royale du Canada Premier trimestre de 2026",
        "28 | RBC – Premier trimestre de 2026",
        "28 Premier trimestre de 2026 – Banque Royale du Canada",
    ],
)
def test_quarterly_chrome_is_removed_without_requiring_word_report(text: str) -> None:
    assert is_quarterly_running_chrome(text)
    decision = cleanup_canonical_fragment(text)
    assert decision.keep is False
    assert decision.reason == "running_header_footer"


@pytest.mark.parametrize(
    "text",
    ["s. o. – sans objet", "n. s. - non significatif", "négl. : négligeable"],
)
def test_standalone_table_marker_definitions_are_removed(text: str) -> None:
    assert is_standalone_table_marker_definition(text)
    assert cleanup_canonical_fragment(text).reason == "table_marker_definition"


def test_narrative_containing_not_applicable_words_is_preserved() -> None:
    decision = cleanup_canonical_fragment("Cette exigence est sans objet pour les filiales vendues.")
    assert decision.keep is True


def test_certain_sentence_continuation_is_merged_without_rewriting() -> None:
    result = repair_block_boundaries(
        [
            RepairableBlock("paragraph", "Le cadre doit"),
            RepairableBlock("paragraph", "faire preuve de cohérence."),
        ]
    )
    assert [block.text for block in result.blocks] == ["Le cadre doit faire preuve de cohérence."]
    assert len(result.merged_boundaries) == 1


def test_complete_sentence_and_short_label_remain_separate() -> None:
    complete = classify_boundary(
        RepairableBlock("paragraph", "Le cadre est approuvé."),
        RepairableBlock("paragraph", "il est revu chaque année."),
    )
    label = classify_boundary(
        RepairableBlock("paragraph", "Gestion des risques"),
        RepairableBlock("paragraph", "la Banque applique trois lignes de défense."),
    )
    assert complete.disposition is BoundaryDisposition.KEEP
    assert label.disposition is BoundaryDisposition.KEEP


def test_missing_punctuation_before_uppercase_is_ambiguous() -> None:
    decision = classify_boundary(
        RepairableBlock("paragraph", "Le cadre demeure applicable"),
        RepairableBlock("paragraph", "La Banque le révise annuellement."),
    )
    assert decision.disposition is BoundaryDisposition.AMBIGUOUS


def test_chunking_repairs_split_sentence_but_not_across_heading() -> None:
    repaired = _chunk_subsection_text("Le cadre doit\n\nfaire preuve de cohérence.")
    separated = _chunk_subsection_text(
        "Le cadre demeure applicable\n\n### Nouveau titre\n\nla surveillance est annuelle."
    )
    assert [chunk.text for chunk in repaired] == ["Le cadre doit faire preuve de cohérence."]
    assert [chunk.text for chunk in separated] == [
        "Le cadre demeure applicable",
        "la surveillance est annuelle.",
    ]


def _empty_audit() -> SectionAudit:
    return SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=1,
        end_page=2,
        anchor_page=1,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[],
        excluded_blocks=[],
    )


def test_cleanup_is_reapplied_after_pdf_safety_reinsertion() -> None:
    events = []
    filtered = _filter_reinserted_section_segments(
        [
            DoclingSegment(kind="paragraph", text="s. o. – sans objet", page=1),
            DoclingSegment(kind="paragraph", text="La Banque surveille ses risques.", page=1),
        ],
        audit=_empty_audit(),
        audit_events=events,
    )
    assert [segment.text for segment in filtered] == ["La Banque surveille ses risques."]
    assert events[0]["reason"] == "table_marker_definition"
    assert events[0]["stage"] == "post_reinsertion"


def test_nonadjacent_dangling_continuation_requires_vision() -> None:
    class _Decision:
        apply_merge = True

        @staticmethod
        def model_dump():
            return {"apply_merge": True, "confidence": 0.98}

    class _Validator:
        def validate(self, previous, current):
            assert previous.text == "Il doit"
            assert current.text.startswith("faire preuve")
            return _Decision()

    segments = [
        DoclingSegment(kind="paragraph", text="Il doit", page=4),
        DoclingSegment(kind="heading", text="Cadre d'appétit", page=4),
        DoclingSegment(kind="list_item", text="Premier objectif.", page=4),
        DoclingSegment(kind="paragraph", text="faire preuve de leadership.", page=4),
    ]
    events = []
    repaired = _repair_nonadjacent_dangling_boundaries(
        segments,
        boundary_validator=_Validator(),
        audit_events=events,
    )
    assert [segment.text for segment in repaired] == [
        "Il doit faire preuve de leadership.",
        "Cadre d'appétit",
        "Premier objectif.",
    ]
    assert events[0]["reason"] == "vision_confirmed_nonadjacent_same_sentence"


def test_docling_image_marker_marks_following_numbered_line_as_visual_note() -> None:
    segments = _parse_docling_markdown(
        "<!-- image -->\n\n"
        "1) Se reporter à la section Gestion des fonds propres pour plus de précisions.\n\n"
        "2) Comprend les dérivés et les transactions assimilées.\n\n"
        "Le cadre de risque demeure applicable."
    )
    assert segments[0].kind == "table"
    assert segments[1].follows_table is True
    assert segments[2].follows_table is True
    assert _should_keep_docling_segment(segments[1]) is False
    assert _should_keep_docling_segment(segments[2]) is False
    assert _should_keep_docling_segment(segments[3]) is True
