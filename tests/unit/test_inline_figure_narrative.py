from __future__ import annotations

from types import SimpleNamespace

import pytest

from vigie.analyse_texte.docling_markdown import (
    _build_text_extraction_markdown_from_docling,
    _parse_docling_markdown,
)
from vigie.analyse_texte.extraction import (
    _annotate_docling_image_markers,
    _merge_missing_fallback_blocks,
    _native_docling_markdown,
)
from vigie.analyse_texte.figure_narrative import (
    FigureNarrative,
    OpenAIFigureNarrator,
)
from vigie.analyse_texte.markdown import (
    _extract_section_text_from_markdown,
    _parse_page_index_from_markdown,
)
from vigie.analyse_texte.models import PDFBlock, SectionAudit


def _audit() -> SectionAudit:
    return SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=116,
        end_page=116,
        anchor_page=116,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.05, 0.05, 0.95, 0.08],
        included_blocks=[
            PDFBlock(
                "p116_d001",
                116,
                [0.05, 0.10, 0.95, 0.20],
                "La direction transmet régulièrement ses attentes aux employés.",
                1,
                "narrative",
                True,
            ),
            PDFBlock(
                "p116_d002",
                116,
                [0.05, 0.68, 0.95, 0.78],
                "Le risque de non-conformité demeure surveillé.",
                2,
                "narrative",
                True,
            ),
        ],
        excluded_blocks=[],
    )


class _FakeNarrator:
    calls_made = 0

    def describe(self, **kwargs):
        self.calls_made += 1
        assert kwargs["page"] == 116
        assert kwargs["bbox_norm"] == pytest.approx([0.12, 0.27, 0.88, 0.56])
        return FigureNarrative(
            visual_type="org_chart",
            title="Culture d’entreprise et conduite",
            summary="Le diagramme relie la direction, la culture, la conduite et les résultats.",
            elements=["Direction de l’organisation", "Conduite individuelle et collective"],
            relationships=["Direction de l’organisation → Culture : détermine"],
            explicit_values=[],
            trends=[],
            confidence=0.96,
        )


def test_annotated_figure_is_inserted_in_the_single_compared_markdown() -> None:
    raw = (
        "La direction transmet régulièrement ses attentes aux employés.\n\n"
        '<!-- image page="116" bbox="0.12,0.27,0.88,0.56" -->\n\n'
        "Le risque de non-conformité demeure surveillé.\n"
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [_audit()],
        raw_docling_markdown=raw,
        figure_narrator=_FakeNarrator(),
    )
    compared_text = _extract_section_text_from_markdown(markdown, "gestion_risques")
    page_index, _starts = _parse_page_index_from_markdown(markdown)

    assert "### Figure — Culture d’entreprise et conduite [pdf.116]" in markdown
    assert "Direction de l’organisation → Culture : détermine" in markdown
    assert "Le diagramme relie la direction" in compared_text
    assert "Direction de l’organisation → Culture : détermine" in compared_text
    assert "visual_comparison" not in markdown
    assert any(page == 116 and "Direction de l’organisation" in text for page, text in page_index["gestion_risques"])


def test_visual_failure_preserves_surrounding_narrative_without_placeholder() -> None:
    raw = (
        "La direction transmet régulièrement ses attentes aux employés.\n\n"
        '<!-- image page="116" bbox="0.12,0.27,0.88,0.56" -->\n\n'
        "Le risque de non-conformité demeure surveillé.\n"
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [_audit()],
        raw_docling_markdown=raw,
        figure_narrator=None,
    )

    assert "La direction transmet régulièrement" in markdown
    assert "Le risque de non-conformité" in markdown
    assert "[visual]" not in markdown
    assert "Figure détectée" not in markdown


def test_parser_keeps_geometry_only_for_annotated_image_markers() -> None:
    annotated = _parse_docling_markdown('<!-- image page="116" bbox="0.12,0.27,0.88,0.56" -->')[0]
    legacy = _parse_docling_markdown("<!-- image -->")[0]

    assert annotated.kind == "visual"
    assert annotated.page == 116
    assert annotated.bbox_norm == pytest.approx([0.12, 0.27, 0.88, 0.56])
    assert legacy.kind == "table"


def test_docling_picture_geometry_is_attached_to_native_marker() -> None:
    picture = SimpleNamespace(
        prov=[
            SimpleNamespace(
                page_no=116,
                bbox=SimpleNamespace(),
            )
        ]
    )
    document = SimpleNamespace(pictures=[picture])

    monkeypatch_bbox = pytest.MonkeyPatch()
    monkeypatch_bbox.setattr(
        "vigie.analyse_texte.extraction._docling_bbox_to_norm",
        lambda _document, _prov: [0.12, 0.27, 0.88, 0.56],
    )
    try:
        enriched = _annotate_docling_image_markers("Avant\n\n<!-- image -->\n\nAprès", document)
    finally:
        monkeypatch_bbox.undo()

    assert '<!-- image page="116" bbox="0.120000,0.270000,0.880000,0.560000" -->' in enriched
    assert _native_docling_markdown(enriched) == "Avant\n\n<!-- image -->\n\nAprès"


def test_marker_region_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    picture = SimpleNamespace(prov=[SimpleNamespace(page_no=116, bbox=SimpleNamespace())])
    document = SimpleNamespace(pictures=[picture])
    monkeypatch.setattr(
        "vigie.analyse_texte.extraction._docling_bbox_to_norm",
        lambda _document, _prov: [0.12, 0.27, 0.88, 0.56],
    )
    markdown = "<!-- image -->\n\n<!-- image -->"

    assert _annotate_docling_image_markers(markdown, document) == markdown


def test_pymupdf_fallback_does_not_duplicate_text_inside_figure() -> None:
    inside = PDFBlock(
        "inside",
        116,
        [0.20, 0.30, 0.80, 0.50],
        "Direction de l’organisation et facteurs de culture",
        1,
    )
    outside = PDFBlock(
        "outside",
        116,
        [0.10, 0.70, 0.90, 0.80],
        "Le risque de non-conformité demeure surveillé par la Banque.",
        2,
    )
    page_blocks = {116: []}

    _merge_missing_fallback_blocks(
        page_blocks,
        {116: [inside, outside]},
        excluded_bboxes_by_page={116: [[0.12, 0.27, 0.88, 0.56]]},
    )

    assert [block.block_id for block in page_blocks[116]] == ["outside"]


def test_figure_at_start_of_section_is_assigned_by_its_docling_page() -> None:
    raw = '<!-- image page="116" bbox="0.12,0.27,0.88,0.56" -->\n\nLe risque de non-conformité demeure surveillé.\n'

    markdown = _build_text_extraction_markdown_from_docling(
        [_audit()],
        raw_docling_markdown=raw,
        figure_narrator=_FakeNarrator(),
    )

    assert "### Figure — Culture d’entreprise et conduite [pdf.116]" in markdown
    assert "Le risque de non-conformité demeure surveillé." in markdown


def test_figure_narrator_keeps_only_structured_visible_content(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = FigureNarrative(
        visual_type="chart",
        title="Revenus et VaR",
        summary="Le graphique combine des barres de revenus et une ligne de VaR.",
        elements=["Barres : revenus de négociation", "Ligne : VaR"],
        relationships=[],
        explicit_values=[],
        trends=["Les revenus fluctuent autour de zéro"],
        confidence=0.94,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **_kwargs: response)))
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.figure_narrative.crop_table_region_to_bytes",
        lambda *_args, **_kwargs: b"png",
    )
    narrator = OpenAIFigureNarrator(
        pdf_path="rapport.pdf",
        client=client,
        model="test-model",
        confidence_threshold=0.75,
        max_calls=2,
        render_dpi=220,
    )

    result = narrator.describe(
        page=34,
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        section_title="Gestion des risques",
        context_before="Le graphique présente les revenus.",
        context_after="Aucune perte nette n’a été enregistrée.",
    )

    assert result == parsed
    assert result.explicit_values == []
    assert narrator.calls_made == 1
