"""Unit tests for conservative table fragment merger."""

from __future__ import annotations

from vigilance.compare.table_fragment_merger import merge_table_fragments
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str,
    title: str,
    page: int,
    rows: list[list[str]],
    bbox: list[float] | None = None,
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "T2 2025"],
        rows=rows,
        first_column_indicators=[row[0] for row in rows if row and str(row[0]).strip()],
        extraction_method="docling",
        table_number=None,
        bbox=bbox,
        quarter="t2-2025",
        pdf_path="dummy.pdf",
    )


def test_merges_adjacent_fragments_with_high_confidence() -> None:
    left = _table(
        table_id="t_left",
        section="risk_management",
        title="Analyse des echeances",
        page=10,
        rows=[["segment a", "1"], ["segment b", "2"]],
        bbox=[0.05, 0.60, 0.95, 0.95],
    )
    right = _table(
        table_id="t_right",
        section="risk_management",
        title="Analyse des echeances (suite)",
        page=11,
        rows=[["segment c", "3"], ["segment d", "4"]],
        bbox=[0.05, 0.10, 0.95, 0.40],
    )

    merged, events = merge_table_fragments([left, right], merge_score_min=0.85)
    assert len(merged) == 1
    assert len(events) == 1
    assert merged[0].table_id == "t_left__t_right"
    assert len(merged[0].rows) == 4


def test_does_not_merge_different_sections() -> None:
    left = _table(
        table_id="t_left",
        section="risk_management",
        title="Analyse des echeances",
        page=10,
        rows=[["segment a", "1"]],
    )
    right = _table(
        table_id="t_right",
        section="capital_management",
        title="Analyse des echeances (suite)",
        page=11,
        rows=[["segment b", "2"]],
    )

    merged, events = merge_table_fragments([left, right], merge_score_min=0.85)
    assert len(merged) == 2
    assert events == []


def test_does_not_merge_when_previous_fragment_looks_complete() -> None:
    left = _table(
        table_id="t_left",
        section="risk_management",
        title="Analyse des echeances",
        page=10,
        rows=[["segment a", "1"], ["total", "999"]],
        bbox=[0.05, 0.60, 0.95, 0.95],
    )
    right = _table(
        table_id="t_right",
        section="risk_management",
        title="Analyse des echeances",
        page=11,
        rows=[["segment b", "2"]],
        bbox=[0.05, 0.10, 0.95, 0.40],
    )

    merged, events = merge_table_fragments([left, right], merge_score_min=0.85)
    assert len(merged) == 2
    assert events == []


def test_does_not_merge_near_duplicates() -> None:
    left = _table(
        table_id="t_left",
        section="risk_management",
        title="Risque de credit",
        page=12,
        rows=[["canada", "1"], ["us", "2"]],
        bbox=[0.05, 0.10, 0.95, 0.45],
    )
    right = _table(
        table_id="t_right",
        section="risk_management",
        title="Risque de credit",
        page=12,
        rows=[["canada", "10"], ["us", "20"]],
        bbox=[0.05, 0.47, 0.95, 0.82],
    )

    merged, events = merge_table_fragments([left, right], merge_score_min=0.85)
    assert len(merged) == 2
    assert events == []


def test_does_not_merge_unknown_sections() -> None:
    left = _table(
        table_id="t_left",
        section="unknown_section",
        title="Bloc 1",
        page=5,
        rows=[["a", "1"]],
    )
    right = _table(
        table_id="t_right",
        section="unknown_section",
        title="Bloc 2",
        page=6,
        rows=[["b", "2"]],
    )

    merged, events = merge_table_fragments([left, right], merge_score_min=0.85)
    assert len(merged) == 2
    assert events == []

