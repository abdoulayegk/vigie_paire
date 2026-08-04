"""Tests de separation entre lignes reelles et signaux discriminants."""

from __future__ import annotations

from vigie.comparaison.io import _is_ghost_table
from vigie.extraction.vision_full import (
    VisionFullResult,
    _grade_extraction_quality,
    _structural_indicator_count,
    _viable_indicator_count,
)


def _result(
    indicators: list[str],
    *,
    headers: list[str] | None = None,
) -> VisionFullResult:
    return VisionFullResult(
        table_title="Répartition géographique",
        table_summary="Répartition par région",
        headers=list(headers or []),
        indicators=indicators,
        footnotes_content=[],
    )


def test_generic_business_rows_are_structural_but_not_discriminating() -> None:
    result = _result(["Canada", "Autres", "Total"])

    assert _structural_indicator_count(result) == 3
    assert _viable_indicator_count(result) == 0


def test_generic_business_rows_do_not_create_a_ghost_table() -> None:
    assert not _is_ghost_table(
        {
            "table_id": "short_table",
            "title": "Répartition géographique",
            "headers": [],
            "indicators": ["Canada", "Autres", "Total"],
        }
    )


def test_narrative_text_without_headers_remains_a_ghost_table() -> None:
    assert _is_ghost_table(
        {
            "table_id": "narrative",
            "title": "Rapport de gestion",
            "headers": [],
            "indicators": [
                "Le tableau présente les principales tendances observées au cours du trimestre."
            ],
        }
    )


def test_two_rows_and_four_columns_are_not_a_truncation_signal() -> None:
    result = _result(
        ["Canada", "Total"],
        headers=["Catégorie", "T1 2026", "T2 2026", "Variation"],
    )

    assert _grade_extraction_quality(result) == []
