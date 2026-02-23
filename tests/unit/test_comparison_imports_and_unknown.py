"""Smoke checks for comparison module imports and unknown section behavior."""

from __future__ import annotations

from vigilance.comparison.structural_comparator import (
    StructuralComparisonResult,
    analyze_and_format_structural_changes_multi_section,
)


def test_comparison_modules_importable() -> None:
    modules = [
        "vigilance.comparison.match_signals",
        "vigilance.comparison.match_decision",
        "vigilance.comparison.robust_table_matcher",
        "vigilance.comparison.displacement_detector",
        "vigilance.comparison.table_preview_matcher",
        "vigilance.comparison.structural_comparator",
        "vigilance.comparison.indicator_comparator",
    ]
    for module in modules:
        __import__(module)


def test_structural_analysis_uses_unknown_section_when_missing_mapping() -> None:
    result = StructuralComparisonResult(
        tables_added=[{"title": "Tableau test", "page_number": 12}],
        bank_code="rbc",
    )

    analyzed = analyze_and_format_structural_changes_multi_section(
        result=result,
        page_to_section_t1={},
        page_to_section_t2={},
        api_key=None,
        use_genai=False,
    )

    assert analyzed.changes
    assert any(change.titre == "unknown_section" for change in analyzed.changes)
