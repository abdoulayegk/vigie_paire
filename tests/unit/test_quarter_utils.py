from __future__ import annotations

import pytest

from app.quarter_utils import build_quarter_context, parse_quarter_ref


@pytest.mark.parametrize(
    ("current_quarter", "year", "expected_previous"),
    [
        ("Q2", 2025, "Q1-2025"),
        ("Q3", 2025, "Q2-2025"),
        ("Q1", 2026, "Q3-2025"),
        ("Q4", 2026, "Q4-2025"),
    ],
)
def test_build_quarter_context_infers_previous_quarter(
    current_quarter: str,
    year: int,
    expected_previous: str,
) -> None:
    ctx = build_quarter_context(current_quarter, year=year)

    assert ctx["current"]["label"] == f"{current_quarter}-{year}"
    assert ctx["previous"]["label"] == expected_previous
    assert ctx["comparison_direction"] == "current_vs_previous"


@pytest.mark.parametrize(
    ("raw_value", "year", "expected_label"),
    [
        ("T2-2025", None, "Q2-2025"),
        ("q2_2025", None, "Q2-2025"),
        ("Q2", 2025, "Q2-2025"),
    ],
)
def test_parse_quarter_ref_accepts_repo_formats(
    raw_value: str,
    year: int | None,
    expected_label: str,
) -> None:
    ref = parse_quarter_ref(raw_value, year=year)

    assert ref.label == expected_label
