from __future__ import annotations

import pytest

from vigie.support.quarter_utils import (
    build_quarter_context,
    format_quarter_display_label,
    parse_quarter_ref,
    quarter_label_from_payload,
)


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


def test_parse_quarter_ref_accepts_year_first_periods() -> None:
    ref = parse_quarter_ref("2025_t3")

    assert ref.label == "Q3-2025"
    assert ref.display_label == "T3 2025"


def test_quarter_label_from_payload_uses_french_display_labels() -> None:
    payload = {
        "year_previous": 2025,
        "quarter_previous": "t2",
        "year_current": 2025,
        "quarter_current": "t3",
    }

    assert quarter_label_from_payload(payload, "current") == "T3 2025"
    assert quarter_label_from_payload(payload, "previous") == "T2 2025"
    assert format_quarter_display_label("2026_t1") == "T1 2026"
