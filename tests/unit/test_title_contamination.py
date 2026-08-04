"""Tests for table title contamination detection and cleaning."""

from __future__ import annotations

import pytest

from vigie.support.utils.indicator_cleaner import (
    clean_table_title_contamination,
    is_table_title_contaminated,
)


@pytest.mark.parametrize(
    "title",
    [
        "Expositions brutes 79 772 76 163",
        "Expositions brutes au risque de credit 79 772 76 163 80 000",
        "Title 79,772 76,163 80,000",
        "Some table 1,234 5,678 9,012",
        "Revenue 100 200 300",
    ],
)
def test_contaminated_title_detected(title: str) -> None:
    """Titles with long numeric runs or multiple trailing numbers are contaminated."""
    assert is_table_title_contaminated(title) is True


@pytest.mark.parametrize(
    "title",
    [
        "Tableau 28",
        "Tableau 28 (1)",
        "Capital réglementaire",
        "Expositions brutes au risque de credit",
        "NOTATIONS DE CREDIT (1)",
        "",
    ],
)
def test_normal_title_not_contaminated(title: str) -> None:
    """Single table number or no amount run: not contaminated."""
    assert is_table_title_contaminated(title) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Expositions brutes 79 772 76 163", "Expositions brutes"),
        ("Title 79,772 76,163", "Title"),
        ("Expositions brutes 79 772 76 163 80 000", "Expositions brutes"),
        ("Capital réglementaire", "Capital réglementaire"),
        ("Tableau 28 (1)", "Tableau 28 (1)"),
    ],
)
def test_clean_table_title_strips_amounts(raw: str, expected: str) -> None:
    """clean_table_title_contamination removes trailing/leading numeric runs."""
    assert clean_table_title_contamination(raw) == expected


def test_clean_table_title_unchanged_when_normal() -> None:
    """Normal title (no contamination) is returned unchanged."""
    normal = "Expositions brutes au risque de credit"
    assert clean_table_title_contamination(normal) == normal


def test_writer_uses_title_clean_when_present() -> None:
    """vision_extraction_writer uses title_clean as primary for export."""
    from vigie.extraction.vision_extraction_writer import _table_entry_indicators

    class Table:
        table_id = "tableau_1"
        title_clean = "Expositions brutes"
        title = "Expositions brutes 79 772 76 163"
        page_pdf = 1
        first_column_indicators = ["Line 1"]
        first_column_indicators_raw = ["Line 1"]
        unit_context = ""

    entry = _table_entry_indicators(Table(), "t1")
    assert entry["title"] == "Expositions brutes"
    assert "79" not in entry["title"] and "772" not in entry["title"]


def test_writer_falls_back_to_title_when_title_clean_missing() -> None:
    """When title_clean is missing, writer uses title."""
    from vigie.extraction.vision_extraction_writer import _table_entry_indicators

    class Table:
        table_id = "tableau_1"
        title_clean = None
        title = "Capital réglementaire"
        page_pdf = 1
        first_column_indicators = []
        first_column_indicators_raw = []
        unit_context = ""

    entry = _table_entry_indicators(Table(), "t1")
    assert entry["title"] == "Capital réglementaire"


# --- indicators_dedupe tests ---


def test_dedupe_removes_exact_duplicates_preserves_order() -> None:
    """Repeated block removed, order preserved."""
    from vigie.support.utils.indicator_cleaner import dedupe_indicators

    raw = ["Total actifs", "Total passifs", "Total actifs", "Fonds propres", "Total actifs"]
    result, ratio, removed = dedupe_indicators(raw, duplicate_ratio_threshold=0.15)
    assert result == ["Total actifs", "Total passifs", "Fonds propres"]
    assert ratio == 0.4  # 1 - 3/5
    assert removed == 2


def test_dedupe_below_threshold_returns_original() -> None:
    """When duplicate_ratio < threshold, list unchanged."""
    from vigie.support.utils.indicator_cleaner import dedupe_indicators

    raw = ["A", "B", "C", "D", "A"]  # 1 dup, ratio 0.2
    result, ratio, _ = dedupe_indicators(raw, duplicate_ratio_threshold=0.5)
    assert result == raw
    assert abs(ratio - 0.2) < 0.01


# --- indicators_line_merge tests ---
