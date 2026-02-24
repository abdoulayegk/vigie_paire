"""Tests for indicator label date/unit stripping (anti false-additions)."""

from __future__ import annotations

import pytest

from vigilance.utils.indicator_cleaner import (
    normalize_indicator_for_comparison,
    strip_dates_from_indicator_label,
    strip_units_currency_from_indicator_label,
)


# Jeu de test minimum (spec)
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Au 30 avril 2025", ""),
        ("Au 31 janvier 2025", ""),
        ("Prets garantis au 30 avril 2025", "prets garantis"),
        ("Prets garantis (en millions de dollars)", "prets garantis"),
        ("Marge de credit (M$)", "marge de credit"),
        ("Ratio CET1 (%)", "ratio cet1"),
        ("Actifs greves - en milliers de dollars canadiens", "actifs greves"),
        ("Pour la periode close le 31/03/2025", ""),
        ("Total (CAD)", "total"),
        ("Note 3", ""),
        ("(1)", ""),
        ("1)", ""),
    ],
)
def test_normalize_indicator_spec_cases(raw: str, expected: str) -> None:
    """normalize_indicator_for_comparison must neutralize dates and units."""
    assert normalize_indicator_for_comparison(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Au 30 avril 2025", ""),
        ("Au 31 janvier 2025", ""),
        ("31 janvier 2025", ""),
        ("30 avril 2025", ""),
        ("Prets garantis au 30 avril 2025", "Prets garantis"),
        ("Prets garantis - 31/01/2025", "Prets garantis"),
        ("Pour la periode close le 31/03/2025", ""),
    ],
)
def test_strip_dates_from_indicator_label(raw: str, expected: str) -> None:
    """strip_dates_from_indicator_label removes date fragments."""
    assert strip_dates_from_indicator_label(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Prets garantis (en millions de dollars)", "Prets garantis"),
        ("Marge de credit (M$)", "Marge de credit"),
        ("Ratio CET1 (%)", "Ratio CET1"),
        ("Total (CAD)", "Total"),
        ("Actifs greves - en milliers de dollars canadiens", "Actifs greves"),
    ],
)
def test_strip_units_currency_from_indicator_label(raw: str, expected: str) -> None:
    """strip_units_currency_from_indicator_label removes unit/currency phrases."""
    assert strip_units_currency_from_indicator_label(raw) == expected


def test_semantic_core_preserved() -> None:
    """Noyau semantique (CET1, Tier 1, etc.) must be preserved."""
    assert normalize_indicator_for_comparison("Ratio CET1 (%)") == "ratio cet1"
    assert normalize_indicator_for_comparison("Tier 1 (M$)") == "tier 1"
    assert "cet1" in normalize_indicator_for_comparison("Ratio CET1 au 30 avril 2025")


def test_same_indicator_different_dates_unify() -> None:
    """Same indicator with different dates must produce same canonical key."""
    a = normalize_indicator_for_comparison("Prets garantis au 30 avril 2025")
    b = normalize_indicator_for_comparison("Prets garantis au 31 janvier 2025")
    assert a == b == "prets garantis"


def test_same_indicator_different_units_unify() -> None:
    """Same indicator with different unit phrases must produce same canonical key."""
    a = normalize_indicator_for_comparison("Prets garantis (en millions de dollars)")
    b = normalize_indicator_for_comparison("Prets garantis (M$)")
    assert a == b == "prets garantis"
