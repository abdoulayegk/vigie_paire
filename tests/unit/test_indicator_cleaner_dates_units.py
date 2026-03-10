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
        ("Prets garantis au 30 avril 2025", "pret garanti"),
        ("Prets garantis (en millions de dollars)", "pret garanti"),
        ("Marge de credit (M$)", "marge de credit"),
        ("Ratio CET1 (%)", "ratio cet1"),
        ("Actifs greves - en milliers de dollars canadiens", "actif greve"),
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
    assert a == b == "pret garanti"


def test_same_indicator_different_units_unify() -> None:
    """Same indicator with different unit phrases must produce same canonical key."""
    a = normalize_indicator_for_comparison("Prets garantis (en millions de dollars)")
    b = normalize_indicator_for_comparison("Prets garantis (M$)")
    assert a == b == "pret garanti"


def test_leading_row_numbers_stripped() -> None:
    """Leading row/line numbers (Basel III, NSFR, LCR tables) must be stripped."""
    a = normalize_indicator_for_comparison("29 Actifs d'instruments dérivés du NSFR")
    b = normalize_indicator_for_comparison("27 Actifs d'instruments dérivés du NSFR")
    assert a == b
    assert "29" not in a and "27" not in a

    # Different row numbers, same text
    a2 = normalize_indicator_for_comparison("32 Éléments hors bilan")
    b2 = normalize_indicator_for_comparison("30 Éléments hors bilan")
    assert a2 == b2

    # Single-digit row number
    c = normalize_indicator_for_comparison("1 Trésorerie")
    assert c == "tresorerie"

    # Must NOT strip regulatory tokens like CET1, Tier 2
    assert normalize_indicator_for_comparison("CET1") == "cet1"
    assert normalize_indicator_for_comparison("Tier 2") == "tier 2"


def test_pension_asset_net_variant_unifies() -> None:
    a = normalize_indicator_for_comparison(
        "Variation de l’actif net des régimes de retraite à prestations définies (déduction faite des passifs d’impôt)"
    )
    b = normalize_indicator_for_comparison(
        "Variation de l’actif des régimes de retraite à prestations définies (déduction faite des passifs d’impôt)"
    )
    assert a == b
def test_impot_impots_same_canonical_key() -> None:
    """d'impôt and d'impôts must produce the same key to avoid false add/remove."""
    a = normalize_indicator_for_comparison(
        "Variation de l'actif des régimes de retraite (déduction faite des passifs d'impôt)"
    )
    b = normalize_indicator_for_comparison(
        "Variation de l'actif des régimes de retraite (déduction faite des passifs d'impôts)"
    )
    assert a == b
    c = normalize_indicator_for_comparison("passifs d'impôt")
    d = normalize_indicator_for_comparison("passifs d'impôts")
    assert c == d


def test_elision_with_space_same_key() -> None:
    """d' impôt (space after apostrophe) must match d'impôts."""
    a = normalize_indicator_for_comparison("déduction faite des passifs d' impôt")
    b = normalize_indicator_for_comparison("déduction faite des passifs d'impôts")
    assert a == b


def test_plan_examples_same_key_across_variants() -> None:
    """Plan examples: same key for variants (spaces, apostrophes, impôt/impôts)."""
    base1 = "Variation de l'actif des régimes de retraite à prestations définies (déduction faite des passifs d'impôt)"
    key1 = normalize_indicator_for_comparison(base1)
    assert key1
    assert (
        normalize_indicator_for_comparison(
            "Variation de l'actif des régimes de retraite à prestations définies (déduction faite des passifs d'impôts)"
        )
        == key1
    )
    base2 = "Autres éléments de fonds propres de catégorie 1"
    key2 = normalize_indicator_for_comparison(base2)
    assert key2
    assert normalize_indicator_for_comparison("Autres elements de fonds propres de categorie 1") == key2
    base3 = "Variation de l'actif net des régimes de retraite à prestations définies (déduction faite des passifs d'impôt)"
    key3 = normalize_indicator_for_comparison(base3)
    assert key3 == key1

