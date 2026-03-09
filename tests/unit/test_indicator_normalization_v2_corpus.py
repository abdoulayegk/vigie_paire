from __future__ import annotations

import pytest

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.indicator_normalizer import (
    get_canonical_text,
    get_token_sorted_text,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Agence de notation¹", "Agence de notation"),
        ("Titres vendus à découvert(4)", "Titres vendus à découvert"),
        ("Série K – tranche 1", "Série K – première tranche"),
        ("Série J – tranche 2", "Série J – deuxième tranche"),
        (
            "Billets subordonnés à 4,800 % avec termes des fonds propres de catégorie 1 supplémentaires",
            "Billets subordonnés à 4,80 % aux termes des fonds propres de catégorie 1 supplémentaires",
        ),
        (
            "Options sur actions et attribution d'actions",
            "Options sur actions et attribution d’actions",
        ),
        (
            "Fonds propres de catégorie 1 sous forme d’actions ordinaires "
            "et fonds propres de catégorie 1 supplémentaires (%)",
            "Fonds propres de catégorie 1 "
            "(fonds propres de catégorie 1 sous forme d’actions ordinaires "
            "et fonds propres de catégorie 1 supplémentaires).",
        ),
        (
            "Passifs et capitaux propres : Passifs et capitaux propres attribuables aux actionnaires",
            "Passifs et capitaux propres attribuables aux actionnaires",
        ),
    ],
)
def test_indicator_normalization_v2_equivalent_cases(left: str, right: str) -> None:
    assert normalize_indicator_for_comparison(left) == normalize_indicator_for_comparison(right)
    assert get_canonical_text(left) == get_canonical_text(right)
    assert get_token_sorted_text(left) == get_token_sorted_text(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Titres vendus à découvert", "Agence de notation"),
        ("Risque de crédit", "Risque de liquidité"),
        ("Passifs et capitaux propres", "Actifs et capitaux propres"),
    ],
)
def test_indicator_normalization_v2_distinct_cases_stay_distinct(
    left: str, right: str
) -> None:
    assert get_canonical_text(left) != get_canonical_text(right)


def test_token_sorted_text_dedupes_repeated_tokens() -> None:
    text = (
        "Fonds propres de catégorie 1 fonds propres de catégorie 1 "
        "sous forme d’actions ordinaires"
    )
    token_sorted = get_token_sorted_text(text)
    assert token_sorted.split().count("fonds") == 1
    assert token_sorted.split().count("categorie") == 1


def test_canonical_strips_chained_parenthesized_footnotes() -> None:
    left = "À dividende non cumulatif, série BW (3), (4),"
    right = "À dividende non cumulatif, série BW (3), (4), (5)"
    assert normalize_indicator_for_comparison(left) == "a dividende non cumulatif serie bw"
    assert normalize_indicator_for_comparison(right) == "a dividende non cumulatif serie bw"
    assert get_canonical_text(left) == get_canonical_text(right)
    assert get_token_sorted_text(left) == get_token_sorted_text(right)
