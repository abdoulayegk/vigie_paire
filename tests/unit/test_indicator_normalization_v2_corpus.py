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
