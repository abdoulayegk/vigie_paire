from __future__ import annotations

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.indicator_normalizer import (
    get_canonical_text,
    get_token_sorted_text,
)


def test_redundant_parenthetical_expansion_unifies_bmo_tier1_case() -> None:
    t1 = (
        "Fonds propres de catégorie 1 sous forme d’actions ordinaires "
        "et fonds propres de catégorie 1 supplémentaires (%)"
    )
    t2 = (
        "Fonds propres de catégorie 1 "
        "(fonds propres de catégorie 1 sous forme d’actions ordinaires "
        "et fonds propres de catégorie 1 supplémentaires)."
    )

    expected = (
        "fonds propre de categorie 1 sous forme daction ordinaire "
        "et fonds propre de categorie 1 supplementaire"
    )

    assert normalize_indicator_for_comparison(t1) == expected
    assert normalize_indicator_for_comparison(t2) == expected
    assert get_canonical_text(t1) == get_canonical_text(t2) == expected
    assert get_token_sorted_text(t1) == get_token_sorted_text(t2)
