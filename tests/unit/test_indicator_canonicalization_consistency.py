"""Test de contrat: tous les chemins de canonicalisation des indicateurs produisent la meme cle."""

from __future__ import annotations

import pytest

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.compare.indicator_comparator import _canonical_indicator_label
from app.comparison_runner import _canonical_indicator_key


@pytest.mark.parametrize(
    "raw",
    [
        "Prets garantis (M$)",
        "Ratio CET1 (%)",
        "Actifs greves - en milliers de dollars canadiens",
        "Prets garantis au 30 avril 2025",
        "Marge de credit",
        "Total (CAD)",
        "Metaux precieux :",
    ],
)
def test_indicator_canonicalization_contract(raw: str) -> None:
    """Same input must yield same output from all canonical entry points."""
    a = normalize_indicator_for_comparison(raw)
    b = _canonical_indicator_label(raw)
    c = _canonical_indicator_key(raw)
    assert a == b == c, (
        f"Divergence for {raw!r}: "
        f"normalize_indicator_for_comparison={a!r}, "
        f"_canonical_indicator_label={b!r}, "
        f"_canonical_indicator_key={c!r}"
    )


def test_empty_and_filtered_produce_empty() -> None:
    """Empty and date-only lines produce empty from all paths."""
    for raw in ["", "   ", "Au 30 avril 2025", "Note 3", "(1)"]:
        a = normalize_indicator_for_comparison(raw)
        b = _canonical_indicator_label(raw)
        c = _canonical_indicator_key(raw)
        assert a == b == c, f"Divergence for {raw!r}: {a!r} vs {b!r} vs {c!r}"


def test_lie_liee_canonicalize_to_same_key() -> None:
    """lie/liee/lies/liees canonicalize to 'lie' so same-phrase variants share key."""
    a = normalize_indicator_for_comparison("Exposition liee au credit")
    b = normalize_indicator_for_comparison("Exposition lie au credit")
    assert a == b, f"lie vs liee should match: {a!r} vs {b!r}"


def test_relatif_relative_canonicalize_to_same_key() -> None:
    """relatif/relative/relatifs/relatives canonicalize to 'relatif'."""
    a = normalize_indicator_for_comparison("Valeur relative au marche")
    b = normalize_indicator_for_comparison("Valeur relatif au marche")
    assert a == b, f"relatif vs relative should match: {a!r} vs {b!r}"


def test_residuel_residuels_canonicalize_to_same_key() -> None:
    """residuel/residuels canonicalize to 'residuel'."""
    a = normalize_indicator_for_comparison("Ecarts residuels")
    b = normalize_indicator_for_comparison("Ecarts residuel")
    assert a == b, f"residuel vs residuels should match: {a!r} vs {b!r}"
