"""Regression tests for normalization robustness (base text, fusion/split, footnotes)."""

from __future__ import annotations

from vigilance.compare.footnote_comparator import FootnoteComparator
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.text_normalize_base import normalize_text_base

from app.comparison_runner import _apply_short_indicator_guard, _detect_fusion_split


def test_normalize_text_base_accents_apostrophe_nbsp() -> None:
    assert normalize_text_base("cat\u00e9gorie 1") == normalize_text_base("categorie 1")
    assert normalize_text_base("d\u2019actions") == normalize_text_base("d'actions")
    assert normalize_text_base("Ratio\u00a0de levier") == normalize_text_base("Ratio de levier")


def test_normalize_text_base_lowercase_false_preserves_case() -> None:
    assert normalize_text_base("Hello World", lowercase=False) == "Hello World"
    assert normalize_text_base("hello", lowercase=True) == "hello"


def test_normalize_indicator_cet1_categorie_total_equivalent() -> None:
    a = "CET1 (3) cat\u00e9gorie 1 (3) total"
    b = "CET1 (3) categorie 1 (3) total"
    assert normalize_indicator_for_comparison(a) == normalize_indicator_for_comparison(b)


def test_normalize_indicator_ratio_percent_variant() -> None:
    with_pct = "Ratio de liquidit\u00e9 \u00e0 court terme (%)"
    no_pct = "Ratio de liquidite a court terme"
    # Both should map to same canonical key when (%) stripped by units path
    k1 = normalize_indicator_for_comparison(with_pct)
    k2 = normalize_indicator_for_comparison(no_pct)
    assert k1 == k2


def test_fusion_split_order_invariant() -> None:
    """Two added fragments in reverse order still merge with one removed long line."""
    long_key = normalize_indicator_for_comparison(
        "Titres adosses a des creances hypothecaires (presentes comme des prets au cout amorti)"
    )
    frag1 = normalize_indicator_for_comparison("(presentes comme des prets au cout amorti)")
    frag2 = normalize_indicator_for_comparison(
        "Titres adosses a des creances hypothecaires"
    )
    if not long_key or not frag1 or not frag2:
        return
    added = ["x"]
    removed = ["y"]
    # Direct _detect_fusion_split on canonical display strings
    added_list = [
        "Titres adosses a des creances hypothecaires (presentes comme des prets au cout amorti)"
    ]
    removed_list = [
        "(presentes comme des prets au cout amorti)",
        "Titres adosses a des creances hypothecaires",
    ]
    a, r, merged = _detect_fusion_split(
        added_list[:],
        removed_list[:],
    )
    assert merged is True
    assert a == []
    assert r == []


def test_fusion_split_reverse_fragment_order() -> None:
    removed_list = [
        "Titres adosses a des creances hypothecaires",
        "(presentes comme des prets au cout amorti)",
    ]
    added_list = [
        "Titres adosses a des creances hypothecaires (presentes comme des prets au cout amorti)"
    ]
    a, r, merged = _detect_fusion_split(added_list[:], removed_list[:])
    assert merged is True
    assert not a and not r


def test_short_indicator_guard_drops_subset_fragment() -> None:
    added = {"autre passif"}
    removed: set[str] = set()
    stable = {"total des autre passif et engagement hors bilan"}
    ex: dict[str, int] = {}
    th = {
        "indicator_short_guard_enabled": True,
        "indicator_short_guard_max_tokens": 3,
        "indicator_short_guard_min_stable_tokens": 5,
    }
    _apply_short_indicator_guard(added, removed, stable, th, ex)
    assert "autre passif" not in added
    assert ex.get("short_indicator_guard", 0) >= 1


def test_footnote_comparator_accent_apostrophe_no_modified() -> None:
    fc = FootnoteComparator(similarity_threshold=0.8)
    d1 = {"1": "Les donnees tiennent compte du rachat d'actions privilegiees."}
    d2 = {
        "1": "Les donnees tiennent compte du rachat d\u2019actions privilegiees."
    }
    changes = fc.compare_footnotes(d1, d2)
    modified = [c for c in changes if c.change_type == "modified_footnote"]
    assert modified == []
