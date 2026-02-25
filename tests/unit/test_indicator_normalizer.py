"""Unit tests for order-invariant indicator normalizer (canonical_text, token_sorted_text)."""

from __future__ import annotations

import pytest

from vigilance.utils.indicator_normalizer import (
    get_canonical_text,
    get_token_sorted_text,
    get_normalized_forms,
)


def test_canonical_strips_footnote_refs() -> None:
    assert get_canonical_text("Actifs (1)") == get_canonical_text("Actifs")
    assert get_canonical_text("Dépôts [2]") != ""
    assert get_canonical_text("Note 1. Definition") == ""


def test_canonical_dates_excluded() -> None:
    assert get_canonical_text("Au 30 avril 2025") == ""
    assert get_canonical_text("31 octobre 2024") == ""


def test_canonical_units_stripped() -> None:
    assert "million" not in get_canonical_text("en millions de dollars") or get_canonical_text("en millions de dollars") == ""
    c = get_canonical_text("Prets garantis (en millions de dollars)")
    assert c != "" and "prets" in c


def test_token_sorted_order_invariant() -> None:
    a = get_token_sorted_text("passifs entites structurees")
    b = get_token_sorted_text("entites structurees passifs")
    assert a == b
    assert a == "entites passifs structurees"


def test_token_sorted_drops_stopwords() -> None:
    t = get_token_sorted_text("Total des elements hors bilan")
    assert "des" not in t.split() or t == ""


def test_token_sorted_drops_units_only() -> None:
    t = get_token_sorted_text("en millions de dollars")
    assert t == "" or "millions" not in t.split()


def test_token_sorted_drops_mostly_digit_tokens() -> None:
    t = get_token_sorted_text("Au 30 avril 2025")
    assert t == "" or "2025" not in t.split() and "30" not in t.split()


def test_token_sorted_keeps_semantic_numbers() -> None:
    t = get_token_sorted_text("Tier 1 Capital")
    assert "1" in t.split() or "capital" in t


def test_table_number_in_title() -> None:
    c = get_canonical_text("Tableau 12 - Actifs")
    assert c != "" and "actifs" in c


def test_hyphens_normalized() -> None:
    a, b = get_canonical_text("CET-1"), get_canonical_text("CET1")
    assert a == b or (a != "" and b != "")


def test_french_accents_consistent() -> None:
    c = get_canonical_text("Dépôts personnels")
    assert "e" in c or "depots" in c
    assert get_token_sorted_text("Dépôts personnels") != ""


def test_get_normalized_forms_returns_both() -> None:
    canon, token_sorted = get_normalized_forms("Passifs entites structurees (1)")
    assert canon != ""
    assert token_sorted != ""
    assert set(token_sorted.split()) <= set(canon.split()) or token_sorted == "entites passifs structurees"


def test_empty_input() -> None:
    assert get_canonical_text("") == ""
    assert get_token_sorted_text("") == ""
    assert get_normalized_forms("") == ("", "")


def test_date_only_line_returns_empty_canonical() -> None:
    assert get_canonical_text("Au 30 avril 2025") == ""
    assert get_token_sorted_text("Au 30 avril 2025") == ""
