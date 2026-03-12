"""Unit tests for indicator_match_helpers (LCS, prefix, structural signals)."""

from __future__ import annotations

import pytest

from vigilance.comparison.indicator_match_helpers import (
    compute_indicator_signals,
    compute_robust_indicator_score_from_signals,
    lcs_length,
    longest_common_prefix_length,
)


def test_lcs_length_identical_lists() -> None:
    a = ["a", "b", "c", "d"]
    b = ["a", "b", "c", "d"]
    assert lcs_length(a, b) == 4


def test_lcs_length_prefix_only_overlap() -> None:
    a = ["cash", "securities", "govt bonds", "mortgages", "loans"]
    b = ["cash", "securities", "govt bonds", "liquid by entity", "liquid by currency"]
    assert lcs_length(a, b) == 3


def test_lcs_length_no_overlap() -> None:
    a = ["x", "y", "z"]
    b = ["p", "q", "r"]
    assert lcs_length(a, b) == 0


def test_lcs_length_one_extra_in_second() -> None:
    a = ["a", "b", "c"]
    b = ["a", "b", "x", "c"]
    assert lcs_length(a, b) == 3


def test_longest_common_prefix_identical() -> None:
    a = ["a", "b", "c"]
    b = ["a", "b", "c"]
    assert longest_common_prefix_length(a, b) == 3


def test_longest_common_prefix_divergence() -> None:
    a = ["cash", "securities", "govt bonds", "mortgages"]
    b = ["cash", "securities", "govt bonds", "liquid assets"]
    assert longest_common_prefix_length(a, b) == 3


def test_longest_common_prefix_empty() -> None:
    assert longest_common_prefix_length([], ["a"]) == 0
    assert longest_common_prefix_length(["a"], []) == 0


def test_compute_indicator_signals_identical() -> None:
    ind = ["a", "b", "c", "d"]
    out = compute_indicator_signals(ind, ind)
    assert out["indicator_jaccard"] == 1.0
    assert out["indicator_containment_min"] == 1.0
    assert out["indicator_lcs_ratio"] == 1.0
    assert out["indicator_size_ratio"] == 1.0
    assert out["indicator_prefix_ratio"] == 1.0


def test_compute_indicator_signals_prefix_only_divergence() -> None:
    a = ["cash", "securities", "govt bonds", "mortgages", "equity", "loans"]
    b = ["cash", "securities", "govt bonds", "liquid by entity", "liquid by currency", "hqla"]
    out = compute_indicator_signals(a, b)
    assert out["indicator_prefix_ratio"] >= 0.4
    assert out["indicator_lcs_ratio"] < 0.6
    assert out["indicator_jaccard"] < 0.5


def test_compute_indicator_signals_size_mismatch() -> None:
    a = ["a", "b", "c"]
    b = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    out = compute_indicator_signals(a, b)
    assert out["indicator_size_ratio"] == 0.3
    assert out["indicator_containment_min"] == 1.0


def test_compute_indicator_signals_empty() -> None:
    out = compute_indicator_signals([], [])
    assert out["indicator_jaccard"] == 0.0
    assert out["indicator_size_ratio"] == 1.0
    assert out["indicator_prefix_ratio"] == 0.0


def test_compute_robust_indicator_score_from_signals_high() -> None:
    signals = {
        "indicator_jaccard": 0.9,
        "indicator_containment_min": 0.9,
        "indicator_lcs_ratio": 0.85,
        "indicator_size_ratio": 0.9,
        "indicator_prefix_ratio": 0.5,
    }
    score = compute_robust_indicator_score_from_signals(signals)
    assert score >= 0.7


def test_compute_robust_indicator_score_from_signals_fallback_overlap() -> None:
    signals = {"indicator_overlap": 0.8}
    score = compute_robust_indicator_score_from_signals(signals)
    assert 0 <= score <= 1.0
