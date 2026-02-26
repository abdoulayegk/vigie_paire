"""Unit tests for Fix B (structure-change suppression) and _jaccard helper."""

from __future__ import annotations

import pytest

from app.comparison_runner import (
    _jaccard,
    _should_suppress_as_structure_change,
)


def test_jaccard_empty_sets() -> None:
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard(set(), {"b"}) == 0.0


def test_jaccard_identical() -> None:
    s = {"x", "y", "z"}
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint() -> None:
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial_overlap() -> None:
    # |intersection|=2, |union|=5
    assert _jaccard({"a", "b", "c"}, {"a", "b", "d", "e"}) == pytest.approx(2.0 / 5.0)


def test_jaccard_one_contains_other() -> None:
    # {a,b} subset of {a,b,c,d}; intersection=2, union=4
    assert _jaccard({"a", "b"}, {"a", "b", "c", "d"}) == pytest.approx(0.5)


def test_should_suppress_disabled_by_config() -> None:
    cfg = {"enable_structure_change_suppression": False}
    set1 = {str(i) for i in range(12)}
    set2 = set1.copy()
    suppress, jaccard, reason = _should_suppress_as_structure_change(
        set1, set2, 0.85, cfg, "td"
    )
    assert suppress is False
    assert jaccard == 1.0
    assert reason == ""


def test_should_suppress_wrong_bank() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {str(i) for i in range(12)}
    set2 = set1.copy()
    suppress, _, _ = _should_suppress_as_structure_change(set1, set2, 0.85, cfg, "cibc")
    assert suppress is False


def test_should_suppress_conditions_met() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {str(i) for i in range(12)}
    set2 = set1.copy()
    set2.discard("11")
    set2.add("extra")
    # 11 overlap, 12 union -> jaccard = 11/12 > 0.65
    suppress, jaccard, reason = _should_suppress_as_structure_change(
        set1, set2, 0.85, cfg, "td"
    )
    assert suppress is True
    assert jaccard >= 0.65
    assert reason == "high_jaccard_suppression"


def test_should_suppress_score_too_low() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {str(i) for i in range(12)}
    set2 = set1.copy()
    suppress, _, _ = _should_suppress_as_structure_change(set1, set2, 0.70, cfg, "td")
    assert suppress is False


def test_should_suppress_too_few_indicators() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {"a", "b", "c", "d", "e"}
    set2 = set1.copy()
    suppress, _, _ = _should_suppress_as_structure_change(set1, set2, 0.90, cfg, "td")
    assert suppress is False


def test_should_suppress_jaccard_too_low() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {str(i) for i in range(12)}
    set2 = {str(i) for i in range(6)}
    suppress, jaccard, _ = _should_suppress_as_structure_change(set1, set2, 0.90, cfg, "td")
    assert jaccard == 0.5
    assert suppress is False


def test_should_suppress_bns_allowed() -> None:
    cfg = {
        "enable_structure_change_suppression": True,
        "structure_change_table_score_threshold": 0.80,
        "structure_change_min_indicators": 10,
        "structure_change_jaccard_threshold": 0.65,
    }
    set1 = {str(i) for i in range(12)}
    set2 = set1.copy()
    suppress, _, reason = _should_suppress_as_structure_change(set1, set2, 0.85, cfg, "bns")
    assert suppress is True
    assert reason == "high_jaccard_suppression"
