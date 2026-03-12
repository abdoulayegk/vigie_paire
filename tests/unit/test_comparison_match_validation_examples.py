"""
Validation-style tests: labeled examples with signal assertions.

Covers true match (with insertion), prefix-only false match, same table number
but different indicators, and generic-title. Asserts both decision and
structural signal relationships for threshold calibration.
"""

from __future__ import annotations

from vigilance.comparison.indicator_match_helpers import (
    compute_robust_indicator_score_from_signals,
)
from vigilance.comparison.match_decision import (
    _corroborated_table_number,
    compute_decision,
)
from vigilance.comparison.match_signals import compute_match_signals


def test_validation_true_match_with_insertion_signals() -> None:
    """True match (same table, one row inserted): robust score high, decision MATCH."""
    t1 = {
        "title": "Table 1",
        "headers": ["A", "B"],
        "indicators": ["Item1", "Item2", "Item3"],
        "section": "capital",
        "page": 1,
    }
    t2 = {
        "title": "Table 1",
        "headers": ["A", "B"],
        "indicators": ["Item1", "Item2", "Item2b", "Item3"],
        "section": "capital",
        "page": 2,
    }
    signals = compute_match_signals(t1, t2, has_headers=True)
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    robust = compute_robust_indicator_score_from_signals(signals)
    assert robust > 0.6
    assert decision["decision"] == "MATCH"


def test_validation_prefix_only_signals() -> None:
    """Prefix-only false match: high prefix_ratio, low lcs_ratio, decision NO_MATCH."""
    t1 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": ["Cash", "Securities", "Govt bonds", "Mortgages", "Equity"],
        "section": "unknown",
        "page": 1,
    }
    t2 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": [
            "Cash",
            "Securities",
            "Govt bonds",
            "Liquid by entity",
            "Liquid by currency",
        ],
        "section": "unknown",
        "page": 2,
    }
    signals = compute_match_signals(t1, t2, has_headers=True)
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    prefix_ratio = signals.get("indicator_prefix_ratio", 0) or 0
    lcs_ratio = signals.get("indicator_lcs_ratio", 0) or 0
    assert prefix_ratio >= 0.4
    assert lcs_ratio < 0.7
    assert decision["decision"] == "NO_MATCH"


def test_validation_reused_table_number_low_corroboration() -> None:
    """Same table number but different indicators: corroborated table number low/partial."""
    signals = {
        "table_number_match": True,
        "header_overlap": 0.5,
        "indicator_jaccard": 0.25,
        "indicator_containment_min": 0.20,
        "indicator_lcs_ratio": 0.18,
        "indicator_size_ratio": 0.55,
        "indicator_prefix_ratio": 0.25,
        "title_similarity": 0.0,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 0,
    }
    corroborated = _corroborated_table_number(signals)
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    assert corroborated <= 0.5
    assert decision["decision"] != "MATCH"


def test_validation_generic_title_case_requires_indicator_support() -> None:
    """Generic title: MATCH only with sufficient indicator support, not header alone."""
    signals_low_indicator = {
        "table_number_match": False,
        "header_overlap": 0.90,
        "indicator_jaccard": 0.22,
        "indicator_containment_min": 0.18,
        "indicator_lcs_ratio": 0.15,
        "indicator_size_ratio": 0.5,
        "indicator_prefix_ratio": 0.28,
        "title_similarity": 0.0,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 0,
    }
    decision = compute_decision(
        signals=signals_low_indicator,
        table_type_t1="unknown",
        table_type_t2="unknown",
        title_t1="Tableau",
        title_t2="Tableau",
        has_headers=True,
        generic_titles=frozenset({"tableau"}),
    )
    assert decision["decision"] != "MATCH"
