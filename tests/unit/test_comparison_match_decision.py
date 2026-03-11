"""Tests for strict section gating in comparison.match_decision."""

from __future__ import annotations

from vigilance.comparison.match_decision import compute_decision


def test_indicator_structure_mismatch_forces_no_match() -> None:
    """Prefix-only pattern (high prefix, low LCS, poor size) triggers NO_MATCH."""
    signals = {
        "table_number_match": False,
        "header_overlap": 0.7,
        "indicator_overlap": 0.5,
        "indicator_jaccard": 0.5,
        "indicator_containment_min": 0.4,
        "indicator_lcs_ratio": 0.20,
        "indicator_size_ratio": 0.35,
        "indicator_prefix_ratio": 0.75,
        "title_similarity": 0.0,
        "section_match": False,
        "section_state": "unknown_present",
        "page_distance": 1,
    }
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    assert decision["decision"] == "NO_MATCH"
    assert decision["reason"] == "indicator_structure_mismatch"


def test_table_number_no_content_support_not_match() -> None:
    """Same table number but low indicator support: corroborated tn is low, no MATCH."""
    signals = {
        "table_number_match": True,
        "header_overlap": 0.3,
        "indicator_overlap": 0.2,
        "indicator_jaccard": 0.2,
        "indicator_containment_min": 0.15,
        "indicator_lcs_ratio": 0.15,
        "indicator_size_ratio": 0.5,
        "indicator_prefix_ratio": 0.3,
        "title_similarity": 0.0,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 0,
    }
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    assert decision["decision"] != "MATCH"


def test_table_number_high_indicator_support_still_matches() -> None:
    """Same table number and high robust indicator support: MATCH."""
    signals = {
        "table_number_match": True,
        "header_overlap": 0.9,
        "indicator_overlap": 0.85,
        "indicator_jaccard": 0.85,
        "indicator_containment_min": 0.9,
        "indicator_lcs_ratio": 0.85,
        "indicator_size_ratio": 0.95,
        "indicator_prefix_ratio": 0.8,
        "title_similarity": 0.9,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 0,
    }
    decision = compute_decision(
        signals=signals,
        table_type_t1="requirements",
        table_type_t2="requirements",
        has_headers=True,
    )
    assert decision["decision"] == "MATCH"


def test_backward_compat_old_signals_dict_still_works() -> None:
    """Signals without new indicator keys (old style) still produce a decision."""
    signals = {
        "table_number_match": True,
        "header_overlap": 0.9,
        "indicator_overlap": 0.8,
        "title_similarity": 0.9,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 1,
    }
    decision = compute_decision(
        signals=signals,
        table_type_t1="requirements",
        table_type_t2="requirements",
        title_t1="Tableau 24",
        title_t2="Tableau 24",
        headers_t1=["T1"],
        headers_t2=["T1"],
        has_headers=True,
    )
    assert decision["decision"] == "MATCH"
    assert "composite_score" in decision


def test_generic_title_strong_header_low_indicator_not_match() -> None:
    """Generic title + high header overlap + low indicator support must not become MATCH."""
    signals = {
        "table_number_match": False,
        "header_overlap": 0.90,
        "indicator_overlap": 0.25,
        "indicator_jaccard": 0.25,
        "indicator_containment_min": 0.20,
        "indicator_lcs_ratio": 0.18,
        "indicator_size_ratio": 0.50,
        "indicator_prefix_ratio": 0.30,
        "title_similarity": 0.0,
        "section_match": True,
        "section_state": "same_known",
        "page_distance": 0,
    }
    decision = compute_decision(
        signals=signals,
        table_type_t1="unknown",
        table_type_t2="unknown",
        title_t1="Tableau",
        title_t2="Tableau",
        has_headers=True,
        generic_titles=frozenset({"tableau"}),
    )
    assert decision["decision"] != "MATCH"


def test_cross_section_forbidden_even_with_strong_signals() -> None:
    decision = compute_decision(
        signals={
            "table_number_match": True,
            "header_overlap": 0.95,
            "indicator_overlap": 0.9,
            "title_similarity": 0.95,
            "section_match": False,
            "section_state": "mismatch_known",
            "page_distance": 0,
        },
        table_type_t1="requirements",
        table_type_t2="requirements",
        title_t1="Tableau 24 - Fonds propres",
        title_t2="Tableau 24 - Risque de credit",
        headers_t1=["T1"],
        headers_t2=["T1"],
        has_headers=True,
    )
    assert decision["decision"] == "NO_MATCH"
    assert decision["reason"] == "cross_section_forbidden"


def test_same_section_with_good_score_matches() -> None:
    decision = compute_decision(
        signals={
            "table_number_match": True,
            "header_overlap": 0.9,
            "indicator_overlap": 0.8,
            "title_similarity": 0.9,
            "section_match": True,
            "section_state": "same_known",
            "page_distance": 1,
        },
        table_type_t1="requirements",
        table_type_t2="requirements",
        title_t1="Tableau 24 - Fonds propres",
        title_t2="Tableau 24 - Fonds propres",
        headers_t1=["T1"],
        headers_t2=["T1"],
        has_headers=True,
    )
    assert decision["decision"] == "MATCH"


def test_validation_prefix_only_rejected_same_structure_accepted() -> None:
    """Targeted validation: prefix-only pair not accepted as MATCH; same table with extra row accepted."""
    from vigilance.comparison.match_signals import compute_match_signals

    prefix_only_t1 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": ["Cash", "Securities", "Govt bonds", "Mortgages", "Equity"],
        "section": "unknown",
        "page": 1,
    }
    prefix_only_t2 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": ["Cash", "Securities", "Govt bonds", "Liquid by entity", "Liquid by currency"],
        "section": "unknown",
        "page": 2,
    }
    signals_bad = compute_match_signals(prefix_only_t1, prefix_only_t2, has_headers=True)
    decision_bad = compute_decision(
        signals=signals_bad,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    assert decision_bad["decision"] == "NO_MATCH"

    same_table_t1 = {
        "title": "Table 1",
        "headers": ["A", "B"],
        "indicators": ["Item1", "Item2", "Item3"],
        "section": "capital",
        "page": 1,
    }
    same_table_t2 = {
        "title": "Table 1",
        "headers": ["A", "B"],
        "indicators": ["Item1", "Item2", "Item2b", "Item3"],
        "section": "capital",
        "page": 2,
    }
    signals_good = compute_match_signals(same_table_t1, same_table_t2, has_headers=True)
    decision_good = compute_decision(
        signals=signals_good,
        table_type_t1="unknown",
        table_type_t2="unknown",
        has_headers=True,
    )
    assert decision_good["decision"] == "MATCH"
