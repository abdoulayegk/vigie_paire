"""Tests for strict section gating in comparison.match_decision."""

from __future__ import annotations

from vigilance.comparison.match_decision import compute_decision


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
