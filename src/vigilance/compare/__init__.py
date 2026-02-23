"""Comparison helpers for T1/T2 table matching."""

from vigilance.compare.indicator_comparator import (
    MatchDecision,
    match_decision,
    match_tables_intra_section,
    run_strict_intra_section_compare,
)

__all__ = [
    "MatchDecision",
    "match_decision",
    "match_tables_intra_section",
    "run_strict_intra_section_compare",
]
