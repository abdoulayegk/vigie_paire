"""Comparison helpers for T1/T2 table matching."""

from vigilance.compare.indicator_comparator import (
    MatchDecision,
    match_decision,
    match_tables_intra_section,
)
from vigilance.compare.footnote_comparator import FootnoteComparator, compare_footnotes
from vigilance.compare.table_pairing_engine import run_strict_intra_section_compare

__all__ = [
    "FootnoteComparator",
    "MatchDecision",
    "compare_footnotes",
    "match_decision",
    "match_tables_intra_section",
    "run_strict_intra_section_compare",
]
