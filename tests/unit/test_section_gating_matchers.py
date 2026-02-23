"""Section gating tests across comparison matchers."""

from __future__ import annotations

from types import SimpleNamespace

from vigilance.comparison.multi_signal_matcher import MultiSignalMatcher, TableSignature
from vigilance.comparison.structural_comparator import StructuralTableComparator
from vigilance.comparison.table_preview_matcher import TablePreviewMatcher


def _signature(table_id: str, section: str, title: str) -> TableSignature:
    return TableSignature(
        table_id=table_id,
        page_number=1,
        title=title,
        first_column_labels=["CET1", "RWA"],
        headers=["Indicateur", "Valeur"],
        num_rows=2,
        num_columns=2,
        section_type=section,
    )


def test_multi_signal_never_scores_cross_section_candidates() -> None:
    matcher = MultiSignalMatcher()
    t1 = _signature("t1", "capital_management", "TABLEAU 12 - Fonds propres")
    t2 = _signature("t2", "risk_management", "TABLEAU 12 - Risque de credit")

    matches, unmatched_t1, unmatched_t2 = matcher.find_best_matches([t1], [t2])
    assert matches == []
    assert len(unmatched_t1) == 1
    assert len(unmatched_t2) == 1


def test_multi_signal_unknown_section_is_soft_and_can_match() -> None:
    matcher = MultiSignalMatcher()
    t1 = _signature("t1", "unknown_section", "TABLEAU 12 - Fonds propres")
    t2 = _signature("t2", "unknown_section", "TABLEAU 12 - Fonds propres")

    matches, unmatched_t1, unmatched_t2 = matcher.find_best_matches([t1], [t2])
    assert len(matches) == 1
    assert unmatched_t1 == []
    assert unmatched_t2 == []


def test_structural_comparator_requires_same_section() -> None:
    comparator = StructuralTableComparator()
    t1 = {"title": "Tableau 12", "rows": [["CET1", "1"]], "section": "capital_management"}
    t2 = {"title": "Tableau 12", "rows": [["CET1", "1"]], "section": "risk_management"}

    strong, probable, *_ = comparator._match_tables([t1], [t2])
    assert strong == []
    assert probable == []


def test_structural_comparator_matches_same_section() -> None:
    comparator = StructuralTableComparator()
    t1 = {"title": "Tableau 12", "rows": [["CET1", "1"]], "section": "capital_management"}
    t2 = {"title": "Tableau 12", "rows": [["CET1", "1"]], "section": "capital_management"}

    strong, probable, *_ = comparator._match_tables([t1], [t2])
    assert len(strong) == 1
    assert probable == []


def test_preview_sections_compatible_is_strict_only_on_known_mismatch() -> None:
    matcher = TablePreviewMatcher(use_robust_matcher=False)
    t_capital = SimpleNamespace(section="capital_management")
    t_risk = SimpleNamespace(section="risk_management")
    t_unknown = SimpleNamespace(section="unknown_section")

    assert matcher._sections_compatible(t_capital, t_risk) is False
    assert matcher._sections_compatible(t_capital, t_unknown) is True
    assert matcher._sections_compatible(t_capital, t_capital) is True
