"""Section gating tests for comparison.table_comparator."""

from __future__ import annotations

from vigilance.comparison.table_comparator import TableMeta, compute_table_match_score


def _meta(table_key: str, section: str, labels: list[str]) -> TableMeta:
    return TableMeta(
        table_key=table_key,
        section_norm=section,
        table_title=table_key,
        first_col_labels=labels,
        page=1,
        row_count=len(labels),
        position_in_section=0.1,
    )


def test_compute_table_match_score_blocks_cross_section() -> None:
    a = _meta("t1", "capital_management", ["Atlantique", "Québec"])
    b = _meta("t2", "risk_management", ["Atlantique", "Québec"])

    score, breakdown = compute_table_match_score(a, b)
    assert score == 0.0
    assert breakdown.get("blocked") == "cross_section_forbidden"


def test_compute_table_match_score_blocks_unknown_section() -> None:
    a = _meta("t1", "unknown_section", ["Atlantique", "Québec"])
    b = _meta("t2", "unknown_section", ["Atlantique", "Québec"])

    score, breakdown = compute_table_match_score(a, b)
    assert score == 0.0
    assert breakdown.get("blocked") == "unknown_section"


def test_compute_table_match_score_allows_same_section() -> None:
    a = _meta("t1", "capital_management", ["Atlantique", "Québec"])
    b = _meta("t2", "capital_management", ["Atlantique", "Québec"])

    score, breakdown = compute_table_match_score(a, b)
    assert score > 0.0
    assert "blocked" not in breakdown
