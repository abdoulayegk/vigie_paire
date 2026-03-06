"""Integration-like checks for strict intra-section matching output reasons."""

from __future__ import annotations

from vigilance.compare.indicator_comparator import (
    match_tables_intra_section,
    run_strict_intra_section_compare,
)
from vigilance.models.table_models import TableArtifact


def _table(
    table_id: str,
    section: str,
    title: str,
    rows: list[list[str]],
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=1,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "Valeur"],
        rows=rows,
        first_column_indicators=[row[0] for row in rows if row],
        extraction_method="docling",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
    )


def test_intra_section_matching_and_reasons() -> None:
    t1_tables = [
        _table("t1_capital_match", "capital_management", "TABLEAU 10", [["CET1", "13"]]),
        _table("t1_capital_nomatch", "capital_management", "TABLEAU 28", [["TLAC", "20"]]),
        _table("t1_unknown", "unknown_section", "Sans section", [["X", "1"]]),
    ]
    t2_tables = [
        _table("t2_capital_match", "capital_management", "TABLEAU 10", [["CET1", "14"]]),
        _table("t2_risk_cross", "risk_management", "TABLEAU 28", [["TLAC", "20"]]),
        _table("t2_unknown", "unknown_section", "Sans section", [["X", "1"]]),
    ]

    result = match_tables_intra_section(t1_tables, t2_tables)

    assert any(
        pair["reason"]
        in ("table_number_match", "indicator_set_hash_exact", "indicator_overlap_match")
        for pair in result["pairs"]
    )
    assert any(
        item["t1_table_id"] == "t1_capital_nomatch"
        and item["section"] == "capital_management"
        and item["reason"] in {"no_candidate_same_section", "low_containment", "weak_signals"}
        for item in result["unmatched_t1"]
    )
    unknown_t1_handled = any(pair["t1_table_id"] == "t1_unknown" for pair in result["pairs"]) or any(
        item["t1_table_id"] == "t1_unknown"
        and item["section"] == "unknown_section"
        and item["reason"] in {"unknown_section", "unknown_section_penalized", "low_containment", "weak_signals"}
        for item in result["unmatched_t1"]
    )
    assert unknown_t1_handled is True
    assert any(
        item["reason"] in {"unknown_section", "unmatched", "ambiguous_candidate"}
        for item in result["unmatched_t2"]
    )


def test_strict_compare_one_to_one_and_added_removed() -> None:
    """Validation 1-to-1: unique pair UIDs, coverage, and added/removed rules."""
    t1_tables = [
        _table("t1_a", "capital_management", "Table A", [["CET1", "x"], ["Tier 1", "y"]]),
        _table("t1_b", "capital_management", "Table B", [["TLAC", "z"]]),
    ]
    t2_tables = [
        _table("t2_a", "capital_management", "Table A", [["CET1", "x"], ["Tier 1", "y"]]),
        _table("t2_c", "capital_management", "Table C", [["Other", "1"]]),
    ]
    result = run_strict_intra_section_compare(t1_tables, t2_tables)

    pairs = result["pairs"]
    pair_t1 = {p["t1_uid"] for p in pairs}
    pair_t2 = {p["t2_uid"] for p in pairs}
    assert len(pair_t1) == len(pairs), "each pair must have unique t1_uid"
    assert len(pair_t2) == len(pairs), "each pair must have unique t2_uid"

    added = result["added_tables"]
    removed = result["removed_tables"]
    unmatched_t2 = result["unmatched_t2"]
    unmatched_t1 = result["unmatched_t1"]
    added_t2_uids = {a["t2_uid"] for a in added}
    for item in unmatched_t2:
        if item.get("reason") == "unmatched":
            assert item["t2_uid"] in added_t2_uids, "unmatched T2 with reason 'unmatched' must appear in added_tables"
    for r in removed:
        assert r.get("reason") == "removed_table"
        assert any(u.get("t1_uid") == r["t1_uid"] for u in unmatched_t1)
