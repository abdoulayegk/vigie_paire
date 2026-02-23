"""Integration-like checks for strict intra-section matching output reasons."""

from __future__ import annotations

from vigilance.compare.indicator_comparator import match_tables_intra_section
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
        pair["reason"] in ("table_number_match", "indicator_set_hash_exact")
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
    assert any(item["reason"] in {"unknown_section", "unmatched"} for item in result["unmatched_t2"])
