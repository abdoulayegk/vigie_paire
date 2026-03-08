"""Synthetic integration suite for strict intra-section policy."""

from __future__ import annotations

from vigilance.compare import run_strict_intra_section_compare
from vigilance.models.table_models import TableArtifact


def _table(table_id: str, section: str, title: str, labels: list[str]) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=1,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in labels],
        first_column_indicators=labels,
        first_column_indicators_raw=labels,
        extraction_method="vision_full_gpt4o",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )


def test_strict_suite_cross_unknown_added_removed_and_same_section() -> None:
    t1 = [
        _table("cap_match", "capital_management", "TABLEAU 10", ["CET1", "RWA"]),
        _table("cap_removed", "capital_management", "TABLEAU 23", ["Atlantique", "Québec"]),
        _table("unknown_t1", "unknown_section", "Sans section", ["X"]),
    ]
    t2 = [
        _table("cap_match_t2", "capital_management", "TABLEAU 10", ["CET1", "RWA"]),
        _table("cap_added", "capital_management", "TABLEAU 24", ["Tier 2", "Levier"]),
        _table("risk_cross", "risk_management", "TABLEAU 23", ["Atlantique", "Québec"]),
        _table("unknown_t2", "unknown_section", "Sans section", ["Y"]),
    ]

    result = run_strict_intra_section_compare(t1, t2)

    assert any(
        pair["t1_table_id"] == "cap_match" and pair["t2_table_id"] == "cap_match_t2"
        for pair in result["pairs"]
    )
    assert any(item["reason"] == "removed_table" for item in result["removed_tables"])
    assert any(item["reason"] == "added_table" for item in result["added_tables"])
    assert any(
        item["reason"] in {"removed_table", "ambiguous_candidate"}
        for item in result["unmatched_t1"]
    )
    assert any(
        item["reason"] in {"no_match", "added_table", "ambiguous_candidate"}
        for item in result["unmatched_t2"]
    )
    assert "cross_section_forbidden" not in result["reasons"]
