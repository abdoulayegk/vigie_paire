"""Compatibility tests for the official strict comparator facade."""

from __future__ import annotations

from vigilance.compare import run_strict_intra_section_compare
from vigilance.comparison.indicator_comparator import (
    run_strict_intra_section_compare as run_strict_intra_section_compare_legacy,
)
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
        extraction_method="docling",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
    )


def test_official_and_legacy_facades_return_same_payload() -> None:
    tables_t1 = [_table("t1_10", "capital_management", "TABLEAU 10", ["CET1", "RWA"])]
    tables_t2 = [_table("t2_10", "capital_management", "TABLEAU 10", ["CET1", "RWA"])]

    official = run_strict_intra_section_compare(tables_t1, tables_t2)
    legacy = run_strict_intra_section_compare_legacy(tables_t1, tables_t2)

    assert official == legacy
