"""Tests for per-section snapshot JSON exports."""

from __future__ import annotations

import json
from pathlib import Path

from app.comparison_runner import write_section_snapshots
from vigilance.models.table_models import TableArtifact


def _table(section: str, table_id: str, page: int, indicators: list[str]) -> TableArtifact:
    rows = [[text, "1"] for text in indicators]
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=f"Table {table_id}",
        headers=["Indicateur", "Valeur"],
        rows=rows,
        first_column_indicators=indicators,
        extraction_method="docling",
        quarter="t1",
        pdf_path="dummy.pdf",
    )


def test_write_section_snapshots_creates_one_file_per_section_and_quarter(tmp_path: Path) -> None:
    tables = [
        _table("capital_management", "tableau_14", 21, ["CET1", "TLAC"]),
        _table("risk_management", "tableau_20", 35, ["Risque de crédit"]),
    ]
    section_ranges = [
        {"section": "capital_management", "start": 21, "end": 30},
        {"section": "risk_management", "start": 31, "end": 47},
    ]

    written = write_section_snapshots(
        bank_code="bnc",
        year=2025,
        quarter="t1",
        source_pdf="data/bnc/t1.pdf",
        section_ranges=section_ranges,
        tables=tables,
        out_root=tmp_path,
    )

    assert set(written.keys()) == {"capital_management", "risk_management"}
    for path in written.values():
        assert Path(path).exists()
    assert Path(written["capital_management"]).name == "gestion_capital_t1.json"
    assert Path(written["risk_management"]).name == "gestion_risques_t1.json"

    capital_payload = json.loads(
        Path(written["capital_management"]).read_text(encoding="utf-8")
    )
    assert capital_payload["schema_version"] == "section_snapshot_v1"
    assert capital_payload["quarter"] == "T1"
    assert capital_payload["section"] == "capital_management"
    assert capital_payload["page_range"] == {"start": 21, "end": 30}
    assert capital_payload["tables"][0]["table_id"] == "tableau_14"
    assert capital_payload["tables"][0]["indicators"] == ["CET1", "TLAC"]
