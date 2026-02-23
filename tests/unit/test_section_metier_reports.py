"""Tests for per-section business report JSON exports."""

from __future__ import annotations

import json
from pathlib import Path

from app.comparison_runner import write_section_metier_reports
from vigilance.models.table_models import TableArtifact


def _table(*, section: str, table_id: str, page: int, indicators: list[str]) -> TableArtifact:
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=f"Tableau {table_id}",
        headers=["Indicateur", "Valeur"],
        rows=[[i, "1"] for i in indicators],
        first_column_indicators=indicators,
        extraction_method="docling",
        quarter="t1",
        pdf_path="dummy.pdf",
    )


def test_write_section_metier_reports_generates_expected_french_file(tmp_path: Path) -> None:
    t1_table = _table(
        section="capital_management",
        table_id="tableau_24",
        page=24,
        indicators=["Fonds propres CET1", "Solde au début"],
    )
    t2_table = _table(
        section="capital_management",
        table_id="tableau_24",
        page=25,
        indicators=["Fonds propres CET1", "Réserve contracyclique"],
    )

    result = {
        "pairs": [
            {
                "section": "capital_management",
                "score": 0.87,
                "t1_uid": "capital_management|tableau_24|p24",
                "t2_uid": "capital_management|tableau_24|p25",
            }
        ],
        "added_tables": [
            {
                "section": "capital_management",
                "t2_table_id": "tableau_30",
                "title_t2": "Tableau 30",
                "page_t2": 30,
            }
        ],
        "removed_tables": [
            {
                "section": "capital_management",
                "t1_table_id": "tableau_10",
                "title_t1": "Tableau 10",
                "page_t1": 10,
            }
        ],
    }

    written = write_section_metier_reports(
        bank_code="bnc",
        year_t1=2025,
        quarter_t1="t1",
        year_t2=2025,
        quarter_t2="t2",
        result=result,
        tables_t1=[t1_table],
        tables_t2=[t2_table],
        out_root=tmp_path,
    )

    path = Path(written["capital_management"])
    assert path.exists()
    assert path.name == "gestion_capital_metier.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "section_metier_v1"
    assert payload["bank"] == "bnc"
    assert payload["quarter_t1"] == "T1"
    assert payload["quarter_t2"] == "T2"
    assert payload["section"] == "gestion_capital"
    assert payload["summary"] == {
        "tables_matched": 1,
        "tables_added": 1,
        "tables_removed": 1,
        "indicators_added": 1,
        "indicators_removed": 1,
        "indicators_modified": 0,
        "uncertain_matches": 0,
    }
    change_types = {c["change_type"] for c in payload["changes"]}
    assert {
        "indicator_added",
        "indicator_removed",
        "table_added",
        "table_removed",
    }.issubset(change_types)
    assert all(str(c["change_id"]).startswith("gc_") for c in payload["changes"])
    indicator_added = [c for c in payload["changes"] if c["change_type"] == "indicator_added"][0]
    assert indicator_added["table_number"] == "24"
    assert indicator_added["indicator_name"] == "Réserve contracyclique"


def test_write_section_metier_reports_detects_indicator_and_table_modifications(
    tmp_path: Path,
) -> None:
    t1_table = TableArtifact(
        bank_code="bnc",
        section="risk_management",
        page_pdf=40,
        table_id="tableau_55",
        title="Tableau 55 - Risques",
        headers=["Indicateur", "Valeur T1"],
        rows=[["Ratio CET1", "12.1 %"], ["Ratio de levier", "4.2 %"]],
        first_column_indicators=["Ratio CET1", "Ratio de levier"],
        extraction_method="docling",
        quarter="t1",
        pdf_path="dummy.pdf",
    )
    t2_table = TableArtifact(
        bank_code="bnc",
        section="risk_management",
        page_pdf=41,
        table_id="tableau_55",
        title="Tableau 55 - Risques et capital",
        headers=["Indicateur", "Valeur T2"],
        rows=[["Ratio CET1", "13.0 %"], ["Ratio de levier", "4.2 %"]],
        first_column_indicators=["Ratio CET1", "Ratio de levier"],
        extraction_method="docling",
        quarter="t2",
        pdf_path="dummy.pdf",
    )

    result = {
        "pairs": [
            {
                "section": "risk_management",
                "score": 0.92,
                "t1_uid": "risk_management|tableau_55|p40",
                "t2_uid": "risk_management|tableau_55|p41",
            }
        ],
        "added_tables": [],
        "removed_tables": [],
    }

    written = write_section_metier_reports(
        bank_code="bnc",
        year_t1=2025,
        quarter_t1="t1",
        year_t2=2025,
        quarter_t2="t2",
        result=result,
        tables_t1=[t1_table],
        tables_t2=[t2_table],
        out_root=tmp_path,
    )

    payload = json.loads(Path(written["risk_management"]).read_text(encoding="utf-8"))

    assert payload["summary"]["indicators_modified"] == 1
    change_types = [c["change_type"] for c in payload["changes"]]
    assert "indicator_modified" in change_types
    assert "table_modified" in change_types

    line_change = [c for c in payload["changes"] if c["change_type"] == "indicator_modified"][0]
    assert line_change["indicator_name"] == "Ratio CET1"
    assert line_change["value_t1"] == ["12.1 %"]
    assert line_change["value_t2"] == ["13.0 %"]

    table_change = [c for c in payload["changes"] if c["change_type"] == "table_modified"][0]
    assert "title_changed" in table_change["details"]["reasons"]
    assert "headers_changed" in table_change["details"]["reasons"]
