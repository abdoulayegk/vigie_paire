from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vigilance.extraction.vision_extraction_writer import write_compact_report_artifacts


def _table(
    *,
    table_id: str,
    page: int,
    section: str | None,
    title: str | None,
    raw: list[str] | None,
    normalized: list[str] | None,
    footnotes: list[dict] | None,
    headers: list[str] | None = None,
    rows: list[list[str]] | None = None,
    table_index_on_page: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        page_number=page,
        section=section,
        title=title,
        title_clean=None,
        headers=headers or [],
        rows=rows or [],
        first_column_indicators_raw=raw,
        first_column_indicators=normalized or [],
        footnotes=footnotes,
        table_index_on_page=table_index_on_page,
    )


def test_write_compact_report_artifacts_outputs_consistent_json(tmp_path: Path) -> None:
    tables = [
        _table(
            table_id="tableau_2",
            page=4,
            section=None,
            title=None,
            raw=None,
            normalized=None,
            footnotes=None,
            headers=["Metric", "Value"],
            rows=[["A", "1"]],
            table_index_on_page=1,
        ),
        _table(
            table_id="tableau_1",
            page=3,
            section="Gestion des risques",
            title="Table risque",
            raw=["Pertes de credit", "Total"],
            normalized=["pertes de credit", "total"],
            footnotes=[{"marker": "1", "text": "Note de risque"}],
            headers=["Indicator", "Amount"],
            rows=[["Pertes de credit", "10"]],
            table_index_on_page=2,
        ),
    ]

    paths = write_compact_report_artifacts(
        tables=tables,
        out_dir=tmp_path,
        bank_code="bnc",
        year=2025,
        quarter="t1",
        meta={"model_version": "gpt-5.4", "prompt_version": "extract_v1"},
    )

    assert set(paths) == {"tables", "indicators", "footnotes"}
    tables_payload = json.loads(paths["tables"].read_text(encoding="utf-8"))
    indicators_payload = json.loads(paths["indicators"].read_text(encoding="utf-8"))
    footnotes_payload = json.loads(paths["footnotes"].read_text(encoding="utf-8"))

    assert [t["table_id"] for t in tables_payload["tables"]] == ["tableau_1", "tableau_2"]
    assert [t["table_id"] for t in indicators_payload["tables"]] == ["tableau_1", "tableau_2"]
    assert [t["table_id"] for t in footnotes_payload["tables"]] == ["tableau_1", "tableau_2"]

    risk_table = tables_payload["tables"][0]
    assert risk_table["section"] == "risk_management"
    assert risk_table["title"] == "Table risque"
    assert risk_table["headers"] == ["Indicator", "Amount"]
    assert risk_table["rows"] == [["Pertes de credit", "10"]]
    assert risk_table["indicators_raw"] == ["Pertes de credit", "Total"]
    assert risk_table["indicators_normalized"] == ["pertes de credit", "total"]
    assert risk_table["footnotes"] == [{"id": "1", "text": "Note de risque"}]

    empty_table = tables_payload["tables"][1]
    assert empty_table["section"] == "unknown_section"
    assert empty_table["title"] == ""
    assert empty_table["indicators_raw"] == []
    assert empty_table["indicators_normalized"] == []
    assert empty_table["footnotes"] == []

    indicators_entry = indicators_payload["tables"][0]
    assert set(indicators_entry) == {
        "table_id",
        "page",
        "section",
        "title",
        "indicators_raw",
        "indicators_normalized",
    }
    assert "sections" not in indicators_entry
    assert "source" not in indicators_entry
    assert "date_reference" not in indicators_entry

    footnotes_entry = footnotes_payload["tables"][0]
    assert set(footnotes_entry) == {
        "table_id",
        "page",
        "section",
        "title",
        "footnotes",
    }
    assert "footnotes_content" not in footnotes_entry
    assert "has_footnotes" not in footnotes_entry
    assert "footnote_markers" not in footnotes_entry

    for payload in (tables_payload, indicators_payload, footnotes_payload):
        assert payload["bank_code"] == "bnc"
        assert payload["year"] == 2025
        assert payload["quarter"] == "t1"
        assert payload["schema_version"] == 3
        assert payload["model_version"] == "gpt-5.4"
        assert payload["prompt_version"] == "extract_v1"
