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
    bbox: list[float] | None,
    raw: list[str] | None,
    normalized: list[str] | None,
    footnotes: list[dict] | None,
    headers: list[str] | None = None,
    rows: list[list[str]] | None = None,
    table_index_on_page: int | None = None,
    extraction_status: str = "ok",
    debug_metrics: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        page_number=page,
        section=section,
        title=title,
        table_summary="",
        title_clean=None,
        bbox=bbox,
        headers=headers or [],
        rows=rows or [],
        first_column_indicators_raw=raw,
        first_column_indicators=normalized or [],
        footnotes=footnotes,
        table_index_on_page=table_index_on_page,
        extraction_status=extraction_status,
        debug_metrics=debug_metrics or {},
    )


def test_write_compact_report_artifacts_outputs_consistent_json(tmp_path: Path) -> None:
    tables = [
        _table(
            table_id="tableau_2",
            page=4,
            section=None,
            title=None,
            bbox=None,
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
            bbox=[0.1, 0.2, 0.8, 0.7],
            raw=["Pertes de credit", "Total"],
            normalized=["pertes de credit", "total"],
            footnotes=[{"marker": "1", "text": "Note de risque"}],
            headers=["Indicator", "Amount"],
            rows=[["Pertes de credit", "10"]],
            table_index_on_page=2,
            debug_metrics={
                "bbox_original": [0.12, 0.22, 0.78, 0.65],
                "bbox_final": [0.1, 0.2, 0.8, 0.7],
                "bbox_source": "page_context_locator",
                "bbox_confidence": 0.96,
                "bbox_verified": True,
                "bbox_verification_reason": "single_region_confirmed",
            },
        ),
    ]

    paths = write_compact_report_artifacts(
        tables=tables,
        out_dir=tmp_path,
        bank_code="bnc",
        year=2025,
        quarter="t1",
        meta={
            "model_version": "gpt-5.4",
        },
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
    assert risk_table["bbox"] == [0.1, 0.2, 0.8, 0.7]
    assert risk_table["extraction_status"] == "ok"
    assert risk_table["table_summary"] == ""
    assert risk_table["headers"] == ["Indicator", "Amount"]
    assert risk_table["row_count"] == 2
    assert risk_table["indicators"] == ["Pertes de credit", "Total"]
    assert risk_table["footnotes"] == [{"id": "1", "text": "Note de risque"}]
    assert risk_table["bbox_provenance"] == {
        "bbox_original": [0.12, 0.22, 0.78, 0.65],
        "bbox_final": [0.1, 0.2, 0.8, 0.7],
        "bbox_source": "page_context_locator",
        "bbox_confidence": 0.96,
        "bbox_verified": True,
        "bbox_verification_reason": "single_region_confirmed",
    }

    empty_table = tables_payload["tables"][1]
    assert empty_table["section"] == "unknown_section"
    assert empty_table["title"] == ""
    assert empty_table["bbox"] is None
    assert empty_table["extraction_status"] == "ok"
    assert empty_table["table_summary"] == ""
    assert empty_table["row_count"] == 0
    assert empty_table["indicators"] == []
    assert empty_table["footnotes"] == []

    indicators_entry = indicators_payload["tables"][0]
    assert set(indicators_entry) == {
        "table_id",
        "page",
        "section",
        "title",
        "indicators",
    }
    assert indicators_entry["indicators"] == ["Pertes de credit", "Total"]
    assert "bbox" not in indicators_entry
    assert "table_summary" not in indicators_entry

    footnotes_entry = footnotes_payload["tables"][0]
    assert set(footnotes_entry) == {
        "table_id",
        "page",
        "section",
        "footnotes",
    }
    assert footnotes_entry["footnotes"] == [{"id": "1", "text": "Note de risque"}]
    assert "title" not in footnotes_entry
    assert "bbox" not in footnotes_entry
    assert "footnotes_content" not in footnotes_entry

    for payload in (tables_payload, indicators_payload, footnotes_payload):
        assert payload["bank_code"] == "bnc"
        assert payload["year"] == 2025
        assert payload["quarter"] == "t1"
        assert payload["schema_version"] == 7
        assert "model_version" not in payload
        assert "prompt_version" not in payload


def test_write_compact_report_artifacts_allows_multiple_tables_on_same_page(
    tmp_path: Path,
) -> None:
    tables = [
        _table(
            table_id="tableau_1",
            page=5,
            section="Gestion des risques",
            title="Top table",
            bbox=[0.1, 0.1, 0.9, 0.4],
            raw=["A"],
            normalized=["a"],
            footnotes=None,
            table_index_on_page=1,
        ),
        _table(
            table_id="tableau_2",
            page=5,
            section="Gestion des risques",
            title="Bottom table",
            bbox=[0.1, 0.5, 0.9, 0.9],
            raw=["B"],
            normalized=["b"],
            footnotes=None,
            table_index_on_page=2,
        ),
    ]

    paths = write_compact_report_artifacts(
        tables=tables,
        out_dir=tmp_path,
        bank_code="bnc",
        year=2025,
        quarter="t1",
    )

    payload = json.loads(paths["tables"].read_text(encoding="utf-8"))
    assert [entry["table_id"] for entry in payload["tables"]] == [
        "tableau_1",
        "tableau_2",
    ]
    assert [entry["page"] for entry in payload["tables"]] == [5, 5]


def test_write_compact_report_artifacts_rejects_duplicate_table_ids(
    tmp_path: Path,
) -> None:
    tables = [
        _table(
            table_id="tableau_1",
            page=5,
            section="Gestion des risques",
            title="Top table",
            bbox=[0.1, 0.1, 0.9, 0.4],
            raw=["A"],
            normalized=["a"],
            footnotes=None,
            table_index_on_page=1,
        ),
        _table(
            table_id="tableau_1",
            page=5,
            section="Gestion des risques",
            title="Bottom table",
            bbox=[0.1, 0.5, 0.9, 0.9],
            raw=["B"],
            normalized=["b"],
            footnotes=None,
            table_index_on_page=2,
        ),
    ]

    try:
        write_compact_report_artifacts(
            tables=tables,
            out_dir=tmp_path,
            bank_code="bnc",
            year=2025,
            quarter="t1",
        )
    except ValueError as exc:
        assert "Duplicate table_id" in str(exc)
        assert "Multiple tables may share a page" in str(exc)
    else:
        raise AssertionError("Expected duplicate table_id validation to fail")
