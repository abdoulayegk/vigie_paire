"""Tests for extraction_status-first quality and eligibility decisions."""

from __future__ import annotations

from vigilance.models.table_models import (
    TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE,
    TABLE_EXTRACTION_STATUS_OK,
    TABLE_EXTRACTION_STATUS_RESCUED,
    TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED,
    TableArtifact,
)
from vigilance.quality.quality_gate import evaluate_extraction_quality


def _table(
    *,
    table_id: str = "t1",
    extraction_status: str = TABLE_EXTRACTION_STATUS_OK,
    indicators: list[str] | None = None,
    debug_metrics: dict | None = None,
) -> TableArtifact:
    ind = list(indicators or ["A", "B", "C"])
    dm = dict(debug_metrics or {})
    dm.setdefault("vision_extraction_applied", True)
    dm.setdefault("vision_extraction_confidence", 0.85)
    return TableArtifact(
        bank_code="bnc",
        section="risk_management",
        page_pdf=1,
        table_id=table_id,
        title="Table",
        headers=["X", "Y"],
        rows=[[x, "1"] for x in ind],
        first_column_indicators=ind,
        first_column_indicators_raw=ind,
        extraction_method="vision_full_gpt4o",
        footnotes=[],
        debug_metrics=dm,
        extraction_status=extraction_status,
    )


def test_extraction_status_drives_comparison_eligibility() -> None:
    ok_table = _table(extraction_status=TABLE_EXTRACTION_STATUS_OK)
    rescued_table = _table(
        table_id="rescued",
        extraction_status=TABLE_EXTRACTION_STATUS_RESCUED,
    )
    suspect_table = _table(
        table_id="suspect",
        extraction_status=TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED,
    )
    artifact_table = _table(
        table_id="artifact",
        extraction_status=TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE,
        indicators=[],
    )

    assert ok_table.comparison_eligible is True
    assert rescued_table.comparison_eligible is True
    assert suspect_table.comparison_eligible is False
    assert artifact_table.comparison_eligible is False
    assert suspect_table.comparison_blockers == [TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED]
    assert artifact_table.comparison_blockers == [TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE]


def test_evaluate_extraction_quality_fails_on_suspect_unresolved() -> None:
    ok_table = _table(table_id="ok", extraction_status=TABLE_EXTRACTION_STATUS_OK)
    suspect_table = _table(
        table_id="suspect",
        extraction_status=TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED,
        debug_metrics={"vision_extraction_confidence": 0.32},
    )

    report = evaluate_extraction_quality([ok_table, suspect_table])

    assert report["status"] == "FAIL"
    assert report["eligible_for_review"] is False
    assert report["summary"]["tables_ok"] == 1
    assert report["summary"]["tables_suspect_unresolved"] == 1
    assert any(
        "extraction_suspect_unresolved_tables=1" in reason
        for reason in report["fail_reasons"]
    )
    assert len(report["suspect_table_evidence"]) == 1
    assert report["suspect_table_evidence"][0]["table_id"] == "suspect"


def test_evaluate_extraction_quality_passes_on_ok_and_rescued() -> None:
    report = evaluate_extraction_quality(
        [
            _table(table_id="ok", extraction_status=TABLE_EXTRACTION_STATUS_OK),
            _table(
                table_id="rescued",
                extraction_status=TABLE_EXTRACTION_STATUS_RESCUED,
            ),
        ]
    )

    assert report["status"] == "PASS"
    assert report["eligible_for_review"] is True
    assert report["summary"]["tables_ok"] == 1
    assert report["summary"]["tables_rescued"] == 1
    assert report["summary"]["tables_suspect_unresolved"] == 0


def test_confirmed_no_table_is_warning_only() -> None:
    report = evaluate_extraction_quality(
        [
            _table(
                table_id="artifact",
                extraction_status=TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE,
                indicators=[],
            )
        ]
    )

    assert report["status"] == "PASS"
    assert report["eligible_for_review"] is True
    assert report["summary"]["tables_confirmed_no_table"] == 1
    assert report["summary"]["tables_suspect_unresolved"] == 0
