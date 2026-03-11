"""Tests for quality-first extraction certification and matching hard rejects."""

from __future__ import annotations

from vigilance.models.table_models import (
    EXTRACTION_STATUS_BLOCKED,
    EXTRACTION_STATUS_CERTIFIED,
    EXTRACTION_STATUS_REVIEW_REQUIRED,
    TableArtifact,
    derive_extraction_blockers,
    get_extraction_status,
    is_auto_compare_eligible,
)
from vigilance.quality.quality_gate import evaluate_extraction_quality


def _table(
    *,
    table_id: str = "t1",
    section: str = "risk",
    indicators: list[str] | None = None,
    debug_metrics: dict | None = None,
) -> TableArtifact:
    ind = list(indicators or ["A", "B", "C"])
    dm = dict(debug_metrics or {})
    dm.setdefault("vision_extraction_applied", True)
    dm.setdefault("vision_extraction_confidence", 0.85)
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=1,
        table_id=table_id,
        title="Table",
        headers=["X", "Y"],
        rows=[[x, "1"] for x in ind],
        first_column_indicators=ind,
        first_column_indicators_raw=ind,
        extraction_method="vision_full_gpt4o",
        footnotes=[],
        debug_metrics=dict(dm),
    )


def test_certified_table_is_auto_compare_eligible() -> None:
    t = _table(
        indicators=["A", "B", "C"],
        debug_metrics={"vision_extraction_applied": True, "vision_extraction_confidence": 0.85},
    )
    assert derive_extraction_blockers(t) == []
    assert get_extraction_status(t) == EXTRACTION_STATUS_CERTIFIED
    assert is_auto_compare_eligible(t) is True


def test_blocked_crop_rejected() -> None:
    t = _table(debug_metrics={"crop_reject_reason": "area_too_large", "vision_extraction_applied": False})
    assert get_extraction_status(t) == EXTRACTION_STATUS_BLOCKED
    assert is_auto_compare_eligible(t) is False
    assert "crop_rejected" in derive_extraction_blockers(t)


def test_blocked_low_confidence() -> None:
    t = _table(debug_metrics={"vision_extraction_applied": True, "vision_extraction_confidence": 0.3})
    assert get_extraction_status(t) == EXTRACTION_STATUS_BLOCKED
    assert "low_extraction_confidence" in derive_extraction_blockers(t)


def test_review_required_medium_confidence() -> None:
    t = _table(debug_metrics={"vision_extraction_applied": True, "vision_extraction_confidence": 0.65})
    assert get_extraction_status(t) == EXTRACTION_STATUS_REVIEW_REQUIRED
    assert is_auto_compare_eligible(t) is False


def test_evaluate_extraction_quality_fail_on_blocked() -> None:
    certified = _table(table_id="c", debug_metrics={"vision_extraction_confidence": 0.9})
    blocked = _table(table_id="b", debug_metrics={"vision_extraction_confidence": 0.2})
    report = evaluate_extraction_quality([certified, blocked], config={"max_tables_blocked": 0})
    assert report["status"] == "FAIL"
    assert report["eligible_for_review"] is False
    assert any("extraction_blocked" in r for r in report["fail_reasons"])
    assert report["summary"]["tables_blocked"] == 1
    assert report["summary"]["tables_certified"] == 1


def test_evaluate_extraction_quality_pass_when_all_certified() -> None:
    tables = [
        _table(table_id="a", debug_metrics={"vision_extraction_confidence": 0.9}),
        _table(table_id="b", debug_metrics={"vision_extraction_confidence": 0.85}),
    ]
    report = evaluate_extraction_quality(tables)
    assert report["status"] == "PASS"
    assert report["summary"]["tables_certified"] == 2
    assert report["summary"]["tables_blocked"] == 0
