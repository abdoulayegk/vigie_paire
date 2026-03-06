"""Tests for deterministic Vision flag propagation in comparison runner."""

from __future__ import annotations

import os

from app.comparison_runner import (
    _compute_extraction_kpis,
    _extract_tables,
    _resolve_vision_primary_mode,
)
from vigilance.models.table_models import TableArtifact


def test_resolve_vision_primary_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("VIGILANCE_VISION_PRIMARY", "1")
    assert _resolve_vision_primary_mode("bnc", False, allow_env_legacy=True) is False
    assert _resolve_vision_primary_mode("bnc", True, allow_env_legacy=True) is True


def test_extract_tables_forwards_flags_without_env_mutation(monkeypatch) -> None:
    import vigilance.extraction.docling_processor as dp

    seen: dict[str, object] = {}

    def fake_extract_tables_docling_by_sections(
        *,
        pdf_path: str,
        bank_code: str,
        quarter: str,
        year: int,
        section_ranges: list[dict[str, object]],
        use_vision_primary: bool | None = None,
    ) -> list[object]:
        seen["pdf_path"] = pdf_path
        seen["bank_code"] = bank_code
        seen["quarter"] = quarter
        seen["year"] = year
        seen["section_ranges"] = section_ranges
        seen["use_vision_primary"] = use_vision_primary
        return []

    monkeypatch.setattr(dp, "extract_tables_docling_by_sections", fake_extract_tables_docling_by_sections)

    _extract_tables(
        pdf_path="/tmp/fake.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[{"section": "s", "start": 1, "end": 1}],
        api_key=None,
        use_vision_primary=False,
        use_stored_extraction_if_available=False,
    )

    assert seen["use_vision_primary"] is False


def test_extraction_kpis_include_vision_primary_contract_fields() -> None:
    table = TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=21,
        table_id="t1",
        title="Table",
        headers=[],
        rows=[],
        first_column_indicators=[],
        extraction_method="docling",
        debug_metrics={
            "vision_primary_attempted": True,
            "vision_primary_applied": False,
            "vision_schema_contract_failed": True,
            "vision_primary_disabled_reason": "Vision schema contract invalid: Missing 'appears_truncated'",
        },
    )
    kpis = _compute_extraction_kpis(
        [table],
        [],
        comparisons=[],
        tables_added=[],
        tables_removed=[],
    )
    assert kpis["vision_primary_attempted_count"] == 1
    assert kpis["vision_primary_applied_count"] == 0
    assert kpis["vision_schema_contract_fail_count"] == 1
    assert "Vision schema contract invalid" in str(
        kpis["vision_primary_disabled_reason"]
    )
