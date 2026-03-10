"""Tests for deterministic Vision flag propagation in comparison runner."""

from __future__ import annotations

import os
from types import SimpleNamespace

from app.comparison_runner import (
    _compute_extraction_kpis,
    _extract_tables,
    _resolve_vision_extraction_enabled,
)
from vigilance.models.table_models import TableArtifact, VISION_CONTENT_SOURCE


def test_resolve_vision_extraction_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("VIGILANCE_VISION_EXTRACTION_ENABLED", "1")
    assert _resolve_vision_extraction_enabled("bnc", False, allow_env_legacy=True) is False
    assert _resolve_vision_extraction_enabled("bnc", True, allow_env_legacy=True) is True


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
        use_vision_extraction: bool | None = None,
    ) -> list[object]:
        seen["pdf_path"] = pdf_path
        seen["bank_code"] = bank_code
        seen["quarter"] = quarter
        seen["year"] = year
        seen["section_ranges"] = section_ranges
        seen["use_vision_extraction"] = use_vision_extraction
        return []

    monkeypatch.setattr(dp, "extract_tables_docling_by_sections", fake_extract_tables_docling_by_sections)

    _extract_tables(
        pdf_path="/tmp/fake.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[{"section": "s", "start": 1, "end": 1}],
        api_key=None,
        use_vision_extraction=False,
        use_stored_extraction_if_available=False,
    )

    assert seen["use_vision_extraction"] is False


def test_extract_tables_reuses_stored_when_fresh_extraction_is_empty(
    monkeypatch,
    tmp_path,
) -> None:
    import app.comparison_runner as cr
    import app.extraction_storage as storage_mod
    import vigilance.extraction.docling_processor as dp

    stored_table = TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=1,
        table_id="stored_ok",
        title="Stored",
        headers=["Indicateur", "Montant"],
        rows=[["Ratio CET1", "13.1"]],
        first_column_indicators=["ratio cet1"],
        first_column_indicators_raw=["Ratio CET1"],
        extraction_method="vision_full_gpt4o",
        footnotes=[],
        quarter="t1",
        pdf_path="/tmp/stored.pdf",
        content_source=VISION_CONTENT_SOURCE,
    )
    fresh_failed = TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=1,
        table_id="fresh_failed",
        title="Fresh",
        headers=[],
        rows=[],
        first_column_indicators=[],
        first_column_indicators_raw=[],
        extraction_method="vision_failed",
        footnotes=[],
        quarter="t1",
        pdf_path="/tmp/fresh.pdf",
        content_source=VISION_CONTENT_SOURCE,
    )

    monkeypatch.setattr(
        dp,
        "extract_tables_docling_by_sections",
        lambda **_kwargs: [SimpleNamespace()],
    )
    monkeypatch.setattr(cr, "_table_to_artifact", lambda *_a, **_k: fresh_failed)
    monkeypatch.setattr(
        storage_mod,
        "load_extraction",
        lambda *_a, **_k: ([stored_table], {"schema_version": 3}),
    )

    save_called = {"count": 0}

    def _fake_save_extraction(**_kwargs):
        save_called["count"] += 1
        raise AssertionError("save_extraction should not run when stored extraction is reused")

    monkeypatch.setattr(storage_mod, "save_extraction", _fake_save_extraction)

    result = _extract_tables(
        pdf_path="/tmp/fake.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[{"section": "s", "start": 1, "end": 1}],
        api_key=None,
        use_vision_extraction=True,
        use_stored_extraction_if_available=False,
        extraction_base_dir=str(tmp_path),
    )

    assert [t.table_id for t in result] == ["stored_ok"]
    assert save_called["count"] == 0


def test_extraction_kpis_include_vision_extraction_contract_fields() -> None:
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
            "vision_extraction_attempted": True,
            "vision_extraction_applied": False,
            "vision_schema_contract_failed": True,
            "vision_extraction_disabled_reason": "Vision schema contract invalid: Missing 'appears_truncated'",
        },
    )
    kpis = _compute_extraction_kpis(
        [table],
        [],
        comparisons=[],
        tables_added=[],
        tables_removed=[],
    )
    assert kpis["vision_extraction_attempted_count"] == 1
    assert kpis["vision_extraction_applied_count"] == 0
    assert kpis["vision_schema_contract_fail_count"] == 1
    assert "Vision schema contract invalid" in str(
        kpis["vision_extraction_disabled_reason"]
    )


def test_table_to_artifact_requires_vision_raw_for_comparison() -> None:
    from types import SimpleNamespace

    from app.comparison_runner import _table_to_artifact

    fake = SimpleNamespace(
        rows=[["Docling label", "1"]],
        headers=["Col1", "Col2"],
        first_column_indicators=["docling label"],
        first_column_indicators_raw=None,
        section="gestion_capital",
        page_number=3,
        table_id="T1",
        title="Table",
        extraction_method="docling",
        table_number=None,
        bbox=None,
        footnotes=[],
    )

    art = _table_to_artifact(fake, bank_code="bnc", quarter="t1", pdf_path="/tmp/fake.pdf")
    assert art.first_column_indicators == []
    assert art.first_column_indicators_raw == []
    assert art.comparison_eligible is False
    assert "non_vision_content_source" in art.comparison_blockers
    assert "missing_vision_indicators" in art.comparison_blockers


def test_table_artifact_explicit_indicator_aliases() -> None:
    artifact = TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=2,
        table_id="T2",
        title="Capital",
        headers=[],
        rows=[],
        first_column_indicators=["ratio cet1"],
        first_column_indicators_raw=["Ratio CET1"],
        extraction_method="vision_full_gpt4o",
        footnotes=[{"id": "1", "text": "Note"}],
        content_source=VISION_CONTENT_SOURCE,
    )

    assert artifact.vision_raw_indicators == ["Ratio CET1"]
    assert artifact.comparison_normalized_indicators == ["ratio cet1"]
    assert artifact.canonical_footnotes == [{"id": "1", "text": "Note"}]
    assert artifact.is_vision_sourced is True
