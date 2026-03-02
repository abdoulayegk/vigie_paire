"""Tests for deterministic Vision flag propagation in comparison runner."""

from __future__ import annotations

import os

from app.comparison_runner import _extract_tables, _resolve_vision_primary_mode


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
        use_vision_fallback: bool | None = None,
    ) -> list[object]:
        seen["pdf_path"] = pdf_path
        seen["bank_code"] = bank_code
        seen["quarter"] = quarter
        seen["year"] = year
        seen["section_ranges"] = section_ranges
        seen["use_vision_primary"] = use_vision_primary
        seen["use_vision_fallback"] = use_vision_fallback
        return []

    monkeypatch.setattr(dp, "extract_tables_docling_by_sections", fake_extract_tables_docling_by_sections)
    monkeypatch.setenv("ENABLE_VISION_FALLBACK", "1")

    _extract_tables(
        pdf_path="/tmp/fake.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[{"section": "s", "start": 1, "end": 1}],
        use_vision_fallback=False,
        api_key=None,
        use_vision_primary=False,
    )

    assert seen["use_vision_primary"] is False
    assert seen["use_vision_fallback"] is False
    assert os.environ.get("ENABLE_VISION_FALLBACK") == "1"
