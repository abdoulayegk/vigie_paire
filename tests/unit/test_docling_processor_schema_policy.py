"""Unit tests for Vision schema failure policy in Docling primary mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vigilance.extraction.docling_processor import DoclingProcessor
from vigilance.extraction.vision_full_extractor import VisionSchemaContractError


class _FakeBBox:
    def as_tuple(self) -> tuple[float, float, float, float]:
        return (0.1, 0.1, 0.9, 0.7)


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no
        self.bbox = _FakeBBox()


class _FakeTable:
    def __init__(self, page_no: int) -> None:
        self.prov = [_FakeProv(page_no)]


class _FailingVisionExtractor:
    calls = 0

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del args, kwargs

    def extract_with_quality_pass(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        _FailingVisionExtractor.calls += 1
        raise VisionSchemaContractError(
            "Invalid schema for response_format 'vision_full_extraction': Missing 'appears_truncated'."
        )


def _build_processor_with_fake_doc(monkeypatch, pages: list[int]) -> DoclingProcessor:
    fake_doc = SimpleNamespace(
        tables=[_FakeTable(p) for p in pages],
        pages={},
        export_to_markdown=lambda: "",
    )
    fake_converter = SimpleNamespace(
        convert=lambda *args, **kwargs: SimpleNamespace(document=fake_doc)
    )

    processor = DoclingProcessor(use_vision_fallback=False, openai_api_key="test-key")
    processor._converter = fake_converter
    processor._initialized = True
    # Vision-only path: no Docling content helpers are called; only Vision extract runs.
    processor._associate_tables_with_sections = (  # type: ignore[method-assign]
        lambda tables, text_content: tables
    )

    monkeypatch.setattr(
        "vigilance.extraction.vision_cache.compute_pdf_sha256",
        lambda pdf_path: "sha-test",
    )
    monkeypatch.setattr(
        "vigilance.extraction.vision_full_extractor.VisionFullExtractor",
        _FailingVisionExtractor,
    )
    monkeypatch.setattr(
        "vigilance.utils.pdf_crop.crop_table_region_to_bytes",
        lambda pdf_path, page_number, bbox_norm, bottom_extension=0.0, dpi=300: b"fake",
    )
    monkeypatch.setattr(
        "vigilance.extraction.docling_processor.logger",
        SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
    )
    return processor


def test_schema_policy_fail_fast_stops_run(monkeypatch, tmp_path: Path) -> None:
    _FailingVisionExtractor.calls = 0
    processor = _build_processor_with_fake_doc(monkeypatch, pages=[1, 2])

    monkeypatch.setattr(
        "vigilance.config.get_vision_extraction_config",
        lambda bank_code=None: {
            "bottom_extension_footnotes": 0.12,
            "fallback_to_docling_on_error": True,
            "schema_failure_policy": "fail_fast",
        },
    )
    with pytest.raises(RuntimeError, match="Vision schema contract invalid"):
        processor._extract_with_docling(
            pdf_path=tmp_path / "dummy.pdf",
            bank_code="bnc",
            quarter="t1",
            year=2026,
            page_ranges=[(1, 3)],
            labels_only=False,
            use_vision_primary=True,
        )
    assert _FailingVisionExtractor.calls == 1


def test_schema_policy_degrade_to_docling_disables_vision(monkeypatch, tmp_path: Path) -> None:
    _FailingVisionExtractor.calls = 0
    processor = _build_processor_with_fake_doc(monkeypatch, pages=[1, 2])

    monkeypatch.setattr(
        "vigilance.config.get_vision_extraction_config",
        lambda bank_code=None: {
            "bottom_extension_footnotes": 0.12,
            "fallback_to_docling_on_error": True,
            "schema_failure_policy": "degrade_to_docling",
        },
    )
    extracted = processor._extract_with_docling(
        pdf_path=tmp_path / "dummy.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2026,
        page_ranges=[(1, 3)],
        labels_only=False,
        use_vision_primary=True,
    )

    assert _FailingVisionExtractor.calls == 1
    assert len(extracted.all_tables) == 2
    first_dm = extracted.all_tables[0].debug_metrics or {}
    second_dm = extracted.all_tables[1].debug_metrics or {}
    assert first_dm.get("vision_schema_contract_failed") is True
    assert "Vision schema contract invalid" in str(
        first_dm.get("vision_primary_disabled_reason", "")
    )
    assert second_dm.get("vision_primary_attempted") is False
    assert "Vision schema contract invalid" in str(
        second_dm.get("vision_primary_disabled_reason", "")
    )
