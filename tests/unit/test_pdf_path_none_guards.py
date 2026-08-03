from __future__ import annotations

from vigilance.ui_detection import _detect_sections_core
from vigilance.ui_io import load_comparison_result
from vigilance.extraction.vision_cache import compute_pdf_sha256


def test_load_comparison_result_returns_none_for_missing_path() -> None:
    assert load_comparison_result(None) is None


def test_compute_pdf_sha256_returns_empty_string_for_missing_path() -> None:
    assert compute_pdf_sha256(None) == ""
    assert compute_pdf_sha256("") == ""


def test_detect_sections_core_returns_fallback_for_missing_path() -> None:
    result = _detect_sections_core(None, "bnc")

    assert result["total_pages"] == 1
    assert isinstance(result["sections"], list)
    assert result["sections"]
