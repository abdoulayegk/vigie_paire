"""Tests for vision extraction quality improvements.

Covers:
- Truncation recovery: rows + footnotes salvaged from truncated JSON
- Cache versioning: _VISION_CACHE_VERSION prefix in cache keys
- Prompt reinforcement: no-dictionary instruction when reference_text absent
- Preprocess flag: explicit enabled= parameter override
- Quality summary: aggregate stats computation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. Truncation recovery
# ---------------------------------------------------------------------------

class TestTruncationRecovery:
    """_try_parse_truncated_result now salvages rows and footnotes."""

    @staticmethod
    def _parse(raw: str):
        from vigilance.extraction.vision_full_extractor import _try_parse_truncated_result
        return _try_parse_truncated_result(raw)

    def test_salvages_rows_and_footnotes(self) -> None:
        data = {
            "table_title": "Tableau 26",
            "headers": ["Indicateur", "T1", "T2"],
            "indicators": ["Ratio CET1", "Total"],
            "rows": [
                ["Ratio CET1", "13,1 %", "12,8 %"],
                ["Total", "100", "200"],
            ],
            "footnotes_content": [
                {"id": "1", "text": "Comprennent les instruments"},
                {"id": "2", "text": "Le ratio de levier"},
            ],
            "footnote_markers": ["1", "2"],
            "confidence": 0.88,
        }
        result = self._parse(json.dumps(data))
        assert result is not None
        assert result.appears_truncated is True
        assert result.vision_status == "partial"
        assert len(result.rows) == 2
        assert result.rows[0] == ["Ratio CET1", "13,1 %", "12,8 %"]
        assert len(result.footnotes_content) == 2
        assert result.footnotes_content[0]["marker"] == "1"
        assert result.footnotes_content[1]["text"] == "Le ratio de levier"

    def test_partial_rows_incomplete_row_dropped(self) -> None:
        """A row that is not a list of strings/numbers is skipped."""
        data = {
            "indicators": ["A", "B"],
            "confidence": 0.9,
            "rows": [
                ["A", "100"],
                {"broken": True},
                ["B", "200"],
            ],
        }
        result = self._parse(json.dumps(data))
        assert result is not None
        assert len(result.rows) == 2
        assert result.rows[0] == ["A", "100"]
        assert result.rows[1] == ["B", "200"]

    def test_footnotes_legacy_dict_format(self) -> None:
        """Legacy dict footnotes are still handled."""
        data = {
            "indicators": ["X"],
            "confidence": 0.85,
            "footnotes_content": {"1": "Note A", "2": "Note B"},
        }
        result = self._parse(json.dumps(data))
        assert result is not None
        assert len(result.footnotes_content) == 2
        assert result.footnotes_content[0]["marker"] == "1"

    def test_missing_rows_and_footnotes_still_works(self) -> None:
        """Tables without rows/footnotes produce empty lists (no regression)."""
        data = {
            "indicators": ["Only indicator"],
            "confidence": 0.7,
        }
        result = self._parse(json.dumps(data))
        assert result is not None
        assert result.rows == []
        assert result.footnotes_content == []

    def test_minimum_required_fields(self) -> None:
        """Without indicators or confidence, returns None."""
        assert self._parse(json.dumps({"confidence": 0.5})) is None
        assert self._parse(json.dumps({"indicators": ["A"]})) is None
        assert self._parse("not json at all") is None


# ---------------------------------------------------------------------------
# 2. Cache versioning
# ---------------------------------------------------------------------------

class TestCacheVersioning:
    def test_cache_key_includes_version_prefix(self) -> None:
        from vigilance.extraction.vision_cache import _VISION_CACHE_VERSION, make_cache_key

        key = make_cache_key("abc123sha", 5, [0.1, 0.2, 0.8, 0.9])
        assert key.startswith(f"{_VISION_CACHE_VERSION}_")
        assert "abc123sha" in key
        assert "_5_" in key

    def test_cache_key_changes_when_version_bumped(self) -> None:
        from vigilance.extraction.vision_cache import make_cache_key

        key_v2 = make_cache_key("sha", 1, [0.0, 0.0, 1.0, 1.0])
        with patch("vigilance.extraction.vision_cache._VISION_CACHE_VERSION", "v99"):
            from vigilance.extraction import vision_cache
            old_ver = vision_cache._VISION_CACHE_VERSION
            vision_cache._VISION_CACHE_VERSION = "v99"
            key_v99 = vision_cache.make_cache_key("sha", 1, [0.0, 0.0, 1.0, 1.0])
            vision_cache._VISION_CACHE_VERSION = old_ver
        assert key_v2 != key_v99

    def test_cache_key_empty_on_invalid_bbox(self) -> None:
        from vigilance.extraction.vision_cache import make_cache_key

        assert make_cache_key("sha", 1, []) == ""
        assert make_cache_key("sha", 1, [0.1, 0.2]) == ""


# ---------------------------------------------------------------------------
# 3. Prompt reinforcement (no dictionary)
# ---------------------------------------------------------------------------

class TestPromptReinforcement:
    @staticmethod
    def _build(reference_text=None, vision_cfg=None):
        from vigilance.extraction.vision_full_extractor import _build_prompt
        return _build_prompt("bnc", vision_cfg or {}, reference_text=reference_text)

    def test_with_dictionary_includes_consigne_dictionnaire(self) -> None:
        prompt = self._build(reference_text="A" * 100)
        assert "DICTIONNAIRE DE RÉFÉRENCE" in prompt
        assert "Transcris les libellés d'indicateurs" in prompt

    def test_without_dictionary_includes_transcris_exactement(self) -> None:
        prompt = self._build(reference_text=None)
        assert "Transcris EXACTEMENT ce que tu vois" in prompt
        assert "DICTIONNAIRE DE RÉFÉRENCE" not in prompt

    def test_short_reference_text_treated_as_no_dictionary(self) -> None:
        prompt = self._build(reference_text="too short")
        assert "Transcris EXACTEMENT ce que tu vois" in prompt

    def test_reference_text_max_chars_respected(self) -> None:
        long_text = "A" * 10000
        prompt = self._build(reference_text=long_text, vision_cfg={"vision_reference_text_max_chars": 50})
        assert "DICTIONNAIRE DE RÉFÉRENCE" in prompt
        assert "A" * 50 in prompt
        assert "A" * 51 not in prompt


# ---------------------------------------------------------------------------
# 4. Preprocess flag
# ---------------------------------------------------------------------------

class TestPreprocessFlag:
    def test_explicit_false_skips_preprocessing(self) -> None:
        from vigilance.extraction.vision_image_preprocessor import preprocess_for_vision

        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = preprocess_for_vision(raw, enabled=False)
        assert result is raw

    def test_explicit_true_attempts_preprocessing(self) -> None:
        from vigilance.extraction.vision_image_preprocessor import preprocess_for_vision

        raw = b"not a real png"
        result = preprocess_for_vision(raw, enabled=True)
        assert result == raw

    def test_none_falls_back_to_env(self, monkeypatch) -> None:
        from vigilance.extraction.vision_image_preprocessor import preprocess_for_vision

        monkeypatch.setenv("VISION_PREPROCESS", "false")
        raw = b"test"
        result = preprocess_for_vision(raw, enabled=None)
        assert result is raw


# ---------------------------------------------------------------------------
# 5. Quality summary
# ---------------------------------------------------------------------------

class TestQualitySummary:
    @staticmethod
    def _make_table(debug_metrics: dict[str, Any]):
        """Create a minimal object with debug_metrics attr."""
        class _T:
            def __init__(self, dm: dict) -> None:
                self.debug_metrics = dm
        return _T(debug_metrics)

    def test_basic_counts(self) -> None:
        from vigilance.extraction.docling_processor import _compute_vision_quality_summary

        tables = [
            self._make_table({
                "vision_status": "ok",
                "vision_extraction_attempted": True,
                "vision_extraction_confidence": 0.95,
                "has_reference_text": True,
            }),
            self._make_table({
                "vision_status": "partial",
                "vision_extraction_attempted": True,
                "vision_extraction_confidence": 0.75,
                "has_reference_text": False,
                "appears_truncated": True,
            }),
            self._make_table({
                "vision_status": "failed",
                "vision_extraction_attempted": True,
                "vision_extraction_confidence": 0.0,
                "has_reference_text": True,
                "crop_reject_reason": "too small",
            }),
        ]
        summary = _compute_vision_quality_summary(tables)
        assert summary["total_tables"] == 3
        assert summary["attempted"] == 3
        assert summary["ok"] == 1
        assert summary["partial"] == 1
        assert summary["failed"] == 1
        assert summary["truncated"] == 1
        assert summary["low_confidence"] == 1
        assert summary["no_reference_text"] == 1
        assert summary["bbox_rejected"] == 1

    def test_empty_tables(self) -> None:
        from vigilance.extraction.docling_processor import _compute_vision_quality_summary

        summary = _compute_vision_quality_summary([])
        assert summary["total_tables"] == 0
        assert summary["ok"] == 0

    def test_all_ok(self) -> None:
        from vigilance.extraction.docling_processor import _compute_vision_quality_summary

        tables = [
            self._make_table({
                "vision_status": "ok",
                "vision_extraction_attempted": True,
                "vision_extraction_confidence": 0.95,
                "has_reference_text": True,
            })
            for _ in range(5)
        ]
        summary = _compute_vision_quality_summary(tables)
        assert summary["total_tables"] == 5
        assert summary["ok"] == 5
        assert summary["failed"] == 0
        assert summary["truncated"] == 0
        assert summary["low_confidence"] == 0
        assert summary["no_reference_text"] == 0
