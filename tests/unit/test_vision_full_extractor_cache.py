"""Tests for Vision full extractor cache: full payload roundtrip and bottom_extension in key.

Ensures cache stores and returns table_title, headers, rows, footnotes_content,
and that different bottom_extension produces different cache keys (no stale reuse).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vigilance.extraction.vision_cache import (
    cache_get,
    cache_put,
    make_cache_key,
)
from vigilance.extraction.vision_full_extractor import VisionFullExtractor, VisionFullResult


def test_cache_key_includes_bottom_extension() -> None:
    """Different bottom_extension must produce different cache keys."""
    pdf_sha = "abc123"
    page = 1
    bbox = [0.1, 0.2, 0.8, 0.5]
    key1 = make_cache_key(pdf_sha, page, bbox)
    bbox_ext = [0.1, 0.2, 0.8, min(1.0, 0.5 + 0.12)]
    key2 = make_cache_key(pdf_sha, page, bbox_ext)
    assert key1 != key2
    assert "0.5" in key1
    assert "0.62" in key2 or "0.6199" in key2


def test_full_payload_roundtrip_restores_all_fields() -> None:
    """Cache put then get must restore table_title, headers, rows, footnotes_content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key = make_cache_key("sha_1", 2, [0.1, 0.2, 0.9, 0.6])
        payload = {
            "table_title": "Tableau 5 - Ratios",
            "headers": ["Période", "T1", "T2"],
            "indicators": ["Ratio CET1", "Ratio Tier 1"],
            "rows": [
                ["Ratio CET1", "13.1%", "13.3%"],
                ["Ratio Tier 1", "14.5%", "14.8%"],
            ],
            "footnotes_content": [
                {"marker": "1", "text": "Note provisoire"},
                {"marker": "2", "text": "Autre note"},
            ],
            "footnote_markers": ["1", "2"],
            "confidence": 0.95,
            "appears_truncated": False,
            "estimated_content_height": 80,
            "vision_status": "ok",
            "warnings": [],
        }
        cache_put(tmpdir, key, payload)
        loaded = cache_get(tmpdir, key)
        assert loaded is not None
        assert loaded.get("table_title") == "Tableau 5 - Ratios"
        assert loaded.get("headers") == ["Période", "T1", "T2"]
        assert loaded.get("rows") == payload["rows"]
        assert loaded.get("footnotes_content") == payload["footnotes_content"]
        assert loaded.get("footnote_markers") == ["1", "2"]
        assert loaded.get("indicators") == payload["indicators"]
        assert loaded.get("confidence") == 0.95
        assert loaded.get("vision_status") == "ok"


def test_extractor_cache_hit_returns_vision_full_result_with_all_fields(
    monkeypatch,
) -> None:
    """When cache hits, VisionFullExtractor.extract returns VisionFullResult with full content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("VISION_CACHE_DIR", tmpdir)
        pdf_sha = "test_sha_456"
        page_number = 3
        bbox_norm = [0.05, 0.15, 0.95, 0.55]
        bbox_with_ext = list(bbox_norm)
        bbox_with_ext[3] = min(1.0, bbox_norm[3] + 0.12)
        key = make_cache_key(pdf_sha, page_number, bbox_with_ext)
        cache_put(
            tmpdir,
            key,
            {
                "table_title": "Tableau 10 - Expositions",
                "headers": ["Catégorie", "Montant"],
                "indicators": ["Détail", "Total"],
                "rows": [["Détail", "100"], ["Total", "100"]],
                "footnotes_content": [{"marker": "(1)", "text": "En millions."}],
                "footnote_markers": ["(1)"],
                "confidence": 0.88,
                "appears_truncated": False,
                "estimated_content_height": None,
                "vision_status": "ok",
                "warnings": [],
            },
        )
        extractor = VisionFullExtractor(api_key="key", use_cache=True)

        result = extractor.extract(
            crop_bytes=b"fake",
            bank_code="bnc",
            pdf_sha=pdf_sha,
            page_number=page_number,
            bbox_norm=bbox_norm,
            bottom_extension_used=0.12,
        )
        assert result is not None
        assert result.table_title == "Tableau 10 - Expositions"
        assert result.headers == ["Catégorie", "Montant"]
        assert result.rows == [["Détail", "100"], ["Total", "100"]]
        assert result.footnotes_content == [{"marker": "(1)", "text": "En millions."}]
        assert result.footnote_markers == ["(1)"]
        assert result.confidence == 0.88
        assert result.vision_status == "ok"


def test_extractor_cache_miss_does_not_use_stale_from_different_extension(
    monkeypatch,
) -> None:
    """Cache key includes bottom_extension; different extension must not hit wrong entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("VISION_CACHE_DIR", tmpdir)
        pdf_sha = "sha_789"
        page_number = 1
        bbox = [0.0, 0.0, 1.0, 0.4]
        key_ext_0 = make_cache_key(pdf_sha, page_number, bbox)
        bbox_ext_012 = [0.0, 0.0, 1.0, min(1.0, 0.4 + 0.12)]
        key_ext_012 = make_cache_key(pdf_sha, page_number, bbox_ext_012)
        cache_put(
            tmpdir,
            key_ext_0,
            {
                "table_title": "Old",
                "headers": [],
                "indicators": ["Only one"],
                "rows": [],
                "footnotes_content": [],
                "footnote_markers": [],
                "confidence": 0.7,
                "vision_status": "ok",
                "warnings": [],
            },
        )
        cached_for_ext_012 = cache_get(tmpdir, key_ext_012)
        assert cached_for_ext_012 is None
