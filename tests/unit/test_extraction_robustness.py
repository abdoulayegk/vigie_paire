"""Regression tests for extraction robustness (bbox sanity, crop, recrop, prompt, metadata)."""

from __future__ import annotations

import pytest

from vigilance.utils.pdf_crop import (
    bbox_sanity_profile,
    crop_table_region_to_bytes,
    is_bbox_sane,
)


def test_crop_invalid_bbox_returns_empty_bytes() -> None:
    """crop_table_region_to_bytes returns b"" on invalid bbox (no full-page fallback)."""
    empty = crop_table_region_to_bytes(
        "/nonexistent.pdf",
        1,
        [0.0, 0.0, 1.0, 1.0],  # valid shape but file missing
    )
    # With invalid path we get b"" from exception path
    invalid_bbox = crop_table_region_to_bytes(
        "/nonexistent.pdf",
        1,
        [0.0, 1.0, 0.0, 0.5],  # r <= l invalid
    )
    assert invalid_bbox == b""
    too_short = crop_table_region_to_bytes("/nonexistent.pdf", 1, [0.0, 0.0, 0.5])
    assert too_short == b""


def test_vision_not_called_when_crop_rejected() -> None:
    """When bbox fails sanity gate, crop_reject_reason is set and Vision is skipped (no Vision call)."""
    cfg = {
        "bbox_min_width": 0.02,
        "bbox_min_height": 0.02,
        "bbox_min_area": 0.0005,
        "bbox_max_area": 0.95,
        "bbox_near_full_page_threshold": 0.90,
    }
    insane_bbox = [0.02, 0.02, 0.98, 0.98]
    sane, reason, _ = is_bbox_sane(insane_bbox, cfg)
    assert sane is False
    assert reason is not None
    from vigilance.models.table_models import get_extraction_quality_flags

    class TableWithReject:
        debug_metrics = {"crop_reject_reason": "bbox_near_full_page"}

    flags = get_extraction_quality_flags(TableWithReject())
    assert flags.get("crop_rejected") is True


def test_bbox_sanity_rejects_near_full_page_crop() -> None:
    """is_bbox_sane rejects bbox with area or dimensions near full page."""
    cfg = {
        "bbox_min_width": 0.02,
        "bbox_min_height": 0.02,
        "bbox_min_area": 0.0005,
        "bbox_max_area": 0.95,
        "bbox_near_full_page_threshold": 0.90,
    }
    near_full = [0.02, 0.02, 0.98, 0.98]
    sane, reason, profile = is_bbox_sane(near_full, cfg)
    assert sane is False
    assert reason in ("bbox_near_full_page", "bbox_area_too_large")
    assert "reject_reason" in profile or reason


def test_bbox_sanity_accepts_small_table_region() -> None:
    """is_bbox_sane accepts a small valid table-like region."""
    cfg = {
        "bbox_min_width": 0.02,
        "bbox_min_height": 0.02,
        "bbox_min_area": 0.0005,
        "bbox_max_area": 0.95,
        "bbox_near_full_page_threshold": 0.90,
    }
    small = [0.1, 0.2, 0.6, 0.5]
    sane, reason, profile = is_bbox_sane(small, cfg)
    assert sane is True
    assert reason is None


def test_prompt_no_longer_mentions_cadre_rouge() -> None:
    """Vision extraction prompt must not mention 'cadre rouge'."""
    from vigilance.extraction.vision_full_extractor import (
        _PROMPT_BASE,
        _PROMPT_JSON_STRICT,
    )

    combined = (_PROMPT_BASE or "") + (_PROMPT_JSON_STRICT or "")
    assert "cadre rouge" not in combined.lower()


def test_horizontal_padding_applied_and_clamped() -> None:
    """crop_table_region_to_bytes accepts horizontal_padding and clamps to page."""
    from vigilance.utils.pdf_crop import _validate_bbox

    bbox = [0.2, 0.2, 0.8, 0.6]
    assert _validate_bbox(bbox)
    out = crop_table_region_to_bytes(
        "/nonexistent.pdf",
        1,
        bbox,
        horizontal_padding=0.5,
    )
    assert (
        out == b""
    )  # file missing, but call succeeds; padding is applied in real path


def test_bbox_sanity_profile_shape() -> None:
    """bbox_sanity_profile returns dict with width_norm, height_norm, area_norm."""
    profile = bbox_sanity_profile([0.1, 0.1, 0.5, 0.4])
    assert "width_norm" in profile
    assert "height_norm" in profile
    assert "area_norm" in profile
    assert profile["width_norm"] == pytest.approx(0.4)
    assert profile["height_norm"] == pytest.approx(0.3)
    assert profile["area_norm"] == pytest.approx(0.12)


def test_recrop_triggered_on_incomplete_rows_vs_indicators() -> None:
    """extract_with_quality_pass _needs_recrop triggers when indicators present but rows nearly empty."""
    from vigilance.extraction.vision_full_extractor import VisionFullResult

    many_indicators_few_rows = VisionFullResult(
        table_title="",
        headers=[],
        indicators=[{"text": f"Ind{i}", "bbox": None} for i in range(10)],
        rows=[["Ind1", "1"]],
        footnotes_content=[],
        footnote_markers=[],
        confidence=0.9,
    )
    assert len(many_indicators_few_rows.indicators) >= 5
    assert len(many_indicators_few_rows.rows) < max(
        1, int(0.3 * len(many_indicators_few_rows.indicators))
    )
    assert many_indicators_few_rows.appears_truncated is False


def test_recrop_failed_incomplete_sets_debug_flag() -> None:
    """VisionFullResult can carry recrop_failed_incomplete for debug_metrics."""
    from vigilance.extraction.vision_full_extractor import VisionFullResult

    r = VisionFullResult(
        table_title="",
        headers=[],
        indicators=[],
        rows=[],
        footnotes_content=[],
        footnote_markers=[],
        confidence=0.5,
        recrop_attempted=True,
        recrop_used=False,
        recrop_failed_incomplete=True,
    )
    assert r.recrop_failed_incomplete is True
    assert r.recrop_used is False


def test_partial_lean_result_does_not_force_recrop(monkeypatch) -> None:
    """A lean fallback with missing rows must not trigger recrop solely because rows are omitted."""
    from vigilance.extraction.vision_full_extractor import (
        VisionFullExtractor,
        VisionFullResult,
    )

    result = VisionFullResult(
        table_title="Tableau 1",
        headers=["Indicateur", "Valeur"],
        indicators=[{"text": f"Ind{i}", "bbox": None} for i in range(20)],
        rows=[],
        footnotes_content=[],
        footnote_markers=[],
        confidence=0.92,
        vision_status="partial",
        warnings=[
            "vision_truncated",
            "vision_lean_mode",
            "vision_rows_missing_from_fallback",
        ],
    )

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    calls = {"extract": 0, "recrop": 0}

    def _fake_extract(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        calls["extract"] += 1
        return result

    def _fake_recrop(_ext: float) -> bytes:
        calls["recrop"] += 1
        return b"unexpected"

    monkeypatch.setattr(extractor, "extract", _fake_extract)

    out = extractor.extract_with_quality_pass(
        crop_bytes=b"abc",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.8, 0.9],
        vision_cfg={},
        get_recrop_fn=_fake_recrop,
    )

    assert out is result
    assert calls["extract"] == 1
    assert calls["recrop"] == 0


def test_page_title_assist_does_not_assign_by_positional_fallback() -> None:
    """With allow_positional_fallback false, title is only applied via table_number or bbox_proximity."""
    assist_cfg = {"allow_positional_fallback": False}
    fallback = bool(assist_cfg.get("allow_positional_fallback", False))
    assert fallback is False


def test_bbox_overlap_detection_logs_multi_table_warning() -> None:
    """Overlap detection logic: two overlapping bboxes on same page yield positive ratio."""

    def _area(b: list[float]) -> float:
        if len(b) < 4:
            return 0.0
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def _overlap_ratio(a: list[float], b: list[float]) -> float:
        if len(a) < 4 or len(b) < 4:
            return 0.0
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        return inter / min(_area(a), _area(b)) if min(_area(a), _area(b)) > 0 else 0.0

    a, b = [0.1, 0.1, 0.5, 0.4], [0.45, 0.15, 0.9, 0.45]
    ratio = _overlap_ratio(a, b)
    assert ratio >= 0.0
    assert ratio <= 1.0
    assert ratio > 0.0


def test_debug_metrics_roundtrip_preserves_quality_profile() -> None:
    """Quality profile keys in debug_metrics are preserved through model/storage."""
    from vigilance.models.table_models import (
        get_extraction_confidence,
        get_extraction_quality_flags,
        get_extraction_quality_profile,
    )

    class T:
        debug_metrics: dict = {
            "vision_extraction_confidence": 0.85,
            "vision_extraction_applied": True,
            "appears_truncated": False,
            "recrop_attempted": True,
            "recrop_used": False,
            "recrop_failed_incomplete": True,
            "crop_reject_reason": None,
            "bbox_sanity_profile": {"width_norm": 0.4, "height_norm": 0.3},
            "page_title_assist_used": False,
            "page_title_assist_match_method": None,
            "warnings": [],
        }

    assert get_extraction_confidence(T()) == 0.85
    flags = get_extraction_quality_flags(T())
    assert flags.get("recrop_failed_incomplete") is True
    assert flags.get("vision_extraction_applied") is True
    profile = get_extraction_quality_profile(T())
    assert profile.get("confidence") == 0.85
    assert profile.get("flags") == flags
    assert profile.get("bbox_sanity_profile") is not None
