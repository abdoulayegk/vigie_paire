"""Tests for Phase-1 exact-ish highlighting (bbox normalization, indicator ordering, crop)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.comparison_runner import (
    _all_indicators_value_clean_ordered,
    _normalize_bbox_ltrb_norm,
)
from vigilance.models.table_models import TableArtifact


def test_normalize_bbox_list() -> None:
    """_normalize_bbox_ltrb_norm accepts list[4]."""
    result = _normalize_bbox_ltrb_norm([0.1, 0.2, 0.5, 0.8])
    assert result == [0.1, 0.2, 0.5, 0.8]


def test_normalize_bbox_x0y0x1y1() -> None:
    """_normalize_bbox_ltrb_norm accepts dict with x0/y0/x1/y1."""
    result = _normalize_bbox_ltrb_norm({"x0": 0.0, "y0": 0.1, "x1": 0.9, "y1": 0.95})
    assert result is not None
    assert result[0] == 0.0
    assert result[1] == 0.1
    assert result[2] == 0.9
    assert result[3] == 0.95


def test_normalize_bbox_ltrb() -> None:
    """_normalize_bbox_ltrb_norm accepts dict with l/t/r/b."""
    result = _normalize_bbox_ltrb_norm({"l": 0.05, "t": 0.15, "r": 0.85, "b": 0.9})
    assert result == [0.05, 0.15, 0.85, 0.9]


def test_normalize_bbox_xywh() -> None:
    """_normalize_bbox_ltrb_norm accepts dict with x/y/width/height."""
    result = _normalize_bbox_ltrb_norm({"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4})
    assert result is not None
    assert result[0] == 0.1
    assert result[1] == 0.2
    assert result[2] == pytest.approx(0.4)
    assert result[3] == pytest.approx(0.6)


def test_normalize_bbox_invalid_returns_none() -> None:
    """Invalid bbox shapes return None."""
    assert _normalize_bbox_ltrb_norm(None) is None
    assert _normalize_bbox_ltrb_norm([]) is None
    assert _normalize_bbox_ltrb_norm([0.1, 0.2]) is None
    assert _normalize_bbox_ltrb_norm({"a": 1}) is None


def test_normalize_bbox_outside_range_returns_none() -> None:
    """Bbox values far outside [0,1] return None."""
    assert _normalize_bbox_ltrb_norm([-0.2, 0.0, 0.5, 0.5]) is None
    assert _normalize_bbox_ltrb_norm([0.0, 0.0, 1.2, 0.5]) is None


def test_normalize_bbox_reversed_returns_none() -> None:
    """r<=l or b<=t returns None."""
    assert _normalize_bbox_ltrb_norm([0.5, 0.2, 0.3, 0.8]) is None
    assert _normalize_bbox_ltrb_norm([0.1, 0.8, 0.5, 0.2]) is None


def test_all_indicators_value_clean_ordered_skips_noise() -> None:
    """Skips lines with empty canonical key (dates, units, footnotes)."""
    table = TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="t1",
        title="Test",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators=["Actif A", "Au 30 avril 2025", "Bilan B"],
        first_column_indicators_raw=["Actif A", "Au 30 avril 2025", "Bilan B"],
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )
    result = _all_indicators_value_clean_ordered(table)
    assert any("actif" in r.lower() for r in result)
    assert any("bilan" in r.lower() for r in result)
    assert not any("30 avril" in r for r in result)


def test_all_indicators_value_clean_ordered_requires_raw_source() -> None:
    """Returns empty when raw Vision indicators are absent."""
    table = TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="t1",
        title="Test",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators_raw=[],
        first_column_indicators=["Indicator One", "Indicator Two"],
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )
    result = _all_indicators_value_clean_ordered(table)
    assert result == []


def test_crop_table_region_to_bytes_invalid_bbox_fallback() -> None:
    """Invalid bbox (r<=l) falls back to full page render."""
    from vigilance.utils.pdf_crop import crop_table_region_to_bytes

    with patch("vigilance.extraction.pdf_preview.render_pdf_page") as mock_render:
        mock_render.return_value = b"\x89PNG"
        result = crop_table_region_to_bytes("/tmp/dummy.pdf", 1, [0.5, 0.5, 0.3, 0.8])
        mock_render.assert_called_once_with("/tmp/dummy.pdf", 1, scale=1.5, format="png")
    assert result == b"\x89PNG"
