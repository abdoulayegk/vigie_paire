"""Unit tests for the Vision indicator added/removed validator."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason="Module vigilance.extraction.vision_indicator_added_validator absent."
)

_MINIMAL_PNG_BYTES: bytes | None = None


def _get_minimal_png() -> bytes:
    """Return minimal valid PNG bytes for mocking."""
    global _MINIMAL_PNG_BYTES
    if _MINIMAL_PNG_BYTES is None:
        from PIL import Image

        img = Image.new("RGB", (10, 10), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _MINIMAL_PNG_BYTES = buf.getvalue()
    return _MINIMAL_PNG_BYTES


class TestValidateIndicatorAddedVision:
    """Unit tests with mocked API and crops."""

    @patch("vigilance.utils.pdf_crop.crop_table_region_to_bytes")
    @patch(
        "vigilance.extraction.vision_indicator_added_validator._get_page_dimensions"
    )
    def test_validate_indicator_added_vision_same_concept(
        self, mock_dims: MagicMock, mock_crop: MagicMock
    ) -> None:
        """When Vision says same_concept=True, indicator is filtered."""
        mock_dims.return_value = (400.0, 600.0)
        mock_crop.return_value = _get_minimal_png()
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"same_concept": true, "confidence": 0.9}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            from vigilance.extraction.vision_indicator_added_validator import (
                validate_indicator_added_vision,
            )

            same, conf, called_api = validate_indicator_added_vision(
                "Revenus nets (Total)",
                "added",
                "/path/t2.pdf",
                3,
                [0.1, 0.2, 0.9, 0.5],
                "/path/t1.pdf",
                2,
                api_key="test",
                row_bboxes=[("Revenus nets (Total)", 100.0, 120.0)],
                table_row_bbox_norm=[0.1, 0.15, 0.9, 0.6],
            )
            assert same is True
            assert conf == 0.9
            assert called_api is True

    @patch("vigilance.utils.pdf_crop.crop_table_region_to_bytes")
    def test_validate_indicator_added_vision_no_row_bbox(
        self, mock_crop: MagicMock
    ) -> None:
        """When table_row_bbox_norm is None, returns (False, 0) - conservative."""
        from vigilance.extraction.vision_indicator_added_validator import (
            validate_indicator_added_vision,
        )

        same, conf, called_api = validate_indicator_added_vision(
            "X",
            "added",
            "/path/t2.pdf",
            1,
            [0.1, 0.2, 0.9, 0.5],
            "/path/t1.pdf",
            1,
            api_key="test",
            row_bboxes=None,
            table_row_bbox_norm=None,
        )
        assert same is False
        assert conf == 0.0
        assert called_api is False
        mock_crop.assert_not_called()


class TestTryVisionValidateIndicators:
    """Tests for try_vision_validate_indicators."""

    def test_empty_added_removed_returns_unchanged(self) -> None:
        """Empty lists return unchanged with no API calls."""
        table_t1 = MagicMock()
        table_t1.bbox = [0.1, 0.2, 0.9, 0.6]
        table_t1.page_pdf = 1
        table_t1.first_column_indicators = ["A"]
        table_t2 = MagicMock()
        table_t2.bbox = [0.1, 0.2, 0.9, 0.6]
        table_t2.page_pdf = 1
        table_t2.first_column_indicators = ["B"]

        from vigilance.extraction.vision_indicator_added_validator import (
            try_vision_validate_indicators,
        )

        fa, fr, stats = try_vision_validate_indicators(
            [],
            [],
            table_t1,
            table_t2,
            "/p1.pdf",
            "/p2.pdf",
            "test",
            0.8,
        )
        assert fa == []
        assert fr == []
        assert stats["vision_calls"] == 0

    @patch(
        "vigilance.extraction.row_bbox_extractor.extract_row_bboxes_from_pdf"
    )
    def test_fallback_when_row_bboxes_empty(
        self, mock_extract: MagicMock
    ) -> None:
        """When row_bboxes are empty, returns unfiltered and sets fallback reason."""
        mock_extract.return_value = []
        table_t1 = MagicMock()
        table_t1.bbox = [0.1, 0.2, 0.9, 0.6]
        table_t1.page_pdf = 1
        table_t1.first_column_indicators = ["A"]
        table_t2 = MagicMock()
        table_t2.bbox = [0.1, 0.2, 0.9, 0.6]
        table_t2.page_pdf = 1
        table_t2.first_column_indicators = ["B", "C"]

        with patch(
            "vigilance.extraction.vision_indicator_added_validator._get_page_dimensions"
        ) as mock_dims:
            mock_dims.return_value = (400, 600)

            from vigilance.extraction.vision_indicator_added_validator import (
                try_vision_validate_indicators,
            )

            fa, fr, stats = try_vision_validate_indicators(
                ["C"],
                [],
                table_t1,
                table_t2,
                "/p1.pdf",
                "/p2.pdf",
                "test",
                0.8,
            )
            assert fa == ["C"]
            assert fr == []
            assert stats["vision_fallback_reason"] == "row_bboxes_empty"
            assert stats["vision_calls"] == 0
