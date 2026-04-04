"""Tests for the Vision added table validator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason="Module vigilance.extraction.vision_added_table_validator absent."
)


class TestValidateAddedTable:
    """Unit tests with mocked API."""

    @patch("vigilance.utils.pdf_crop.crop_table_region_to_bytes")
    def test_validate_added_table_real_new_accepted(
        self, mock_crop: MagicMock
    ) -> None:
        """Tables with is_real_new=True are accepted."""
        mock_crop.return_value = b"fake_png"
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_real_new": true, "confidence": 0.92}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            from vigilance.extraction.vision_added_table_validator import (
                validate_added_table,
            )

            is_real, conf = validate_added_table(
                "/path/to/t2.pdf",
                5,
                [0.1, 0.2, 0.9, 0.8],
                api_key="test",
            )
            assert is_real is True
            assert conf == 0.92

    @patch("vigilance.utils.pdf_crop.crop_table_region_to_bytes")
    def test_validate_added_table_duplicate_rejected(
        self, mock_crop: MagicMock
    ) -> None:
        """Tables with is_real_new=False are rejected."""
        mock_crop.return_value = b"fake_png"
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_real_new": false, "confidence": 0.88}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            from vigilance.extraction.vision_added_table_validator import (
                validate_added_table,
            )

            is_real, conf = validate_added_table(
                "/path/to/t2.pdf",
                3,
                [0.0, 0.1, 1.0, 0.5],
                api_key="test",
            )
            assert is_real is False
            assert conf == 0.88

    def test_validate_added_table_no_bbox_conservative(self) -> None:
        """No bbox returns (True, 0.0) - conservative."""
        from vigilance.extraction.vision_added_table_validator import (
            validate_added_table,
        )

        is_real, conf = validate_added_table(
            "/path/to.pdf",
            1,
            None,
            api_key="test",
        )
        assert is_real is True
        assert conf == 0.0
