"""Tests for T-1 anchoring (vision_t1_anchor module)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from vigie.extraction.vision_t1_anchor import (
    anchor_against_previous,
)


class TestAnchorBelowThreshold:
    """When row count difference is below threshold, no GPT call should be made."""

    def test_no_gpt_call_below_threshold(self) -> None:
        """Difference of 10% (< 20% default) should skip without calling GPT."""
        prev = [f"ind_{i}" for i in range(10)]
        curr = [f"ind_{i}" for i in range(11)]  # 10% diff

        result = anchor_against_previous(
            table_id="tbl_test",
            table_title="Test Table",
            current_indicators=curr,
            previous_indicators=prev,
            api_key="test-key",
        )

        assert result.skipped is True
        assert result.skip_reason == "below_threshold"

    def test_exact_same_count_skips(self) -> None:
        prev = [f"ind_{i}" for i in range(10)]
        curr = [f"ind_{i}" for i in range(10)]

        result = anchor_against_previous(
            table_id="tbl_test",
            table_title="Test Table",
            current_indicators=curr,
            previous_indicators=prev,
            api_key="test-key",
        )

        assert result.skipped is True
        assert result.diff_ratio == 0.0

    def test_both_empty_skips(self) -> None:
        result = anchor_against_previous(
            table_id="tbl_test",
            table_title="Test Table",
            current_indicators=[],
            previous_indicators=[],
            api_key="test-key",
        )

        assert result.skipped is True
        assert result.skip_reason == "both_empty"

    def test_no_previous_skips(self) -> None:
        result = anchor_against_previous(
            table_id="tbl_test",
            table_title="Test Table",
            current_indicators=["A", "B"],
            previous_indicators=[],
            api_key="test-key",
        )

        assert result.skipped is True
        assert result.skip_reason == "no_previous_indicators"


class TestAnchorAboveThreshold:
    """When row count difference exceeds threshold, GPT should be called."""

    def test_flags_likely_extraction_error(self) -> None:
        """Large difference should trigger GPT call and potentially flag error."""
        prev = [f"ind_{i}" for i in range(10)]
        curr = [f"ind_{i}" for i in range(15)]  # 33% diff

        mock_parsed = MagicMock()
        mock_parsed.likely_extraction_error = True
        mock_parsed.explanation = "Missing section headers suggest extraction error"

        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = anchor_against_previous(
                table_id="tbl_test",
                table_title="Test Table",
                current_indicators=curr,
                previous_indicators=prev,
                api_key="test-key",
            )

        assert result.skipped is False
        assert result.likely_extraction_error is True
        assert result.current_count == 15
        assert result.previous_count == 10
        assert result.diff_ratio > 0.20

    def test_graceful_failure_on_exception(self) -> None:
        """Any exception during GPT call should return skipped result."""
        prev = [f"ind_{i}" for i in range(10)]
        curr = [f"ind_{i}" for i in range(20)]  # 50% diff

        with patch("openai.OpenAI", side_effect=RuntimeError("API down")):
            result = anchor_against_previous(
                table_id="tbl_test",
                table_title="Test Table",
                current_indicators=curr,
                previous_indicators=prev,
                api_key="test-key",
            )

        assert result.skipped is True
        assert "exception" in result.skip_reason

    def test_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        prev = [f"ind_{i}" for i in range(10)]
        curr = [f"ind_{i}" for i in range(11)]  # 10% diff

        # With 5% threshold, this should trigger GPT (10% > 5%)
        # We patch GPT to fail, so we get a skipped result with exception, not below_threshold
        with patch("openai.OpenAI", side_effect=RuntimeError("mock")):
            result = anchor_against_previous(
                table_id="tbl_test",
                table_title="Test Table",
                current_indicators=curr,
                previous_indicators=prev,
                diff_threshold=0.05,
                api_key="test-key",
            )

        # With 0.05 threshold, 10% diff should NOT be skipped due to below_threshold
        assert result.skip_reason != "below_threshold"
