"""Tests for the rename validator (GenAI post-matching filter)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vigilance.genai.rename_validator import (
    RenameValidator,
    validate_rename_pairs,
)


class TestRenameValidator:
    """Unit tests with mocked API."""

    @patch("vigilance.genai.rename_validator.RenameValidator._ensure_client")
    def test_validate_batch_same_concept_accepted(
        self, mock_client: MagicMock
    ) -> None:
        """Pairs with same_concept=True and high confidence are accepted."""
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"results": [{"same_concept": true, "confidence": 0.95}]}'
                )
            )
        ]
        mock_client.return_value.chat.completions.create.return_value = mock_resp

        validator = RenameValidator(api_key="test")
        validator._client = mock_client.return_value

        results = validator.validate_batch([("Ratio CET1", "Ratio CET1 (phase-in)")])
        assert len(results) == 1
        assert results[0]["same_concept"] is True
        assert results[0]["confidence"] == 0.95
        assert validator.stats["accepted"] == 1
        assert validator.stats["rejected"] == 0

    @patch("vigilance.genai.rename_validator.RenameValidator._ensure_client")
    def test_validate_batch_different_concept_rejected(
        self, mock_client: MagicMock
    ) -> None:
        """Pairs with same_concept=False are rejected."""
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"results": [{"same_concept": false, "confidence": 0.92}]}'
                )
            )
        ]
        mock_client.return_value.chat.completions.create.return_value = mock_resp

        validator = RenameValidator(api_key="test")
        validator._client = mock_client.return_value

        results = validator.validate_batch([
            ("Depots personnels", "Prets hypothecaires"),
        ])
        assert len(results) == 1
        assert results[0]["same_concept"] is False
        assert validator.stats["rejected"] == 1
        assert validator.stats["accepted"] == 0

    @patch("vigilance.genai.rename_validator.RenameValidator._ensure_client")
    def test_validate_batch_api_error_conservative(
        self, mock_client: MagicMock
    ) -> None:
        """On API error, return same_concept=True (keep renames, conservative)."""
        mock_client.return_value.chat.completions.create.side_effect = Exception(
            "API unavailable"
        )

        validator = RenameValidator(api_key="test")
        validator._client = mock_client.return_value

        results = validator.validate_batch([("A", "B")])
        assert len(results) == 1
        assert results[0]["same_concept"] is True
        assert results[0]["confidence"] == 0.0
        assert validator.stats["errors"] == 1


class TestValidateRenamePairs:
    """Integration of validate_rename_pairs."""

    @patch("vigilance.genai.rename_validator.RenameValidator.validate_batch")
    def test_validate_rename_pairs_filters_low_confidence(
        self, mock_validate: MagicMock
    ) -> None:
        """Pairs with same_concept=True but confidence < min are rejected."""
        mock_validate.return_value = [
            {"same_concept": True, "confidence": 0.5},
        ]
        accepted, rejected, stats = validate_rename_pairs(
            [("A", "B")],
            api_key="test",
            batch_size=10,
            confidence_min=0.8,
        )
        assert accepted == []
        assert rejected == [("A", "B")]
        assert stats.get("rejected", 0) >= 0

    @patch("vigilance.genai.rename_validator.RenameValidator.validate_batch")
    def test_validate_rename_pairs_accepts_high_confidence(
        self, mock_validate: MagicMock
    ) -> None:
        """Pairs with same_concept=True and confidence >= min are accepted."""
        mock_validate.return_value = [
            {"same_concept": True, "confidence": 0.9},
        ]
        accepted, rejected, stats = validate_rename_pairs(
            [("Ratio CET1", "Ratio CET1 (phase-in)")],
            api_key="test",
            batch_size=10,
            confidence_min=0.8,
        )
        assert accepted == [("Ratio CET1", "Ratio CET1 (phase-in)")]
        assert rejected == []

    @patch("vigilance.genai.rename_validator.RenameValidator.validate_batch")
    def test_validate_rename_pairs_empty_input(
        self, mock_validate: MagicMock
    ) -> None:
        """Empty input returns empty lists."""
        accepted, rejected, stats = validate_rename_pairs(
            [],
            api_key="test",
        )
        assert accepted == []
        assert rejected == []
        assert stats["calls"] == 0
        mock_validate.assert_not_called()
