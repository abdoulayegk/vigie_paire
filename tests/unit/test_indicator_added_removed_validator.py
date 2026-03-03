"""Unit tests for the GenAI indicator added/removed validator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vigilance.genai.indicator_added_removed_validator import (
    IndicatorAddedRemovedValidator,
    validate_indicator_added_removed,
)


class TestIndicatorAddedRemovedValidator:
    """Unit tests with mocked API."""

    def test_validate_batch_added_filters_false_positives(self) -> None:
        """Indicators that exist in T1 are filtered from added."""
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"results": [{"same_concept": true, "confidence": 0.9}]}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            validator = IndicatorAddedRemovedValidator(api_key="test")
            results = validator._validate_batch(
                ["Revenus nets (Total)"],
                ["Revenus nets - Total", "Autre ligne"],
                "added",
            )
            assert len(results) == 1
            assert results[0]["same_concept"] is True
            assert results[0]["confidence"] == 0.9

    def test_validate_batch_added_keeps_true_additions(self) -> None:
        """Indicators that don't exist in T1 are kept."""
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"results": [{"same_concept": false, "confidence": 0.85}]}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            validator = IndicatorAddedRemovedValidator(api_key="test")
            results = validator._validate_batch(
                ["Nouveau ratio TLAC"],
                ["Ratio CET1", "Ratio Tier 1"],
                "added",
            )
            assert len(results) == 1
            assert results[0]["same_concept"] is False

    def test_validate_indicator_added_removed_filters(self) -> None:
        """validate_indicator_added_removed filters false positives."""
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"results": [{"same_concept": true, "confidence": 0.9}]}'
                )
            )
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            filtered_added, filtered_removed, stats = validate_indicator_added_removed(
                ["Revenus nets (Total)"],
                [],
                ["Revenus nets - Total"],
                ["Revenus nets (Total)"],
                api_key="test",
                batch_size=8,
                confidence_min=0.8,
            )
            assert filtered_added == []
            assert filtered_removed == []
            assert stats["filtered_added"] == 1

    def test_validate_indicator_added_removed_empty_input(self) -> None:
        """Empty added/removed returns unchanged with zero stats."""
        filtered_added, filtered_removed, stats = validate_indicator_added_removed(
            [],
            [],
            ["A"],
            ["B"],
            api_key="test",
        )
        assert filtered_added == []
        assert filtered_removed == []
        assert stats["calls"] == 0

    def test_circuit_breaker_keeps_all_on_failure(self) -> None:
        """On API error, returns same_concept=False (keep all - conservative)."""
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = (
                Exception("API error")
            )

            validator = IndicatorAddedRemovedValidator(
                api_key="test", circuit_breaker_threshold=1
            )
            results = validator._validate_batch(
                ["X", "Y"],
                ["A"],
                "added",
            )
            assert len(results) == 2
            assert all(r["same_concept"] is False for r in results)
            assert validator.circuit_open

    def test_batch_size_respected(self) -> None:
        """Batching is applied: 4 added with batch_size=2 = 2 API calls."""
        import json

        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            n = 2
            results = [{"same_concept": False, "confidence": 0.5}] * n
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content=json.dumps({"results": results}))
                    )
                ]
            )

        with patch("openai.OpenAI") as mock_openai:
            client = MagicMock()
            client.chat.completions.create = mock_create
            mock_openai.return_value = client

            filtered_added, _, stats = validate_indicator_added_removed(
                ["A", "B", "C", "D"],
                [],
                ["X"],
                ["Y"],
                api_key="test",
                batch_size=2,
                confidence_min=0.9,
            )
            assert len(filtered_added) == 4
            assert stats["calls"] == 2
            assert call_count[0] == 2
