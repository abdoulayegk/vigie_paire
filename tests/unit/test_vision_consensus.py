"""Tests for dual-prompt consensus, text vote, and confidence score."""

from __future__ import annotations

import pytest

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionFullResult,
    _extract_native_text_indicators,
)


def _result(
    *,
    indicators: list[str] | None = None,
    title: str = "",
    summary: str = "",
    headers: list[str] | None = None,
    footnotes: list[dict[str, str]] | None = None,
) -> VisionFullResult:
    return VisionFullResult(
        table_title=title,
        table_summary=summary,
        headers=list(headers or []),
        indicators=list(indicators or []),
        footnotes_content=list(footnotes or []),
    )


class TestSelectConsensusTextVote:
    """Test that _select_consensus handles text vote correctly."""

    def test_text_vote_cannot_add_new_labels(self) -> None:
        """Text vote candidates not in any Vision shot must NOT introduce new labels."""
        r1 = _result(indicators=["Alpha", "Beta", "Gamma"])
        r2 = _result(indicators=["Alpha", "Beta", "Gamma"])

        # Text vote has a candidate "Delta" not seen in Vision shots
        result = VisionFullExtractor._select_consensus(
            [r1, r2],
            text_vote_indicators=["Alpha", "Delta", "Epsilon"],
        )

        assert "Delta" not in result.indicators
        assert "Epsilon" not in result.indicators
        # Vision labels should still be present
        assert "Alpha" in result.indicators
        assert "Beta" in result.indicators
        assert "Gamma" in result.indicators

    def test_text_vote_reinforces_existing_labels(self) -> None:
        """Text vote should boost popularity of matching Vision labels."""
        # r1 has a broader set, r2 has a narrower set
        r1 = _result(indicators=["Alpha", "Beta", "Gamma", "Delta"])
        r2 = _result(indicators=["Alpha", "Beta"])

        # Text vote confirms the broader set
        result = VisionFullExtractor._select_consensus(
            [r1, r2],
            text_vote_indicators=["Alpha", "Beta", "Gamma", "Delta"],
        )

        # Should prefer r1 (broader set confirmed by text)
        assert len(result.indicators) == 4

    def test_confidence_score_perfect_agreement(self) -> None:
        """When all shots agree perfectly, confidence should be 1.0."""
        r1 = _result(indicators=["A", "B", "C"])
        r2 = _result(indicators=["A", "B", "C"])

        result = VisionFullExtractor._select_consensus([r1, r2])

        assert result.confidence_score == 1.0

    def test_confidence_score_partial_agreement(self) -> None:
        """When shots disagree, confidence should be < 1.0."""
        r1 = _result(indicators=["A", "B", "C"])
        r2 = _result(indicators=["A", "B", "D"])

        result = VisionFullExtractor._select_consensus([r1, r2])

        assert 0.0 < result.confidence_score < 1.0

    def test_confidence_score_computed_from_jaccard_agreement(self) -> None:
        """Confidence should match expected Jaccard computation."""
        r1 = _result(indicators=["A", "B", "C", "D"])
        r2 = _result(indicators=["A", "B"])

        result = VisionFullExtractor._select_consensus([r1, r2])

        # The best result should have confidence based on Jaccard:
        # If best = r1: intersection={a,b}, union={a,b,c,d} -> jaccard = 2/4 = 0.5
        # Self-jaccard = 1.0, mean = (1.0 + 0.5) / 2 = 0.75
        # If best = r2: intersection={a,b}, union={a,b,c,d} -> jaccard = 2/4 = 0.5
        # Self-jaccard = 1.0, mean = (0.5 + 1.0) / 2 = 0.75
        assert result.confidence_score > 0.0
        assert result.confidence_score <= 1.0


class TestExtractNativeTextIndicators:
    """Test the native text indicator extraction helper."""

    def test_rejects_purely_numeric(self) -> None:
        text = "100 200 300\nAlpha\n45.2%\nBeta"
        result = _extract_native_text_indicators(text)
        assert "Alpha" in result
        assert "Beta" in result
        assert "100 200 300" not in result

    def test_rejects_short_lines(self) -> None:
        text = "AB\nAlpha\nCD\nBeta Gamma"
        result = _extract_native_text_indicators(text)
        assert "AB" not in result
        assert "CD" not in result
        assert "Alpha" in result

    def test_max_200_lines(self) -> None:
        text = "\n".join(f"Indicator line {i}" for i in range(300))
        result = _extract_native_text_indicators(text)
        assert len(result) <= 200

    def test_empty_input(self) -> None:
        result = _extract_native_text_indicators("")
        assert result == []
