"""Tests for VisionFullExtractor.extract_with_quality_pass (second pass and scoring).

Covers: _needs_recrop logic, expected_markers driving recrop, first vs second pass
scoring, and that best result is returned without degrading quality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionFullResult,
)


def _make_result(
    confidence: float = 0.9,
    indicators: list[str] | None = None,
    footnote_markers: list[str] | None = None,
    appears_truncated: bool = False,
) -> VisionFullResult:
    return VisionFullResult(
        table_title="Tableau 1",
        headers=[],
        indicators=indicators if indicators is not None else ["A", "B"],
        rows=[],
        footnotes_content=[],
        footnote_markers=footnote_markers if footnote_markers is not None else [],
        confidence=confidence,
        appears_truncated=appears_truncated,
    )


def test_quality_pass_returns_first_when_good_no_recrop(monkeypatch: pytest.MonkeyPatch) -> None:
    """When first result has confidence >= 0.85 and indicators, no second pass."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.9, indicators=["X", "Y"])
    recrop_calls: list[float] = []

    def fake_recrop(ext: float) -> bytes:
        recrop_calls.append(ext)
        return b"recrop"

    with patch.object(extractor, "extract", return_value=first):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=fake_recrop,
        )
    assert out is first
    assert len(recrop_calls) == 0


def test_quality_pass_triggers_second_when_confidence_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """When first result has confidence < 0.85, second pass is run."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.80, indicators=["A"])
    second = _make_result(confidence=0.92, indicators=["A", "B"])

    def fake_recrop(ext: float) -> bytes:
        return b"recrop"

    with patch.object(extractor, "extract", side_effect=[first, second]):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            initial_bottom_extension=0.12,
            get_recrop_fn=fake_recrop,
        )
    assert out is second
    assert out.confidence == 0.92


def test_quality_pass_triggers_second_when_indicators_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """When first result has no indicators, second pass is run and returns non-empty indicators."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.90, indicators=[])
    second = _make_result(confidence=0.92, indicators=["Only one"])
    mock_extract = MagicMock(side_effect=[first, second])

    with patch.object(extractor, "extract", mock_extract):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=lambda ext: b"recrop",
        )
    assert mock_extract.call_count == 2
    assert out is not None
    assert len(out.indicators) >= 1
    assert out.indicators[0] == "Only one"


def test_quality_pass_expected_markers_drives_recrop(monkeypatch: pytest.MonkeyPatch) -> None:
    """When expected_markers is set and first has no overlap, second pass is run."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(
        confidence=0.90,
        indicators=["A"],
        footnote_markers=["*", "dagger"],
    )
    second = _make_result(
        confidence=0.90,
        indicators=["A"],
        footnote_markers=["(1)", "(2)"],
    )

    with patch.object(extractor, "extract", side_effect=[first, second]):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            vision_cfg={"expected_markers": ["(1)", "(2)", "(3)"]},
            get_recrop_fn=lambda ext: b"recrop",
        )
    assert out is second
    assert set(out.footnote_markers) == {"(1)", "(2)"}


def test_quality_pass_returns_first_when_second_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When second extract returns None, first is returned."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.80, indicators=["A"])

    with patch.object(extractor, "extract", side_effect=[first, None]):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=lambda ext: b"recrop",
        )
    assert out is first


def test_quality_pass_returns_first_when_recrop_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """When get_recrop_fn returns empty bytes, first is returned and no second extract."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.80, indicators=["A"])

    with patch.object(extractor, "extract", return_value=first):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=lambda ext: b"",
        )
    assert out is first


def test_quality_pass_scoring_prefers_higher_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both passes succeed, the one with higher score (confidence + markers) is returned."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    # First triggers recrop (confidence just below threshold) so both passes run
    first = _make_result(confidence=0.84, indicators=["A"] * 5, footnote_markers=["(1)"])
    second = _make_result(confidence=0.90, indicators=["A"] * 5, footnote_markers=["(1)", "(2)"])

    with patch.object(extractor, "extract", side_effect=[first, second]):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            vision_cfg={"expected_markers": ["(1)", "(2)", "(3)"]},
            get_recrop_fn=lambda ext: b"recrop",
        )
    assert out is second


def test_quality_pass_no_recrop_when_get_recrop_fn_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When get_recrop_fn is None, first (even weak) is returned without second pass."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.80, indicators=["A"])

    with patch.object(extractor, "extract", return_value=first):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=None,
        )
    assert out is first


def test_quality_pass_appears_truncated_triggers_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """When first has appears_truncated=True, second pass runs; higher-scoring result returned."""
    extractor = VisionFullExtractor(api_key="test", use_cache=False)
    first = _make_result(confidence=0.92, indicators=["A"], appears_truncated=True)
    second = _make_result(confidence=0.94, indicators=["A"], appears_truncated=False)

    with patch.object(extractor, "extract", side_effect=[first, second]):
        out = extractor.extract_with_quality_pass(
            crop_bytes=b"img",
            bank_code="bnc",
            get_recrop_fn=lambda ext: b"recrop",
        )
    assert out is second
    assert out.appears_truncated is False
