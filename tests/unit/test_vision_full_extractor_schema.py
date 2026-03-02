"""Unit tests for Vision full extractor schema validation."""

from __future__ import annotations

from vigilance.extraction.vision_full_extractor import (
    _build_openai_json_schema,
    _parse_vision_result,
)


def test_parse_vision_result_valid_with_defaults() -> None:
    raw = {
        "indicators": [" Ratio CET1 ", "", "Total"],
        "footnotes_content": {"1": " Note A ", "2": "Note B"},
        "footnote_markers": ["1", " 2 "],
        "confidence": 0.91,
    }
    parsed = _parse_vision_result(raw)
    assert parsed is not None
    assert parsed.indicators == ["Ratio CET1", "Total"]
    assert parsed.footnotes_content == {"1": "Note A", "2": "Note B"}
    assert parsed.footnote_markers == ["1", "2"]
    assert parsed.appears_truncated is False
    assert parsed.estimated_content_height is None


def test_parse_vision_result_valid_full_payload() -> None:
    raw = {
        "indicators": ["A"],
        "footnotes_content": {"(1)": "Texte"},
        "footnote_markers": ["(1)"],
        "confidence": 0.8,
        "appears_truncated": True,
        "estimated_content_height": 72,
    }
    parsed = _parse_vision_result(raw)
    assert parsed is not None
    assert parsed.appears_truncated is True
    assert parsed.estimated_content_height == 72


def test_parse_vision_result_rejects_missing_required() -> None:
    raw = {
        "indicators": ["A"],
        "footnotes_content": {},
        # missing footnote_markers + confidence
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_wrong_types() -> None:
    raw = {
        "indicators": "not-a-list",
        "footnotes_content": [],
        "footnote_markers": {},
        "confidence": "high",
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_height_out_of_range() -> None:
    raw = {
        "indicators": ["A"],
        "footnotes_content": {},
        "footnote_markers": [],
        "confidence": 0.5,
        "estimated_content_height": 150,
    }
    assert _parse_vision_result(raw) is None


def test_openai_json_schema_contains_strict_contract() -> None:
    schema = _build_openai_json_schema()
    assert schema["type"] == "json_schema"
    block = schema["json_schema"]
    assert block["strict"] is True
    s = block["schema"]
    assert s["additionalProperties"] is False
    required = set(s["required"])
    assert {"indicators", "footnotes_content", "footnote_markers", "confidence"} <= required
