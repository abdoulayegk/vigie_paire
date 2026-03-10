"""Step 2+3 contract test: when Vision returns empty title/headers/rows, ExtractedTable
has empty values — never backfilled from Docling.

Rule 5: Vision failures result in empty content, not Docling content bleed-through.
"""

from __future__ import annotations

from vigilance.extraction.vision_full_extractor import (
    _parse_vision_result,
)


def test_parse_returns_empty_table_title_when_not_provided() -> None:
    """When Vision doesn't include a title, table_title must be empty string."""
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "Ratio CET1", "bbox": None}],
        "footnotes_content": [],
        "footnote_markers": [],
        "confidence": 0.80,
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.table_title == "", f"Expected empty title, got {result.table_title!r}"


def test_parse_returns_empty_headers_when_missing() -> None:
    """When Vision doesn't include headers, headers must be empty list."""
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "Indicateur A", "bbox": None}],
        "footnotes_content": [],
        "footnote_markers": [],
        "confidence": 0.80,
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.headers == [], f"Expected empty headers, got {result.headers!r}"


def test_parse_returns_empty_rows_when_missing() -> None:
    """When Vision doesn't include rows, rows must be empty list."""
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "Indicateur A", "bbox": None}],
        "footnotes_content": [],
        "footnote_markers": [],
        "confidence": 0.80,
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.rows == [], f"Expected empty rows, got {result.rows!r}"


def test_vision_failed_result_has_no_docling_content() -> None:
    """A failed Vision result (None) must not be backfilled — caller must keep content empty."""
    raw = {
        # Missing required 'indicators' and 'confidence' fields — should fail validation
        "footnotes_content": [],
        "footnote_markers": [],
    }
    result = _parse_vision_result(raw)
    # When Vision parse fails, result is None (no content, no backfill)
    assert result is None, "Expected None for invalid Vision response"


def test_vision_status_ok_when_parse_succeeds() -> None:
    """vision_status must be 'ok' on successful parse."""
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "Total", "bbox": None}],
        "footnotes_content": [],
        "footnote_markers": [],
        "confidence": 0.95,
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.vision_status == "ok"
    assert result.warnings == []
