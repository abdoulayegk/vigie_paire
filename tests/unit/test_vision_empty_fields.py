"""Step 2+3 contract test: when Vision returns empty title/headers/rows, ExtractedTable
has empty values — never backfilled from Docling.

Rule 5: Vision failures result in empty content, not Docling content bleed-through.
"""

from __future__ import annotations

from vigie.extraction.vision_full import (
    _parse_vision_result,
)


def test_parse_returns_empty_table_title_when_not_provided() -> None:
    """When Vision doesn't include a title, table_title must be empty string."""
    raw = {
        "table_summary": "Capital réglementaire",
        "indicators": ["Ratio CET1"],
        "footnotes_content": [],
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.table_title == "", f"Expected empty title, got {result.table_title!r}"


def test_parse_returns_empty_headers_when_missing() -> None:
    """When Vision doesn't include headers, headers must be empty list."""
    raw = {
        "table_summary": "Indicateur A",
        "indicators": ["Indicateur A"],
        "footnotes_content": [],
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.headers == [], f"Expected empty headers, got {result.headers!r}"


def test_parse_returns_summary_when_provided() -> None:
    """The minimal schema keeps GPT-provided table_summary."""
    raw = {
        "table_summary": "Risque de crédit",
        "indicators": ["Indicateur A"],
        "footnotes_content": [],
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.table_summary == "Risque de crédit"


def test_vision_failed_result_has_no_docling_content() -> None:
    """A failed Vision result (None) must not be backfilled — caller must keep content empty."""
    raw: dict[str, object] = {
        # Missing required 'indicators' and 'table_summary' fields — should fail validation
        "footnotes_content": [],
    }
    result = _parse_vision_result(raw)
    # When Vision parse fails, result is None (no content, no backfill)
    assert result is None, "Expected None for invalid Vision response"


def test_vision_status_ok_when_parse_succeeds() -> None:
    """vision_status must be 'ok' on successful parse."""
    raw = {
        "table_summary": "Capital",
        "indicators": ["Total"],
        "footnotes_content": [],
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.vision_status == "ok"
    assert result.warnings == []
