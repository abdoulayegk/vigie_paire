"""Step 1 contract test: VisionFullResult.to_footnotes_list() preserves visual order.

Rule 3: footnotes MUST be returned in visual order (top to bottom), never sorted by marker.
"""

from __future__ import annotations

from vigilance.extraction.vision_full_extractor import (
    VisionFullResult,
    _parse_vision_result,
)


def test_footnotes_preserve_insertion_order_not_sorted() -> None:
    """Footnotes must preserve visual order (haut -> bas), not be sorted by marker key."""
    # Markers out of numerical order to verify they are NOT sorted
    raw = {
        "indicators": ["Ratio CET1"],
        "footnotes_content": [
            {"id": "3", "text": "Troisième note"},
            {"id": "1", "text": "Première note"},
            {"id": "2", "text": "Deuxième note"},
        ],
        "footnote_markers": ["3", "1", "2"],
        "confidence": 0.90,
    }
    result = _parse_vision_result(raw)
    assert result is not None

    footnotes = result.to_footnotes_list()
    assert len(footnotes) == 3, f"Expected 3 footnotes, got {len(footnotes)}"

    # Must preserve visual order (3, 1, 2) — NOT sorted (1, 2, 3)
    assert footnotes[0]["marker"] == "3", (
        f"Expected '3' first, got {footnotes[0]['marker']}"
    )
    assert footnotes[1]["marker"] == "1", (
        f"Expected '1' second, got {footnotes[1]['marker']}"
    )
    assert footnotes[2]["marker"] == "2", (
        f"Expected '2' third, got {footnotes[2]['marker']}"
    )


def test_footnotes_from_legacy_dict_preserve_insertion_order() -> None:
    """Legacy dict footnotes_content (migration shim) must also preserve insertion order."""
    raw = {
        "indicators": ["Indicateur A"],
        "footnotes_content": {
            "(3)": "Troisième note",
            "(1)": "Première note",
        },
        "footnote_markers": ["(3)", "(1)"],
        "confidence": 0.85,
    }
    result = _parse_vision_result(raw)
    assert result is not None

    footnotes = result.to_footnotes_list()
    assert len(footnotes) == 2

    # Python 3.7+ dicts preserve insertion order — must NOT be sorted
    markers = [fn["marker"] for fn in footnotes]
    assert markers == ["(3)", "(1)"], f"Expected [(3), (1)] order, got {markers}"


def test_to_footnotes_list_returns_copy() -> None:
    """to_footnotes_list() must return a copy (mutations don't affect the result)."""
    raw = {
        "indicators": ["X"],
        "footnotes_content": [{"id": "1", "text": "Note"}],
        "footnote_markers": ["1"],
        "confidence": 0.9,
    }
    result = _parse_vision_result(raw)
    assert result is not None

    list1 = result.to_footnotes_list()
    list2 = result.to_footnotes_list()
    assert list1 == list2
    list1.clear()
    assert len(result.to_footnotes_list()) == 1  # original not mutated


def test_vision_full_result_has_new_content_fields() -> None:
    """VisionFullResult must have table_title, headers, rows, vision_status, warnings."""
    raw = {
        "table_title": "Tableau 5 : Ratios de fonds propres",
        "headers": ["Période", "T1 2025", "T2 2025"],
        "indicators": ["Ratio CET1", "Ratio Tier 1"],
        "rows": [
            ["Ratio CET1", "13.1%", "13.3%"],
            ["Ratio Tier 1", "14.5%", "14.8%"],
        ],
        "footnotes_content": [{"id": "1", "text": "Note provisoire"}],
        "footnote_markers": ["1"],
        "confidence": 0.95,
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.table_title == "Tableau 5 : Ratios de fonds propres"
    assert result.headers == ["Période", "T1 2025", "T2 2025"]
    assert result.rows == [
        ["Ratio CET1", "13.1%", "13.3%"],
        ["Ratio Tier 1", "14.5%", "14.8%"],
    ]
    assert result.vision_status == "ok"
    assert result.warnings == []
