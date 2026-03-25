"""Step 1 contract test: VisionFullResult.to_footnotes_list() preserves visual order.

Rule 3: footnotes MUST be returned in visual order (top to bottom), never sorted by marker.
"""

from __future__ import annotations

from vigilance.extraction.vision_full_extractor import (
    _parse_vision_result,
)


def test_footnotes_preserve_insertion_order_not_sorted() -> None:
    """Footnotes must preserve visual order (haut -> bas), not be sorted by marker key."""
    # Markers out of numerical order to verify they are NOT sorted
    raw = {
        "table_summary": "Capital réglementaire",
        "indicators": ["Ratio CET1"],
        "footnotes_content": [
            {"id": "3", "text": "Troisième note"},
            {"id": "1", "text": "Première note"},
            {"id": "2", "text": "Deuxième note"},
        ],
    }
    result = _parse_vision_result(raw)
    assert result is not None

    footnotes = result.to_footnotes_list()
    assert len(footnotes) == 3, f"Expected 3 footnotes, got {len(footnotes)}"

    # Must preserve visual order (3, 1, 2) — NOT sorted (1, 2, 3)
    assert footnotes[0]["id"] == "3", (
        f"Expected '3' first, got {footnotes[0]['id']}"
    )
    assert footnotes[1]["id"] == "1", (
        f"Expected '1' second, got {footnotes[1]['id']}"
    )
    assert footnotes[2]["id"] == "2", (
        f"Expected '2' third, got {footnotes[2]['id']}"
    )


def test_footnotes_from_legacy_dict_preserve_insertion_order() -> None:
    """Legacy dict footnotes_content (migration shim) must also preserve insertion order."""
    raw = {
        "table_summary": "Indicateur A",
        "indicators": ["Indicateur A"],
        "footnotes_content": {
            "(3)": "Troisième note",
            "(1)": "Première note",
        },
    }
    result = _parse_vision_result(raw)
    assert result is not None

    footnotes = result.to_footnotes_list()
    assert len(footnotes) == 2

    # Python 3.7+ dicts preserve insertion order — must NOT be sorted
    markers = [fn["id"] for fn in footnotes]
    assert markers == ["(3)", "(1)"], f"Expected [(3), (1)] order, got {markers}"


def test_to_footnotes_list_returns_copy() -> None:
    """to_footnotes_list() must return a copy (mutations don't affect the result)."""
    raw = {
        "table_summary": "X",
        "indicators": ["X"],
        "footnotes_content": [{"id": "1", "text": "Note"}],
    }
    result = _parse_vision_result(raw)
    assert result is not None

    list1 = result.to_footnotes_list()
    list2 = result.to_footnotes_list()
    assert list1 == list2
    list1.clear()
    assert len(result.to_footnotes_list()) == 1  # original not mutated


def test_vision_full_result_has_new_content_fields() -> None:
    """VisionFullResult must expose the new minimal GPT fields."""
    raw = {
        "table_title": "Tableau 5 : Ratios de fonds propres",
        "table_summary": "Ratios de fonds propres réglementaires",
        "headers": ["Période", "T1 2025", "T2 2025"],
        "indicators": [
            {"text": "Ratio CET1", "bbox": [0.1, 0.2, 0.4, 0.25]},
            {"text": "Ratio Tier 1", "bbox": [0.1, 0.3, 0.4, 0.34]},
        ],
        "footnotes_content": [{"id": "1", "text": "Note provisoire"}],
    }
    result = _parse_vision_result(raw)
    assert result is not None
    assert result.table_title == "Tableau 5 : Ratios de fonds propres"
    assert result.table_summary == "Ratios de fonds propres réglementaires"
    assert result.headers == ["Période", "T1 2025", "T2 2025"]
    assert result.indicators == ["Ratio CET1", "Ratio Tier 1"]
    assert result.vision_status == "ok"
    assert result.warnings == []
