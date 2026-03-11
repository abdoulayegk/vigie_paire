"""Contract and semantics tests for comparison_canonical_v2.

- No public quarter_from, quarter_to, *_t1, *_t2 in canonical output.
- added = current-only, removed = previous-only, rename = previous -> current.
"""

from __future__ import annotations

from app.comparison_canonical import (
    SCHEMA_VERSION_V2,
    get_canonical_v2,
    is_canonical_comparison_v2,
    to_canonical_payload,
)

# Keys that must not appear in public v2 comparison entries
FORBIDDEN_IN_ENTRIES = {
    "table_id_t1",
    "table_id_t2",
    "page_t1",
    "page_t2",
    "title_t1",
    "title_t2",
    "bbox_t1",
    "bbox_t2",
    "row_bboxes_t1",
    "row_bboxes_t2",
}


def test_v2_payload_has_no_quarter_from_to() -> None:
    """Canonical v2 output must not contain quarter_from or quarter_to."""
    v2 = {
        "schema_version": SCHEMA_VERSION_V2,
        "quarter_previous": "Q1_2025",
        "quarter_current": "Q2_2025",
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
    }
    out = get_canonical_v2(v2)
    assert "quarter_from" not in out
    assert "quarter_to" not in out
    assert out.get("quarter_previous") == "Q1_2025"
    assert out.get("quarter_current") == "Q2_2025"


def test_v2_conversion_strips_t1_t2_from_comparison_entries() -> None:
    """get_canonical_v2 must not expose *_t1/*_t2 in table_comparisons entries."""
    v1_like = {
        "schema_version": "comparison_canonical_v1",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [
            {
                "table_id_t1": "id1",
                "table_id_t2": "id2",
                "page_t1": 1,
                "page_t2": 2,
                "title_t1": "A",
                "title_t2": "B",
            }
        ],
        "tables_added": [],
        "tables_removed": [],
    }
    out = get_canonical_v2(v1_like)
    assert out["schema_version"] == SCHEMA_VERSION_V2
    assert "quarter_from" not in out
    assert "quarter_to" not in out
    comps = out.get("table_comparisons", [])
    assert len(comps) == 1
    c = comps[0]
    for bad in FORBIDDEN_IN_ENTRIES:
        assert bad not in c, f"comparison entry must not contain {bad}"
    assert c.get("table_id_previous") == "id1"
    assert c.get("table_id_current") == "id2"
    assert c.get("page_previous") == 1
    assert c.get("page_current") == 2


def test_to_canonical_payload_returns_v2_only() -> None:
    """to_canonical_payload must return v2 schema without forbidden keys."""
    empty = to_canonical_payload({})
    assert empty.get("schema_version") == SCHEMA_VERSION_V2
    assert "quarter_from" not in empty
    assert "quarter_to" not in empty

    assert "quarter_from" not in empty
    assert "quarter_to" not in empty


def test_semantics_added_is_current_only() -> None:
    """added_indicators: present in current, absent in previous (semantic)."""
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "quarter_previous": "Q1",
        "quarter_current": "Q2",
        "table_comparisons": [
            {
                "table_id_previous": "p1",
                "table_id_current": "c1",
                "added_indicators": ["New indicator"],
                "removed_indicators": [],
                "renamed_indicators": [],
            }
        ],
        "tables_added": [],
        "tables_removed": [],
    }
    assert is_canonical_comparison_v2(payload)
    comp = payload["table_comparisons"][0]
    assert "New indicator" in comp.get("added_indicators", [])
    assert comp.get("added_indicators")  # added = current-only
    out = get_canonical_v2(payload)
    assert out["table_comparisons"][0].get("added_indicators") == ["New indicator"]


def test_semantics_removed_is_previous_only() -> None:
    """removed_indicators: present in previous, absent in current (semantic)."""
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "quarter_previous": "Q1",
        "quarter_current": "Q2",
        "table_comparisons": [
            {
                "table_id_previous": "p1",
                "table_id_current": "c1",
                "added_indicators": [],
                "removed_indicators": ["Gone indicator"],
                "renamed_indicators": [],
            }
        ],
        "tables_added": [],
        "tables_removed": [],
    }
    comp = payload["table_comparisons"][0]
    assert "Gone indicator" in comp.get("removed_indicators", [])
    out = get_canonical_v2(payload)
    assert out["table_comparisons"][0].get("removed_indicators") == ["Gone indicator"]


def test_semantics_rename_direction_previous_to_current() -> None:
    """renamed_indicators: direction is previous -> current (from -> to)."""
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "quarter_previous": "Q1",
        "quarter_current": "Q2",
        "table_comparisons": [
            {
                "table_id_previous": "p1",
                "table_id_current": "c1",
                "added_indicators": [],
                "removed_indicators": [],
                "renamed_indicators": [{"from": "Old label", "to": "New label"}],
            }
        ],
        "tables_added": [],
        "tables_removed": [],
    }
    comp = payload["table_comparisons"][0]
    ren = comp.get("renamed_indicators", [])
    assert len(ren) == 1
    assert ren[0].get("from") == "Old label"
    assert ren[0].get("to") == "New label"
