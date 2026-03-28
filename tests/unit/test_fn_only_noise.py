"""Test that footnote-only 'modifie' tables skip the redundant table-level ReviewItem."""

from __future__ import annotations

from app.review_adapters import build_review_items_from_indicator_result
from app.review_models import (
    CHANGE_TYPE_FOOTNOTE,
    CHANGE_TYPE_MODIFIED,
)


def _make_payload(
    *, table_status: str, has_indicators: bool, has_footnotes: bool
) -> dict:
    comp: dict = {
        "table_id_t1": "tbl_p037_i01",
        "table_id_t2": "tbl_p031_i01",
        "title_t1": "EXIGENCES DE FONDS PROPRES MINIMALES",
        "title_t2": "EXIGENCES DE FONDS PROPRES MINIMALES",
        "section": "capital_management",
        "page_t1": 37,
        "page_t2": 31,
        "table_status": table_status,
        "match_score": 0.95,
        "added_indicators": [],
        "removed_indicators": [],
        "renamed_indicators": [],
        "added_indicators_raw": [],
        "removed_indicators_raw": [],
        "renamed_indicators_raw": [],
        "footnotes_counts": {},
        "footnotes_diff": {
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
        },
    }
    if has_indicators:
        comp["added_indicators"] = ["Indicateur ajouté"]
        comp["added_indicators_raw"] = [
            {"value": "Indicateur ajouté", "reason": "test"}
        ]
    if has_footnotes:
        comp["footnotes_counts"] = {"footnotes_renamed": 1}
        comp["footnotes_diff"] = {
            "added": [],
            "removed": [],
            "modified": [
                {
                    "footnote_ref": "1",
                    "change_type": "modified",
                    "old_text": "Ancienne note.",
                    "new_text": "Nouvelle note.",
                }
            ],
            "counts": {"added": 0, "removed": 0, "modified": 1},
        }
    return {
        "bank_code": "TD",
        "quarter_from": "T3-2025",
        "quarter_to": "T1-2026",
        "table_comparisons": [comp],
        "tables_added": [],
        "tables_removed": [],
    }


def test_fn_only_modifie_table_skips_table_level_modified_item():
    """A 'modifie' table with only footnote changes should NOT produce a
    redundant 'modified' ReviewItem — only footnote ReviewItems."""
    payload = _make_payload(
        table_status="modifie", has_indicators=False, has_footnotes=True
    )
    items = build_review_items_from_indicator_result(
        payload,
        bank_code="TD",
        quarter_from="T3-2025",
        quarter_to="T1-2026",
        pdf_path_t1="/fake/t1.pdf",
        pdf_path_t2="/fake/t2.pdf",
    )
    modified_items = [i for i in items if i.change_type == CHANGE_TYPE_MODIFIED]
    footnote_items = [i for i in items if i.change_type == CHANGE_TYPE_FOOTNOTE]
    assert len(modified_items) == 0, (
        "Redundant 'modified' ReviewItem should not be created for FN-only tables"
    )
    assert len(footnote_items) >= 1, "Footnote ReviewItem should exist"


def test_modifie_table_without_footnotes_still_gets_table_level_item():
    """A 'modifie' table with no indicators AND no footnotes should still get
    a table-level ReviewItem (edge case: GPT says modifie but no details)."""
    payload = _make_payload(
        table_status="modifie", has_indicators=False, has_footnotes=False
    )
    items = build_review_items_from_indicator_result(
        payload,
        bank_code="TD",
        quarter_from="T3-2025",
        quarter_to="T1-2026",
        pdf_path_t1="/fake/t1.pdf",
        pdf_path_t2="/fake/t2.pdf",
    )
    modified_items = [i for i in items if i.change_type == CHANGE_TYPE_MODIFIED]
    assert len(modified_items) == 1, (
        "Table-level 'modified' item should exist when no footnotes either"
    )


def test_modifie_table_with_indicators_and_footnotes_keeps_both():
    """A 'modifie' table with BOTH indicators and footnotes should produce
    indicator ReviewItems AND footnote ReviewItems (no filtering)."""
    payload = _make_payload(
        table_status="modifie", has_indicators=True, has_footnotes=True
    )
    items = build_review_items_from_indicator_result(
        payload,
        bank_code="TD",
        quarter_from="T3-2025",
        quarter_to="T1-2026",
        pdf_path_t1="/fake/t1.pdf",
        pdf_path_t2="/fake/t2.pdf",
    )
    # Should have indicator items (not change_type MODIFIED, but indicator-level items)
    footnote_items = [i for i in items if i.change_type == CHANGE_TYPE_FOOTNOTE]
    # The table with indicators goes through the indicator-level path, not the table-level path
    assert len(footnote_items) >= 1
    # Should NOT have a redundant 'modified' table-level item
    modified_items = [i for i in items if i.change_type == CHANGE_TYPE_MODIFIED]
    assert len(modified_items) == 0
