"""Contract tests for the current canonical UI comparison payload.

This suite intentionally validates the actively supported Dash/UI contract,
which is ``comparison_canonical_v1``. The previous v2 expectations were
historical and no longer match the public payload emitted by the app.
"""

from __future__ import annotations

from vigilance.comparison_canonical import (
    UI_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    is_ui_comparison_payload,
    new_empty_ui_comparison_payload,
    to_canonical_payload,
)


def _raw_report_comparison() -> dict:
    return {
        "artifact_type": "report_comparison",
        "run_id": "20260325_120000",
        "bank_code": "bnc",
        "year_previous": 2025,
        "quarter_previous": "t3",
        "year_current": 2026,
        "quarter_current": "t1",
        "source_pdf_previous": "/tmp/prev.pdf",
        "source_pdf_current": "/tmp/curr.pdf",
        "archived_pdf_previous": "/archive/prev.pdf",
        "archived_pdf_current": "/archive/curr.pdf",
        "matching": {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 0.98,
                    "reason": "Même concept.",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        },
        "pair_comparisons": [
            {
                "previous_table_id": "prev_1",
                "current_table_id": "curr_1",
                "match_confidence": 0.98,
                "match_reason": "Même concept.",
                "previous_table": {
                    "table_id": "prev_1",
                    "title": "Capital précédent",
                    "page": 8,
                    "section": "capital",
                    "bbox": [0.1, 0.2, 0.8, 0.7],
                    "indicators": ["Ratio CET1"],
                    "footnotes": [],
                },
                "current_table": {
                    "table_id": "curr_1",
                    "title": "Capital courant",
                    "page": 10,
                    "section": "capital",
                    "bbox": [0.1, 0.2, 0.8, 0.7],
                    "indicators": ["Ratio CET1"],
                    "footnotes": [],
                },
                "technical_diff": {
                    "indicators_added": [],
                    "indicators_removed": [],
                    "indicators_renamed": [],
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [],
                    "table_level_change": "inchange",
                },
                "analyst_assessment": {},
                "reason": "Aucun changement.",
            }
        ],
        "summary": {
            "matched_pairs_total": 1,
            "tables_added_total": 0,
            "tables_removed_total": 0,
            "indicator_changes_total": 0,
            "footnote_changes_total": 0,
            "high_priority_items_total": 0,
        },
    }


def test_new_empty_ui_payload_uses_current_schema_version() -> None:
    payload = new_empty_ui_comparison_payload()

    assert payload["schema_version"] == UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    assert is_ui_comparison_payload(payload)


def test_to_canonical_payload_returns_current_ui_schema() -> None:
    payload = to_canonical_payload({})

    assert payload["schema_version"] == UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    assert is_ui_comparison_payload(payload)
    assert payload["meta"]["source_format"] == "unknown"


def test_report_comparison_conversion_keeps_current_dash_entry_shape() -> None:
    payload = to_canonical_payload(_raw_report_comparison())

    assert payload["schema_version"] == UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    assert payload["quarter_from"] == "Q3-2025"
    assert payload["quarter_to"] == "Q1-2026"

    entry = payload["table_comparisons"][0]
    assert entry["table_id_t1"] == "prev_1"
    assert entry["table_id_t2"] == "curr_1"
    assert entry["page_t1"] == 8
    assert entry["page_t2"] == 10
    assert "table_id_previous" not in entry
    assert "table_id_current" not in entry
