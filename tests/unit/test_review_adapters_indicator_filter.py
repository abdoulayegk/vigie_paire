"""Tests: exclude date/unit lines from indicators for tables_added and tables_removed."""

from __future__ import annotations

import pytest

from app.review_adapters import build_review_items_from_indicator_result
from app.review_models import CHANGE_TYPE_TABLE_ADDED, CHANGE_TYPE_TABLE_REMOVED
from app.ui_indicators import build_indicator_change_rows


def test_tables_added_excludes_date_only_indicators() -> None:
    """build_review_items_from_indicator_result filters date-only lines for tables_added."""
    indicator_result = {
        "bank_code": "BMO",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [],
        "tables_added": [
            {
                "title": "Marges de credit",
                "table_id": "t2_1",
                "section": "Risque",
                "page": 43,
                "first_column_indicators": [
                    "Au 30 avril 2025",
                    "Au 31 janvier 2025",
                    "Prets garantis par un bien immobilier",
                ],
            }
        ],
        "tables_removed": [],
    }
    items = build_review_items_from_indicator_result(
        indicator_result,
        bank_code="BMO",
        quarter_from="t1",
        quarter_to="t2",
        pdf_path_t1="/fake/t1.pdf",
        pdf_path_t2="/fake/t2.pdf",
    )
    table_added_items = [i for i in items if i.change_type == CHANGE_TYPE_TABLE_ADDED]
    assert len(table_added_items) == 1
    indicators = table_added_items[0].indicators
    assert len(indicators) == 1
    assert indicators[0]["name"] == "Prets garantis par un bien immobilier"
    assert indicators[0]["type"] == "added"


def test_tables_removed_excludes_date_only_indicators() -> None:
    """build_review_items_from_indicator_result filters date-only lines for tables_removed."""
    indicator_result = {
        "bank_code": "BMO",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [
            {
                "title": "Ancien tableau",
                "table_id": "t1_1",
                "section": "Risque",
                "page": 10,
                "first_column_indicators": [
                    "(en millions de dollars canadiens) Au 31 janvier 2025",
                    "Dépôts personnels",
                ],
            }
        ],
    }
    items = build_review_items_from_indicator_result(
        indicator_result,
        bank_code="BMO",
        quarter_from="t1",
        quarter_to="t2",
        pdf_path_t1="/fake/t1.pdf",
        pdf_path_t2="/fake/t2.pdf",
    )
    table_removed_items = [i for i in items if i.change_type == CHANGE_TYPE_TABLE_REMOVED]
    assert len(table_removed_items) == 1
    indicators = table_removed_items[0].indicators
    assert len(indicators) == 1
    assert indicators[0]["name"] == "Dépôts personnels"
    assert indicators[0]["type"] == "removed"


def test_build_indicator_change_rows_excludes_date_only_for_added_removed_tables() -> None:
    """build_indicator_change_rows does not list date-only lines in Indicateur for tables_added/removed."""
    payload = {
        "table_comparisons": [],
        "tables_added": [
            {
                "title": "Table T2",
                "table_id": "t2_1",
                "section": "Autres",
                "page": 43,
                "first_column_indicators": [
                    "Au 30 avril 2025",
                    "Prets garantis",
                ],
            }
        ],
        "tables_removed": [
            {
                "title": "Table T1",
                "table_id": "t1_1",
                "section": "Autres",
                "page": 10,
                "first_column_indicators": [
                    "Au 31 janvier 2025",
                    "Dépôts personnels",
                ],
            }
        ],
    }
    rows = build_indicator_change_rows(payload)
    assert len(rows) == 2
    indicator_col_added = next((r["Indicateur"] for r in rows if r.get("Page T2") == 43), "")
    indicator_col_removed = next((r["Indicateur"] for r in rows if r.get("Page T1") == 10), "")
    assert "Au 30 avril 2025" not in indicator_col_added
    assert "Prets garantis" in indicator_col_added
    assert "Au 31 janvier 2025" not in indicator_col_removed
    assert "Dépôts personnels" in indicator_col_removed
