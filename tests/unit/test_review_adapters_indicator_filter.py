"""Tests: whole-table events have no indicator list; matched pairs have indicators."""

from __future__ import annotations

import pytest

from vigilance.review_adapters import build_review_items_from_indicator_result
from vigilance.review_models import (
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    EVENT_TYPE_MATCHED_PAIR,
    EVENT_TYPE_TABLE_ADDED,
    EVENT_TYPE_TABLE_REMOVED,
)
from vigilance.ui_indicators import build_indicator_change_rows


def test_tables_added_has_no_indicators_and_event_type() -> None:
    """Whole-table events: table_added has indicators=[] and event_type=table_added (no indicator list in UI)."""
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
    item = table_added_items[0]
    assert item.event_type == EVENT_TYPE_TABLE_ADDED
    assert item.indicators == []


def test_tables_removed_has_no_indicators_and_event_type() -> None:
    """Whole-table events: table_removed has indicators=[] and event_type=table_removed (no indicator list in UI)."""
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
    item = table_removed_items[0]
    assert item.event_type == EVENT_TYPE_TABLE_REMOVED
    assert item.indicators == []


def test_tables_added_event_type_and_all_indicators_t2() -> None:
    """table_added item has event_type and all_indicators_t2 from source (no indicator list)."""
    indicator_result = {
        "bank_code": "BMO",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [],
        "tables_added": [
            {
                "title": "New table",
                "table_id": "t2_1",
                "section": "Risque",
                "page": 43,
                "first_column_indicators": ["Prets garantis", "Depots"],
                "all_indicators_t1": [],
                "all_indicators_t2": ["Prets garantis", "Depots"],
                "bbox_t1": None,
                "bbox_t2": [0.1, 0.2, 0.9, 0.8],
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
    item = table_added_items[0]
    assert item.event_type == EVENT_TYPE_TABLE_ADDED
    assert item.indicators == []
    assert item.all_indicators_t2 == ["Prets garantis", "Depots"]


def test_tables_removed_event_type_and_all_indicators_t1() -> None:
    """table_removed item has event_type and all_indicators_t1 from source (no indicator list)."""
    indicator_result = {
        "bank_code": "BMO",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [
            {
                "title": "Old table",
                "table_id": "t1_1",
                "section": "Risque",
                "page": 10,
                "first_column_indicators": ["Dépôts personnels", "Bilan"],
                "all_indicators_t1": ["Depots personnels", "Bilan"],
                "all_indicators_t2": [],
                "bbox_t1": [0.05, 0.1, 0.95, 0.9],
                "bbox_t2": None,
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
    item = table_removed_items[0]
    assert item.event_type == EVENT_TYPE_TABLE_REMOVED
    assert item.indicators == []
    assert item.all_indicators_t1 == ["Depots personnels", "Bilan"]


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
                "indicators": [
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
                "indicators": [
                    "Au 31 janvier 2025",
                    "Dépôts personnels",
                ],
            }
        ],
    }
    rows = build_indicator_change_rows(payload)
    assert len(rows) == 2
    indicator_col_added = next(
        (r["Indicateur"] for r in rows if r.get("Page courante") == 43),
        "",
    )
    indicator_col_removed = next(
        (r["Indicateur"] for r in rows if r.get("Page précédente") == 10),
        "",
    )
    assert "Au 30 avril 2025" not in indicator_col_added
    assert "Prets garantis" in indicator_col_added
    assert "Au 31 janvier 2025" not in indicator_col_removed
    assert "Dépôts personnels" in indicator_col_removed


def test_review_item_prefers_raw_display_for_matched_table_changes() -> None:
    indicator_result = {
        "bank_code": "BMO",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "table_comparisons": [
            {
                "section": "Risque",
                "table_id_t1": "t1_1",
                "table_id_t2": "t2_1",
                "title_t1": "Tableau test",
                "title_t2": "Tableau test",
                "table_status": "modifie",
                "match_score": 0.98,
                "added_indicators": ["fonds propre de categorie 1"],
                "removed_indicators": ["actif instrument tlac disponible apres ajustement"],
                "renamed_indicators": [
                    {
                        "from": "autre instrument tlac disponible apres ajustement",
                        "to": "instrument tlac disponible apres ajustement",
                    }
                ],
                "added_indicators_raw": ["fonds propre de categorie 1¹"],
                "removed_indicators_raw": ["actif instrument tlac disponible apres ajustement¹"],
                "renamed_indicators_raw": [
                    {
                        "from": "autre instrument tlac disponible apres ajustement¹",
                        "to": "instrument tlac disponible apres ajustement²",
                    }
                ],
            }
        ],
        "tables_added": [],
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
    assert len(items) == 1
    item = items[0]
    assert item.event_type == EVENT_TYPE_MATCHED_PAIR
    # Keep clean lists for highlight/matching compatibility.
    assert item.added_indicators == ["fonds propre de categorie 1"]
    assert item.removed_indicators == ["actif instrument tlac disponible apres ajustement"]
    names = [ind.get("name", "") for ind in item.indicators]
    assert "fonds propre de categorie 1¹" in names
    assert "actif instrument tlac disponible apres ajustement¹" in names
    renamed = next(ind for ind in item.indicators if ind.get("type") == "renamed")
    assert renamed.get("from") == "autre instrument tlac disponible apres ajustement¹"
    assert renamed.get("to") == "instrument tlac disponible apres ajustement²"
