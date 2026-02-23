"""Adapters from comparison payloads to analyst review items."""

from __future__ import annotations

from typing import Any

from app.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    ReviewItem,
)


def _make_change_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def build_review_items_from_indicator_result(
    indicator_result: dict[str, Any],
    *,
    bank_code: str,
    quarter_from: str,
    quarter_to: str,
    pdf_path_t1: str,
    pdf_path_t2: str,
) -> list[ReviewItem]:
    """Build review queue items grouped by table from canonical indicator comparison payload.

    Each table with changes produces ONE ReviewItem containing all its
    added/removed/renamed indicators in the ``indicators`` list.
    """
    items: list[ReviewItem] = []
    table_comparisons = indicator_result.get("table_comparisons", [])
    seq = 1

    for comp in table_comparisons:
        if not isinstance(comp, dict):
            continue

        added = comp.get("added_indicators", []) or []
        removed = comp.get("removed_indicators", []) or []
        renamed = comp.get("renamed_indicators", []) or []

        if not added and not removed and not renamed:
            continue

        section = str(comp.get("section", ""))
        table_name = str(
            comp.get("title_t2")
            or comp.get("title_t1")
            or comp.get("table_id_t2")
            or comp.get("table_id_t1")
            or ""
        )
        page_t1 = comp.get("page_t1")
        page_t2 = comp.get("page_t2")
        table_id_t1 = str(comp.get("table_id_t1", ""))
        table_id_t2 = str(comp.get("table_id_t2", ""))
        confidence = float(comp.get("match_score", 0.0) or 0.0)

        indicators: list[dict[str, str]] = []

        for ind in added:
            indicators.append({"name": str(ind), "type": CHANGE_TYPE_ADDED})

        for ind in removed:
            indicators.append({"name": str(ind), "type": CHANGE_TYPE_REMOVED})

        for ren in renamed:
            if isinstance(ren, dict):
                old_val = str(ren.get("from", ""))
                new_val = str(ren.get("to", ""))
                label = f"{old_val} -> {new_val}" if old_val or new_val else ""
            else:
                label = str(ren)
            indicators.append({"name": label, "type": CHANGE_TYPE_RENAMED})

        n_added = len(added)
        n_removed = len(removed)
        n_renamed = len(renamed)
        parts = []
        if n_added:
            parts.append(f"{n_added} ajout(s)")
        if n_removed:
            parts.append(f"{n_removed} suppression(s)")
        if n_renamed:
            parts.append(f"{n_renamed} renommage(s)")
        summary_indicator = ", ".join(parts)

        if n_removed >= n_added and n_removed >= n_renamed:
            primary_type = CHANGE_TYPE_REMOVED
        elif n_added >= n_renamed:
            primary_type = CHANGE_TYPE_ADDED
        else:
            primary_type = CHANGE_TYPE_RENAMED

        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl", seq),
                change_type=primary_type,
                indicator=summary_indicator,
                section=section,
                table_name=table_name,
                table_id_t1=table_id_t1,
                table_id_t2=table_id_t2,
                page_t1=page_t1,
                page_t2=page_t2,
                source_ref_t1=pdf_path_t1,
                source_ref_t2=pdf_path_t2,
                confidence=confidence,
                table_title_raw=table_name,
                table_status=str(comp.get("table_status", "")),
                indicators=indicators,
            )
        )
        seq += 1

    tables_added = indicator_result.get("tables_added", []) or []
    for table in tables_added:
        if not isinstance(table, dict):
            continue

        table_name = str(table.get("title") or table.get("table_id", ""))
        section = str(table.get("section", ""))
        page_t2 = table.get("page")
        table_id_t2 = str(table.get("table_id", ""))

        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl_add", seq),
                change_type=CHANGE_TYPE_TABLE_ADDED,
                indicator="Tableau entier ajouté",
                section=section,
                table_name=table_name,
                table_id_t2=table_id_t2,
                page_t2=page_t2,
                source_ref_t2=pdf_path_t2,
                confidence=1.0,
                table_title_raw=table_name,
                table_status="ajoute",
                indicators=[],
            )
        )
        seq += 1

    tables_removed = indicator_result.get("tables_removed", []) or []
    for table in tables_removed:
        if not isinstance(table, dict):
            continue

        table_name = str(table.get("title") or table.get("table_id", ""))
        section = str(table.get("section", ""))
        page_t1 = table.get("page")
        table_id_t1 = str(table.get("table_id", ""))

        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl_rem", seq),
                change_type=CHANGE_TYPE_TABLE_REMOVED,
                indicator="Tableau entier supprimé",
                section=section,
                table_name=table_name,
                table_id_t1=table_id_t1,
                page_t1=page_t1,
                source_ref_t1=pdf_path_t1,
                confidence=1.0,
                table_title_raw=table_name,
                table_status="supprime",
                indicators=[],
            )
        )
        seq += 1

    return items
