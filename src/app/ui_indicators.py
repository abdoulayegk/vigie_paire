"""UI flattening helpers for indicator change tables."""

from __future__ import annotations

from typing import Any

from app.i18n import status_fr, t
from vigilance.utils.matching_normalizer import _classify_excluded_line


def get_display_indicators(item: dict) -> list[str]:
    """Return Vision raw indicators when available, else normalized compare labels.

    This helper is UI-only. Matching/diff still use the normalized comparison
    labels stored in ``first_column_indicators``.
    """
    raw = item.get("first_column_indicators_raw") or item.get(
        "first_column_indicators_raw_list"
    )
    if isinstance(raw, list) and any(str(x).strip() for x in raw):
        return [str(x) for x in raw if str(x).strip()]
    clean = item.get("indicators") or item.get("first_column_indicators") or []
    return [str(x) for x in clean if str(x).strip()]


def build_indicator_change_rows(
    payload: dict[str, Any],
    *,
    include_uncertain: bool = False,
    include_review_status: bool = False,
) -> list[dict[str, Any]]:
    """Flatten canonical comparison payload into table rows for Dash."""
    rows: list[dict[str, Any]] = []

    for comp in payload.get("table_comparisons", []) or []:
        if not isinstance(comp, dict):
            continue
        if not include_uncertain and bool(comp.get("uncertain_diff", False)):
            continue

        section = comp.get("section", "")
        table_name = (
            comp.get("title_t2")
            or comp.get("title_t1")
            or comp.get("table_id_t2")
            or comp.get("table_id_t1")
            or ""
        )
        page_t1 = comp.get("page_t1")
        page_t2 = comp.get("page_t2")
        status = comp.get("table_status", "")

        added_display = (
            comp.get("added_indicators_raw", [])
            or comp.get("added_indicators", [])
            or []
        )
        for indicator in added_display:
            row = {
                "Type": t("indicator_add"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": str(indicator),
                "Page précédente": page_t1,
                "Page courante": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

        removed_display = (
            comp.get("removed_indicators_raw", [])
            or comp.get("removed_indicators", [])
            or []
        )
        for indicator in removed_display:
            row = {
                "Type": t("indicator_removal"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": str(indicator),
                "Page précédente": page_t1,
                "Page courante": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

        renamed_display = (
            comp.get("renamed_indicators_raw", [])
            or comp.get("renamed_indicators", [])
            or []
        )
        for renamed in renamed_display:
            if isinstance(renamed, dict):
                label = f"{renamed.get('from', '')} -> {renamed.get('to', '')}"
            else:
                label = str(renamed)
            row = {
                "Type": t("indicator_rename"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": label,
                "Page précédente": page_t1,
                "Page courante": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

    for table in payload.get("tables_added", []) or []:
        display_indicators = [
            n
            for n in get_display_indicators(table)
            if _classify_excluded_line(n) is None
        ]
        rows.append(
            {
                "Type": t("table_added"),
                "Section": table.get("section", ""),
                "Tableau": table.get("title") or table.get("table_id") or "",
                "Indicateur": ", ".join(display_indicators)
                if display_indicators
                else "",
                "Page précédente": "",
                "Page courante": table.get("page", ""),
                "Statut": status_fr("ajoute"),
            }
        )

    for table in payload.get("tables_removed", []) or []:
        display_indicators = [
            n
            for n in get_display_indicators(table)
            if _classify_excluded_line(n) is None
        ]
        rows.append(
            {
                "Type": t("table_removed"),
                "Section": table.get("section", ""),
                "Tableau": table.get("title") or table.get("table_id") or "",
                "Indicateur": ", ".join(display_indicators)
                if display_indicators
                else "",
                "Page précédente": table.get("page", ""),
                "Page courante": "",
                "Statut": status_fr("supprime"),
            }
        )

    return rows


def run_indicator_auto_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """Small helper kept for backward compatibility with older callbacks."""
    rows = build_indicator_change_rows(
        payload, include_uncertain=True, include_review_status=False
    )
    return {"rows": rows, "count": len(rows)}
