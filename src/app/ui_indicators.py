"""UI flattening helpers for indicator change tables."""

from __future__ import annotations

from typing import Any

from app.i18n import status_fr, t
from vigilance.utils.matching_normalizer import _classify_excluded_line


def get_display_indicators(item: dict) -> list[str]:
    """Return raw indicators when available, else clean; for UI display only.
    Matching/diff still use first_column_indicators (clean) only."""
    raw = item.get("first_column_indicators_raw") or item.get("first_column_indicators_raw_list")
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
        table_name = comp.get("title_t2") or comp.get("title_t1") or comp.get("table_id_t2") or comp.get("table_id_t1") or ""
        page_t1 = comp.get("page_t1")
        page_t2 = comp.get("page_t2")
        status = comp.get("table_status", "")

        for indicator in comp.get("added_indicators", []) or []:
            row = {
                "Type": t("indicator_add"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": str(indicator),
                "Page T1": page_t1,
                "Page T2": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

        for indicator in comp.get("removed_indicators", []) or []:
            row = {
                "Type": t("indicator_removal"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": str(indicator),
                "Page T1": page_t1,
                "Page T2": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

        for renamed in comp.get("renamed_indicators", []) or []:
            if isinstance(renamed, dict):
                label = f"{renamed.get('from', '')} -> {renamed.get('to', '')}"
            else:
                label = str(renamed)
            row = {
                "Type": t("indicator_rename"),
                "Section": section,
                "Tableau": table_name,
                "Indicateur": label,
                "Page T1": page_t1,
                "Page T2": page_t2,
                "Statut": status_fr(status),
            }
            if include_review_status:
                row["Review"] = comp.get("review_status", "")
            rows.append(row)

    for table in payload.get("tables_added", []) or []:
        display_indicators = [
            n for n in get_display_indicators(table) if _classify_excluded_line(n) is None
        ]
        rows.append(
            {
                "Type": t("table_added"),
                "Section": table.get("section", ""),
                "Tableau": table.get("title") or table.get("table_id") or "",
                "Indicateur": ", ".join(display_indicators) if display_indicators else "",
                "Page T1": "",
                "Page T2": table.get("page", ""),
                "Statut": status_fr("ajoute"),
            }
        )

    for table in payload.get("tables_removed", []) or []:
        display_indicators = [
            n for n in get_display_indicators(table) if _classify_excluded_line(n) is None
        ]
        rows.append(
            {
                "Type": t("table_removed"),
                "Section": table.get("section", ""),
                "Tableau": table.get("title") or table.get("table_id") or "",
                "Indicateur": ", ".join(display_indicators) if display_indicators else "",
                "Page T1": table.get("page", ""),
                "Page T2": "",
                "Statut": status_fr("supprime"),
            }
        )

    return rows


def run_indicator_auto_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """Small helper kept for backward compatibility with older callbacks."""
    rows = build_indicator_change_rows(payload, include_uncertain=True, include_review_status=False)
    return {"rows": rows, "count": len(rows)}
