"""Helpers for canonical comparison payloads used by the Dash UI."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Changed-tables metrics
# ---------------------------------------------------------------------------

def _is_comparison_changed(c: dict[str, Any]) -> bool:
    """Return True when a matched-pair comparison entry has any detected change.

    Covers indicator-level diffs (table_status != 'stable') and footnote-only
    changes that do not alter table_status.
    """
    if c.get("table_status", "stable") != "stable":
        return True
    fn = c.get("footnotes_counts") or {}
    return bool(fn.get("added", 0) or fn.get("removed", 0) or fn.get("modified", 0))


def compute_changed_tables_t1(result: dict[str, Any]) -> int:
    """Count distinct T1 tables involved in at least one change.

    A T1 table is "changed" if it participates in:
    - a matched pair with indicator/footnote diffs or structure change,
    - OR it was removed (present in T1, absent in T2).

    Uses ``table_id_t1`` (matched pairs) and ``table_id`` (tables_removed)
    as stable de-duplication keys.
    """
    changed: set[str] = set()
    for c in result.get("table_comparisons", []):
        if _is_comparison_changed(c):
            tid = c.get("table_id_t1")
            if tid:
                changed.add(tid)
    for t in result.get("tables_removed", []):
        tid = t.get("table_id")
        if tid:
            changed.add(tid)
    return len(changed)


def compute_changed_tables_t2(result: dict[str, Any]) -> int:
    """Count distinct T2 tables involved in at least one change.

    A T2 table is "changed" if it participates in:
    - a matched pair with indicator/footnote diffs or structure change,
    - OR it was added (absent in T1, present in T2).

    Uses ``table_id_t2`` (matched pairs) and ``table_id`` (tables_added)
    as stable de-duplication keys.
    """
    changed: set[str] = set()
    for c in result.get("table_comparisons", []):
        if _is_comparison_changed(c):
            tid = c.get("table_id_t2")
            if tid:
                changed.add(tid)
    for t in result.get("tables_added", []):
        tid = t.get("table_id")
        if tid:
            changed.add(tid)
    return len(changed)


def get_meta_value(meta: dict[str, Any] | None, *keys: str) -> Any:
    """Safely fetch a nested value from metadata."""
    cur: Any = meta or {}
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def is_canonical_comparison(payload: Any) -> bool:
    """Return True when payload follows ``comparison_canonical_v1`` contract."""
    return isinstance(payload, dict) and payload.get("schema_version") == "comparison_canonical_v1"


def _empty_canonical() -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": "comparison_canonical_v1",
        "bank_code": "",
        "quarter_from": "t1",
        "quarter_to": "t2",
        "year": datetime.now().year,
        "summary": {
            "tables_t1": 0,
            "tables_t2": 0,
            "tables_matched": 0,
            "tables_added": 0,
            "tables_removed": 0,
            "tables_changed_t1": 0,
            "tables_changed_t2": 0,
            "total_added_indicators": 0,
            "total_removed_indicators": 0,
            "total_renamed_indicators": 0,
            "status_counts": {
                "stable": 0,
                "modifie": 0,
                "renommage_probable": 0,
                "incertain": 0,
                "needs_review": 0,
                "structure_change": 0,
                "ajoute": 0,
                "supprime": 0,
            },
        },
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
        "meta": {
            "generated_at": now,
            "provenance": "dash_adapter",
            "source_format": "fallback",
            "executive_summary": {"content": "Aucune comparaison disponible."},
        },
    }


def to_canonical_payload(payload: Any) -> dict[str, Any]:
    """Best-effort conversion to canonical payload used by the Dash app."""
    if is_canonical_comparison(payload):
        return deepcopy(payload)

    canonical = _empty_canonical()
    if not isinstance(payload, dict):
        return canonical

    if payload.get("result_type") == "metier_tableaux":
        canonical["bank_code"] = str(payload.get("bank_code", ""))
        canonical["year"] = int(payload.get("year") or canonical["year"])
        changes = payload.get("changes") or []
        table_comparisons: list[dict[str, Any]] = []
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                continue
            ctype = str(change.get("change_type", ""))
            added = [str(change.get("indicator_name"))] if ctype in {"added", "table_added"} and change.get("indicator_name") else []
            removed = [str(change.get("indicator_name"))] if ctype in {"removed", "table_removed"} and change.get("indicator_name") else []
            table_comparisons.append(
                {
                    "table_id_t1": f"legacy_t1_{index}",
                    "table_id_t2": f"legacy_t2_{index}",
                    "title_t1": change.get("table_title", ""),
                    "title_t2": change.get("table_title", ""),
                    "page_t1": change.get("page_t1"),
                    "page_t2": change.get("page_t2"),
                    "section": change.get("section", "unknown_section"),
                    "added_indicators": added,
                    "removed_indicators": removed,
                    "renamed_indicators": [],
                    "counts": {
                        "added": len(added),
                        "removed": len(removed),
                        "renamed": 0,
                        "renamed_probable": 0,
                    },
                    "uncertain_diff": False,
                    "table_status": "modifie" if added or removed else "stable",
                }
            )
        canonical["table_comparisons"] = table_comparisons
        canonical["summary"]["tables_matched"] = len(table_comparisons)
        canonical["summary"]["total_added_indicators"] = sum(
            len(item.get("added_indicators", [])) for item in table_comparisons
        )
        canonical["summary"]["total_removed_indicators"] = sum(
            len(item.get("removed_indicators", [])) for item in table_comparisons
        )
        canonical["summary"]["status_counts"]["modifie"] = sum(
            1 for item in table_comparisons if item.get("table_status") == "modifie"
        )
        canonical["summary"]["tables_changed_t1"] = compute_changed_tables_t1(canonical)
        canonical["summary"]["tables_changed_t2"] = compute_changed_tables_t2(canonical)
        canonical["meta"]["source_format"] = "legacy_metier"
        canonical["meta"]["executive_summary"] = {
            "content": "Conversion depuis un format metier legacy."
        }
        return canonical

    canonical["bank_code"] = str(payload.get("bank_code", ""))
    canonical["quarter_from"] = str(payload.get("quarter_from", "t1"))
    canonical["quarter_to"] = str(payload.get("quarter_to", "t2"))
    try:
        canonical["year"] = int(payload.get("year") or canonical["year"])
    except (TypeError, ValueError):
        pass

    canonical["meta"]["source_format"] = "unknown"
    canonical["meta"]["executive_summary"] = {
        "content": "Format de comparaison non reconnu. Resultat vide genere."
    }
    return canonical
