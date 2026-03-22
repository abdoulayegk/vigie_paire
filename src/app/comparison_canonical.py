"""Helpers for the canonical UI/export comparison payload used by Dash.

Important distinction:
- ``TableArtifact`` is the canonical in-memory comparison object.
- This module handles the canonical UI/export payload shape consumed by Dash.

The legacy function names are kept as wrappers for compatibility, but the
preferred names in this module explicitly mention the UI payload role.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from app.quarter_utils import get_payload_quarter_context

UI_COMPARISON_PAYLOAD_SCHEMA_VERSION = "comparison_canonical_v1"


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
    """Safely fetch a nested value from metadata.

    Walks the nested dict path given by keys. Returns None if meta is None,
    if any intermediate value is not a dict, or if any key is missing.

    Args:
        meta: Metadata dict, or None (treated as empty).
        *keys: One or more keys for nested lookup (e.g., ``("section", "title")``).

    Returns:
        The value at the nested path, or None if the path cannot be traversed.
    """
    cur: Any = meta or {}
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def is_ui_comparison_payload(payload: Any) -> bool:
    """Return True when payload follows the canonical Dash/UI comparison contract."""
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    )


def new_empty_ui_comparison_payload() -> dict[str, Any]:
    """Build a fresh canonical UI comparison payload with default structure.

    Returns a dict containing all required top-level keys for the Dash app:
    schema_version, bank_code, quarter fields, summary (zeroed counts and
    status_counts), empty lists for table_comparisons/tables_added/tables_removed
    and their variants, and meta (generated_at, provenance, source_format,
    executive_summary). The executive_summary defaults to a French message
    indicating no comparison is available.

    Returns:
        A fully-initialized payload dict conforming to the canonical schema.
    """
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": UI_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "bank_code": "",
        "quarter_from": "",
        "quarter_to": "",
        "previous_quarter": "",
        "current_quarter": "",
        "comparison_direction": "current_vs_previous",
        "year": datetime.now().year,
        "summary": {
            "tables_t1": 0,
            "tables_t2": 0,
            "tables_extracted_t1": 0,
            "tables_extracted_t2": 0,
            "tables_comparable_t1": 0,
            "tables_comparable_t2": 0,
            "tables_matched": 0,
            "tables_added": 0,
            "tables_removed": 0,
            "tables_added_confirmed": 0,
            "tables_removed_confirmed": 0,
            "tables_added_pending_review": 0,
            "tables_removed_pending_review": 0,
            "ambiguous_tables": 0,
            "review_candidates": 0,
            "ambiguous_pairs": 0,
            "pairing_coverage": 0.0,
            "indicator_change_pairs": 0,
            "footnote_change_pairs": 0,
            "pairing_low_confidence": False,
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
                "review_candidate": 0,
                "needs_review": 0,
                "structure_change": 0,
                "ajoute": 0,
                "supprime": 0,
                "ajoute_pending_review": 0,
                "supprime_pending_review": 0,
            },
        },
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
        "tables_added_confirmed": [],
        "tables_removed_confirmed": [],
        "tables_added_pending_review": [],
        "tables_removed_pending_review": [],
        "review_candidates": [],
        "meta": {
            "generated_at": now,
            "provenance": "dash_adapter",
            "source_format": "fallback",
            "executive_summary": {"content": "Aucune comparaison disponible."},
        },
    }


def to_ui_comparison_payload(payload: Any) -> dict[str, Any]:
    """Best-effort conversion to the canonical payload used by the Dash app.

    If the payload is already canonical (has schema_version matching
    UI_COMPARISON_PAYLOAD_SCHEMA_VERSION), returns a deep copy.
    If it is a legacy ``metier_tableaux`` result, converts to the canonical
    shape and populates table_comparisons from the ``changes`` list.
    Otherwise, returns an empty canonical payload with basic fields filled
    from the input when possible.

    Args:
        payload: Raw comparison data. May be a canonical dict, a legacy
            metier dict with result_type ``metier_tableaux``, or any other
            dict/object. Non-dict inputs yield an empty canonical payload.

    Returns:
        A dict conforming to the canonical Dash UI comparison schema.
    """
    if is_ui_comparison_payload(payload):
        return deepcopy(payload)

    ui_payload = new_empty_ui_comparison_payload()
    if not isinstance(payload, dict):
        return ui_payload

    if payload.get("result_type") == "metier_tableaux":
        ui_payload["bank_code"] = str(payload.get("bank_code", ""))
        ui_payload["year"] = int(payload.get("year") or ui_payload["year"])
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
        ui_payload["table_comparisons"] = table_comparisons
        ui_payload["summary"]["tables_matched"] = len(table_comparisons)
        ui_payload["summary"]["total_added_indicators"] = sum(
            len(item.get("added_indicators", [])) for item in table_comparisons
        )
        ui_payload["summary"]["total_removed_indicators"] = sum(
            len(item.get("removed_indicators", [])) for item in table_comparisons
        )
        ui_payload["summary"]["status_counts"]["modifie"] = sum(
            1 for item in table_comparisons if item.get("table_status") == "modifie"
        )
        ui_payload["summary"]["tables_changed_t1"] = compute_changed_tables_t1(ui_payload)
        ui_payload["summary"]["tables_changed_t2"] = compute_changed_tables_t2(ui_payload)
        ui_payload["meta"]["source_format"] = "legacy_metier"
        ui_payload["meta"]["executive_summary"] = {
            "content": "Conversion depuis un format metier legacy."
        }
        return ui_payload

    ui_payload["bank_code"] = str(payload.get("bank_code", ""))
    ui_payload["quarter_from"] = str(payload.get("quarter_from", ""))
    ui_payload["quarter_to"] = str(payload.get("quarter_to", ""))
    ui_payload["previous_quarter"] = str(
        payload.get("previous_quarter", ui_payload["quarter_from"])
    )
    ui_payload["current_quarter"] = str(
        payload.get("current_quarter", ui_payload["quarter_to"])
    )
    ui_payload["comparison_direction"] = str(
        payload.get("comparison_direction", "current_vs_previous")
    )
    try:
        ui_payload["year"] = int(payload.get("year") or ui_payload["year"])
    except (TypeError, ValueError):
        pass

    ui_payload["meta"]["source_format"] = "unknown"
    ui_payload["meta"]["quarter_context"] = get_payload_quarter_context(payload)
    ui_payload["meta"]["executive_summary"] = {
        "content": "Format de comparaison non reconnu. Resultat vide genere."
    }
    return ui_payload


def is_canonical_comparison(payload: Any) -> bool:
    """Backward-compatible alias for :func:`is_ui_comparison_payload`."""
    return is_ui_comparison_payload(payload)


def _empty_canonical() -> dict[str, Any]:
    """Backward-compatible alias for :func:`new_empty_ui_comparison_payload`."""
    return new_empty_ui_comparison_payload()


def to_canonical_payload(payload: Any) -> dict[str, Any]:
    """Backward-compatible alias for :func:`to_ui_comparison_payload`."""
    return to_ui_comparison_payload(payload)
