"""Export and review item helpers for Dash callbacks.

Extracted from dash_app/app.py. app.py re-exports all names from this module
so that all existing monkeypatches (setattr on dash_app) continue to work.
"""

from __future__ import annotations

from vigilance.dash_app.services.review_navigation import (
    _review_id,
    _table_decision_bucket,
)
from vigilance.quarter_utils import quarter_label_from_payload
from vigilance.review_adapters import build_review_items_from_indicator_result
from vigilance.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    EVENT_TYPE_FOOTNOTE_ONLY,
    EVENT_TYPE_TABLE_ADDED,
    EVENT_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    ReviewItem,
)
from vigilance.review_models_v2 import ChangeType


def _indicator_change_total(comp: dict) -> int:
    return (
        len(comp.get("added_indicators", []) or [])
        + len(comp.get("removed_indicators", []) or [])
        + len(comp.get("renamed_indicators", []) or [])
    )


def _footnote_change_total(comp: dict) -> int:
    footnotes = comp.get("footnotes_counts", {}) or {}
    return sum(
        int(footnotes.get(key, 0) or 0) for key in ("added", "removed", "modified")
    )


def _comparison_change_total(comp: dict) -> int:
    return _indicator_change_total(comp) + _footnote_change_total(comp)


def _comparison_has_changes(comp: dict) -> bool:
    return _comparison_change_total(comp) > 0


def _review_priority_of(item: dict) -> str:
    match_meta = item.get("match_metadata", {}) or {}
    genai = item.get("genai_analysis", {}) or {}
    return (
        str(match_meta.get("review_priority") or genai.get("review_priority") or "")
        .strip()
        .lower()
    )


def _is_high_priority_item(item: dict) -> bool:
    return _review_priority_of(item) in {"critique", "prioritaire"}


def _is_low_confidence_comparison(comp: dict) -> bool:
    try:
        match_score = float(comp.get("match_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        match_score = 0.0
    table_status = str(comp.get("table_status", "") or "").strip().lower()
    match_meta = comp.get("match_metadata", {}) or {}
    return (
        match_score < 0.85
        or table_status in {"incertain", "needs_review"}
        or bool(match_meta.get("drastic_row_drop", False))
    )


def _review_items_from_v2_queue(queue: list[dict]) -> list[ReviewItem]:
    """Convert canonical V2 queue to legacy ReviewItem list for exports."""
    items: list[ReviewItem] = []
    for idx, table in enumerate(queue or [], start=1):
        review_id = _review_id(table) or f"tbl_{idx:04d}"
        section = str(table.get("section", ""))
        table_name = str(table.get("table_name") or table.get("table_title") or "")
        table_number = str(table.get("table_number") or "")
        table_id_t1 = str(table.get("table_id_t1") or "")
        table_id_t2 = str(table.get("table_id_t2") or "")
        page_t1 = table.get("page_t1")
        page_t2 = table.get("page_t2")
        changes = table.get("changes", []) or []

        def _export_status_label(status: str) -> str:
            s = str(status or "pending")
            if s == "approved" or s == "skipped":
                return REVIEW_STATUS_APPROVED
            if s == "rejected":
                return REVIEW_STATUS_REJECTED
            return REVIEW_STATUS_PENDING

        has_table_added = any(
            str(c.get("change_type", ""))
            in (ChangeType.TABLE_ADDED.value, "table_added")
            for c in changes
        )
        has_table_removed = any(
            str(c.get("change_type", ""))
            in (ChangeType.TABLE_REMOVED.value, "table_removed")
            for c in changes
        )
        table_bucket = _table_decision_bucket(table)
        review_status = (
            REVIEW_STATUS_APPROVED
            if table_bucket == "approved"
            else REVIEW_STATUS_REJECTED
            if table_bucket == "rejected"
            else REVIEW_STATUS_PENDING
        )

        indicators: list[dict[str, str]] = []
        item_type = "indicator"
        event_type = "matched_pair"
        review_comments: list[str] = []
        review_timestamps: list[str] = []
        review_users: list[str] = []
        for change in changes:
            note = str(change.get("validation_notes", "")).strip()
            if note:
                review_comments.append(note)
            validated_at = str(change.get("validated_at", "")).strip()
            if validated_at:
                review_timestamps.append(validated_at)
            validated_by = str(change.get("validated_by", "")).strip()
            if validated_by:
                review_users.append(validated_by)
        if has_table_added or has_table_removed:
            change_type = (
                CHANGE_TYPE_TABLE_ADDED
                if has_table_added
                else CHANGE_TYPE_TABLE_REMOVED
            )
            event_type = (
                EVENT_TYPE_TABLE_ADDED if has_table_added else EVENT_TYPE_TABLE_REMOVED
            )
            summary_indicator = (
                "Tableau entier ajouté" if has_table_added else "Tableau entier retiré"
            )
        else:
            n_added = n_removed = n_renamed = 0
            for change in changes:
                ctype = str(change.get("change_type", ""))
                payload = change.get("payload", {}) or {}
                c_status = _export_status_label(
                    change.get("validation_status", "pending")
                )
                if ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
                    n_added += 1
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", "")),
                            "type": CHANGE_TYPE_ADDED,
                            "review_status": c_status,
                        }
                    )
                elif ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
                    n_removed += 1
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", "")),
                            "type": CHANGE_TYPE_REMOVED,
                            "review_status": c_status,
                        }
                    )
                elif ctype in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
                    n_renamed += 1
                    from_val = str(payload.get("from", ""))
                    to_val = str(payload.get("to", ""))
                    indicators.append(
                        {
                            "name": f"{from_val} -> {to_val}".strip(" ->"),
                            "type": CHANGE_TYPE_RENAMED,
                            "from": from_val,
                            "to": to_val,
                            "review_status": c_status,
                        }
                    )
                elif "footnote" in ctype:
                    item_type = "footnote"
                    event_type = EVENT_TYPE_FOOTNOTE_ONLY
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", ""))
                            or str(payload.get("new_text", "")),
                            "type": CHANGE_TYPE_MODIFIED,
                            "review_status": c_status,
                        }
                    )
            if n_removed >= n_added and n_removed >= n_renamed:
                change_type = CHANGE_TYPE_REMOVED
            elif n_added >= n_renamed:
                change_type = CHANGE_TYPE_ADDED
            else:
                change_type = CHANGE_TYPE_RENAMED
            summary_parts = []
            if n_added:
                summary_parts.append(f"{n_added} ajouté(s)")
            if n_removed:
                summary_parts.append(f"{n_removed} retiré(s)")
            if n_renamed:
                summary_parts.append(f"{n_renamed} renommé(s)")
            summary_indicator = ", ".join(summary_parts)

        items.append(
            ReviewItem(
                change_id=review_id,
                change_type=change_type,
                indicator=summary_indicator,
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t1=table_id_t1,
                table_id_t2=table_id_t2,
                page_t1=page_t1,
                page_t2=page_t2,
                source_ref_t1=str(table.get("source_pdf_t1", "")),
                source_ref_t2=str(table.get("source_pdf_t2", "")),
                review_status=review_status,
                comment=" | ".join(dict.fromkeys(review_comments)),
                review_user=" | ".join(dict.fromkeys(review_users)),
                review_timestamp=max(review_timestamps) if review_timestamps else "",
                confidence=float(table.get("confidence", 0.0) or 0.0),
                table_title_raw=str(table.get("table_title") or table_name),
                table_status=str(table.get("table_status", "")),
                indicators=indicators,
                match_method=str(table.get("match_method", "")),
                bbox_t1=table.get("bbox_t1"),
                bbox_t2=table.get("bbox_t2"),
                genai_analysis=table.get("genai_analysis") or {},
                match_metadata=table.get("match_metadata") or {},
                item_type=item_type,
                event_type=event_type,
            )
        )
    return items


def _resolve_export_review_items(
    review_items_data, review_queue_data, indicator_result, paths
):
    """Resolve export items from the current review state with queue priority."""
    ir = indicator_result or {}
    items = []
    if review_queue_data:
        items = _review_items_from_v2_queue(review_queue_data)
    elif review_items_data:
        try:
            items = [ReviewItem.from_dict(d) for d in review_items_data]
        except Exception:
            items = []
    if not items and ir:
        paths = paths or {}
        path_t1 = (
            paths.get("pdf_previous", "") or paths.get("pdf_t1", "")
            if isinstance(paths, dict)
            else ""
        )
        path_t2 = (
            paths.get("pdf_current", "") or paths.get("pdf_t2", "")
            if isinstance(paths, dict)
            else ""
        )
        items = build_review_items_from_indicator_result(
            ir,
            bank_code=str(ir.get("bank_code", "")),
            quarter_from=quarter_label_from_payload(ir, "previous"),
            quarter_to=quarter_label_from_payload(ir, "current"),
            pdf_path_t1=path_t1 or "",
            pdf_path_t2=path_t2 or "",
        )
    return items
