"""Review queue normalizer: deduplication and grouping logic.

This module transforms raw ReviewItems (which may contain duplicates)
into a canonical list of ReviewTableItems where each table appears
exactly once with all its changes grouped.

Key functions:
- normalize_review_queue: Main entry point
- _change_exists: Deduplication helper
"""

from __future__ import annotations

import logging
from typing import Any

from vigilance.review_models_v2 import (
    ChangeItem,
    ChangeType,
    ReviewTableItem,
    compute_table_key,
    legacy_change_type_to_new,
)

logger = logging.getLogger(__name__)


def normalize_review_queue(
    raw_items: list[dict[str, Any]],
    bank_code: str,
    quarter_from: str,
    quarter_to: str,
) -> list[ReviewTableItem]:
    """Canonicalize raw review items into deduplicated, grouped table items.

    This is the main entry point for review queue normalization. It:
    1. Computes a stable table_key for each raw item
    2. Groups items by table_key
    3. Merges all changes under one ReviewTableItem per key
    4. Filters out tables with 0 changes
    5. Sorts by section, then page, then table_name

    Args:
        raw_items: List of raw review item dicts (from build_review_items_from_indicator_result)
        bank_code: Bank identifier (e.g., "rbc", "td")
        quarter_from: Source quarter label (e.g., "Q3_2025")
        quarter_to: Target quarter label (e.g., "Q4_2025")

    Returns:
        Deduplicated list of ReviewTableItem, sorted by section/page/name
    """
    groups: dict[str, ReviewTableItem] = {}
    change_counter = 1

    for raw in raw_items:
        # Step 1: Compute stable table_key
        table_key = compute_table_key(
            bank_code=bank_code,
            section=raw.get("section", ""),
            table_id_t1=raw.get("table_id_t1", ""),
            table_id_t2=raw.get("table_id_t2", ""),
            table_title=raw.get("table_name", "") or raw.get("table_title_raw", ""),
            page_t1=raw.get("page_t1"),
            page_t2=raw.get("page_t2"),
        )

        # Step 2: Get or create group
        if table_key not in groups:
            # Extract proof image path - handle both old and new field names
            proof_path = raw.get("proof_image_path", "")

            groups[table_key] = ReviewTableItem(
                table_key=table_key,
                section=raw.get("section", ""),
                table_name=raw.get("table_name", "") or raw.get("table_title_raw", ""),
                table_number=raw.get("table_number", ""),
                table_id_t1=raw.get("table_id_t1", ""),
                table_id_t2=raw.get("table_id_t2", ""),
                page_t1=raw.get("page_t1"),
                page_t2=raw.get("page_t2"),
                proof_image_t1=proof_path,  # Will be set properly later
                proof_image_t2=proof_path,
                bbox_t1=raw.get("bbox_t1"),
                bbox_t2=raw.get("bbox_t2"),
                source_pdf_t1=raw.get("source_ref_t1", ""),
                source_pdf_t2=raw.get("source_ref_t2", ""),
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                match_method=raw.get("match_method", ""),
                genai_analysis=raw.get("genai_analysis", {}),
                match_metadata=raw.get("match_metadata", {}),
            )

        table_item = groups[table_key]

        # Update table_item fields if this raw item has better data
        _merge_table_metadata(table_item, raw)

        change_type_raw = str(raw.get("change_type", "") or "")
        is_table_only_change = change_type_raw in {"table_added", "table_removed"}

        if is_table_only_change:
            table_change_type = _map_table_level_change_type(change_type_raw)
            table_item.changes = []

            change_item = ChangeItem(
                change_id=f"chg_{change_counter:05d}",
                change_type=table_change_type,
                payload={
                    "description": raw.get("indicator", ""),
                    "table_status": raw.get("table_status", ""),
                },
                validation_status="pending",
                is_required=True,
            )

            table_item.changes.append(change_item)
            change_counter += 1
            continue

        if _has_table_only_change(table_item.changes):
            continue

        # Step 3: Extract and append changes
        item_type = raw.get("item_type", "indicator")
        indicators = raw.get("indicators", [])

        for ind in indicators:
            # Determine change type
            ind_type = ind.get("type", "added")
            change_type = legacy_change_type_to_new(item_type, ind_type)

            # Build payload based on change type
            payload = _build_change_payload(ind, item_type, ind_type)

            # Get existing validation status from indicator if present
            existing_status = ind.get("review_status", "pending")

            change_item = ChangeItem(
                change_id=f"chg_{change_counter:05d}",
                change_type=change_type,
                payload=payload,
                validation_status=existing_status,
                is_required=item_type != "footnote",  # Footnotes optional by default
            )

            # Dedupe: check if identical change already exists
            if not _change_exists(table_item.changes, change_item):
                table_item.changes.append(change_item)
                change_counter += 1

        # Handle table-level changes (table_added, table_removed, structure_change)
        if change_type_raw in (
            "table_added",
            "table_removed",
            "structure_change",
            "uncertain",
            "modified",
        ):
            # Only add table-level change if no indicators were present
            # (otherwise the indicators themselves represent the changes)
            if not indicators:
                table_change_type = _map_table_level_change_type(change_type_raw)

                change_item = ChangeItem(
                    change_id=f"chg_{change_counter:05d}",
                    change_type=table_change_type,
                    payload={
                        "description": raw.get("indicator", ""),
                        "table_status": raw.get("table_status", ""),
                    },
                    validation_status="pending",
                    is_required=True,
                )

                if not _change_exists(table_item.changes, change_item):
                    table_item.changes.append(change_item)
                    change_counter += 1

    # Step 4: Filter out tables with 0 changes
    result = [t for t in groups.values() if len(t.changes) > 0]

    # Log deduplication stats
    n_raw = len(raw_items)
    n_deduped = len(result)
    n_removed = n_raw - n_deduped
    if n_removed > 0:
        logger.info(
            "[normalize_review_queue] Deduplicated %d raw items -> %d tables (%d merged/removed)",
            n_raw,
            n_deduped,
            n_removed,
        )

    # Step 5: Sort by section, page, table_name
    result.sort(
        key=lambda t: (
            t.section.lower(),
            t.page_t2 or t.page_t1 or 999,
            t.table_name.lower(),
        )
    )

    # Update status for each table
    for table_item in result:
        table_item.update_status()

    return result


def _build_change_payload(
    ind: dict[str, Any], item_type: str, ind_type: str
) -> dict[str, Any]:
    """Build the payload dict for a ChangeItem based on type."""
    payload: dict[str, Any] = {
        "indicator_name": ind.get("name", ""),
        "indicator_name_clean": (
            ind.get("name_clean", ind.get("name", "")).lower().strip()
        ),
    }

    if ind_type == "renamed":
        payload["from"] = ind.get("from", "")
        payload["to"] = ind.get("to", "")
        payload["from_clean"] = ind.get("from_clean", "")
        payload["to_clean"] = ind.get("to_clean", "")

    if item_type == "footnote":
        payload["footnote_ref"] = ind.get("footnote_ref", "")
        payload["old_text"] = ind.get("old_text", "")
        payload["new_text"] = ind.get("new_text", "")

    return payload


def _map_table_level_change_type(change_type_raw: str) -> str:
    """Map raw table-level change type to ChangeType value."""
    mapping = {
        "table_added": ChangeType.TABLE_ADDED.value,
        "table_removed": ChangeType.TABLE_REMOVED.value,
        "structure_change": ChangeType.STRUCTURE_CHANGE.value,
        "uncertain": ChangeType.UNCERTAIN.value,
        "modified": ChangeType.MODIFIED.value,
    }
    return mapping.get(change_type_raw, ChangeType.MODIFIED.value)


def _merge_table_metadata(table_item: ReviewTableItem, raw: dict[str, Any]) -> None:
    """Merge metadata from raw item into table_item if better."""
    # Update page numbers if missing
    if table_item.page_t1 is None and raw.get("page_t1") is not None:
        table_item.page_t1 = raw.get("page_t1")
    if table_item.page_t2 is None and raw.get("page_t2") is not None:
        table_item.page_t2 = raw.get("page_t2")

    # Update bbox if missing
    if table_item.bbox_t1 is None and raw.get("bbox_t1"):
        table_item.bbox_t1 = raw.get("bbox_t1")
    if table_item.bbox_t2 is None and raw.get("bbox_t2"):
        table_item.bbox_t2 = raw.get("bbox_t2")

    # Merge genai_analysis (prefer non-empty)
    if not table_item.genai_analysis and raw.get("genai_analysis"):
        table_item.genai_analysis = raw.get("genai_analysis", {})

    # Update confidence if higher
    raw_conf = float(raw.get("confidence", 0.0) or 0.0)
    if raw_conf > table_item.confidence:
        table_item.confidence = raw_conf

    # Update match_method if not set
    if not table_item.match_method and raw.get("match_method"):
        table_item.match_method = raw.get("match_method", "")


def _change_exists(changes: list[ChangeItem], new_change: ChangeItem) -> bool:
    """Check if an equivalent change already exists (dedupe).

    Returns True if a change with the same type and key payload fields
    already exists in the list.
    """
    for c in changes:
        if c.change_type != new_change.change_type:
            continue

        # Compare payload keys that matter based on change type
        if c.change_type in (
            ChangeType.INDICATOR_ADDED.value,
            ChangeType.INDICATOR_REMOVED.value,
            "indicator_added",
            "indicator_removed",
        ):
            if c.payload.get("indicator_name_clean") == new_change.payload.get(
                "indicator_name_clean"
            ):
                return True

        elif c.change_type in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
            if c.payload.get("from_clean") == new_change.payload.get(
                "from_clean"
            ) and c.payload.get("to_clean") == new_change.payload.get("to_clean"):
                return True

        elif c.change_type in (
            ChangeType.FOOTNOTE_ADDED.value,
            ChangeType.FOOTNOTE_REMOVED.value,
            ChangeType.FOOTNOTE_MODIFIED.value,
            "footnote_added",
            "footnote_removed",
            "footnote_modified",
        ):
            if c.payload.get("footnote_ref") == new_change.payload.get("footnote_ref"):
                return True

        elif c.change_type in (
            ChangeType.TABLE_ADDED.value,
            ChangeType.TABLE_REMOVED.value,
            ChangeType.STRUCTURE_CHANGE.value,
            ChangeType.UNCERTAIN.value,
            ChangeType.MODIFIED.value,
            "table_added",
            "table_removed",
            "structure_change",
            "uncertain",
            "modified",
        ):
            # Table-level changes: compare description
            if c.payload.get("description") == new_change.payload.get("description"):
                return True

    return False


def _has_table_only_change(changes: list[ChangeItem]) -> bool:
    for change in changes:
        if change.change_type in (
            ChangeType.TABLE_ADDED.value,
            ChangeType.TABLE_REMOVED.value,
            "table_added",
            "table_removed",
        ):
            return True
    return False


def sort_review_tables_by_priority(
    tables: list[ReviewTableItem],
) -> list[ReviewTableItem]:
    """Sort review tables by priority (highest first).

    Priority is determined by:
    1. Regulatory relevance (REGLEMENTAIRE > NOUVELLE_DIVULGATION > STRUCTUREL > others)
    2. Risk level (ELEVE > MODERE > FAIBLE)
    3. Number of changes (more changes = higher priority)
    4. Confidence (lower confidence = needs more review)
    """
    relevance_order = {
        "REGLEMENTAIRE": 0,
        "NOUVELLE_DIVULGATION": 1,
        "STRUCTUREL": 2,
        "NON_SIGNIFICATIF": 3,
        "NON_CLASSIFIE": 4,
        "": 5,
    }

    risk_order = {
        "ELEVE": 0,
        "MODERE": 1,
        "FAIBLE": 2,
        "": 3,
    }

    def sort_key(t: ReviewTableItem) -> tuple:
        rel = relevance_order.get(t.relevance.upper(), 5)
        risk = risk_order.get(t.risk_level.upper(), 3)
        n_changes = len(t.changes)
        conf = t.confidence

        # Sort by: relevance (asc), risk (asc), changes (desc), confidence (asc)
        return (rel, risk, -n_changes, conf)

    return sorted(tables, key=sort_key)


def build_normalized_review_queue(
    indicator_result: dict[str, Any],
    raw_items: list[dict[str, Any]],
    pdf_path_t1: str = "",
    pdf_path_t2: str = "",
) -> list[ReviewTableItem]:
    """Build a normalized, deduplicated review queue from raw items.

    This is a convenience wrapper that extracts metadata from indicator_result
    and calls normalize_review_queue.

    Args:
        indicator_result: The canonical comparison result
        raw_items: List of raw ReviewItem dicts
        pdf_path_t1: Path to T1 PDF
        pdf_path_t2: Path to T2 PDF

    Returns:
        Sorted, deduplicated list of ReviewTableItems
    """
    bank_code = str(indicator_result.get("bank_code", ""))
    quarter_from = str(indicator_result.get("quarter_from", "t1"))
    quarter_to = str(indicator_result.get("quarter_to", "t2"))

    # Normalize
    tables = normalize_review_queue(
        raw_items,
        bank_code=bank_code,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
    )

    # Enrich with PDF paths
    for table in tables:
        if not table.source_pdf_t1:
            table.source_pdf_t1 = pdf_path_t1
        if not table.source_pdf_t2:
            table.source_pdf_t2 = pdf_path_t2

    # Extract priority signals from genai_analysis if available
    for table in tables:
        if table.genai_analysis:
            table.relevance = str(table.genai_analysis.get("relevance", ""))
            table.risk_level = str(table.genai_analysis.get("risk_level", ""))

    # Sort by priority
    tables = sort_review_tables_by_priority(tables)

    return tables
