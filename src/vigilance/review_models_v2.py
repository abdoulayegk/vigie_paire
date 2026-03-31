"""Review queue models V2 with deduplication support.

This module provides the new data model for the review queue:
- ReviewTableItem: One item per table (or matched pair), grouping all changes
- ChangeItem: A single atomic change (indicator/footnote/structural)

Key features:
- Stable table_key for deduplication
- Grouped changes under one queue item
- Per-change validation tracking
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChangeType(str, Enum):
    """Types of changes that can occur in a table."""

    INDICATOR_ADDED = "indicator_added"
    INDICATOR_REMOVED = "indicator_removed"
    INDICATOR_RENAMED = "indicator_renamed"
    FOOTNOTE_ADDED = "footnote_added"
    FOOTNOTE_REMOVED = "footnote_removed"
    FOOTNOTE_MODIFIED = "footnote_modified"
    TABLE_ADDED = "table_added"
    TABLE_REMOVED = "table_removed"
    TITLE_CHANGED = "title_changed"
    PAGE_MOVED = "page_moved"
    STRUCTURE_CHANGE = "structure_change"
    UNCERTAIN = "uncertain"
    MODIFIED = "modified"


class ValidationStatus(str, Enum):
    """Validation status for a change."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


# Mapping from legacy change_type strings to ChangeType enum
_LEGACY_CHANGE_TYPE_MAP = {
    "added": ChangeType.INDICATOR_ADDED,
    "removed": ChangeType.INDICATOR_REMOVED,
    "renamed": ChangeType.INDICATOR_RENAMED,
    "table_added": ChangeType.TABLE_ADDED,
    "table_removed": ChangeType.TABLE_REMOVED,
    "structure_change": ChangeType.STRUCTURE_CHANGE,
    "uncertain": ChangeType.UNCERTAIN,
    "modified": ChangeType.MODIFIED,
    "footnote": ChangeType.FOOTNOTE_MODIFIED,
}


def compute_table_key(
    bank_code: str,
    section: str,
    table_id_t1: str,
    table_id_t2: str,
    table_title: str,
    page_t1: int | None = None,
    page_t2: int | None = None,
) -> str:
    """Compute a deterministic unique key for a table or matched pair.

    The key uniquely identifies a table across the pipeline, enabling
    deduplication when the same table appears multiple times (e.g.,
    once for indicator changes, once for footnote changes).

    Args:
        bank_code: Bank identifier (e.g., "rbc", "td")
        section: Section name (e.g., "Credit Risk")
        table_id_t1: Table ID from T1 document (may be empty)
        table_id_t2: Table ID from T2 document (may be empty)
        table_title: Table title for fallback hashing

    Returns:
        Stable unique key string
    """
    normalized_section = (section or "").lower().strip()

    # Prefer ID-based pair key if available
    if table_id_t1 or table_id_t2:
        pair_id = f"{table_id_t1 or ''}|{table_id_t2 or ''}"
    else:
        # Fallback to title hash for tables without IDs
        normalized_title = " ".join((table_title or "").lower().strip().split())[:120]
        if normalized_title:
            page_sig = f"{page_t1 if page_t1 is not None else ''}|{page_t2 if page_t2 is not None else ''}"
            fallback_sig = f"{normalized_section}|{normalized_title}|{page_sig}"
            pair_id = f"title:{hashlib.sha256(fallback_sig.encode()).hexdigest()[:16]}"
        else:
            # Last resort: generate unique key (will not dedupe)
            unknown_sig = f"{normalized_section}|{page_t1}|{page_t2}|{table_title}"
            pair_id = f"unknown:{hashlib.sha256(unknown_sig.encode()).hexdigest()[:16]}"

    return f"{bank_code.lower()}::{normalized_section}::{pair_id}"


@dataclass(slots=True)
class ChangeItem:
    """A single atomic change within a table.

    Represents one indicator added/removed/renamed, one footnote change,
    or one structural change. Each change can be validated independently.
    """

    change_id: str
    change_type: str  # ChangeType value as string for JSON serialization
    payload: dict[str, Any]  # Type-specific data
    validation_status: str = "pending"  # ValidationStatus value as string
    is_required: bool = True  # If False, can skip without blocking table completion
    validation_decision: str = ""  # Final decision
    validation_notes: str = ""
    validated_at: str = ""
    validated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for dcc.Store."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "payload": dict(self.payload) if self.payload else {},
            "validation_status": self.validation_status,
            "is_required": self.is_required,
            "validation_decision": self.validation_decision,
            "validation_notes": self.validation_notes,
            "validated_at": self.validated_at,
            "validated_by": self.validated_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeItem":
        """Deserialize from dictionary."""
        return cls(
            change_id=str(data.get("change_id", "")),
            change_type=str(data.get("change_type", "indicator_added")),
            payload=dict(data.get("payload", {})),
            validation_status=str(data.get("validation_status", "pending")),
            is_required=bool(data.get("is_required", True)),
            validation_decision=str(data.get("validation_decision", "")),
            validation_notes=str(data.get("validation_notes", "")),
            validated_at=str(data.get("validated_at", "")),
            validated_by=str(data.get("validated_by", "")),
        )

    def is_validated(self) -> bool:
        """Check if this change has been validated (approved/rejected/skipped)."""
        return self.validation_status in ("approved", "rejected", "skipped")


@dataclass(slots=True)
class ReviewTableItem:
    """A single table/pair in the review queue, grouping all its changes.

    This is the canonical unit in the review queue. Each table appears
    exactly once, with all its changes (indicators, footnotes, structural)
    grouped in the `changes` list.
    """

    table_key: str  # Canonical unique key
    section: str
    table_name: str
    table_number: str
    table_id_t1: str
    table_id_t2: str
    page_t1: int | None
    page_t2: int | None

    # Proof images
    proof_image_t1: str = ""
    proof_image_t2: str = ""
    proof_image_combined: str = ""  # Side-by-side
    bbox_t1: list[float] | None = None
    bbox_t2: list[float] | None = None
    source_pdf_t1: str = ""
    source_pdf_t2: str = ""

    # Grouped changes
    changes: list[ChangeItem] = field(default_factory=list)

    # Status tracking
    table_status: str = "pending"  # "pending", "partial", "completed"
    confidence: float = 0.0
    match_method: str = ""

    # Additional metadata
    genai_analysis: dict[str, Any] = field(default_factory=dict)
    match_metadata: dict[str, Any] = field(default_factory=dict)

    # Priority signals (from GenAI or rules)
    relevance: str = (
        ""  # REGLEMENTAIRE, NOUVELLE_DIVULGATION, STRUCTUREL, NON_SIGNIFICATIF
    )
    risk_level: str = ""  # ELEVE, MODERE, FAIBLE

    def compute_summary(self) -> dict[str, int]:
        """Compute summary counts for display."""
        counts = {
            "total_changes": len(self.changes),
            "indicators_added": 0,
            "indicators_removed": 0,
            "indicators_renamed": 0,
            "footnotes_changed": 0,
            "validated": 0,
            "pending": 0,
        }
        for c in self.changes:
            ct = c.change_type
            if ct == ChangeType.INDICATOR_ADDED.value or ct == "indicator_added":
                counts["indicators_added"] += 1
            elif ct == ChangeType.INDICATOR_REMOVED.value or ct == "indicator_removed":
                counts["indicators_removed"] += 1
            elif ct == ChangeType.INDICATOR_RENAMED.value or ct == "indicator_renamed":
                counts["indicators_renamed"] += 1
            elif ct in (
                ChangeType.FOOTNOTE_ADDED.value,
                ChangeType.FOOTNOTE_REMOVED.value,
                ChangeType.FOOTNOTE_MODIFIED.value,
                "footnote_added",
                "footnote_removed",
                "footnote_modified",
            ):
                counts["footnotes_changed"] += 1

            if c.is_validated():
                counts["validated"] += 1
            else:
                counts["pending"] += 1

        return counts

    def is_complete(self) -> bool:
        """Check if all required changes are validated."""
        for c in self.changes:
            if c.is_required and not c.is_validated():
                return False
        return True

    def update_status(self) -> None:
        """Update table_status based on changes validation state."""
        if not self.changes:
            self.table_status = "completed"
            return

        summary = self.compute_summary()
        if summary["pending"] == 0:
            self.table_status = "completed"
        elif summary["validated"] > 0:
            self.table_status = "partial"
        else:
            self.table_status = "pending"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for dcc.Store."""
        changes_dict = [c.to_dict() for c in self.changes]
        change_types = {c.get("change_type", "") for c in changes_dict}
        view_mode = (
            "table_only"
            if (
                ChangeType.TABLE_ADDED.value in change_types
                or ChangeType.TABLE_REMOVED.value in change_types
                or "table_added" in change_types
                or "table_removed" in change_types
            )
            else "change_list"
        )
        return {
            "review_id": self.table_key,
            "table_key": self.table_key,
            "table_title": self.table_name,
            "view_mode": view_mode,
            "section": self.section,
            "table_name": self.table_name,
            "table_number": self.table_number,
            "table_id_t1": self.table_id_t1,
            "table_id_t2": self.table_id_t2,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "proof_image_t1": self.proof_image_t1,
            "proof_image_t2": self.proof_image_t2,
            "proof_image_combined": self.proof_image_combined,
            "bbox_t1": list(self.bbox_t1) if self.bbox_t1 else None,
            "bbox_t2": list(self.bbox_t2) if self.bbox_t2 else None,
            "source_pdf_t1": self.source_pdf_t1,
            "source_pdf_t2": self.source_pdf_t2,
            "changes": changes_dict,
            "table_status": self.table_status,
            "confidence": self.confidence,
            "match_method": self.match_method,
            "genai_analysis": dict(self.genai_analysis) if self.genai_analysis else {},
            "match_metadata": dict(self.match_metadata) if self.match_metadata else {},
            "relevance": self.relevance,
            "risk_level": self.risk_level,
            "summary": self.compute_summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewTableItem":
        """Deserialize from dictionary."""
        changes_raw = data.get("changes", [])
        changes = [ChangeItem.from_dict(c) for c in changes_raw]

        item = cls(
            table_key=str(data.get("table_key", "")),
            section=str(data.get("section", "")),
            table_name=str(data.get("table_name", "")),
            table_number=str(data.get("table_number", "")),
            table_id_t1=str(data.get("table_id_t1", "")),
            table_id_t2=str(data.get("table_id_t2", "")),
            page_t1=data.get("page_t1"),
            page_t2=data.get("page_t2"),
            proof_image_t1=str(data.get("proof_image_t1", "")),
            proof_image_t2=str(data.get("proof_image_t2", "")),
            proof_image_combined=str(data.get("proof_image_combined", "")),
            bbox_t1=data.get("bbox_t1"),
            bbox_t2=data.get("bbox_t2"),
            source_pdf_t1=str(data.get("source_pdf_t1", "")),
            source_pdf_t2=str(data.get("source_pdf_t2", "")),
            changes=changes,
            table_status=str(data.get("table_status", "pending")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            match_method=str(data.get("match_method", "")),
            genai_analysis=dict(data.get("genai_analysis", {})),
            match_metadata=dict(data.get("match_metadata", {})),
            relevance=str(data.get("relevance", "")),
            risk_level=str(data.get("risk_level", "")),
        )
        return item


def legacy_change_type_to_new(item_type: str, ind_type: str) -> str:
    """Convert legacy change type to new ChangeType value.

    Args:
        item_type: "indicator" or "footnote"
        ind_type: The type field from indicators list ("added", "removed", "renamed", etc.)

    Returns:
        New ChangeType value as string
    """
    if item_type == "footnote":
        if ind_type == "added":
            return ChangeType.FOOTNOTE_ADDED.value
        elif ind_type == "removed":
            return ChangeType.FOOTNOTE_REMOVED.value
        else:
            return ChangeType.FOOTNOTE_MODIFIED.value
    else:
        if ind_type == "added":
            return ChangeType.INDICATOR_ADDED.value
        elif ind_type == "removed":
            return ChangeType.INDICATOR_REMOVED.value
        elif ind_type == "renamed":
            return ChangeType.INDICATOR_RENAMED.value
        else:
            return ChangeType.INDICATOR_ADDED.value
