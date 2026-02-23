"""Review item model used by the Dash analyst workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHANGE_TYPE_ADDED = "added"
CHANGE_TYPE_REMOVED = "removed"
CHANGE_TYPE_RENAMED = "renamed"
CHANGE_TYPE_TABLE_ADDED = "table_added"
CHANGE_TYPE_TABLE_REMOVED = "table_removed"

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_APPROVED = "approved"
REVIEW_STATUS_REJECTED = "rejected"


@dataclass(slots=True)
class ReviewItem:
    change_id: str
    change_type: str
    indicator: str
    section: str = ""
    table_name: str = ""
    table_id_t1: str = ""
    table_id_t2: str = ""
    page_t1: int | None = None
    page_t2: int | None = None
    source_ref_t1: str = ""
    source_ref_t2: str = ""
    review_status: str = REVIEW_STATUS_PENDING
    comment: str = ""
    confidence: float = 0.0
    proof_image_path: str = ""
    proof_mode: str = ""
    unit_context_t1: str = ""
    unit_context_t2: str = ""
    title_resolution_method_t1: str = ""
    title_resolution_method_t2: str = ""
    table_title_raw: str = ""
    table_status: str = ""
    indicators: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "indicator": self.indicator,
            "section": self.section,
            "table_name": self.table_name,
            "table_id_t1": self.table_id_t1,
            "table_id_t2": self.table_id_t2,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "source_ref_t1": self.source_ref_t1,
            "source_ref_t2": self.source_ref_t2,
            "review_status": self.review_status,
            "comment": self.comment,
            "confidence": float(self.confidence),
            "proof_image_path": self.proof_image_path,
            "proof_mode": self.proof_mode,
            "unit_context_t1": self.unit_context_t1,
            "unit_context_t2": self.unit_context_t2,
            "title_resolution_method_t1": self.title_resolution_method_t1,
            "title_resolution_method_t2": self.title_resolution_method_t2,
            "table_title_raw": self.table_title_raw,
            "table_status": self.table_status,
            "indicators": list(self.indicators),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewItem":
        return cls(
            change_id=str(data.get("change_id", "")),
            change_type=str(data.get("change_type", CHANGE_TYPE_ADDED)),
            indicator=str(data.get("indicator", "")),
            section=str(data.get("section", "")),
            table_name=str(data.get("table_name", "")),
            table_id_t1=str(data.get("table_id_t1", "")),
            table_id_t2=str(data.get("table_id_t2", "")),
            page_t1=data.get("page_t1"),
            page_t2=data.get("page_t2"),
            source_ref_t1=str(data.get("source_ref_t1", "")),
            source_ref_t2=str(data.get("source_ref_t2", "")),
            review_status=str(data.get("review_status", REVIEW_STATUS_PENDING)),
            comment=str(data.get("comment", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            proof_image_path=str(data.get("proof_image_path", "")),
            proof_mode=str(data.get("proof_mode", "")),
            unit_context_t1=str(data.get("unit_context_t1", "")),
            unit_context_t2=str(data.get("unit_context_t2", "")),
            title_resolution_method_t1=str(data.get("title_resolution_method_t1", "")),
            title_resolution_method_t2=str(data.get("title_resolution_method_t2", "")),
            table_title_raw=str(data.get("table_title_raw", "")),
            table_status=str(data.get("table_status", "")),
            indicators=list(data.get("indicators", [])),
        )
