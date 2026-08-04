"""Modele d'element de revue utilise par le workflow analyste Dash."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHANGE_TYPE_ADDED = "added"
CHANGE_TYPE_REMOVED = "removed"
CHANGE_TYPE_RENAMED = "renamed"
CHANGE_TYPE_TABLE_ADDED = "table_added"
CHANGE_TYPE_TABLE_REMOVED = "table_removed"
CHANGE_TYPE_MODIFIED = "modified"
CHANGE_TYPE_UNCERTAIN = "uncertain"
CHANGE_TYPE_STRUCTURE = "structure_change"
CHANGE_TYPE_FOOTNOTE = "footnote"

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_APPROVED = "approved"
REVIEW_STATUS_REJECTED = "rejected"

# Review event type: drives UI (table-only vs indicator-diff) and payload rules.
EVENT_TYPE_MATCHED_PAIR = "matched_pair"
EVENT_TYPE_TABLE_ADDED = "table_added"
EVENT_TYPE_TABLE_REMOVED = "table_removed"
EVENT_TYPE_FOOTNOTE_ONLY = "footnote_only"


def _parse_all_indicators(val: Any) -> list[str]:
    """Convertir une valeur en liste de chaines d'indicateurs."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    return []


def _parse_bbox(val: Any) -> list[float] | None:
    """Parser une boite englobante en liste de 4 floats."""
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 4:
        try:
            return [float(x) for x in val]
        except (TypeError, ValueError):
            pass
    return None


@dataclass(slots=True)
class ReviewItem:
    """Element unitaire de la file de revue pour le workflow analyste.

    Represente un changement individuel (indicateur, note de bas de page ou
    evenement structurel) avec son statut de validation et ses metadonnees
    de preuve.
    """

    change_id: str
    change_type: str
    indicator: str
    section: str = ""
    table_name: str = ""
    table_number: str = ""
    table_id_t1: str = ""
    table_id_t2: str = ""
    page_t1: int | None = None
    page_t2: int | None = None
    source_ref_t1: str = ""
    source_ref_t2: str = ""
    review_status: str = REVIEW_STATUS_PENDING
    comment: str = ""
    review_user: str = ""
    review_timestamp: str = ""
    edited_value: str = ""
    confidence: float = 0.0
    proof_image_path: str = ""
    proof_mode: str = ""
    unit_context_t1: str = ""
    unit_context_t2: str = ""
    title_resolution_method_t1: str = ""
    title_resolution_method_t2: str = ""
    table_title_raw: str = ""
    table_status: str = ""
    indicators: list[dict[str, Any]] = field(default_factory=list)
    match_method: str = ""
    all_indicators_t1: list[str] = field(default_factory=list)
    all_indicators_t2: list[str] = field(default_factory=list)
    bbox_t1: list[float] | None = None
    bbox_t2: list[float] | None = None
    added_indicators: list[str] = field(default_factory=list)
    removed_indicators: list[str] = field(default_factory=list)
    item_type: str = "indicator"  # "indicator" or "footnote"
    event_type: str = EVENT_TYPE_MATCHED_PAIR  # matched_pair | table_added | table_removed | footnote_only
    footnote_changes: list[dict[str, Any]] = field(default_factory=list)
    genai_analysis: dict[str, Any] = field(default_factory=dict)
    match_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialiser en dictionnaire pour dcc.Store."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "indicator": self.indicator,
            "section": self.section,
            "table_name": self.table_name,
            "table_number": self.table_number,
            "table_id_t1": self.table_id_t1,
            "table_id_t2": self.table_id_t2,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "source_ref_t1": self.source_ref_t1,
            "source_ref_t2": self.source_ref_t2,
            "review_status": self.review_status,
            "comment": self.comment,
            "review_user": self.review_user,
            "review_timestamp": self.review_timestamp,
            "edited_value": self.edited_value,
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
            "match_method": self.match_method,
            "all_indicators_t1": list(self.all_indicators_t1),
            "all_indicators_t2": list(self.all_indicators_t2),
            "bbox_t1": list(self.bbox_t1) if self.bbox_t1 else None,
            "bbox_t2": list(self.bbox_t2) if self.bbox_t2 else None,
            "added_indicators": list(self.added_indicators),
            "removed_indicators": list(self.removed_indicators),
            "item_type": self.item_type,
            "event_type": self.event_type,
            "footnote_changes": list(self.footnote_changes),
            "genai_analysis": dict(self.genai_analysis) if self.genai_analysis else {},
            "match_metadata": dict(self.match_metadata) if self.match_metadata else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewItem":
        """Deserialiser depuis un dictionnaire."""
        return cls(
            change_id=str(data.get("change_id", "")),
            change_type=str(data.get("change_type", CHANGE_TYPE_ADDED)),
            indicator=str(data.get("indicator", "")),
            section=str(data.get("section", "")),
            table_name=str(data.get("table_name", "")),
            table_number=str(data.get("table_number", "") or ""),
            table_id_t1=str(data.get("table_id_t1", "")),
            table_id_t2=str(data.get("table_id_t2", "")),
            page_t1=data.get("page_t1"),
            page_t2=data.get("page_t2"),
            source_ref_t1=str(data.get("source_ref_t1", "")),
            source_ref_t2=str(data.get("source_ref_t2", "")),
            review_status=str(data.get("review_status", REVIEW_STATUS_PENDING)),
            comment=str(data.get("comment", "")),
            review_user=str(data.get("review_user", "")),
            review_timestamp=str(data.get("review_timestamp", "")),
            edited_value=str(data.get("edited_value", "")),
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
            match_method=str(data.get("match_method", "")),
            all_indicators_t1=_parse_all_indicators(data.get("all_indicators_t1")),
            all_indicators_t2=_parse_all_indicators(data.get("all_indicators_t2")),
            bbox_t1=_parse_bbox(data.get("bbox_t1")),
            bbox_t2=_parse_bbox(data.get("bbox_t2")),
            added_indicators=_parse_all_indicators(data.get("added_indicators")),
            removed_indicators=_parse_all_indicators(data.get("removed_indicators")),
            item_type=str(data.get("item_type", "indicator")),
            event_type=str(data.get("event_type", EVENT_TYPE_MATCHED_PAIR)),
            footnote_changes=list(data.get("footnote_changes", [])),
            genai_analysis=dict(data.get("genai_analysis") or {}),
            match_metadata=dict(data.get("match_metadata") or {}),
        )
