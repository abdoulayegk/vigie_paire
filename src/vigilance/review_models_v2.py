"""Modeles de file de revue V2 avec support de deduplication.

Ce module fournit le nouveau modele de donnees pour la file de revue :
- ReviewTableItem : un element par table (ou paire appariee), regroupant tous les changements
- ChangeItem : un changement atomique (indicateur/note de bas de page/structurel)

Fonctionnalites cles :
- table_key stable pour la deduplication
- changements regroupes sous un seul element de file
- suivi de validation par changement
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChangeType(str, Enum):
    """Types de changements pouvant survenir dans une table."""

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
    """Statut de validation d'un changement."""

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
    """Calculer une cle unique deterministe pour une table ou paire appariee.

    La cle identifie de maniere unique une table dans le pipeline, permettant
    la deduplication quand la meme table apparait plusieurs fois.

    Args:
        bank_code: Identifiant de la banque (ex. "rbc", "td").
        section: Nom de la section (ex. "Credit Risk").
        table_id_t1: ID de la table du document T1 (peut etre vide).
        table_id_t2: ID de la table du document T2 (peut etre vide).
        table_title: Titre de la table pour le hachage de repli.
        page_t1: Numero de page dans le document T1 (optionnel).
        page_t2: Numero de page dans le document T2 (optionnel).

    Returns:
        Chaine de cle unique et stable.
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
    """Changement atomique au sein d'une table.

    Represente un indicateur ajoute/supprime/renomme, un changement de note
    de bas de page ou un changement structurel. Chaque changement peut etre
    valide independamment.
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
        """Serialiser en dictionnaire pour dcc.Store."""
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
        """Deserialiser depuis un dictionnaire."""
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
        """Verifier si ce changement a ete valide (approuve/rejete/ignore)."""
        return self.validation_status in ("approved", "rejected", "skipped")


@dataclass(slots=True)
class ReviewTableItem:
    """Table ou paire dans la file de revue, regroupant tous ses changements.

    Unite canonique de la file de revue. Chaque table apparait exactement
    une fois, avec tous ses changements (indicateurs, notes de bas de page,
    structurels) regroupes dans la liste ``changes``.
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
        """Calculer les compteurs de synthese pour l'affichage."""
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
        """Verifier si tous les changements requis sont valides."""
        for c in self.changes:
            if c.is_required and not c.is_validated():
                return False
        return True

    def update_status(self) -> None:
        """Mettre a jour table_status selon l'etat de validation des changements."""
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
        """Serialiser en dictionnaire pour dcc.Store."""
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
        """Deserialiser depuis un dictionnaire."""
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
    """Convertir un type de changement ancien vers la nouvelle valeur ChangeType.

    Args:
        item_type: "indicator" ou "footnote".
        ind_type: Champ type de la liste d'indicateurs ("added", "removed", "renamed", etc.).

    Returns:
        Valeur ChangeType sous forme de chaine.
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
