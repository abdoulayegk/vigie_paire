"""
Systeme de flags pour revue manuelle des cas incertains.

Ce module gere les cas ou le matching automatique ne peut pas
garantir une decision avec confiance suffisante.

Categories de flags:
- UNCERTAIN_MATCH: Score 0.65-0.84
- MULTI_PAGE_MERGE: Fusion detectee necessitant confirmation
- NO_MATCH_FOUND: Tableau nouveau ou supprime
- DUPLICATE_CANDIDATES: Plusieurs matchs possibles avec scores proches
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Types de flags pour revue manuelle."""

    UNCERTAIN_MATCH = "uncertain_match"
    MULTI_PAGE_MERGE = "multi_page_merge"
    NO_MATCH_FOUND = "no_match_found"
    DUPLICATE_CANDIDATES = "duplicate_candidates"
    STRUCTURE_MISMATCH = "structure_mismatch"
    TITLE_CHANGE = "title_change"


class FlagSeverity(Enum):
    """Niveau de severite du flag."""

    LOW = "low"  # Info, auto-resolvable
    MEDIUM = "medium"  # Revue recommandee
    HIGH = "high"  # Revue requise
    CRITICAL = "critical"  # Bloquant


@dataclass
class ReviewFlag:
    """Flag pour revue manuelle."""

    flag_id: str
    flag_type: FlagType
    severity: FlagSeverity
    description: str
    table_t1_id: Optional[str] = None
    table_t2_id: Optional[str] = None
    candidates: list[dict] = field(default_factory=list)
    match_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertir en dictionnaire."""
        return {
            "flag_id": self.flag_id,
            "flag_type": self.flag_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "table_t1_id": self.table_t1_id,
            "table_t2_id": self.table_t2_id,
            "candidates": self.candidates,
            "match_score": self.match_score,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "context": self.context,
        }


@dataclass
class ReviewSession:
    """Session de revue avec ensemble de flags."""

    session_id: str
    bank_code: str
    pdf_t1_name: str
    pdf_t2_name: str
    flags: list[ReviewFlag] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed: bool = False

    @property
    def pending_count(self) -> int:
        """Nombre de flags en attente de revue."""
        return sum(1 for f in self.flags if not f.resolved)

    @property
    def critical_count(self) -> int:
        """Nombre de flags critiques."""
        return sum(1 for f in self.flags if f.severity == FlagSeverity.CRITICAL and not f.resolved)

    def to_dict(self) -> dict:
        """Convertir en dictionnaire."""
        return {
            "session_id": self.session_id,
            "bank_code": self.bank_code,
            "pdf_t1_name": self.pdf_t1_name,
            "pdf_t2_name": self.pdf_t2_name,
            "flags": [f.to_dict() for f in self.flags],
            "created_at": self.created_at,
            "completed": self.completed,
            "stats": {
                "total": len(self.flags),
                "pending": self.pending_count,
                "critical": self.critical_count,
            },
        }


class UncertaintyFlagger:
    """
    Gestionnaire de flags pour les cas incertains.

    Cree et gere les flags pour revue manuelle.
    """

    def __init__(self):
        self._flag_counter = 0

    def create_uncertain_match_flag(
        self,
        table_t1_id: str,
        table_t2_id: str,
        match_score: float,
        score_details: dict,
    ) -> ReviewFlag:
        """Creer un flag pour un match incertain (score 0.65-0.84)."""
        self._flag_counter += 1

        severity = FlagSeverity.MEDIUM
        if match_score < 0.70:
            severity = FlagSeverity.HIGH

        return ReviewFlag(
            flag_id=f"flag_{self._flag_counter:04d}",
            flag_type=FlagType.UNCERTAIN_MATCH,
            severity=severity,
            description=(
                f"Match probable mais non confirme (score: {match_score:.2f}). Revue recommandee."
            ),
            table_t1_id=table_t1_id,
            table_t2_id=table_t2_id,
            match_score=match_score,
            context=score_details,
        )

    def create_no_match_flag(
        self,
        table_id: str,
        is_t1: bool,
        best_candidate: Optional[dict] = None,
    ) -> ReviewFlag:
        """Creer un flag pour un tableau sans correspondance."""
        self._flag_counter += 1

        report = "T1 (ancien)" if is_t1 else "T2 (nouveau)"
        action = "supprime" if is_t1 else "ajoute"

        return ReviewFlag(
            flag_id=f"flag_{self._flag_counter:04d}",
            flag_type=FlagType.NO_MATCH_FOUND,
            severity=FlagSeverity.MEDIUM,
            description=(f"Tableau {report} sans correspondance. Probablement {action}."),
            table_t1_id=table_id if is_t1 else None,
            table_t2_id=table_id if not is_t1 else None,
            candidates=[best_candidate] if best_candidate else [],
            context={"source_report": "t1" if is_t1 else "t2"},
        )

    def create_duplicate_candidates_flag(
        self,
        table_t1_id: str,
        candidates: list[dict],
    ) -> ReviewFlag:
        """Creer un flag pour plusieurs candidats avec scores proches."""
        self._flag_counter += 1

        return ReviewFlag(
            flag_id=f"flag_{self._flag_counter:04d}",
            flag_type=FlagType.DUPLICATE_CANDIDATES,
            severity=FlagSeverity.HIGH,
            description=(
                f"Plusieurs correspondances possibles ({len(candidates)} candidats). "
                f"Selection manuelle requise."
            ),
            table_t1_id=table_t1_id,
            candidates=candidates,
        )

    def create_multi_page_merge_flag(
        self,
        page_range: tuple[int, int],
        fragment_count: int,
        confidence: float,
    ) -> ReviewFlag:
        """Creer un flag pour une fusion multi-pages."""
        self._flag_counter += 1

        severity = FlagSeverity.LOW if confidence >= 0.80 else FlagSeverity.MEDIUM

        return ReviewFlag(
            flag_id=f"flag_{self._flag_counter:04d}",
            flag_type=FlagType.MULTI_PAGE_MERGE,
            severity=severity,
            description=(
                f"Tableau fusionne depuis {fragment_count} fragments "
                f"(pages {page_range[0]}-{page_range[1]}). "
                f"Confiance: {confidence:.0%}"
            ),
            context={
                "page_range": list(page_range),
                "fragment_count": fragment_count,
                "confidence": confidence,
            },
        )

    def create_structure_mismatch_flag(
        self,
        table_t1_id: str,
        table_t2_id: str,
        t1_cols: int,
        t2_cols: int,
        t1_rows: int,
        t2_rows: int,
    ) -> ReviewFlag:
        """Creer un flag pour une difference de structure significative."""
        self._flag_counter += 1

        return ReviewFlag(
            flag_id=f"flag_{self._flag_counter:04d}",
            flag_type=FlagType.STRUCTURE_MISMATCH,
            severity=FlagSeverity.MEDIUM,
            description=(
                f"Structure differente: T1 ({t1_cols}x{t1_rows}) vs T2 ({t2_cols}x{t2_rows}). "
                f"Verifier correspondance."
            ),
            table_t1_id=table_t1_id,
            table_t2_id=table_t2_id,
            context={
                "t1_structure": {"columns": t1_cols, "rows": t1_rows},
                "t2_structure": {"columns": t2_cols, "rows": t2_rows},
            },
        )

    def create_review_session(
        self,
        bank_code: str,
        pdf_t1_name: str,
        pdf_t2_name: str,
    ) -> ReviewSession:
        """Creer une nouvelle session de revue."""
        session_id = f"{bank_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ReviewSession(
            session_id=session_id,
            bank_code=bank_code,
            pdf_t1_name=pdf_t1_name,
            pdf_t2_name=pdf_t2_name,
        )

    def resolve_flag(
        self,
        flag: ReviewFlag,
        resolution: str,
        resolved_by: str = "user",
    ) -> None:
        """Marquer un flag comme resolu."""
        flag.resolved = True
        flag.resolution = resolution
        flag.resolved_by = resolved_by
        flag.resolved_at = datetime.now().isoformat()

        logger.info(f"Flag {flag.flag_id} resolu: {resolution}")

    def save_session(
        self,
        session: ReviewSession,
        output_path: str,
    ) -> None:
        """Sauvegarder une session en JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Session sauvegardee: {output_path}")

    def load_session(self, input_path: str) -> ReviewSession:
        """Charger une session depuis JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        flags = []
        for f_data in data.get("flags", []):
            flags.append(
                ReviewFlag(
                    flag_id=f_data["flag_id"],
                    flag_type=FlagType(f_data["flag_type"]),
                    severity=FlagSeverity(f_data["severity"]),
                    description=f_data["description"],
                    table_t1_id=f_data.get("table_t1_id"),
                    table_t2_id=f_data.get("table_t2_id"),
                    candidates=f_data.get("candidates", []),
                    match_score=f_data.get("match_score", 0.0),
                    created_at=f_data.get("created_at", ""),
                    resolved=f_data.get("resolved", False),
                    resolution=f_data.get("resolution"),
                    resolved_by=f_data.get("resolved_by"),
                    resolved_at=f_data.get("resolved_at"),
                    context=f_data.get("context", {}),
                )
            )

        return ReviewSession(
            session_id=data["session_id"],
            bank_code=data["bank_code"],
            pdf_t1_name=data["pdf_t1_name"],
            pdf_t2_name=data["pdf_t2_name"],
            flags=flags,
            created_at=data.get("created_at", ""),
            completed=data.get("completed", False),
        )
