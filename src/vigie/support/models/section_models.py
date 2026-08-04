"""Dataclasses decrivant les sections localisees dans un PDF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SectionRange:
    """Plage d'une section detectee dans un PDF."""

    section: str
    start_page_pdf: int
    end_page_pdf: int
    method: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialiser la plage en dictionnaire."""
        return asdict(self)


@dataclass(slots=True)
class SectionRangesResult:
    """Resultat agrege de detection de sections pour un PDF d'entree."""

    bank_code: str
    pdf_path: str
    quarter: str
    ranges: list[SectionRange] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialiser le resultat en dictionnaire."""
        payload = asdict(self)
        payload["ranges"] = [r.to_dict() for r in self.ranges]
        return payload
