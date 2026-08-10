"""Modèles de données internes du pipeline d'analyse textuelle.

Les structures décrivent les sections résolues, les blocs PDF, les audits et
les unités sémantiques échangés entre les étapes. Elles restent indépendantes
des entrées-sorties et des fournisseurs de modèles de langage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TextAnalysisQualityError(RuntimeError):
    """Signale qu'une section ciblée ne produit aucune unité sémantique exploitable."""


@dataclass(slots=True)
class SemanticUnit:
    """Unité sémantique extraite d'une section narratif d'un rapport bancaire."""

    unit_id: str
    section_key: str
    theme: str
    semantic_text: str
    source_text: str
    source_block_ids: list[str]
    source_resolution: str
    evidence_pages: list[int]
    evidence_snippet: str


@dataclass(slots=True)
class ResolvedSection:
    """Section narrative résolue avec ses coordonnées de pages."""

    section_key: str
    title: str
    start_page: int
    end_page: int
    anchor_page: int | None = None
    anchor_text: str | None = None
    anchor_bbox_norm: list[float] | None = None
    end_anchor_page: int | None = None
    end_anchor_text: str | None = None
    end_anchor_bbox_norm: list[float] | None = None

    @property
    def pages(self) -> list[int]:
        """Retourne la liste des pages couvertes par la section."""
        return list(range(self.start_page, self.end_page + 1))


@dataclass(slots=True)
class PDFBlock:
    """Bloc de texte extrait d'un PDF avec ses métadonnées de position."""

    block_id: str
    page: int
    bbox_norm: list[float]
    text: str
    line_number: int
    block_type: str = "other"
    included: bool = False
    exclusion_reason: str = ""
    source_label: str = ""
    heading_level: int | None = None

    @property
    def y0(self) -> float:
        """Coordonnée Y du bord supérieur du bloc (normalisée)."""
        return float(self.bbox_norm[1])

    @property
    def y1(self) -> float:
        """Coordonnée Y du bord inférieur du bloc (normalisée)."""
        return float(self.bbox_norm[3])


@dataclass(slots=True)
class SectionAudit:
    """Audit complet d'une section narrative : blocs inclus, exclus et unités sémantiques."""

    section_key: str
    section_title: str
    start_page: int
    end_page: int
    anchor_page: int | None
    anchor_text: str | None
    anchor_bbox_norm: list[float] | None
    included_blocks: list[PDFBlock]
    excluded_blocks: list[PDFBlock]
    semantic_units: list[SemanticUnit] = field(default_factory=list)
    table_regions: list[dict[str, Any]] = field(default_factory=list)
    end_anchor_page: int | None = None
    end_anchor_text: str | None = None
    end_anchor_bbox_norm: list[float] | None = None
