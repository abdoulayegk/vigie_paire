"""Modeles internes et erreurs du pipeline texte."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field as PydanticField

class TextAnalysisQualityError(RuntimeError):
    """Raised when a targeted text section cannot yield analyzable semantic units."""


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
class NarrativeUnit:
    """Unité narrative courte issue du markdown et utilisée pour l'alignement."""

    section_key: str
    heading: str
    canonical_topic: str
    unit_text: str
    unit_index: int
    source_heading: str
    char_len: int
    word_count: int
    hierarchy_path: str = ""


@dataclass(slots=True)
class _SubsectionRecord:
    """Sous-section markdown enrichie pour le matching local."""

    section_key: str
    heading: str
    body: str
    canonical_topic: str
    tokens: set[str]
    units: list[NarrativeUnit]
    hierarchy_path: str = ""


@dataclass(slots=True)
class _AlignmentCandidate:
    """Candidat d'alignement entre deux sous-sections ou deux unités."""

    score: float
    alignment_type: str
    canonical_topic: str
    title_similarity: float
    content_similarity: float
    canonical_match: bool


@dataclass(slots=True)
class _ComparisonTask:
    """Travail de comparaison prêt à être exécuté localement/GPT."""

    heading_t1: str | None
    body_t1: str
    heading_t2: str | None
    body_t2: str
    alignment_type: str
    canonical_topic: str
    alignment_confidence: float
    emit_rename: bool = False
    restructure_group_id: str | None = None
    previous_unit_index: int | None = None
    current_unit_index: int | None = None
    previous_unit_indexes: list[int] | None = None
    current_unit_indexes: list[int] | None = None
    synthetic_diff_type: str | None = None


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


class TextObservationGroup(BaseModel):
    """Groupe d'observation proposé par le LLM."""

    model_config = ConfigDict(extra="forbid")

    source_change_ids: list[str]
    observation_title: str
    analyst_summary: str
    rationale: str
    impact_level: str
    action_requise: str
    nouvelle_idee: bool
    themes_amf: list[str] = PydanticField(default_factory=list)
    nouvelle_idee_justification: str = ""


class TextAtomicObservation(BaseModel):
    """Changement que le LLM recommande de conserver séparé."""

    model_config = ConfigDict(extra="forbid")

    change_id: str
    non_grouping_reason: str


class TextObservationConsolidationBatch(BaseModel):
    """Réponse structurée de consolidation d'observations texte."""

    model_config = ConfigDict(extra="forbid")

    observations: list[TextObservationGroup]
    atomic_changes: list[TextAtomicObservation] = PydanticField(default_factory=list)


class TextSubsectionAlignmentMatch(BaseModel):
    """Alignement LLM proposé entre deux sous-sections markdown."""

    model_config = ConfigDict(extra="forbid")

    heading_t1: str
    heading_t2: str
    alignment_type: str
    confidence: str
    reason: str = ""


class TextSubsectionAlignmentPlan(BaseModel):
    """Plan structuré d'alignement des sous-sections."""

    model_config = ConfigDict(extra="forbid")

    matches: list[TextSubsectionAlignmentMatch] = PydanticField(default_factory=list)


class TextUnitAlignmentMatch(BaseModel):
    """Alignement LLM 1-à-1 entre deux unités narratives."""

    model_config = ConfigDict(extra="forbid")

    previous_unit_index: int
    current_unit_index: int
    confidence: str
    reason: str = ""


class TextUnitGroupAlignmentMatch(BaseModel):
    """Alignement LLM local N-à-M entre unités narratives d'une même sous-section."""

    model_config = ConfigDict(extra="forbid")

    previous_unit_indexes: list[int]
    current_unit_indexes: list[int]
    confidence: str
    reason: str = ""


class TextUnitAlignmentPlan(BaseModel):
    """Plan structuré d'alignement des unités narratives."""

    model_config = ConfigDict(extra="forbid")

    matches: list[TextUnitAlignmentMatch] = PydanticField(default_factory=list)
    group_matches: list[TextUnitGroupAlignmentMatch] = PydanticField(default_factory=list)
    removed_unit_indexes: list[int] = PydanticField(default_factory=list)
    added_unit_indexes: list[int] = PydanticField(default_factory=list)
    ambiguous_previous_unit_indexes: list[int] = PydanticField(default_factory=list)
    ambiguous_current_unit_indexes: list[int] = PydanticField(default_factory=list)


class TextComparisonChangeItem(BaseModel):
    """Changement textuel retourné par le LLM avant normalisation interne."""

    model_config = ConfigDict(extra="forbid")

    diff_type: str
    status: str = ""
    topic: str = ""
    text_t1: str = ""
    text_t2: str = ""
    change_summary: str = ""


class TextComparisonBatch(BaseModel):
    """Réponse structurée pour une comparaison texte simple."""

    model_config = ConfigDict(extra="forbid")

    changes: list[TextComparisonChangeItem] = PydanticField(default_factory=list)


class TextComparisonTaskResult(BaseModel):
    """Résultat structuré d'une tâche de comparaison batched."""

    model_config = ConfigDict(extra="forbid")

    task_index: int
    changes: list[TextComparisonChangeItem] = PydanticField(default_factory=list)


class TextComparisonResultsBatch(BaseModel):
    """Réponse structurée pour une comparaison texte par lots."""

    model_config = ConfigDict(extra="forbid")

    results: list[TextComparisonTaskResult] = PydanticField(default_factory=list)
