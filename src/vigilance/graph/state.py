"""Définition de l'état unifié du graphe multi-agents (ComparisonState)."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class TextTriageResponse(BaseModel):
    """Réponse typée Pydantic pour le triage d'un fragment ou d'une section textuelle."""

    is_relevant: bool = Field(default=True, description="Indique si le changement textuel est pertinent au niveau prudentiel/AMF")
    themes_amf: list[str] = Field(default_factory=list, description="Liste des thèmes AMF v2 associés (multi-labels)")
    posture_change: str = Field(default="RENFORCEMENT", description="Changement de posture: RENFORCEMENT, ALLEGEMENT, NOUVEAU_DISPOSITIF, RETRAIT_DISPOSITIF, AUCUN")
    impact_level: str = Field(default="MODERE", description="Niveau d'impact: MAJEUR, MODERE, MINEUR")
    explanation: str = Field(default="", description="Explication et note synthétique pour l'analyste")


class ComparisonState(BaseModel):
    """État unifié échangé entre tous les nœuds du graphe LangGraph."""

    bank_code: str = Field(default="", description="Code de la banque (ex: RBC, BMO)")
    year_current: int = Field(default=0, description="Année courante")
    year_previous: int = Field(default=0, description="Année précédente")
    quarter_current: str = Field(default="", description="Trimestre courant (ex: T4)")
    quarter_previous: str = Field(default="", description="Trimestre précédent")

    previous_cards: list[dict[str, Any]] = Field(default_factory=list)
    current_cards: list[dict[str, Any]] = Field(default_factory=list)

    matched_pairs: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_previous: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_current: list[dict[str, Any]] = Field(default_factory=list)

    hybrid_recovery_applied: bool = Field(default=False)
    devil_advocate_applied: bool = Field(default=False)

    pair_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    text_section_triages: list[dict[str, Any]] = Field(default_factory=list)
    global_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
