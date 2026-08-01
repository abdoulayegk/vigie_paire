"""Définition de l'état unifié du graphe multi-agents (ComparisonState)."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ComparisonState(BaseModel):
    """État unifié échangé entre tous les nœuds du graphe LangGraph.

    Cet objet conserve la mémoire d'exécution, le statut des sous-agents,
    et les résultats cumulés de l'analyse (tableaux, diffs, triage AMF).
    """

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
    global_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
