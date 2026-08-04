"""Etat Pydantic du rapprochement de tableaux entre stages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MatchedPair(BaseModel):
    """Paire de tableaux apparies entre le trimestre precedent et le courant."""

    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    match_confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    def __getitem__(self, key: str) -> Any:
        """Acces dict-like pour compatibilite avec les appelants legacies."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Lecture optionnelle dict-like pour compatibilite legacy."""
        return getattr(self, key, default)


class TableRef(BaseModel):
    """Reference a un tableau ajoute ou supprime avec motif."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    reason: str = ""

    def __getitem__(self, key: str) -> Any:
        """Acces dict-like pour compatibilite avec les appelants legacies."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Lecture optionnelle dict-like pour compatibilite legacy."""
        return getattr(self, key, default)


class MatchingState(BaseModel):
    """Etat intermediaire passe entre primary, inspecteur et recovery."""

    model_config = ConfigDict(extra="allow")

    previous_cards: list[Any] = Field(default_factory=list)
    current_cards: list[Any] = Field(default_factory=list)
    confirmed_pairs: list[MatchedPair] = Field(default_factory=list)
    rejected_pairs: list[MatchedPair] = Field(default_factory=list)
    unresolved_current_ids: list[str] = Field(default_factory=list)
    remaining_previous_ids: list[str] = Field(default_factory=list)
    tables_added: list[TableRef] = Field(default_factory=list)
    tables_removed: list[TableRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class MatchingResult(BaseModel):
    """Resultat final du processus d'appariement des tableaux."""

    model_config = ConfigDict(extra="allow")

    executed: bool = False
    matched_pairs: list[MatchedPair] = Field(default_factory=list)
    tables_added: list[TableRef] = Field(default_factory=list)
    tables_removed: list[TableRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matching_passes_total: int = 0
    matching_pairs_llm_duplicates_total: int = 0
    matching_pairs_llm_deduped_total: int = 0
    validation_retries_total: int = 0
    matching_validation_failures_total: int = 0
    stage1_validation_retries_total: int = 0
    stage2_validation_retries_total: int = 0
    unresolved_after_stage1_total: int = 0
    matched_in_stage2_total: int = 0
    inspector_passes_total: int = 0
    unmatched_after_primary_total: int = 0
    unmatched_after_rescue_total: int = 0
    inspector_rejected_total: int = 0
    inspector_confirmed_total: int = 0
    hybrid_recovery_executed: int = 0
    hybrid_primary_pairs_released_total: int = 0
    hybrid_candidate_pairs_total: int = 0
    hybrid_judge_calls_total: int = 0
    hybrid_final_inspector_calls_total: int = 0
    hybrid_pairs_rejected_total: int = 0
    hybrid_embedding_calls_total: int = 0

    def to_legacy_dict(self) -> dict[str, Any]:
        """Dump JSON-compatible pour les appelants encore bases sur dict."""
        return self.model_dump(mode="json")
