"""Modeles et parametres de la comparaison de sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from vigilance.text_analysis.chunk_alignment import ChunkAlignment


_MAX_COMPARISON_LLM_WORKERS = 6


_EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD = 0.98


_COMPARISON_BATCH_SIZES = {
    "matched_strong": 5,
    "matched_grouped": 1,
    "matched_weak": 3,
    "ambiguous": 1,
    "possible_added": 1,
    "possible_removed": 1,
}


_CHUNK_COMPARISON_VALIDATION_RETRY_MESSAGE = (
    "Corrige la réponse et renvoie le batch COMPLET en respectant strictement le schéma. "
    "Chaque changement doit inclure alignment_id obligatoire, diff_type parmi "
    "unchanged|modified|added|removed, text_t1/text_t2 sans balises [a00]/[c00], "
    "alignment_decision parmi same_disclosure|distinct_disclosures|moved_text|uncertain, "
    "alignment_confidence parmi high|medium|low et alignment_rationale non vide, "
    "modified et unchanged doivent avoir text_t1 et text_t2 non vides, added doit "
    "avoir text_t2 non vide, removed doit avoir text_t1 non vide. Ne fusionne jamais "
    "plusieurs alignments dans un même changement."
)


class ChunkComparisonLLMChange(BaseModel):
    """Changement brut validé à la frontière LLM pour un seul alignment."""

    model_config = ConfigDict(extra="forbid")

    alignment_id: str
    diff_type: Literal["unchanged", "modified", "added", "removed"]
    text_t1: str
    text_t2: str
    change_summary: str
    # Kept optional for compatibility with existing cached responses.  The
    # prompt requires all three fields; missing values are handled
    # conservatively from the deterministic alignment type below.
    alignment_decision: Literal[
        "same_disclosure", "distinct_disclosures", "moved_text", "uncertain", ""
    ] = ""
    alignment_confidence: Literal["high", "medium", "low", ""] = ""
    alignment_rationale: str = ""

    @field_validator(
        "alignment_id",
        "text_t1",
        "text_t2",
        "change_summary",
        "alignment_rationale",
        mode="before",
    )
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_by_diff_type(self) -> "ChunkComparisonLLMChange":
        if not self.alignment_id:
            raise ValueError("alignment_id est obligatoire pour chaque changement chunké")
        if self.diff_type in {"unchanged", "modified"} and not (self.text_t1 and self.text_t2):
            raise ValueError("unchanged/modified exigent text_t1 et text_t2 non vides")
        if self.diff_type == "added" and not self.text_t2:
            raise ValueError("added exige text_t2 non vide")
        if self.diff_type == "removed" and not self.text_t1:
            raise ValueError("removed exige text_t1 non vide")
        return self


class ChunkComparisonLLMResponse(BaseModel):
    """Réponse structurée du LLM pour un batch d'alignements chunkés."""

    model_config = ConfigDict(extra="forbid")

    changes: list[ChunkComparisonLLMChange]


@dataclass(slots=True)
class ComparisonBatch:
    """Lot d'alignements envoyé dans un appel LLM de comparaison."""

    batch_id: str
    alignment_type: str
    alignments: list[ChunkAlignment]
    heading_label: str
    heading_slug: str
    idx_offset: int
