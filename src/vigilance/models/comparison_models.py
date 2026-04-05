"""Modeles Pydantic pour le pipeline de comparaison GPT (Structured Outputs).

Ces modeles definissent les schemas de reponse stricts utilises par :
- la revue Devil's Advocate
- le diff d'indicateurs
- le diff de notes de bas de page
- le filtre d'artefacts de l'inspecteur
- la verification visuelle (Visual Sanity Check)

Ils remplacent l'ancien pattern de parsing ``dict[str, Any]`` libre par des
sorties garanties typees via OpenAI Structured Outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class AnalystAssessment(BaseModel):
    """Evaluation destinee a l'analyste, attachee a chaque changement du diff."""

    model_config = ConfigDict(extra="forbid")

    relevance_level: int = Field(description="1=Critical/Regulatory, 2=High/Structural, 3=Low/Cosmetic")
    justification: str = Field(description="Business impact justification for the analyst")


# ---------------------------------------------------------------------------
# Match Inspector (Stage 1.5 — pair-level GenAI verification)
# ---------------------------------------------------------------------------


class MatchInspectorVerdict(BaseModel):
    """Verdict pour une paire appariee revue par l'inspecteur."""

    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    verdict: str = Field(description="'confirmed' or 'rejected'")
    shared_indicators: list[str] = Field(
        default_factory=list,
        description="Indicator labels found in BOTH tables (exact or semantic match)",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class MatchInspectorResponse(BaseModel):
    """Schema de reponse strict pour la revue par lot de l'inspecteur de matching."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[MatchInspectorVerdict]


class SinglePairInspectorResponse(BaseModel):
    """Schema de reponse pour l'inspection d'une paire unique (pas d'IDs — le code les connait)."""

    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(description="'confirmed' or 'rejected'")
    shared_indicators: list[str] = Field(
        default_factory=list,
        description="Indicator labels found in BOTH tables (exact or semantic match)",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


# ---------------------------------------------------------------------------
# Devil's Advocate
# ---------------------------------------------------------------------------


class DevilAdvocateMatch(BaseModel):
    """Paire proposee par le Devil's Advocate."""

    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    match_confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class DevilAdvocateConfirmation(BaseModel):
    """Confirmation d'une paire a faible confiance par le Devil's Advocate."""

    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    verdict: str = Field(description="'confirmed'")


class DevilAdvocateContestation(BaseModel):
    """Contestation d'une paire par le Devil's Advocate."""

    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    verdict: str = Field(description="'contested'")
    reason: str


class DevilAdvocateResponse(BaseModel):
    """Schema de reponse strict pour la revue en second avis du Devil's Advocate."""

    model_config = ConfigDict(extra="forbid")

    new_matches: list[DevilAdvocateMatch] = Field(default_factory=list)
    confirmed_low_confidence: list[DevilAdvocateConfirmation] = Field(default_factory=list)
    contested_pairs: list[DevilAdvocateContestation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicator diff
# ---------------------------------------------------------------------------


class IndicatorChange(BaseModel):
    """Indicateur ajoute ou supprime entre deux trimestres."""

    model_config = ConfigDict(extra="forbid")

    value: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class IndicatorRename(BaseModel):
    """Indicateur renomme entre deux trimestres."""

    model_config = ConfigDict(extra="forbid")

    previous: str
    current: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class IndicatorDiffResponse(BaseModel):
    """Schema de reponse strict pour l'appel GPT de diff d'indicateurs."""

    model_config = ConfigDict(extra="forbid")

    indicators_added: list[IndicatorChange] = Field(default_factory=list)
    indicators_removed: list[IndicatorChange] = Field(default_factory=list)
    indicators_renamed: list[IndicatorRename] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Footnote diff
# ---------------------------------------------------------------------------


class FootnoteChange(BaseModel):
    """Note de bas de page ajoutee ou supprimee."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class FootnoteRename(BaseModel):
    """Note de bas de page modifiee (reformulation materielle)."""

    model_config = ConfigDict(extra="forbid")

    previous_id: str
    current_id: str
    previous_text: str
    current_text: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class FootnoteDiffResponse(BaseModel):
    """Schema de reponse strict pour l'appel GPT de diff de notes de bas de page."""

    model_config = ConfigDict(extra="forbid")

    footnotes_added: list[FootnoteChange] = Field(default_factory=list)
    footnotes_removed: list[FootnoteChange] = Field(default_factory=list)
    footnotes_renamed: list[FootnoteRename] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Inspector artifact filter
# ---------------------------------------------------------------------------


class InspectorVerdict(BaseModel):
    """Verdict pour un indicateur individuel dans la passe d'inspection."""

    model_config = ConfigDict(extra="forbid")

    value: str
    verdict: str = Field(description="'real' or 'artifact'")
    reason: str = ""


class InspectorArtifactPair(BaseModel):
    """Paire appariee d'indicateurs artefactuels (supprime / ajoute)."""

    model_config = ConfigDict(extra="forbid")

    removed: str
    added: str
    reason: str = ""


class InspectorRenamedVerdict(BaseModel):
    """Verdict pour un indicateur renomme dans la passe d'inspection."""

    model_config = ConfigDict(extra="forbid")

    previous: str
    current: str
    verdict: str = Field(description="'real' or 'artifact'")
    reason: str = ""


class InspectorResponse(BaseModel):
    """Schema de reponse strict pour l'inspecteur d'artefacts post-diff."""

    model_config = ConfigDict(extra="forbid")

    added_verdicts: list[InspectorVerdict] = Field(default_factory=list)
    removed_verdicts: list[InspectorVerdict] = Field(default_factory=list)
    renamed_verdicts: list[InspectorRenamedVerdict] = Field(default_factory=list)
    artifact_pairs: list[InspectorArtifactPair] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Visual Sanity Check
# ---------------------------------------------------------------------------


class SanityCheckItem(BaseModel):
    """Verdict pour un element de diff dans la verification visuelle."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Exact opaque identifier provided in the input items list")
    item_type: str = Field(
        description=(
            "'indicator_added', 'indicator_removed', 'indicator_renamed', "
            "'footnote_added', 'footnote_removed', 'footnote_modified', "
            "'table_added', or 'table_removed'"
        )
    )
    verdict: str = Field(description="'confirmed' or 'rejected'")
    reason: str = ""


class VisualSanityCheckResponse(BaseModel):
    """Schema de reponse strict pour l'agent de verification visuelle."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[SanityCheckItem] = Field(default_factory=list)
    overall_assessment: str = Field(
        default="",
        description="Brief overall assessment of the diff quality",
    )
