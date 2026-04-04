"""Pydantic models for the GPT comparison pipeline (Structured Outputs).

These models define the strict response schemas used by:
- Devil's Advocate review
- Indicator diff
- Footnote diff
- Inspector artifact filter
- Visual Sanity Check

They replace the previous pattern of free-form ``dict[str, Any]`` parsing
with guaranteed-typed outputs via OpenAI Structured Outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class AnalystAssessment(BaseModel):
    """Analyst-facing assessment attached to every diff change."""

    model_config = ConfigDict(extra="forbid")

    relevance_level: int = Field(
        description="1=Critical/Regulatory, 2=High/Structural, 3=Low/Cosmetic"
    )
    justification: str = Field(
        description="Business impact justification for the analyst"
    )


# ---------------------------------------------------------------------------
# Match Inspector (Stage 1.5 — pair-level GenAI verification)
# ---------------------------------------------------------------------------


class MatchInspectorVerdict(BaseModel):
    """Verdict for a single matched pair reviewed by the inspector."""

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
    """Strict response schema for the Match Inspector batch review."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[MatchInspectorVerdict]


# ---------------------------------------------------------------------------
# Devil's Advocate
# ---------------------------------------------------------------------------


class DevilAdvocateMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    match_confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class DevilAdvocateConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    verdict: str = Field(description="'confirmed'")


class DevilAdvocateContestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    current_table_id: str
    verdict: str = Field(description="'contested'")
    reason: str


class DevilAdvocateResponse(BaseModel):
    """Strict response schema for the Devil's Advocate second-opinion review."""

    model_config = ConfigDict(extra="forbid")

    new_matches: list[DevilAdvocateMatch] = Field(default_factory=list)
    confirmed_low_confidence: list[DevilAdvocateConfirmation] = Field(
        default_factory=list
    )
    contested_pairs: list[DevilAdvocateContestation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicator diff
# ---------------------------------------------------------------------------


class IndicatorChange(BaseModel):
    """An indicator added or removed."""

    model_config = ConfigDict(extra="forbid")

    value: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class IndicatorRename(BaseModel):
    """An indicator renamed between quarters."""

    model_config = ConfigDict(extra="forbid")

    previous: str
    current: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class IndicatorDiffResponse(BaseModel):
    """Strict response schema for indicator diff GPT call."""

    model_config = ConfigDict(extra="forbid")

    indicators_added: list[IndicatorChange] = Field(default_factory=list)
    indicators_removed: list[IndicatorChange] = Field(default_factory=list)
    indicators_renamed: list[IndicatorRename] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Footnote diff
# ---------------------------------------------------------------------------


class FootnoteChange(BaseModel):
    """A footnote added or removed."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class FootnoteRename(BaseModel):
    """A footnote renamed (materially revised wording)."""

    model_config = ConfigDict(extra="forbid")

    previous_id: str
    current_id: str
    previous_text: str
    current_text: str
    reason: str = ""
    analyst_assessment: AnalystAssessment


class FootnoteDiffResponse(BaseModel):
    """Strict response schema for footnote diff GPT call."""

    model_config = ConfigDict(extra="forbid")

    footnotes_added: list[FootnoteChange] = Field(default_factory=list)
    footnotes_removed: list[FootnoteChange] = Field(default_factory=list)
    footnotes_renamed: list[FootnoteRename] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Inspector artifact filter
# ---------------------------------------------------------------------------


class InspectorVerdict(BaseModel):
    """Verdict for a single indicator in the inspector pass."""

    model_config = ConfigDict(extra="forbid")

    value: str
    verdict: str = Field(description="'real' or 'artifact'")
    reason: str = ""


class InspectorArtifactPair(BaseModel):
    """A matched pair of artifact indicators (removed ↔ added)."""

    model_config = ConfigDict(extra="forbid")

    removed: str
    added: str
    reason: str = ""


class InspectorResponse(BaseModel):
    """Strict response schema for the post-diff artifact inspector."""

    model_config = ConfigDict(extra="forbid")

    added_verdicts: list[InspectorVerdict] = Field(default_factory=list)
    removed_verdicts: list[InspectorVerdict] = Field(default_factory=list)
    artifact_pairs: list[InspectorArtifactPair] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Visual Sanity Check
# ---------------------------------------------------------------------------


class SanityCheckItem(BaseModel):
    """Verdict for a single diff item in the visual sanity check."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(
        description="Exact opaque identifier provided in the input items list"
    )
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
    """Strict response schema for the Visual Sanity Check agent."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[SanityCheckItem] = Field(default_factory=list)
    overall_assessment: str = Field(
        default="",
        description="Brief overall assessment of the diff quality",
    )
