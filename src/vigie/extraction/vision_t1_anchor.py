"""T-1 Anchoring: compare current extraction against previous period to detect extraction drift.

When the indicator count between two periods diverges by more than a configurable
threshold (default 20%), a GPT-based judgment call determines whether the difference
is a real structural change or a likely extraction error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ROW_COUNT_DIFF_THRESHOLD = 0.20  # 20% triggers GPT check


@dataclass
class AnchorResult:
    """Result of a T-1 anchoring check for a single table pair."""

    table_id: str
    likely_extraction_error: bool = False
    explanation: str = ""
    skipped: bool = False
    skip_reason: str = ""
    current_count: int = 0
    previous_count: int = 0
    diff_ratio: float = 0.0


def anchor_against_previous(
    *,
    table_id: str,
    table_title: str,
    current_indicators: list[str],
    previous_indicators: list[str],
    model: str = "gpt-4o",
    api_key: str | None = None,
    diff_threshold: float = _ROW_COUNT_DIFF_THRESHOLD,
) -> AnchorResult:
    """Compare current indicators against T-1 and flag likely extraction errors.

    Only calls GPT when the row count difference exceeds ``diff_threshold``.
    Always fails gracefully — returns a skipped result on any exception.

    Args:
        table_id: Identifier of the table being checked.
        table_title: Title of the table for context.
        current_indicators: Indicator labels from the current period extraction.
        previous_indicators: Indicator labels from the previous period extraction.
        model: OpenAI model to use for judgment.
        api_key: OpenAI API key (falls back to env var).
        diff_threshold: Minimum relative difference to trigger GPT check.

    Returns:
        AnchorResult with the assessment.
    """
    current_count = len(current_indicators)
    previous_count = len(previous_indicators)

    if previous_count == 0 and current_count == 0:
        return AnchorResult(
            table_id=table_id,
            skipped=True,
            skip_reason="both_empty",
            current_count=current_count,
            previous_count=previous_count,
        )

    if previous_count == 0:
        return AnchorResult(
            table_id=table_id,
            skipped=True,
            skip_reason="no_previous_indicators",
            current_count=current_count,
            previous_count=previous_count,
        )

    max_count = max(current_count, previous_count)
    diff_ratio = abs(current_count - previous_count) / max_count

    if diff_ratio <= diff_threshold:
        return AnchorResult(
            table_id=table_id,
            skipped=True,
            skip_reason="below_threshold",
            current_count=current_count,
            previous_count=previous_count,
            diff_ratio=round(diff_ratio, 3),
        )

    # GPT-based judgment
    try:
        from pydantic import BaseModel, ConfigDict, Field

        class AnchorJudgment(BaseModel):
            """Sortie GPT validée : décision sur la nature d'une divergence d'ancrage."""

            model_config = ConfigDict(extra="forbid")
            likely_extraction_error: bool = Field(
                ...,
                description="True if the difference is likely due to an extraction error, False if it reflects a real structural change.",
            )
            explanation: str = Field(
                ...,
                description="Brief explanation of why this is or is not an extraction error.",
            )

        if api_key is None:
            from vigie.support.utils.genai import get_openai_api_key
            api_key = get_openai_api_key()

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Build concise indicator lists for GPT
        prev_sample = previous_indicators[:30]
        curr_sample = current_indicators[:30]

        prompt = (
            f"You are auditing a Canadian bank quarterly report table extraction.\n\n"
            f"Table: \"{table_title}\"\n"
            f"Previous period had {previous_count} indicators, current period has {current_count}.\n"
            f"Difference: {abs(current_count - previous_count)} indicators "
            f"({'more' if current_count > previous_count else 'fewer'} in current).\n\n"
            f"Previous indicators (first {len(prev_sample)}):\n"
            + "\n".join(f"  - {ind}" for ind in prev_sample)
            + f"\n\nCurrent indicators (first {len(curr_sample)}):\n"
            + "\n".join(f"  - {ind}" for ind in curr_sample)
            + "\n\nIs this difference:\n"
            "(A) A real structural change in the table (new rows added, rows removed due to business events), or\n"
            "(B) A likely extraction error (the extractor missed rows, duplicated sections, or captured wrong content)?\n\n"
            "Consider: bank tables rarely change by more than 20% between quarters. "
            "Missing section headers, duplicate group headings, or absent sub-items suggest extraction error."
        )

        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial data quality auditor."},
                {"role": "user", "content": prompt},
            ],
            response_format=AnchorJudgment,
            temperature=0.0,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            return AnchorResult(
                table_id=table_id,
                skipped=True,
                skip_reason="gpt_parse_failed",
                current_count=current_count,
                previous_count=previous_count,
                diff_ratio=round(diff_ratio, 3),
            )

        return AnchorResult(
            table_id=table_id,
            likely_extraction_error=parsed.likely_extraction_error,
            explanation=parsed.explanation,
            current_count=current_count,
            previous_count=previous_count,
            diff_ratio=round(diff_ratio, 3),
        )

    except Exception as exc:
        logger.warning("T-1 anchor check failed for %s (non-fatal): %s", table_id, exc)
        return AnchorResult(
            table_id=table_id,
            skipped=True,
            skip_reason=f"exception: {type(exc).__name__}",
            current_count=current_count,
            previous_count=previous_count,
            diff_ratio=round(diff_ratio, 3),
        )
