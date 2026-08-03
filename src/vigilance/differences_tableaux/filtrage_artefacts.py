"""Inspection et filtrage des artefacts d extraction du diff."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from vigilance.differences_tableaux.normalisation_elements import _table_context
from vigilance.models.comparison_models import InspectorResponse

logger = logging.getLogger(__name__)


INSPECTOR_SYSTEM_PROMPT = """
You are a Senior Risk Analyst quality-control inspector for a banking table diff pipeline.

You receive a diff produced by a first-pass GPT model that compared two already-matched
canonical banking tables from adjacent quarterly reports. Your ONLY job is to classify
each indicator flagged as "added", "removed", or "renamed" as either a REAL semantic
change or an extraction ARTIFACT that should be suppressed.

CRITICAL PRINCIPLE — artifact vs real:
An indicator should ONLY be classified as "artifact" when you can clearly identify its
counterpart on the opposite side (added↔removed) and the ONLY difference between them
is extraction noise (footnote markers, unicode, OCR, whitespace, hierarchical prefixes).
If a removed indicator has NO plausible counterpart among added indicators (and vice versa),
it MUST be "real".

For RENAMED indicators: a rename is "artifact" when the previous and current names refer
to the EXACT same business concept and differ ONLY due to extraction noise. Common patterns:
- Hierarchical prefix added/removed: 'CET1' vs 'Ratios des fonds propres – CET1'
- Footnote markers: 'catégorie 1 (3)' vs 'catégorie 1'
- Superscript vs parenthesized: 'titrisation²' vs 'titrisation(2)'
- Punctuation: em dash vs hyphen, trailing colon, comma
- Minor OCR variance: 'RSLT' vs 'RSLLT'

Common artifact patterns for added/removed you MUST catch:
1. Footnote marker variance: 'Goodwill³' vs 'Goodwill', 'catégorie 1 (4)' vs 'catégorie 1'.
2. Footnote reference shift on the SAME business concept: 'Série 3⁶' vs 'Série 3⁴' — same
   indicator name, only the superscript reference number changed. But 'Série 5' vs 'Série 7'
   are DIFFERENT business entities — never pair them.
3. Disambiguation label / date-prefix / bloc-suffix variance (CRITICAL):
   Tables with repeated sub-sections are often extracted with disambiguation tags.
   One quarter may use '(bloc 2)', '(bloc 3)' suffixes; another may use date prefixes
   like '31 octobre 2025 – ', '31 janvier 2025 – ', etc. To detect these artifacts:
     a) Strip any leading date pattern (e.g. '31 octobre 2025 – ', '30 avril 2025 – ').
     b) Strip any trailing '(bloc N)' tag.
     c) Compare the remaining core indicator name.
   If the core names match, mark BOTH as artifact and pair them.
4. Unicode apostrophe / quote variance: '\u2019' (U+2019) vs "'" (U+0027), « » vs ".
5. Minor OCR / whitespace / punctuation noise: trailing dots, extra spaces, accented char variance.
6. Hierarchical parent-prefix variance: Banking tables have group headers (e.g. "Ratios des fonds propres")
   followed by sub-indicators (e.g. "CET1"). One extraction may fully qualify sub-indicators with the
   group header prefix (e.g. "Ratios des fonds propres – CET1") while the other extracts them bare
   (e.g. "CET1"). If the bare name is the suffix of the qualified name and both occupy the same
   structural position (pos), mark as artifact.
7. Punctuation-only differences: "–" vs "-", trailing colons, comma presence/absence — these are
   extraction noise, not real changes. Mark as artifact.

Rules:
- For each indicator in added_indicators, removed_indicators, AND renamed_indicators, output a verdict.
- verdict MUST be one of: "real" or "artifact".
- An indicator can ONLY be "artifact" if it has a clear extraction-noise counterpart on the
  other side (for added/removed) or if the rename is pure extraction noise (for renamed).
- A removed indicator with no matching added indicator is ALWAYS "real".
  An added indicator with no matching removed indicator is ALWAYS "real".
- If an added indicator and a removed indicator are the SAME business concept (just extraction noise),
  mark BOTH as "artifact" and pair them in artifact_pairs.
- For renamed indicators: if previous and current differ only by extraction noise, mark as "artifact".
  If the rename reflects a genuine business concept change, mark as "real".
- CONTEXTUAL CROSS-CHECK: When unsure whether an added/removed pair are the same indicator,
  look at their NEIGHBORS in the previous_table and current_table indicator lists. If the
  indicators directly above and below are the same on both sides, it strongly confirms
  the flagged indicators occupy the same structural position and are the same business concept.
- When in doubt, mark as "real". Missing a genuine change is worse than surfacing a false positive.
- Cross-check against the provided previous_table and current_table indicator lists. If an
  indicator genuinely does not appear in the other quarter's list, it is "real".

Output must be valid JSON following the response_schema.
"""


def _inspect_diff_artifacts_gpt(
    indicator_diff: dict[str, Any],
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Filtre les artefacts d'extraction du diff d'indicateurs via un appel GPT Inspector.

    Audits added, removed, AND renamed indicators for extraction artifacts.

    Returns:
        Copie nettoyee de *indicator_diff* dont les artefacts ont ete retires.
    """
    added = list(indicator_diff.get("indicators_added", []) or [])
    removed = list(indicator_diff.get("indicators_removed", []) or [])
    renamed = list(indicator_diff.get("indicators_renamed", []) or [])

    # Nothing to inspect — skip the call entirely
    if not added and not removed and not renamed:
        return indicator_diff

    prev_ctx = _table_context(previous_table)
    curr_ctx = _table_context(current_table)

    prompt: dict[str, Any] = {
        "task": (
            "Inspect each added/removed/renamed indicator from a first-pass diff and classify "
            "it as a REAL semantic change or an extraction ARTIFACT."
        ),
        "rules": [
            "Return JSON only and strictly follow the response_schema.",
            "For each entry in added_indicators, removed_indicators, AND renamed_indicators, provide a verdict: 'real' or 'artifact'.",
            "If an added and a removed indicator refer to the same business concept (extraction noise), "
            "mark BOTH as 'artifact' and list them in artifact_pairs.",
            "For renamed indicators: if previous and current differ only by extraction noise "
            "(footnote markers, hierarchical prefix, punctuation, OCR), mark as 'artifact'.",
            "Be strict: when in doubt, prefer 'artifact'. False positives are worse than missing a real change.",
        ],
        "response_schema": {
            "added_verdicts": [
                {
                    "value": "string",
                    "verdict": "'real' or 'artifact'",
                    "reason": "string",
                }
            ],
            "removed_verdicts": [
                {
                    "value": "string",
                    "verdict": "'real' or 'artifact'",
                    "reason": "string",
                }
            ],
            "renamed_verdicts": [
                {
                    "previous": "string",
                    "current": "string",
                    "verdict": "'real' or 'artifact'",
                    "reason": "string",
                }
            ],
            "artifact_pairs": [{"removed": "string", "added": "string", "reason": "string"}],
        },
        "added_indicators": [item.get("value", "") for item in added],
        "removed_indicators": [item.get("value", "") for item in removed],
        "renamed_indicators": [{"previous": r.get("previous", ""), "current": r.get("current", "")} for r in renamed],
        "previous_table": {
            "title": prev_ctx["title"],
            "indicators": prev_ctx["indicators"],
        },
        "current_table": {
            "title": curr_ctx["title"],
            "indicators": curr_ctx["indicators"],
        },
    }

    data = call_openai_json(
        model=model,
        messages=[
            {"role": "system", "content": INSPECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        usage_recorder=usage_recorder,
        call_kind="inspect_artifacts",
        response_model=InspectorResponse,
    )

    # --- Parse verdicts and filter ---
    try:
        artifact_added = {
            v["value"] for v in data.get("added_verdicts", []) if v.get("verdict", "").lower() == "artifact"
        }
        artifact_removed = {
            v["value"] for v in data.get("removed_verdicts", []) if v.get("verdict", "").lower() == "artifact"
        }
        # Build set of artifact renames keyed by (previous, current)
        artifact_renamed: set[tuple[str, str]] = set()
        for v in data.get("renamed_verdicts", []):
            if v.get("verdict", "").lower() == "artifact":
                artifact_renamed.add((v.get("previous", ""), v.get("current", "")))
    except (KeyError, TypeError, AttributeError):
        logger.warning("Inspector returned malformed response — skipping artifact filtering.")
        return indicator_diff

    filtered_added = [item for item in added if str(item.get("value", "")).strip() not in artifact_added]
    filtered_removed = [item for item in removed if str(item.get("value", "")).strip() not in artifact_removed]
    filtered_renamed = [
        item for item in renamed if (item.get("previous", ""), item.get("current", "")) not in artifact_renamed
    ]

    n_filtered = (
        (len(added) - len(filtered_added))
        + (len(removed) - len(filtered_removed))
        + (len(renamed) - len(filtered_renamed))
    )
    if n_filtered:
        logger.info(
            "Inspector filtered %d artifact(s) from indicator diff.",
            n_filtered,
        )

    return {
        **indicator_diff,
        "indicators_added": filtered_added,
        "indicators_removed": filtered_removed,
        "indicators_renamed": filtered_renamed,
    }
