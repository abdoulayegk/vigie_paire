"""Revue en second avis (Devil's Advocate) pour le pipeline de comparaison.

Extrait de compare_gpt.py. Ce dernier re-exporte tous les noms de ce module
afin que les imports existants restent valides.

Meme patron d'injection que comparison_matching.py : ``call_openai_json`` est
injecte pour que le monkeypatching de ``vigilance.compare_gpt._call_openai_json``
continue de fonctionner.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from vigilance.models.comparison_models import DevilAdvocateResponse

logger = logging.getLogger(__name__)

DEVIL_ADVOCATE_SYSTEM_PROMPT = """\
You are a second-opinion financial table matcher reviewing the work of a first analyst.

You receive:
1. Tables from the Previous Quarter (PQ) that the first analyst marked as "removed" (i.e., not matched to anything).
2. Tables from the Current Quarter (CQ) that the first analyst marked as "added" (i.e., not matched to anything).
3. Optionally, pairs the first analyst matched with LOW confidence (< 0.90).

Your mission:
- Look for pairs among the "removed" PQ and "added" CQ tables that the first analyst MISSED.
- IMPORTANT: Extraction can be TRUNCATED. A CQ table with only 1-2 indicators could be a truncated version of a PQ table with 17 indicators. Do NOT reject a match solely because of a large row_count difference. Focus on whether the EXISTING indicators in the shorter table appear in the longer one.
- If two tables share even a few matching indicator labels, column headers, or footnotes, they are likely the same table with a truncated extraction.
- For low-confidence pairs: confirm or contest them. If you agree, say "confirmed". If you disagree, say "contested".

ABSOLUTE DISQUALIFIERS (check BEFORE proposing any new match):
- TABLE ORIENTATION: Check `first_indicator`. NEVER propose a match between a table whose first_indicator is "Actif" (or starts with "Actif") and one whose first_indicator is "Passif" (or "Passif et capitaux propres"). These are fundamentally different business entities even if they share a few generic indicators (e.g., "Instruments financiers dérivés", "Créances sur cartes de crédit") or a similar theme (e.g., both about "échéances").
- FOOTNOTE COUNT: NEVER propose a match if `footnote_count` differs by >= 5 between PQ and CQ. Tables about the same business entity maintain near-identical footnote counts across quarters. A gap of 10 vs 1 means different tables.
- SHARED GENERIC INDICATORS ARE NOT ENOUGH: Indicators like "Autres", "Provisions", "Instruments financiers dérivés" appear in many unrelated tables. A match requires the CORE indicator structure to align, not just 2-3 generic rows.

OUTPUT FORMAT (JSON):
{
  "new_matches": [
    {
      "previous_table_id": "tbl_pXXX_iYY",
      "current_table_id": "tbl_pXXX_iYY",
      "match_confidence": 0.75,
      "reason": "Both tables share indicators X, Y, Z. CQ version appears truncated."
    }
  ],
  "confirmed_low_confidence": [
    {"previous_table_id": "...", "current_table_id": "...", "verdict": "confirmed"}
  ],
  "contested_pairs": [
    {"previous_table_id": "...", "current_table_id": "...", "verdict": "contested", "reason": "..."}
  ]
}

If you find NO new matches and have nothing to contest, return:
{"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []}
"""


def _devil_advocate_review(
    tables_added_cards: list[dict[str, Any]],
    tables_removed_cards: list[dict[str, Any]],
    low_confidence_pairs: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute une revue en second avis sur les tableaux non apparies et a faible confiance.

    Args:
        tables_added_cards: Fiches descriptives des tableaux CQ non apparies.
        tables_removed_cards: Fiches descriptives des tableaux PQ non apparies.
        low_confidence_pairs: Paires appariees avec une confiance inferieure a 0.90.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Fonction injectable pour les appels OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer la consommation API.

    Returns:
        Dictionnaire contenant ``new_matches``, ``confirmed_low_confidence``
        et ``contested_pairs``.
    """
    if not tables_added_cards and not tables_removed_cards and not low_confidence_pairs:
        logger.info(
            "Devil's Advocate: nothing to review (all matched with high confidence)"
        )
        return {
            "new_matches": [],
            "confirmed_low_confidence": [],
            "contested_pairs": [],
        }

    user_payload = {
        "unmatched_previous_tables": tables_removed_cards,
        "unmatched_current_tables": tables_added_cards,
        "low_confidence_pairs": low_confidence_pairs,
    }

    logger.info(
        "Devil's Advocate: reviewing %d unmatched PQ + %d unmatched CQ + %d low-confidence pairs",
        len(tables_removed_cards),
        len(tables_added_cards),
        len(low_confidence_pairs),
    )

    try:
        result = call_openai_json(
            model=model,
            messages=[
                {"role": "system", "content": DEVIL_ADVOCATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            usage_recorder=usage_recorder,
            call_kind="devil_advocate",
            response_model=DevilAdvocateResponse,
        )
    except Exception as exc:
        logger.warning("Devil's Advocate: API call failed: %s", exc)
        return {
            "new_matches": [],
            "confirmed_low_confidence": [],
            "contested_pairs": [],
        }

    logger.info(
        "Devil's Advocate: found %d new matches, %d confirmed, %d contested",
        len(result.get("new_matches", [])),
        len(result.get("confirmed_low_confidence", [])),
        len(result.get("contested_pairs", [])),
    )

    return result
