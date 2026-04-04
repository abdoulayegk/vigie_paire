"""Match Inspector — pair-level GenAI verification (Stage 1.5).

After Stage 1 batch matching, this module reviews each matched pair individually
with a focused 1-on-1 GPT call. The reduced context window eliminates the
batch-level confusion that causes hallucinated matches (e.g., GPT claiming
"indicators match exactly" when they share 0% overlap).

Rejected pairs are returned to the unresolved pool for Stage 2 recovery.

Same injection pattern as comparison_matching.py: call_openai_json is injected
so that monkeypatching "vigilance.compare_gpt._call_openai_json" continues to work.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from vigilance.models.comparison_models import MatchInspectorResponse

logger = logging.getLogger(__name__)

MATCH_INSPECTOR_SYSTEM_PROMPT = """\
You are a rigorous quality inspector for financial table matching in Canadian bank quarterly reports.

You receive a SINGLE matched pair: one Previous Quarter (PQ) table and one Current Quarter (CQ) table.
A first-pass analyst claims they are the same business table across quarters.

Your job: VERIFY or REJECT this match by examining the actual content.

VERIFICATION PROCEDURE (follow strictly):
1. LIST every indicator label that appears in BOTH tables (exact text or obvious semantic equivalent, e.g. "Total des actifs" ≈ "Total de l'actif"). Put these in `shared_indicators`.
2. COUNT the shared indicators vs total unique indicators across both tables.
3. CHECK the structural signals:
   - Do the titles match or are they semantically equivalent?
   - Do the column headers share the same structure (ignoring date shifts)?
   - Is the row_count difference reasonable (≤ 3 rows)?
   - Are the sections logically compatible?
   - Is the footnote_count difference < 5?

DECISION RULES:
- If shared_indicators contains >= 3 indicators AND covers >= 40% of the smaller table's indicators → "confirmed"
- If the titles are an exact match AND headers align AND row_count difference ≤ 3 → "confirmed" (even with few shared indicators, as indicator extraction can be noisy)
- If shared_indicators is EMPTY (0 overlap) → "rejected" (ALWAYS, regardless of title similarity)
- If first_indicator of one table is "Actif" and the other is "Passif" → "rejected"
- If footnote_count difference >= 5 AND shared_indicators < 3 → "rejected"

CRITICAL ANTI-HALLUCINATION RULE:
You MUST list the actual shared indicator labels in `shared_indicators`. Do NOT fabricate indicator names. If you cannot find real shared indicators in the provided data, the list must be empty and the verdict must be "rejected".

Output must follow the response_schema strictly.
"""


def _build_single_pair_payload(
    pair_with_card: dict[str, Any],
) -> dict[str, Any]:
    """Build the user prompt for a single pair to inspect."""
    return {
        "pair_to_inspect": {
            "previous_table": pair_with_card["previous_card"],
            "current_table": pair_with_card["current_card"],
            "original_confidence": pair_with_card["match_confidence"],
            "original_reason": pair_with_card["reason"],
        }
    }


def _inspect_matched_pairs(
    matched_pairs: list[dict[str, Any]],
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect all Stage 1 matched pairs and return confirmed/rejected lists.

    Returns a dict with:
      - confirmed_pairs: list of original pair dicts that passed inspection
      - rejected_pairs: list of original pair dicts that failed inspection
      - inspector_verdicts: raw verdicts for logging/debugging
      - inspection_stats: summary counters
    """
    if not matched_pairs:
        return {
            "confirmed_pairs": [],
            "rejected_pairs": [],
            "inspector_verdicts": [],
            "inspection_stats": {
                "total_inspected": 0,
                "confirmed": 0,
                "rejected": 0,
            },
        }

    # Build lookup maps for cards
    prev_card_map = {card["table_id"]: card for card in previous_cards}
    cur_card_map = {card["table_id"]: card for card in current_cards}

    # Enrich pairs with their full table cards
    pairs_with_cards = []
    for pair in matched_pairs:
        prev_id = pair.get("previous_table_id", "")
        cur_id = pair.get("current_table_id", "")
        prev_card = prev_card_map.get(prev_id)
        cur_card = cur_card_map.get(cur_id)
        if not prev_card or not cur_card:
            logger.warning(
                "Inspector: skipping pair %s <-> %s — card not found", prev_id, cur_id
            )
            continue
        pairs_with_cards.append(
            {
                "previous_table_id": prev_id,
                "current_table_id": cur_id,
                "match_confidence": pair.get("match_confidence", 0.0),
                "reason": pair.get("reason", ""),
                "previous_card": prev_card,
                "current_card": cur_card,
            }
        )

    if not pairs_with_cards:
        return {
            "confirmed_pairs": list(matched_pairs),
            "rejected_pairs": [],
            "inspector_verdicts": [],
            "inspection_stats": {
                "total_inspected": 0,
                "confirmed": len(matched_pairs),
                "rejected": 0,
            },
        }

    logger.info(
        "Match Inspector: reviewing %d matched pairs (1 call per pair)",
        len(pairs_with_cards),
    )

    # --- Per-pair GPT calls (no batch confusion possible) ---
    all_verdicts: list[dict[str, Any]] = []
    api_errors: list[str] = []

    for item in pairs_with_cards:
        prev_id = item["previous_table_id"]
        cur_id = item["current_table_id"]
        user_payload = _build_single_pair_payload(item)

        try:
            result = call_openai_json(
                model=model,
                messages=[
                    {"role": "system", "content": MATCH_INSPECTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                max_completion_tokens=800,
                temperature=0.0,
                usage_recorder=usage_recorder,
                call_kind="match_inspector",
                response_model=MatchInspectorResponse,
            )
            all_verdicts.extend(result.get("verdicts", []))
        except Exception as exc:
            logger.warning(
                "Match Inspector: API call failed for %s <-> %s (%s) — keeping as confirmed",
                prev_id,
                cur_id,
                exc,
            )
            api_errors.append(f"{prev_id}<->{cur_id}: {exc}")

    # Parse verdicts from all per-pair calls
    verdicts = all_verdicts
    rejected_keys: set[tuple[str, str]] = set()
    for verdict in verdicts:
        prev_id = str(verdict.get("previous_table_id", "")).strip()
        cur_id = str(verdict.get("current_table_id", "")).strip()
        decision = str(verdict.get("verdict", "")).strip().lower()
        shared = verdict.get("shared_indicators", [])
        confidence = float(verdict.get("confidence", 0.0))
        reason = str(verdict.get("reason", ""))

        if decision == "rejected":
            rejected_keys.add((prev_id, cur_id))
            logger.info(
                "Match Inspector REJECTED: %s <-> %s (shared=%d, conf=%.2f) — %s",
                prev_id,
                cur_id,
                len(shared),
                confidence,
                reason,
            )
        else:
            logger.debug(
                "Match Inspector confirmed: %s <-> %s (shared=%d, conf=%.2f)",
                prev_id,
                cur_id,
                len(shared),
                confidence,
            )

    # Partition original pairs
    confirmed_pairs = []
    rejected_pairs = []
    for pair in matched_pairs:
        key = (
            pair.get("previous_table_id", ""),
            pair.get("current_table_id", ""),
        )
        if key in rejected_keys:
            rejected_pairs.append(pair)
        else:
            confirmed_pairs.append(pair)

    stats = {
        "total_inspected": len(pairs_with_cards),
        "confirmed": len(confirmed_pairs),
        "rejected": len(rejected_pairs),
    }
    logger.info(
        "Match Inspector: %d confirmed, %d rejected out of %d inspected",
        stats["confirmed"],
        stats["rejected"],
        stats["total_inspected"],
    )

    return {
        "confirmed_pairs": confirmed_pairs,
        "rejected_pairs": rejected_pairs,
        "inspector_verdicts": verdicts,
        "inspection_stats": stats,
    }
