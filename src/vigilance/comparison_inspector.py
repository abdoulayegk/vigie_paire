"""Match Inspector -- verification GenAI paire par paire (Etape 1.5).

Apres l'appariement par lot de l'Etape 1, ce module reexamine chaque paire
individuellement via un appel GPT dedie. La fenetre de contexte reduite elimine
la confusion par lot qui cause des appariements hallucines (par ex., GPT affirmant
que les indicateurs correspondent exactement alors qu'ils partagent 0 % de recouvrement).

Les paires rejetees sont retournees au bassin non resolu pour la recuperation en Etape 2.

Meme patron d'injection que comparison_matching.py : ``call_openai_json`` est injecte
afin que le monkeypatching de ``vigilance.compare_gpt._call_openai_json`` continue de
fonctionner.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any, Callable

from vigilance.models.comparison_models import SinglePairInspectorResponse

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

GROUND TRUTH ANCHOR:
The payload includes `system_computed_facts` with pre-computed metrics based on exact string matching.
- `exact_indicator_overlap`: number of indicator labels that appear verbatim in BOTH tables.
- `shared_indicator_names`: the actual verbatim labels found in both tables.
- `pq_indicator_count` / `cq_indicator_count`: total indicators per side.
- `row_count_delta`: absolute difference in row_count.
- `title_exact_match`: whether the titles match exactly (case-insensitive).
These numbers are OBJECTIVE TRUTH. Use them as your starting point:
- If `exact_indicator_overlap` is 0, you may ONLY add semantic equivalents to `shared_indicators` if you can explicitly name the PQ and CQ labels that mean the same thing. Simple thematic proximity (e.g. "liquidité" appearing in both summaries) is NOT a semantic equivalence.
- If `row_count_delta` > 10 AND `exact_indicator_overlap` < 3, this is an extremely strong rejection signal.
- If `exact_indicator_overlap` >= 5 AND `row_count_delta` <= 3, this is a strong confirmation signal.

Output must follow the response_schema strictly.
Return a SINGLE verdict object (not a list). Do NOT include table IDs in your response — the system already knows which pair you are inspecting.
"""


def _compute_pair_facts(prev_card: dict[str, Any], cur_card: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute objective metrics for a single matched pair."""
    pq_inds = [
        unicodedata.normalize("NFC", str(i).strip().lower())
        for i in (prev_card.get("indicators") or [])
        if str(i).strip()
    ]
    cq_inds = [
        unicodedata.normalize("NFC", str(i).strip().lower())
        for i in (cur_card.get("indicators") or [])
        if str(i).strip()
    ]
    pq_set = set(pq_inds)
    cq_set = set(cq_inds)
    exact_overlap = sorted(pq_set & cq_set)

    pq_title = str(prev_card.get("title", "")).strip().lower()
    cq_title = str(cur_card.get("title", "")).strip().lower()
    title_match = pq_title == cq_title and pq_title != ""

    pq_rows = int(prev_card.get("row_count", 0) or 0)
    cq_rows = int(cur_card.get("row_count", 0) or 0)

    return {
        "exact_indicator_overlap": len(exact_overlap),
        "pq_indicator_count": len(pq_inds),
        "cq_indicator_count": len(cq_inds),
        "row_count_delta": abs(pq_rows - cq_rows),
        "title_exact_match": title_match,
        "shared_indicator_names": exact_overlap,
    }


def _build_single_pair_payload(
    pair_with_card: dict[str, Any],
) -> dict[str, Any]:
    """Construit le prompt utilisateur pour une paire unique a inspecter."""
    facts = _compute_pair_facts(pair_with_card["previous_card"], pair_with_card["current_card"])
    return {
        "pair_to_inspect": {
            "previous_table": pair_with_card["previous_card"],
            "current_table": pair_with_card["current_card"],
            "original_confidence": pair_with_card["match_confidence"],
            "original_reason": pair_with_card["reason"],
            "system_computed_facts": facts,
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
    """Inspecte toutes les paires appariees de l'Etape 1 et retourne les listes confirmees/rejetees.

    Args:
        matched_pairs: Paires appariees issues de l'Etape 1.
        previous_cards: Fiches resumees des tableaux du trimestre precedent.
        current_cards: Fiches resumees des tableaux du trimestre courant.
        model: Identifiant du modele OpenAI.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.

    Returns:
        Dictionnaire contenant ``confirmed_pairs``, ``rejected_pairs``,
        ``inspector_verdicts`` et ``inspection_stats``.
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
            logger.warning("Inspector: skipping pair %s <-> %s — card not found", prev_id, cur_id)
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
                response_model=SinglePairInspectorResponse,
            )
            # Scalar response — inject the IDs we already know
            result["previous_table_id"] = prev_id
            result["current_table_id"] = cur_id
            all_verdicts.append(result)
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
