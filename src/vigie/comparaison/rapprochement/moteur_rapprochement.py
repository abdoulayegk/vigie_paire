"""Execution du rapprochement primaire, de la recuperation et du flux RBC."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from vigie.comparaison.inspector import _inspect_matched_pairs
from vigie.comparaison.io import _coerce_int, table_view_as_dict
from vigie.comparaison.rapprochement.contrats import (
    _MATCHING_VALIDATION_ATTEMPTS,
    MATCHING_ADJUDICATOR_SYSTEM_PROMPT,
    MATCHING_REPAIR_SYSTEM_PROMPT,
    PRIMARY_MATCH_SYSTEM_PROMPT,
    RECOVERY_MATCH_SYSTEM_PROMPT,
    _MatchingValidationError,
)
from vigie.comparaison.rapprochement.correction_reponses import (
    _analyze_matching_candidate,
    _build_matching_fail_soft_response,
    _build_matching_repair_prompt,
    _build_matching_repair_response_model,
    _merge_matching_repair_response,
)
from vigie.comparaison.rapprochement.etat import MatchingResult, MatchingState
from vigie.comparaison.rapprochement.normalisation_reponses import (
    _empty_matching_result,
    _matching_decisions_to_pairs,
    _matching_decisions_to_table_refs,
    _normalize_matching_response,
    _normalize_matching_warnings,
    _pairs_from_dicts,
    _refs_from_dicts,
    _sort_matched_pairs,
)
from vigie.comparaison.rbc_hybrid_matching import (
    partition_trusted_rbc_primary_pairs,
    run_rbc_hybrid_recovery,
)
from vigie.support.models.comparison_models import (
    PrimaryMatchResponse,
    RecoveryMatchResponse,
)

logger = logging.getLogger(__name__)


def _build_matching_stage_prompt(
    *,
    stage: str,
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    current_ids: set[str],
    previous_ids: set[str],
    allowed_decisions: set[str],
    validation_feedback: str,
) -> dict[str, Any]:
    """Construit le prompt utilisateur JSON pour une etape d'appariement.

    Args:
        stage: Etape d'appariement (``"primary"`` ou ``"recovery"``).
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        current_ids: Identifiants des tableaux courants.
        previous_ids: Identifiants des tableaux precedents.
        allowed_decisions: Decisions autorisees pour cette etape.
        validation_feedback: Retour de validation a inclure si la tentative
            precedente a echoue.

    Returns:
        Dictionnaire JSON representant le prompt utilisateur complet.
    """
    decision_values = sorted(allowed_decisions)
    response_item: dict[str, Any] = {
        "current_table_id": "string",
        "decision": f"one_of_{decision_values}",
        "reason": "short explanation grounded in indicators, headers, row_count, title, table_summary, footnotes, section, and page only if needed",
    }
    if "matched" in allowed_decisions:
        response_item["previous_table_id"] = "string_required_when_decision_is_matched"
        response_item["match_confidence"] = "number_0_to_1_required_when_decision_is_matched"

    if stage == "primary":
        task = (
            "Perform a strict global reconciliation of business banking tables between the previous and current reports. "
            "For each current table, return either matched or unresolved."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "This is the strict precision-first pass: if any material doubt remains, return unresolved instead of forcing a match.",
            "Do not classify any table as added in this stage.",
        ]
    else:
        task = (
            "Resolve the remaining ambiguous current business tables against the remaining unused previous business tables. "
            "For each current table, return either matched or added."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "This is a recovery pass over unresolved current tables only.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "If the best remaining candidate is still ambiguous, return added instead of forcing a speculative match.",
            "Do not skip any current table.",
        ]

    prompt = {
        "stage": stage,
        "task": task,
        "rules": rules,
        "response_schema": {
            "current_table_decisions": [response_item],
            "warnings": ["string"],
        },
        "allowed_decisions": decision_values,
        "required_current_table_ids": sorted(current_ids),
        "allowed_previous_table_ids": sorted(previous_ids),
        "previous_tables": [table_view_as_dict(card) for card in previous_cards],
        "current_tables": [table_view_as_dict(card) for card in current_cards],
    }
    if validation_feedback:
        prompt["validation_feedback"] = validation_feedback
        prompt["rules"].append(
            "Your previous response was structurally invalid. Fix the listed validation problem and return a corrected JSON object."
        )
    return prompt


def _run_matching_stage(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    stage: str,
    allowed_decisions: set[str],
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Execute une etape d'appariement (primaire ou recuperation) avec validation.

    Appelle d'abord le matcher existant sans modifier son comportement. Si sa
    reponse est structurellement invalide, repare uniquement les decisions en
    conflit avec des identifiants PQ/CQ fermes, puis fait arbitrer les conflits
    persistants. Une derniere degradation non bloquante garde les decisions
    valides et classe le reliquat en ``unresolved`` ou ``added``.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        stage: Etape d'appariement (``"primary"`` ou ``"recovery"``).
        allowed_decisions: Decisions autorisees pour cette etape.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.
        consumed_previous_ids: Identifiants precedents deja consommes.

    Returns:
        Dictionnaire normalise contenant les decisions, avertissements et
        metriques de validation.

    """
    previous_ids = {card["table_id"] for card in previous_cards}
    current_ids = {card["table_id"] for card in current_cards}
    system_prompt = PRIMARY_MATCH_SYSTEM_PROMPT if stage == "primary" else RECOVERY_MATCH_SYSTEM_PROMPT
    validation_feedback = ""
    validation_retries_total = 0
    matching_validation_failures_total = 0
    matching_pairs_llm_duplicates_total = 0
    candidate_data: dict[str, Any] = {}

    for attempt in range(_MATCHING_VALIDATION_ATTEMPTS):
        if attempt == 0:
            prompt = _build_matching_stage_prompt(
                stage=stage,
                previous_cards=previous_cards,
                current_cards=current_cards,
                current_ids=current_ids,
                previous_ids=previous_ids,
                allowed_decisions=allowed_decisions,
                validation_feedback="",
            )
            stage_response_model = PrimaryMatchResponse if stage == "primary" else RecoveryMatchResponse
            data = call_openai_json(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False),
                    },
                ],
                usage_recorder=usage_recorder,
                call_kind="matching",
                response_model=stage_response_model,
            )
        else:
            repair_state = _analyze_matching_candidate(
                candidate_data,
                previous_ids=previous_ids,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                consumed_previous_ids=consumed_previous_ids,
            )
            repair_prompt = _build_matching_repair_prompt(
                stage=stage,
                repair_round=attempt,
                previous_cards=previous_cards,
                current_cards=current_cards,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                validation_feedback=validation_feedback,
                repair_state=repair_state,
            )
            response_model = _build_matching_repair_response_model(
                current_aliases=list(repair_prompt["required_repair_current_table_ids"]),
                previous_aliases=list(repair_prompt["allowed_previous_table_ids"]),
                allowed_decisions=set(repair_prompt["allowed_decisions"]),
            )
            repair_data = call_openai_json(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            MATCHING_REPAIR_SYSTEM_PROMPT if attempt == 1 else MATCHING_ADJUDICATOR_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(repair_prompt, ensure_ascii=False),
                    },
                ],
                usage_recorder=usage_recorder,
                call_kind="matching",
                response_model=response_model,
            )
            data = _merge_matching_repair_response(
                repair_data,
                repair_state=repair_state,
            )
        candidate_data = data
        try:
            normalized = _normalize_matching_response(
                data,
                previous_ids=previous_ids,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                consumed_previous_ids=consumed_previous_ids,
            )
            normalized["executed"] = True
            normalized["validation_retries_total"] = validation_retries_total
            normalized["matching_validation_failures_total"] = matching_validation_failures_total
            normalized["matching_pairs_llm_duplicates_total"] += matching_pairs_llm_duplicates_total
            return normalized
        except _MatchingValidationError as exc:
            validation_feedback = str(exc)
            validation_retries_total += 1
            matching_validation_failures_total += int(getattr(exc, "validation_failures", 1))
            matching_pairs_llm_duplicates_total += int(getattr(exc, "duplicate_count", 0))
            if attempt + 1 >= _MATCHING_VALIDATION_ATTEMPTS:
                fallback_data = _build_matching_fail_soft_response(
                    candidate_data,
                    stage=stage,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    allowed_decisions=allowed_decisions,
                    consumed_previous_ids=consumed_previous_ids,
                )
                normalized = _normalize_matching_response(
                    fallback_data,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    allowed_decisions=allowed_decisions,
                    consumed_previous_ids=consumed_previous_ids,
                )
                normalized["executed"] = True
                normalized["validation_retries_total"] = validation_retries_total
                normalized["matching_validation_failures_total"] = matching_validation_failures_total
                normalized["matching_pairs_llm_duplicates_total"] += matching_pairs_llm_duplicates_total
                return normalized

    raise RuntimeError("Unreachable matching validation loop")


def _match_tables(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    hybrid_recovery_enabled: bool = False,
    hybrid_embedding_model: str = "text-embedding-3-large",
    hybrid_top_k: int = 5,
    hybrid_min_confidence: float = 0.75,
    call_openai_embeddings: Callable[..., list[list[float]]] | None = None,
) -> MatchingResult:
    """Orchestre l'appariement complet en deux etapes avec inspection intermediaire.

    Etape 1 (primaire) : appariement strict precision-first.
    Etape 1.5 : inspection GenAI des paires appariees (rejet des faux positifs).
    Etape 2 (recuperation) : resolution des tableaux non resolus restants.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.
        hybrid_recovery_enabled: Active la recuperation hybride RBC opt-in.
        hybrid_embedding_model: Modele utilise pour la recherche de candidats.
        hybrid_top_k: Nombre de candidats embeddings presentes par tableau courant.
        hybrid_min_confidence: Confiance LLM minimale avant l'affectation globale.
        call_openai_embeddings: Callable injecte pour calculer les embeddings.

    Returns:
        ``MatchingResult`` contenant les paires appariees, tableaux ajoutes/supprimes,
        avertissements et metriques detaillees des deux etapes.
    """
    if not previous_cards and not current_cards:
        return _empty_matching_result()

    if not current_cards:
        tables_removed = [
            {
                "table_id": str(card.get("table_id", "") or ""),
                "reason": "No current business table was available for matching.",
            }
            for card in previous_cards
        ]
        return _empty_matching_result(tables_removed=tables_removed)

    state = MatchingState(
        previous_cards=list(previous_cards),
        current_cards=list(current_cards),
    )

    stage1 = _run_matching_stage(
        previous_cards,
        current_cards,
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    stage1_decisions = list(stage1.get("current_table_decisions", []) or [])
    stage1_pairs = _matching_decisions_to_pairs(stage1_decisions)

    # --- Stage 1.5: Match Inspector (pair-level GenAI verification) ----------
    inspector_result = _inspect_matched_pairs(
        stage1_pairs,
        previous_cards,
        current_cards,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    confirmed_stage1_pairs = inspector_result["confirmed_pairs"]
    rejected_stage1_pairs = inspector_result["rejected_pairs"]
    inspector_stats = inspector_result.get("inspection_stats", {})

    released_primary_pairs: list[dict[str, Any]] = []
    if hybrid_recovery_enabled:
        confirmed_stage1_pairs, released_primary_pairs = partition_trusted_rbc_primary_pairs(
            confirmed_stage1_pairs,
            previous_cards,
            current_cards,
        )
        rejected_stage1_pairs = rejected_stage1_pairs + released_primary_pairs
        if released_primary_pairs:
            logger.info(
                "RBC hybrid audit released %d weak primary pairs back to recovery",
                len(released_primary_pairs),
            )

    if rejected_stage1_pairs:
        logger.info(
            "Match Inspector rejected %d pairs — returning to unresolved pool",
            len(rejected_stage1_pairs),
        )

    state.confirmed_pairs = _pairs_from_dicts(confirmed_stage1_pairs)
    state.rejected_pairs = _pairs_from_dicts(rejected_stage1_pairs)

    # Use only confirmed pairs as consumed; rejected pairs go back to pools
    used_previous_stage1 = {
        item["previous_table_id"]
        for item in confirmed_stage1_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    # Unresolved = original unresolved + rejected CQ tables
    unresolved_ids = list(
        dict.fromkeys(
            [item["current_table_id"] for item in stage1_decisions if item.get("decision") == "unresolved"]
            + [item["current_table_id"] for item in rejected_stage1_pairs]
        )
    )
    unresolved_lookup = {card["table_id"]: card for card in current_cards}
    unresolved_current_cards = [
        unresolved_lookup[table_id] for table_id in unresolved_ids if table_id in unresolved_lookup
    ]
    remaining_previous_cards = [card for card in previous_cards if card["table_id"] not in used_previous_stage1]
    state.unresolved_current_ids = list(unresolved_ids)
    state.remaining_previous_ids = [str(card.get("table_id", "") or "") for card in remaining_previous_cards]

    stage2_decisions: list[dict[str, Any]] = []
    stage2_metrics = {
        "executed": False,
        "validation_retries_total": 0,
        "matching_validation_failures_total": 0,
        "matching_pairs_llm_duplicates_total": 0,
        "matching_pairs_llm_deduped_total": 0,
        "hybrid_recovery_executed": 0,
        "hybrid_candidate_pairs_total": 0,
        "hybrid_judge_calls_total": 0,
        "hybrid_final_inspector_calls_total": 0,
        "hybrid_pairs_rejected_total": 0,
        "hybrid_embedding_calls_total": 0,
        "warnings": [],
    }
    tables_added: list[dict[str, str]] = []

    if unresolved_current_cards and remaining_previous_cards:
        if hybrid_recovery_enabled and call_openai_embeddings is not None:
            stage2 = run_rbc_hybrid_recovery(
                remaining_previous_cards,
                unresolved_current_cards,
                model=model,
                embedding_model=hybrid_embedding_model,
                top_k=hybrid_top_k,
                min_confidence=hybrid_min_confidence,
                call_openai_json=call_openai_json,
                call_openai_embeddings=call_openai_embeddings,
                usage_recorder=usage_recorder,
            )
        elif hybrid_recovery_enabled:
            stage2 = {
                **stage2_metrics,
                "executed": True,
                "hybrid_recovery_executed": 1,
                "warnings": ["rbc_hybrid_embeddings_callback_missing"],
                "current_table_decisions": [
                    {
                        "current_table_id": str(card.get("table_id", "")),
                        "decision": "added",
                        "reason": "RBC hybrid embedding retrieval was unavailable; left unmatched for review.",
                    }
                    for card in unresolved_current_cards
                ],
            }
        else:
            stage2 = _run_matching_stage(
                remaining_previous_cards,
                unresolved_current_cards,
                stage="recovery",
                allowed_decisions={"matched", "added"},
                model=model,
                call_openai_json=call_openai_json,
                usage_recorder=usage_recorder,
                consumed_previous_ids=used_previous_stage1,
            )
        stage2_metrics = stage2
        stage2_decisions = list(stage2.get("current_table_decisions", []) or [])
        tables_added = _matching_decisions_to_table_refs(
            stage2_decisions,
            decision="added",
        )
    elif unresolved_current_cards:
        tables_added = [
            {
                "table_id": str(card.get("table_id", "") or ""),
                "reason": "No previous business table remained available for matching.",
            }
            for card in unresolved_current_cards
        ]

    matched_pairs = confirmed_stage1_pairs + _matching_decisions_to_pairs(stage2_decisions)
    used_previous_all = {
        str(item.get("previous_table_id", "") or "").strip()
        for item in matched_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    tables_removed = [
        {
            "table_id": str(card.get("table_id", "") or ""),
            "reason": "No current business table was matched to this previous table.",
        }
        for card in previous_cards
        if card["table_id"] not in used_previous_all
    ]
    warnings = _normalize_matching_warnings(
        list(stage1.get("warnings", []) or []) + list(stage2_metrics.get("warnings", []) or [])
    )
    state.tables_added = _refs_from_dicts(tables_added)
    state.tables_removed = _refs_from_dicts(tables_removed)
    state.warnings = warnings

    return MatchingResult(
        executed=bool(stage1.get("executed") or stage2_metrics.get("executed")),
        matched_pairs=_pairs_from_dicts(matched_pairs),
        tables_added=state.tables_added,
        tables_removed=state.tables_removed,
        warnings=warnings,
        matching_pairs_llm_duplicates_total=_coerce_int(stage1.get("matching_pairs_llm_duplicates_total"))
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_duplicates_total")),
        matching_pairs_llm_deduped_total=_coerce_int(stage1.get("matching_pairs_llm_deduped_total"))
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_deduped_total")),
        validation_retries_total=_coerce_int(stage1.get("validation_retries_total"))
        + _coerce_int(stage2_metrics.get("validation_retries_total")),
        matching_validation_failures_total=_coerce_int(stage1.get("matching_validation_failures_total"))
        + _coerce_int(stage2_metrics.get("matching_validation_failures_total")),
        stage1_validation_retries_total=_coerce_int(stage1.get("validation_retries_total")),
        stage2_validation_retries_total=_coerce_int(stage2_metrics.get("validation_retries_total")),
        unresolved_after_stage1_total=len(unresolved_current_cards),
        matched_in_stage2_total=len([item for item in stage2_decisions if item.get("decision") == "matched"]),
        matching_passes_total=int(bool(stage1.get("executed"))) + int(bool(stage2_metrics.get("executed"))),
        inspector_passes_total=_coerce_int(inspector_stats.get("total_inspected")),
        unmatched_after_primary_total=len(unresolved_current_cards) + len(remaining_previous_cards),
        unmatched_after_rescue_total=len(tables_added) + len(tables_removed),
        inspector_rejected_total=_coerce_int(inspector_stats.get("rejected")),
        inspector_confirmed_total=_coerce_int(inspector_stats.get("confirmed")),
        hybrid_recovery_executed=_coerce_int(stage2_metrics.get("hybrid_recovery_executed")),
        hybrid_primary_pairs_released_total=len(released_primary_pairs),
        hybrid_candidate_pairs_total=_coerce_int(stage2_metrics.get("hybrid_candidate_pairs_total")),
        hybrid_judge_calls_total=_coerce_int(stage2_metrics.get("hybrid_judge_calls_total")),
        hybrid_final_inspector_calls_total=_coerce_int(stage2_metrics.get("hybrid_final_inspector_calls_total")),
        hybrid_pairs_rejected_total=_coerce_int(stage2_metrics.get("hybrid_pairs_rejected_total")),
        hybrid_embedding_calls_total=_coerce_int(stage2_metrics.get("hybrid_embedding_calls_total")),
    )


def _run_table_matching(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    hybrid_recovery_enabled: bool = False,
    hybrid_embedding_model: str = "text-embedding-3-large",
    hybrid_top_k: int = 5,
    hybrid_min_confidence: float = 0.75,
    call_openai_embeddings: Callable[..., list[list[float]]] | None = None,
) -> dict[str, Any]:
    """Point d'entree principal de l'appariement des tableaux.

    Appelle ``_match_tables`` puis trie les paires et structure le resultat
    final avec toutes les metriques de suivi.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.
        hybrid_recovery_enabled: Active la recuperation hybride RBC opt-in.
        hybrid_embedding_model: Modele utilise pour la recherche de candidats.
        hybrid_top_k: Nombre de candidats embeddings presentes par tableau courant.
        hybrid_min_confidence: Confiance LLM minimale avant l'affectation globale.
        call_openai_embeddings: Callable injecte pour calculer les embeddings.

    Returns:
        Dictionnaire structure (dump de ``MatchingResult`` enrichi) contenant
        ``matched_pairs``, ``tables_added``, ``tables_removed``, ``warnings``
        et toutes les metriques de suivi.
    """
    result = _match_tables(
        previous_cards,
        current_cards,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        hybrid_recovery_enabled=hybrid_recovery_enabled,
        hybrid_embedding_model=hybrid_embedding_model,
        hybrid_top_k=hybrid_top_k,
        hybrid_min_confidence=hybrid_min_confidence,
        call_openai_embeddings=call_openai_embeddings,
    )
    sorted_pairs = _sort_matched_pairs(
        [pair.model_dump(mode="json") for pair in result.matched_pairs],
        previous_cards,
    )
    result.matched_pairs = _pairs_from_dicts(sorted_pairs)
    tables_added = [ref.model_dump(mode="json") for ref in result.tables_added]
    tables_removed = [ref.model_dump(mode="json") for ref in result.tables_removed]
    payload = result.to_legacy_dict()
    payload.update(
        {
            "matched_pairs": sorted_pairs,
            "tables_added": tables_added,
            "tables_removed": tables_removed,
            "matching_passes_total": _coerce_int(result.matching_passes_total),
            "inspector_passes_total": _coerce_int(result.inspector_passes_total),
            "audit_passes_total": 0,
            "matching_output_retries_total": _coerce_int(result.validation_retries_total),
            "matching_validation_failures_total": _coerce_int(result.matching_validation_failures_total),
            "stage1_validation_retries_total": _coerce_int(result.stage1_validation_retries_total),
            "stage2_validation_retries_total": _coerce_int(result.stage2_validation_retries_total),
            "unresolved_after_stage1_total": _coerce_int(result.unresolved_after_stage1_total),
            "matched_in_stage2_total": _coerce_int(result.matched_in_stage2_total),
            "unmatched_previous_table_ids": [item["table_id"] for item in tables_removed],
            "unmatched_current_table_ids": [item["table_id"] for item in tables_added],
            "unmatched_after_primary_total": _coerce_int(result.unmatched_after_primary_total),
            "unmatched_after_rescue_total": _coerce_int(result.unmatched_after_rescue_total),
            "matching_pairs_llm_duplicates_total": _coerce_int(result.matching_pairs_llm_duplicates_total),
            "matching_pairs_llm_deduped_total": _coerce_int(result.matching_pairs_llm_deduped_total),
            "inspector_rejected_total": _coerce_int(result.inspector_rejected_total),
            "inspector_confirmed_total": _coerce_int(result.inspector_confirmed_total),
            "hybrid_recovery_executed": _coerce_int(result.hybrid_recovery_executed),
            "hybrid_primary_pairs_released_total": _coerce_int(result.hybrid_primary_pairs_released_total),
            "hybrid_candidate_pairs_total": _coerce_int(result.hybrid_candidate_pairs_total),
            "hybrid_judge_calls_total": _coerce_int(result.hybrid_judge_calls_total),
            "hybrid_final_inspector_calls_total": _coerce_int(result.hybrid_final_inspector_calls_total),
            "hybrid_pairs_rejected_total": _coerce_int(result.hybrid_pairs_rejected_total),
            "hybrid_embedding_calls_total": _coerce_int(result.hybrid_embedding_calls_total),
            "warnings": _normalize_matching_warnings(result.warnings),
        }
    )
    return payload
