"""Correction ciblee des reponses de rapprochement invalides."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, create_model

from vigie.comparaison.rapprochement.normalisation_reponses import (
    _alias_table_card,
    _canonical_matching_item,
    _current_alias,
    _decode_current_alias,
    _decode_previous_alias,
    _normalize_matching_warnings,
    _previous_alias,
)


def _analyze_matching_candidate(
    data: dict[str, Any],
    *,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Separe les decisions verrouillables des decisions a reparer.

    Cette analyse est strictement structurelle : elle ne choisit jamais une
    correspondance metier. Elle preserve seulement les decisions deja valides.
    """
    consumed_previous = set(consumed_previous_ids or ())
    raw_items = [item for item in list(data.get("current_table_decisions", []) or []) if isinstance(item, dict)]

    current_occurrences: dict[str, list[dict[str, Any]]] = {}
    previous_occurrences: dict[str, list[str]] = {}
    unknown_current_ids: set[str] = set()
    unknown_previous_ids: set[str] = set()

    for item in raw_items:
        current_id = _decode_current_alias(item.get("current_table_id"))
        if current_id not in current_ids:
            if current_id:
                unknown_current_ids.add(current_id)
            continue
        current_occurrences.setdefault(current_id, []).append(item)
        decision = str(item.get("decision", "") or "").strip().lower()
        if decision != "matched":
            continue
        previous_id = _decode_previous_alias(item.get("previous_table_id"))
        if previous_id in previous_ids:
            previous_occurrences.setdefault(previous_id, []).append(current_id)
        elif previous_id:
            unknown_previous_ids.add(previous_id)

    duplicate_current_ids = {current_id for current_id, items in current_occurrences.items() if len(items) != 1}
    duplicate_previous_assignments = {
        previous_id: sorted(set(assigned_current_ids))
        for previous_id, assigned_current_ids in previous_occurrences.items()
        if len(assigned_current_ids) > 1
    }
    missing_current_ids = current_ids - set(current_occurrences)
    repair_current_ids = set(missing_current_ids) | set(duplicate_current_ids)
    locked_decisions: list[dict[str, Any]] = []

    for current_id in sorted(current_ids):
        items = current_occurrences.get(current_id, [])
        if len(items) != 1 or current_id in repair_current_ids:
            repair_current_ids.add(current_id)
            continue
        item = items[0]
        decision = str(item.get("decision", "") or "").strip().lower()
        previous_id = _decode_previous_alias(item.get("previous_table_id"))
        confidence_raw = item.get("match_confidence")
        confidence_supplied = confidence_raw is not None and str(confidence_raw).strip() != ""

        is_valid = decision in allowed_decisions
        if decision == "matched":
            is_valid = bool(
                is_valid
                and previous_id in previous_ids
                and previous_id not in consumed_previous
                and previous_id not in duplicate_previous_assignments
            )
        else:
            is_valid = bool(is_valid and not previous_id and not confidence_supplied)

        if not is_valid:
            repair_current_ids.add(current_id)
            continue
        locked_decisions.append(_canonical_matching_item(item))

    used_locked_previous_ids = {
        str(item.get("previous_table_id", "") or "") for item in locked_decisions if item.get("decision") == "matched"
    }
    available_previous_ids = previous_ids - consumed_previous - used_locked_previous_ids

    invalid_decisions = [
        dict(item)
        for item in raw_items
        if _decode_current_alias(item.get("current_table_id")) in repair_current_ids
        or _decode_current_alias(item.get("current_table_id")) not in current_ids
    ]
    diagnostics = {
        "missing_current_table_ids": sorted(missing_current_ids),
        "duplicate_current_table_ids": sorted(duplicate_current_ids),
        "duplicate_previous_assignments": duplicate_previous_assignments,
        "unknown_current_table_ids": sorted(unknown_current_ids),
        "unknown_previous_table_ids": sorted(unknown_previous_ids),
        "wrong_namespace_previous_ids": sorted(unknown_previous_ids & current_ids),
    }
    return {
        "locked_decisions": locked_decisions,
        "repair_current_ids": repair_current_ids,
        "available_previous_ids": available_previous_ids,
        "invalid_decisions": invalid_decisions,
        "diagnostics": diagnostics,
        "warnings": _normalize_matching_warnings(data.get("warnings", [])),
    }


def _build_matching_repair_response_model(
    *,
    current_aliases: list[str],
    previous_aliases: list[str],
    allowed_decisions: set[str],
) -> type:
    """Construit un schema OpenAI dont les IDs sont des enums fermes PQ/CQ."""
    current_id_type = Literal.__getitem__(tuple(current_aliases))
    previous_id_type = Literal.__getitem__(tuple(["", *previous_aliases]))
    decision_type = Literal.__getitem__(tuple(sorted(allowed_decisions)))
    decision_model = create_model(
        "MatchingRepairDecision",
        __config__=ConfigDict(extra="forbid"),
        current_table_id=(current_id_type, ...),
        decision=(decision_type, ...),
        reason=(str, ...),
        previous_table_id=(previous_id_type, ""),
        match_confidence=(float | None, Field(default=None, ge=0.0, le=1.0)),
    )
    return create_model(
        "MatchingRepairResponse",
        __config__=ConfigDict(extra="forbid"),
        current_table_decisions=(list[decision_model], ...),
        warnings=(list[str], Field(default_factory=list)),
    )


def _build_matching_repair_prompt(
    *,
    stage: str,
    repair_round: int,
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    current_ids: set[str],
    allowed_decisions: set[str],
    validation_feedback: str,
    repair_state: dict[str, Any],
) -> dict[str, Any]:
    """Construit le prompt cible du reparateur ou de l'arbitre."""
    repair_current_ids = set(repair_state["repair_current_ids"])
    available_previous_ids = set(repair_state["available_previous_ids"])
    repair_allowed_decisions = set(allowed_decisions)
    if not available_previous_ids:
        repair_allowed_decisions.discard("matched")

    current_aliases = [_current_alias(table_id) for table_id in sorted(repair_current_ids)]
    previous_aliases = [_previous_alias(table_id) for table_id in sorted(available_previous_ids)]
    return {
        "stage": stage,
        "agent_role": "matching_structure_repair" if repair_round == 1 else "matching_final_adjudicator",
        "task": "Repair only the structurally invalid decisions. Locked decisions are preserved by the application and must not be returned.",
        "rules": [
            "Return exactly one decision for every required_repair_current_table_id.",
            "Use CQ:: identifiers only in current_table_id.",
            "Use PQ:: identifiers only in previous_table_id.",
            "Use each PQ:: identifier at most once.",
            "For a non-matched decision, return previous_table_id as an empty string and match_confidence as null.",
            "Do not return locked decisions or any unrequested current table.",
            "Prefer unresolved or added when the evidence is ambiguous.",
        ],
        "validation_feedback": validation_feedback,
        "validation_diagnostics": repair_state["diagnostics"],
        "invalid_decisions": repair_state["invalid_decisions"],
        "locked_decisions": repair_state["locked_decisions"],
        # Kept for observability and backward-compatible diagnostics.
        "required_current_table_ids": sorted(current_ids),
        "required_repair_current_table_ids": current_aliases,
        "allowed_previous_table_ids": previous_aliases,
        "allowed_decisions": sorted(repair_allowed_decisions),
        "previous_tables": [
            _alias_table_card(card, previous=True)
            for card in previous_cards
            if str(card.get("table_id", "") or "") in available_previous_ids
        ],
        "current_tables": [
            _alias_table_card(card, previous=False)
            for card in current_cards
            if str(card.get("table_id", "") or "") in repair_current_ids
        ],
        "response_schema": {
            "current_table_decisions": [
                {
                    "current_table_id": f"one_of_{current_aliases}",
                    "decision": f"one_of_{sorted(repair_allowed_decisions)}",
                    "reason": "short evidence-grounded explanation",
                    "previous_table_id": f"one_of_{['', *previous_aliases]}",
                    "match_confidence": "number_0_to_1_or_null",
                }
            ],
            "warnings": ["string"],
        },
    }


def _merge_matching_repair_response(
    repair_data: dict[str, Any],
    *,
    repair_state: dict[str, Any],
) -> dict[str, Any]:
    """Fusionne les seules decisions reparees avec le registre verrouille."""
    repair_current_ids = set(repair_state["repair_current_ids"])
    repaired_decisions: list[dict[str, Any]] = []
    for item in list(repair_data.get("current_table_decisions", []) or []):
        if not isinstance(item, dict):
            continue
        decoded = dict(item)
        decoded["current_table_id"] = _decode_current_alias(item.get("current_table_id"))
        decoded["previous_table_id"] = _decode_previous_alias(item.get("previous_table_id"))
        if decoded["current_table_id"] in repair_current_ids:
            repaired_decisions.append(decoded)
    return {
        "current_table_decisions": [
            *list(repair_state["locked_decisions"]),
            *repaired_decisions,
        ],
        "warnings": _normalize_matching_warnings(
            [
                *list(repair_state.get("warnings", []) or []),
                *list(repair_data.get("warnings", []) or []),
            ]
        ),
    }


def _build_matching_fail_soft_response(
    data: dict[str, Any],
    *,
    stage: str,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Preserve les decisions valides et place les conflits en revue non bloquante."""
    repair_state = _analyze_matching_candidate(
        data,
        previous_ids=previous_ids,
        current_ids=current_ids,
        allowed_decisions=allowed_decisions,
        consumed_previous_ids=consumed_previous_ids,
    )
    fallback_decision = "unresolved" if "unresolved" in allowed_decisions else "added"
    repair_ids = sorted(set(repair_state["repair_current_ids"]))
    fallback_items = [
        {
            "current_table_id": current_id,
            "decision": fallback_decision,
            "reason": (
                f"Structural matching repair exhausted; this table requires review after the {stage} matching stage."
            ),
        }
        for current_id in repair_ids
    ]
    return {
        "current_table_decisions": [
            *list(repair_state["locked_decisions"]),
            *fallback_items,
        ],
        "warnings": _normalize_matching_warnings(
            [
                *list(repair_state.get("warnings", []) or []),
                f"matching_structure_repair_exhausted:{','.join(repair_ids)}",
            ]
        ),
    }
