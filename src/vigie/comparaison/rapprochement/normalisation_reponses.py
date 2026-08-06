"""Normalisation et validation des reponses de rapprochement."""

from __future__ import annotations

from typing import Any

from vigie.comparaison.io import _coerce_float, _require_string
from vigie.comparaison.rapprochement.contrats import (
    _CURRENT_ID_PREFIX,
    _MatchingValidationError,
    _PREVIOUS_ID_PREFIX,
)
from vigie.comparaison.rapprochement.etat import MatchedPair, MatchingResult, TableRef


def _normalize_matching_warnings(items: Any) -> list[str]:
    """Filtre et normalise une liste brute d'avertissements d'appariement."""
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _previous_alias(table_id: str) -> str:
    """Retourne l'identifiant explicitement espace de noms du trimestre precedent."""
    return f"{_PREVIOUS_ID_PREFIX}{table_id}"


def _current_alias(table_id: str) -> str:
    """Retourne l'identifiant explicitement espace de noms du trimestre courant."""
    return f"{_CURRENT_ID_PREFIX}{table_id}"


def _decode_previous_alias(value: Any) -> str:
    """Decode un alias PQ tout en acceptant les IDs bruts des anciens appelants."""
    text = str(value or "").strip()
    return text[len(_PREVIOUS_ID_PREFIX) :] if text.startswith(_PREVIOUS_ID_PREFIX) else text


def _decode_current_alias(value: Any) -> str:
    """Decode un alias CQ tout en acceptant les IDs bruts des anciens appelants."""
    text = str(value or "").strip()
    return text[len(_CURRENT_ID_PREFIX) :] if text.startswith(_CURRENT_ID_PREFIX) else text


def _alias_table_card(card: Any, *, previous: bool) -> dict[str, Any]:
    """Copie une fiche de tableau en remplacant son ID par un alias PQ/CQ."""
    from vigie.comparaison.io import table_view_as_dict

    aliased = table_view_as_dict(card)
    table_id = str(aliased.get("table_id", "") or "")
    aliased["table_id"] = _previous_alias(table_id) if previous else _current_alias(table_id)
    return aliased


def _canonical_matching_item(item: dict[str, Any]) -> dict[str, Any]:
    """Construit une decision canonique a partir d'un item structurellement valide."""
    decision = str(item.get("decision", "") or "").strip().lower()
    normalized: dict[str, Any] = {
        "current_table_id": _decode_current_alias(item.get("current_table_id")),
        "decision": decision,
        "reason": str(item.get("reason", "") or "").strip(),
    }
    if decision == "matched":
        normalized["previous_table_id"] = _decode_previous_alias(item.get("previous_table_id"))
        normalized["match_confidence"] = max(
            0.0,
            min(1.0, _coerce_float(item.get("match_confidence"))),
        )
    return normalized


def _sort_matched_pairs(
    pairs: list[dict[str, Any]],
    previous_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Trie les paires appariees selon l'ordre d'apparition des tableaux precedents."""
    order = {str(card.get("table_id", "") or ""): index for index, card in enumerate(previous_cards)}
    return sorted(
        pairs,
        key=lambda item: (
            order.get(str(item.get("previous_table_id", "") or ""), 10**9),
            str(item.get("previous_table_id", "") or ""),
            str(item.get("current_table_id", "") or ""),
        ),
    )


def _coerce_matching_payload(data: Any) -> dict[str, Any]:
    """Accepte un BaseModel Pydantic ou un dict brut pour la normalisation."""
    if hasattr(data, "model_dump") and callable(data.model_dump):
        dumped = data.model_dump(mode="python")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(data, dict):
        return data
    raise _MatchingValidationError("Matching response must be an object")


def _normalize_matching_response(
    data: Any,
    *,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Valide et normalise la reponse JSON brute du LLM pour l'appariement.

    Verifie les contraintes d'unicite, les identifiants valides et la
    couverture complete des tableaux courants. Leve ``_MatchingValidationError``
    en cas de violation.

    Args:
        data: Reponse JSON brute du LLM (dict ou modele Pydantic) contenant
            ``current_table_decisions``.
        previous_ids: Ensemble des identifiants de tableaux du trimestre precedent.
        current_ids: Ensemble des identifiants de tableaux du trimestre courant.
        allowed_decisions: Decisions autorisees (ex. ``{"matched", "unresolved"}``).
        consumed_previous_ids: Identifiants precedents deja consommes par une
            etape anterieure.

    Returns:
        Dictionnaire normalise avec ``current_table_decisions``, ``warnings``
        et metriques de doublons/deduplication.

    Raises:
        _MatchingValidationError: Si la reponse viole les contraintes structurelles.
    """
    payload = _coerce_matching_payload(data)
    current_table_decisions: list[dict[str, Any]] = []
    used_previous: set[str] = set()
    used_current: set[str] = set()
    duplicate_total = 0
    raw_total = 0
    consumed_previous = set(consumed_previous_ids or ())

    for item in list(payload.get("current_table_decisions", []) or []):
        if hasattr(item, "model_dump") and callable(item.model_dump):
            item = item.model_dump(mode="python")
        if not isinstance(item, dict):
            raise _MatchingValidationError("current_table_decisions items must be objects")
        current_table_id = _decode_current_alias(_require_string(item.get("current_table_id"), "current_table_id"))
        decision = _require_string(item.get("decision"), "decision").lower()
        if decision not in allowed_decisions:
            raise _MatchingValidationError(
                f"Invalid matching decision returned by GPT: {decision!r}; allowed={sorted(allowed_decisions)}"
            )
        if current_table_id not in current_ids:
            raise _MatchingValidationError(f"Unknown current_table_id returned by GPT: {current_table_id}")
        if current_table_id in used_current:
            duplicate_total += 1
            raise _MatchingValidationError(
                f"Duplicate current_table_id returned by GPT: {current_table_id}",
                duplicate_count=duplicate_total,
            )
        used_current.add(current_table_id)

        normalized_item: dict[str, Any] = {
            "current_table_id": current_table_id,
            "decision": decision,
            "reason": str(item.get("reason", "") or "").strip(),
        }

        previous_table_id = _decode_previous_alias(item.get("previous_table_id"))
        confidence_raw = item.get("match_confidence")
        confidence_supplied = confidence_raw is not None and str(confidence_raw).strip() != ""

        if decision == "matched":
            if not previous_table_id:
                raise _MatchingValidationError("Matched decisions must include previous_table_id.")
            if previous_table_id not in previous_ids:
                raise _MatchingValidationError(f"Unknown previous_table_id returned by GPT: {previous_table_id}")
            raw_total += 1
            if previous_table_id in consumed_previous or previous_table_id in used_previous:
                duplicate_total += 1
                raise _MatchingValidationError(
                    f"Duplicate or reused previous_table_id returned by GPT: {previous_table_id}",
                    duplicate_count=duplicate_total,
                )
            try:
                confidence = float(confidence_raw or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            normalized_item["previous_table_id"] = previous_table_id
            normalized_item["match_confidence"] = max(0.0, min(1.0, confidence))
            used_previous.add(previous_table_id)
        else:
            if previous_table_id:
                raise _MatchingValidationError(f"{decision!r} decisions must not include previous_table_id.")
            if confidence_supplied:
                raise _MatchingValidationError(f"{decision!r} decisions must not include match_confidence.")

        current_table_decisions.append(normalized_item)

    if used_current != current_ids:
        missing = sorted(current_ids - used_current)
        extra = sorted(used_current - current_ids)
        raise _MatchingValidationError(
            f"Matching output must cover exactly the business current tables. missing={missing} extra={extra}"
        )

    return {
        "current_table_decisions": current_table_decisions,
        "warnings": _normalize_matching_warnings(payload.get("warnings", [])),
        "matching_pairs_llm_duplicates_total": duplicate_total,
        "matching_pairs_llm_deduped_total": max(0, raw_total - len(used_previous)),
    }


def _matching_decisions_to_pairs(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extrait les paires appariees depuis la liste des decisions d'appariement."""
    out: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("decision") != "matched":
            continue
        out.append(
            {
                "previous_table_id": str(item.get("previous_table_id", "") or ""),
                "current_table_id": str(item.get("current_table_id", "") or ""),
                "match_confidence": _coerce_float(item.get("match_confidence")),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _matching_decisions_to_table_refs(
    decisions: list[dict[str, Any]],
    *,
    decision: str,
) -> list[dict[str, str]]:
    """Extrait les references de tableaux pour un type de decision donne."""
    out: list[dict[str, str]] = []
    for item in decisions:
        if item.get("decision") != decision:
            continue
        out.append(
            {
                "table_id": str(item.get("current_table_id", "") or ""),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _empty_matching_result(
    *,
    tables_removed: list[dict[str, str]] | None = None,
) -> MatchingResult:
    """Construit un resultat d'appariement vide avec toutes les metriques a zero."""
    return MatchingResult(
        executed=False,
        tables_removed=[TableRef.model_validate(item) for item in (tables_removed or [])],
    )


def _pairs_from_dicts(items: list[dict[str, Any]]) -> list[MatchedPair]:
    """Convertit une liste de dicts en paires MatchedPair validees."""
    return [MatchedPair.model_validate(item) for item in items]


def _refs_from_dicts(items: list[dict[str, str]]) -> list[TableRef]:
    """Convertit une liste de dicts en references TableRef validees."""
    return [TableRef.model_validate(item) for item in items]
