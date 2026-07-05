"""Detection des objectifs de vigie sur les changements texte."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

from .constants import _VIGIE_OBJECTIVES
from .text_normalization import _normalize_match_text

def _vigie_objectives_prompt() -> str:
    """Formate les objectifs de vigie comme contexte de comparaison/triage."""
    return "\n".join(
        f"- {objective['label']}: {objective['objective']}" for objective in _VIGIE_OBJECTIVES
    )


def _detect_vigie_objectives_for_text(text: str) -> list[dict[str, str]]:
    """Détecte localement les objectifs possiblement concernés par un texte."""
    normalized = f" {_normalize_match_text(text)} "
    matches: list[dict[str, str]] = []
    for objective in _VIGIE_OBJECTIVES:
        for keyword in objective["keywords"]:
            keyword_norm = f" {_normalize_match_text(str(keyword))} "
            if keyword_norm.strip() and keyword_norm in normalized:
                matches.append(
                    {
                        "tag": str(objective["tag"]),
                        "label": str(objective["label"]),
                        "objective": str(objective["objective"]),
                    }
                )
                break
    return matches


def _detect_vigie_objectives_for_change(change: dict[str, Any]) -> list[dict[str, str]]:
    """Détecte les objectifs touchés par un changement existant."""
    text = " ".join(
        str(change.get(key) or "")
        for key in (
            "subsection_heading",
            "previous_subsection_heading",
            "current_subsection_heading",
            "canonical_topic",
            "change_summary",
            "source_text_t1",
            "source_text_t2",
            "semantic_text_t1",
            "semantic_text_t2",
        )
    )
    return _detect_vigie_objectives_for_text(text)


def _objective_note(objectives: list[dict[str, str]]) -> str:
    labels = [objective["label"] for objective in objectives]
    if not labels:
        return ""
    return "Objectif vigie détecté : " + ", ".join(labels) + "."


def _objective_labels_for_change(change: dict[str, Any]) -> list[str]:
    """Retourne les libellés d'objectifs déjà détectés sur un changement."""
    labels: list[str] = []
    for objective in change.get("objective_matches") or []:
        label = str(objective.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    triage = change.get("genai_triage") or {}
    for objective in triage.get("objective_matches") or []:
        label = str(objective.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _attach_vigie_objective_context(change: dict[str, Any]) -> dict[str, Any]:
    """Ajoute les objectifs au changement sans modifier l'affichage Dash."""
    objectives = _detect_vigie_objectives_for_change(change)
    chunk_topic = str(change.get("chunk_topic") or "").strip()
    if not objectives and not chunk_topic:
        return change

    enriched = dict(change)
    if objectives:
        enriched["objective_tags"] = [objective["tag"] for objective in objectives]
        enriched["objective_matches"] = objectives

    objective_note = _objective_note(objectives)
    topic_note = f"Thème métier détecté : {chunk_topic}." if chunk_topic else ""
    notes = [note for note in (objective_note, topic_note) if note]
    triage = dict(enriched.get("genai_triage") or {})
    if triage:
        explanation = str(triage.get("explanation") or "").strip()
        justification = str(triage.get("nouvelle_idee_justification") or "").strip()
        for note in notes:
            if explanation and note not in explanation:
                explanation = f"{note} {explanation}".strip()
            if justification and note not in justification:
                justification = f"{justification}\n\n{note}".strip()
        if explanation:
            triage["explanation"] = explanation
        if justification:
            triage["nouvelle_idee_justification"] = justification
        if objectives:
            triage["objective_tags"] = enriched["objective_tags"]
            triage["objective_matches"] = objectives
            triage["objective_priority"] = True
            if bool(triage.get("is_relevant", False)) and str(triage.get("impact_level") or "").upper() == "MINEUR":
                triage["impact_level"] = "MODERE"
                if str(triage.get("action_requise") or "").lower() in {"", "aucune", "information"}:
                    triage["action_requise"] = "investigation"
        if chunk_topic:
            triage["chunk_topic"] = chunk_topic
        enriched["genai_triage"] = triage
    return enriched
