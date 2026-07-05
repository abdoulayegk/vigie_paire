"""Retention, consolidation et resume global des changements texte."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

from vigilance.text_comparison.consolidation import (
    build_atomic_observations,
    build_observations_from_group_specs,
    candidate_batches_for_llm,
)

from .constants import _STRONG_AMF_THEMES_FOR_MODERE_RETENTION
from .models import TextObservationConsolidationBatch
from .openai_gateway import _call_structured_completion_with_correction
from .text_normalization import _normalize_match_text
from .vigie_objectives import _objective_labels_for_change

def _is_new_major_or_allowed_moderate(triage: dict[str, Any]) -> bool:
    """Retourne True si le triage indique un changement majeur ou un changement modéré significatif.

    Un changement modéré est retenu uniquement s'il introduit une nouvelle idée
    ou s'il porte un thème AMF méthodologique/réglementaire fort. Utilisée pour
    filtrer les changements à prioriser pour revue.
    """
    if not triage.get("is_relevant", False):
        return False
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    if impact == "MAJEUR":
        return True
    if impact != "MODERE":
        return False
    if triage.get("nouvelle_idee", False):
        return True
    themes = set(triage.get("themes_amf") or [])
    return bool(themes & _STRONG_AMF_THEMES_FOR_MODERE_RETENTION)


def _is_non_cosmetic_change(triage: dict[str, Any]) -> bool:
    """Retourne True si le triage retient le changement (pertinent et thématisé AMF)."""
    return bool(triage.get("is_relevant")) and bool(triage.get("themes_amf"))


_RETAINED_MINOR_VIGIE_TOPICS: frozenset[str] = frozenset({"limites de risque"})


def _is_retained_minor_vigie_topic(change: dict[str, Any]) -> bool:
    """Retourne True pour les sujets mineurs à garder visibles malgré le triage IA."""
    text = " ".join(
        str(change.get(key) or "")
        for key in (
            "subsection_heading",
            "previous_subsection_heading",
            "current_subsection_heading",
            "canonical_topic",
        )
    )
    normalized = _normalize_match_text(text)
    padded = f" {normalized} "
    return any(f" {topic} " in padded for topic in _RETAINED_MINOR_VIGIE_TOPICS)


def _is_retained_text_change(change: dict[str, Any]) -> bool:
    """Détermine si un changement doit apparaître dans ``block_comparisons``."""
    triage = change.get("genai_triage") or {}
    if _is_non_cosmetic_change(triage):
        return True
    return _is_retained_minor_vigie_topic(change)


def _retained_change_sort_key(change: dict[str, Any]) -> tuple[int, int, int, int, str, str, str]:
    """Clé de tri pour ordonner les changements retenus dans le rapport final.

    Ordre de priorité : nouvelles idées en premier, puis impact décroissant
    (MAJEUR → MODERE → MINEUR), puis objectifs de vigie détectés, puis nombre
    de thèmes AMF décroissant (un changement multi-label étant a priori plus
    structurant), puis section, page et type de diff. Les objectifs ne créent
    jamais un changement; ils priorisent seulement les vrais changements déjà
    détectés et triés.
    """
    triage = change.get("genai_triage") or {}
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    diff_type = str(change.get("diff_type") or "").lower()
    pages = change.get("pages_t2") or change.get("pages_t1") or []
    first_page = ""
    if pages:
        try:
            first_page = f"{int(pages[0]):06d}"
        except (TypeError, ValueError):
            first_page = str(pages[0])
    impact_order = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}.get(impact, 99)
    themes_count = len(triage.get("themes_amf") or [])
    objective_order = 0 if _objective_labels_for_change(change) else 1
    return (
        0 if triage.get("nouvelle_idee", False) else 1,
        impact_order,
        objective_order,
        -themes_count,
        str(change.get("section_key") or ""),
        first_page,
        diff_type,
    )


def _consolidate_section_observations_with_llm(
    client: Any,
    *,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Demande au LLM de consolider les chunks en observations analyste.

    Aucun regroupement sémantique n'est fait localement. Le code prépare des
    lots techniques par taille pour limiter le contexte, puis assemble les
    groupes et changements atomiques retournés par GPT. Si GPT échoue, le
    résultat reste atomique.
    """
    if not changes:
        return []

    group_specs: list[dict[str, Any]] = []
    atomic_specs: list[dict[str, Any]] = []
    for batch in candidate_batches_for_llm(changes):
        if len(batch) < 2:
            continue
        try:
            response = _call_structured_completion_with_correction(
                client,
                model=model,
                messages=_build_observation_consolidation_messages(
                    section_key=section_key,
                    changes=batch,
                ),
                response_format=TextObservationConsolidationBatch,
                max_tokens=6_144,
                max_retries=1,
                max_length_retries=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "observation consolidation skipped section=%s batch_size=%d error=%s",
                section_key,
                len(batch),
                exc,
            )
            continue
        group_specs.extend(group.model_dump() for group in response.observations)
        atomic_specs.extend(item.model_dump() for item in response.atomic_changes)

    if not group_specs and not atomic_specs:
        return build_atomic_observations(changes)
    return build_observations_from_group_specs(changes, group_specs, atomic_specs=atomic_specs)


def _build_observation_consolidation_messages(
    *,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system = (
        "Tu es un expert senior en vigie de pairs bancaires. "
        "Ta tâche est de transformer des changements textuels atomiques en "
        "observations analyste consolidées. Tu dois comprendre le sens métier, "
        "pas seulement les mots-clés."
    )
    user = {
        "section_key": section_key,
        "instructions": [
            "Tu reçois les changements d'une même grande section, ou un lot technique découpé seulement par taille.",
            "Décide librement les observations métier consolidées selon le sens, sans te limiter à la page ou à la sous-section.",
            "Tu peux regrouper des changements de pages ou sous-sections différentes s'ils expriment la même observation de vigie.",
            "Un bon groupe doit pouvoir être résumé comme une seule observation de vigie.",
            "Ne groupe pas deux changements seulement parce qu'ils sont sur la même page.",
            "Ne groupe pas des sujets différents dans une sous-section générique.",
            "Si deux changements ont des directions différentes ou des implications métier différentes, laisse-les séparés.",
            "Retourne les groupes dans observations et les changements à garder séparés dans atomic_changes avec une raison courte.",
            "Retourne chaque change_id au plus une fois, soit dans observations, soit dans atomic_changes.",
            "Utilise seulement les change_id fournis; n'invente aucun identifiant.",
            "Rédige tous les champs textuels libres en français professionnel; les clés JSON, codes AMF et valeurs d'énumération restent inchangés.",
            "Pour chaque observation consolidée, décide aussi impact_level, action_requise, nouvelle_idee, themes_amf et nouvelle_idee_justification.",
            "impact_level doit être MAJEUR, MODERE ou MINEUR; action_requise doit être revue_prioritaire, investigation, confirmation, information ou aucune.",
            "Le résumé doit être court, concret, et rédigé comme une note d'analyste.",
            "La justification doit commencer par OUI ou NON et expliquer pourquoi c'est, ou non, une nouvelle idée de vigie.",
        ],
        "changes": [_consolidation_change_payload(change) for change in changes],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _consolidation_change_payload(change: dict[str, Any]) -> dict[str, Any]:
    triage = change.get("genai_triage") or {}
    return {
        "change_id": str(change.get("change_id") or ""),
        "diff_type": str(change.get("diff_type") or ""),
        "subsection": str(
            change.get("subsection_heading")
            or change.get("current_subsection_heading")
            or change.get("previous_subsection_heading")
            or ""
        ),
        "hierarchy_path": str(
            change.get("current_hierarchy_path")
            or change.get("previous_hierarchy_path")
            or ""
        ),
        "pages_current": list(change.get("pages_t2") or []),
        "pages_previous": list(change.get("pages_t1") or []),
        "change_summary": str(change.get("change_summary") or "")[:700],
        "chunk_topic": str(change.get("chunk_topic") or ""),
        "canonical_topic": str(change.get("canonical_topic") or ""),
        "objective_labels": _objective_labels_for_change(change),
        "themes_amf": list(triage.get("themes_amf") or []),
        "impact_level": str(triage.get("impact_level") or ""),
        "nouvelle_idee": bool(triage.get("nouvelle_idee", False)),
        "source_text_t1": str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")[:900],
        "source_text_t2": str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")[:900],
    }


def _build_global_summary(
    section_comparisons: list[dict[str, Any]],
    *,
    bucket: str = "block_comparisons",
) -> dict[str, Any]:
    """Agrège les statistiques de toutes les sections en un résumé global.

    Calcule les comptages par impact, catégorie et action requise, extrait
    les cinq premiers résumés de changements comme points saillants, et détermine
    la pertinence globale (FAIBLE / MOYENNE / ELEVEE) selon le nombre de changements
    majeurs. Utilisé pour les deux champs ``global_summary`` et ``all_changes_summary``
    du payload final.
    """
    all_changes = [block for section in section_comparisons for block in (section.get(bucket) or [])]
    by_impact: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}
    highlights: list[str] = []
    relevant_count = 0
    relevant_major_count = 0

    for change in all_changes:
        triage = change.get("genai_triage") or {}
        is_relevant = bool(triage.get("is_relevant", False))
        impact = str(triage.get("impact_level") or "MINEUR").upper()
        category = str(triage.get("category") or "INCONNU").upper()
        action = str(triage.get("action_requise") or "aucune").lower()
        if is_relevant:
            relevant_count += 1
            if impact == "MAJEUR":
                relevant_major_count += 1
        by_impact[impact] = by_impact.get(impact, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        summary = str(change.get("change_summary") or "").strip()
        if summary and len(highlights) < 5:
            highlights.append(summary)

    if not all_changes:
        overview = "Aucun changement textuel détecté."
    elif relevant_count == len(all_changes):
        overview = f"{len(all_changes)} changement(s) textuel(s) substantiel(s) retenu(s) pour revue experte."
    else:
        overview = (
            f"{len(all_changes)} changement(s) textuel(s) détecté(s), "
            f"dont {relevant_count} substantiel(s) selon l'IA. "
            "Tous restent disponibles pour validation analyste."
        )
    pertinence = "FAIBLE"
    if relevant_major_count >= 3:
        pertinence = "ELEVEE"
    elif relevant_count:
        pertinence = "MOYENNE"

    return {
        "executive_overview": overview,
        "key_highlights": highlights,
        "pertinence_globale": pertinence,
        "counts": {
            "total": len(all_changes),
            "total_detected": len(all_changes),
            "total_relevant": relevant_count,
            "by_impact": by_impact,
            "by_category": by_category,
            "by_action": by_action,
        },
    }
