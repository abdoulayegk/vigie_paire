"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from typing import Any


_STRONG_AMF_THEMES_FOR_MODERE_RETENTION: frozenset[str] = frozenset(
    {
        "MODIFICATION_METHODOLOGIE",
        "NOUVELLE_MENTION_REGLEMENTAIRE",
        "EXIGENCES_REGLEMENTAIRES",
        "FACTEUR_RISQUE_CHANGEMENT",
        "RISQUE_EMERGENT",
        "RISQUE_DONNEES",
        "RISQUE_TIERS_CLOUD",
        "RISQUE_MACRO_GEOPOLITIQUE",
    }
)


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


def _retained_change_sort_key(change: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    """Clé de tri pour ordonner les changements retenus dans le rapport final.

    Ordre de priorité : nouvelles idées en premier, puis impact décroissant
    (MAJEUR → MODERE → MINEUR), puis nombre de thèmes AMF décroissant (un
    changement multi-label étant a priori plus structurant), puis section,
    puis page, puis type de diff.
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
    return (
        0 if triage.get("nouvelle_idee", False) else 1,
        impact_order,
        -themes_count,
        str(change.get("section_key") or ""),
        first_page,
        diff_type,
    )


def _build_global_summary(section_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrège les statistiques de toutes les sections en un résumé global.

    Calcule les comptages par impact, catégorie et action requise, extrait
    les cinq premiers résumés de changements comme points saillants, et détermine
    la pertinence globale (FAIBLE / MOYENNE / ELEVEE) selon le nombre de changements
    majeurs. Utilisé pour les deux champs ``global_summary`` et ``all_changes_summary``
    du payload final.
    """
    all_changes = [block for section in section_comparisons for block in (section.get("block_comparisons") or [])]
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
