"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from typing import Any

from vigilance.analyst_change_presentation import build_change_presentation


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


def _build_global_summary(
    section_comparisons: list[dict[str, Any]],
    *,
    bank_code: str = "",
) -> dict[str, Any]:
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
        summary = str(
            triage.get("changement_constate")
            or change.get("change_summary")
            or ""
        ).strip()
        if summary and bank_code:
            summary = build_change_presentation(
                change,
                bank_code=bank_code,
                candidate_summary=summary,
            ).summary
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


def _build_semantic_quality_metrics(
    *,
    section_comparisons: list[dict[str, Any]],
    reconciliation_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Agrège des métriques de qualité pour l'alignement hybride et le triage."""
    all_changes = [
        block
        for section in section_comparisons
        for block in (section.get("all_block_comparisons") or section.get("block_comparisons") or [])
    ]
    alignment_types: dict[str, int] = {}
    ambiguous_count = 0
    human_review_count = 0
    triage_sent_count = 0
    triage_prefiltered_count = 0
    triage_dedup_groups = 0
    triage_dedup_members = 0
    seen_dedup_groups: set[str] = set()

    for change in all_changes:
        alignment_type = str(change.get("alignment_type") or "unknown")
        alignment_types[alignment_type] = alignment_types.get(alignment_type, 0) + 1
        if alignment_type == "ambiguous" or str(change.get("alignment_decision") or "") == "uncertain":
            ambiguous_count += 1
        triage = change.get("genai_triage") or {}
        if triage.get("alignment_review_required"):
            human_review_count += 1
        source = str(triage.get("source") or "")
        if source == "deterministic_prefilter":
            triage_prefiltered_count += 1
        elif source and source not in {
            "alignment_review_required",
            "semantic_alignment_decision",
        }:
            triage_sent_count += 1
        dedup = change.get("triage_dedup") or {}
        group_id = str(dedup.get("group_id") or triage.get("triage_group_id") or "")
        if group_id:
            if group_id not in seen_dedup_groups:
                seen_dedup_groups.add(group_id)
                triage_dedup_groups += 1
            triage_dedup_members += 1

    audit_rows = list(reconciliation_audit or [])
    applied_reconciliations = sum(1 for row in audit_rows if row.get("applied"))
    total_changes = len(all_changes)
    return {
        "total_changes": total_changes,
        "alignment_type_counts": dict(sorted(alignment_types.items())),
        "ambiguous_alignment_rate": (
            round(ambiguous_count / total_changes, 4) if total_changes else 0.0
        ),
        "ambiguous_alignment_count": ambiguous_count,
        "reconciliation_component_count": len(audit_rows),
        "reconciliation_applied_count": applied_reconciliations,
        "reconciliation_applied_rate": (
            round(applied_reconciliations / len(audit_rows), 4) if audit_rows else 0.0
        ),
        "triage_prefiltered_count": triage_prefiltered_count,
        "triage_sent_count": triage_sent_count,
        "triage_dedup_group_count": triage_dedup_groups,
        "triage_dedup_member_count": triage_dedup_members,
        "triage_duplicate_rate": (
            round(
                max(0, triage_dedup_members - triage_dedup_groups) / total_changes,
                4,
            )
            if total_changes and triage_dedup_groups
            else 0.0
        ),
        "human_review_count": human_review_count,
        "human_review_rate": (
            round(human_review_count / total_changes, 4) if total_changes else 0.0
        ),
    }
