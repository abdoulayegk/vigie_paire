"""Agregation des indicateurs du cockpit sur l'ensemble des comparaisons.

Extrait de ``vigie_dashboard_flow.py`` sans modification.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from vigilance.dash_app.services.export_helpers import _is_high_priority_item
from vigilance.dash_app.services.review_navigation import _table_decision_bucket
from vigilance.vigie_columns import build_text_vigie_display_row

from .formatting import (
    _change_total,
    _comparisons,
    _count_list,
    _footnote_counts,
    _indicator_confidence,
    _low_confidence,
    _safe_float,
    _safe_int,
)

def _text_changes(text_data: dict | None, *, relevant_only: bool = True) -> list[tuple[dict, str]]:
    """Aplatit les changements texte d'un ``text_comparison.json`` en liste (changement, section)."""
    if not isinstance(text_data, dict):
        return []
    rows: list[tuple[dict, str]] = []
    for section in text_data.get("section_comparisons") or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("section_title") or section.get("section_key") or "Section")
        for change in section.get("all_block_comparisons") or []:
            if not isinstance(change, dict):
                continue
            triage = change.get("genai_triage") or {}
            if change.get("diff_type") == "unchanged" or triage.get("source") == "skip":
                continue
            if relevant_only:
                if triage and not bool(triage.get("is_relevant", False)):
                    continue
            rows.append((change, section_title))
    return rows


def _text_metrics(text_data: dict | None) -> dict[str, Any]:
    """Agrège les KPIs du pipeline texte (compteurs par impact, mots impactés, top 5)."""
    summary = (text_data or {}).get("global_summary") or (text_data or {}).get("all_changes_summary") or {}
    counts = summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}
    exportable_changes = _text_changes(text_data, relevant_only=False)
    relevant_changes = [
        (change, section)
        for change, section in exportable_changes
        if bool((change.get("genai_triage") or {}).get("is_relevant", False))
    ]
    sections = {section for _, section in exportable_changes}
    added_words = 0
    removed_words = 0
    added_changes = 0
    removed_changes = 0
    modified = 0
    renamed_changes = 0
    regulatory = 0
    confidences: list[float] = []
    top: list[dict[str, str]] = []
    bank_code = str((text_data or {}).get("bank_code") or "")
    for change, section in exportable_changes:
        triage = change.get("genai_triage") or {}
        display = build_text_vigie_display_row(
            change,
            section_title=section,
            bank_code=bank_code,
        )
        diff_type = str(change.get("diff_type") or "")
        if diff_type == "added":
            added_changes += 1
        if diff_type == "removed":
            removed_changes += 1
        if diff_type == "modified":
            modified += 1
        if diff_type == "renamed":
            renamed_changes += 1
        themes = {str(v).upper() for v in triage.get("themes_amf") or []}
        if "EXIGENCES_REGLEMENTAIRES" in themes or str(triage.get("category", "")).upper() == "REGLEMENTAIRE":
            regulatory += 1
        added_words += len(str(change.get("source_text_t2") or change.get("semantic_text_t2") or "").split())
        removed_words += len(str(change.get("source_text_t1") or change.get("semantic_text_t1") or "").split())
        if triage.get("confidence") is not None:
            confidences.append(_safe_float(triage.get("confidence")))
        top.append(
            {
                "summary": str(display.get("what_changed") or "Changement textuel détecté"),
                "impact": str(triage.get("impact_level") if triage.get("is_relevant") else "NON_PERTINENT").upper(),
                "section": section,
            }
        )
    top.sort(key=lambda item: {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2, "NON_PERTINENT": 3}.get(item["impact"], 9))
    return {
        "major": _safe_int(by_impact.get("MAJEUR")),
        "moderate": _safe_int(by_impact.get("MODERE")),
        "total": len(exportable_changes),
        "relevant": len(relevant_changes),
        "analyzed": len(exportable_changes),
        "sections": len(sections),
        "modified": modified,
        "renamed_changes": renamed_changes,
        "added_changes": added_changes,
        "removed_changes": removed_changes,
        "added_mentions": added_words,
        "removed_mentions": removed_words,
        "words_impacted": added_words + removed_words,
        "regulatory": regulatory or _safe_int((counts.get("by_category") or {}).get("REGLEMENTAIRE")),
        "pertinence": str(summary.get("pertinence_globale") or "N/D").upper(),
        "confidence_values": confidences,
        "top": top[:5],
    }


def _indicator_metrics(indicator: dict | None) -> dict[str, Any]:
    """Agrège les KPIs du pipeline tableaux (indicateurs, footnotes, tableaux ajoutés / retirés)."""
    if not isinstance(indicator, dict):
        return {"comparisons": [], "confidence_values": [], "total_changes": 0}
    summary = indicator.get("summary", indicator.get("kpi_metier", {})) or {}
    comparisons = _comparisons(indicator)
    tables_added = indicator.get("tables_added", []) or []
    tables_removed = indicator.get("tables_removed", []) or []
    added = sum(_count_list(comp, "added_indicators", "indicators_added") for comp in comparisons)
    removed = sum(_count_list(comp, "removed_indicators", "indicators_removed") for comp in comparisons)
    indicator_renamed = sum(_count_list(comp, "renamed_indicators", "indicators_renamed") for comp in comparisons)
    footnote_added = sum(_footnote_counts(comp).get("added", 0) for comp in comparisons)
    footnote_removed = sum(_footnote_counts(comp).get("removed", 0) for comp in comparisons)
    footnote_modified = sum(_footnote_counts(comp).get("modified", 0) for comp in comparisons)
    footnote_renamed = sum(_footnote_counts(comp).get("renamed", 0) for comp in comparisons)
    renamed = indicator_renamed + footnote_renamed
    notes = footnote_added + footnote_removed + footnote_modified + footnote_renamed
    confidence_values = [score for score in (_indicator_confidence(comp) for comp in comparisons) if score is not None]
    tables_added_count = len(tables_added) or _safe_int(summary.get("tables_added_total"))
    tables_removed_count = len(tables_removed) or _safe_int(summary.get("tables_removed_total"))
    indicator_added_count = added or _safe_int(summary.get("total_added_indicators"))
    indicator_removed_count = removed or _safe_int(summary.get("total_removed_indicators"))
    indicator_renamed_count = indicator_renamed or _safe_int(summary.get("total_renamed_indicators"))
    renamed_count = indicator_renamed_count + footnote_renamed
    notes_count = notes or _safe_int(summary.get("footnote_changes_total"))
    return {
        "matched": _safe_int(
            summary.get("tables_matched"), _safe_int(summary.get("matched_pairs_total"), len(comparisons))
        ),
        "tables_removed": tables_removed_count,
        "tables_added": tables_added_count,
        "indicator_added": indicator_added_count,
        "indicator_removed": indicator_removed_count,
        "indicator_renamed": indicator_renamed_count,
        "footnote_added": footnote_added,
        "footnote_removed": footnote_removed,
        "footnote_modified": footnote_modified,
        "footnote_renamed": footnote_renamed,
        "renamed": renamed_count,
        "notes": notes_count,
        "priority": _safe_int(
            summary.get("high_priority_items_total"),
            sum(1 for comp in comparisons if _change_total(comp) and _is_high_priority_item(comp)),
        ),
        "low_confidence": sum(1 for comp in comparisons if _low_confidence(comp)),
        "total_changes": (
            indicator_added_count
            + indicator_removed_count
            + renamed_count
            + footnote_added
            + footnote_removed
            + footnote_modified
            + tables_added_count
            + tables_removed_count
        ),
        "confidence_values": confidence_values,
        "comparisons": comparisons,
    }


def _review_counts(review_queue: list | None, review_items: list | None, indicator: dict | None) -> dict[str, int]:
    """Compte les décisions de revue (total / approved / rejected / pending) par fallback de sources."""
    queue = review_queue if isinstance(review_queue, list) else []
    if queue:
        total = len(queue)
        approved = sum(1 for item in queue if _table_decision_bucket(item) == "approved")
        rejected = sum(1 for item in queue if _table_decision_bucket(item) == "rejected")
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": max(0, total - approved - rejected),
        }
    items = review_items if isinstance(review_items, list) else []
    if items:
        statuses = Counter(str(item.get("status", "pending")).lower() for item in items if isinstance(item, dict))
        total = len(items)
        approved = statuses.get("approved", 0) + statuses.get("validated", 0) + statuses.get("valide", 0)
        rejected = statuses.get("rejected", 0) + statuses.get("rejete", 0)
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": max(0, total - approved - rejected),
        }
    summary = (indicator or {}).get("review_decisions_summary", {}) if isinstance(indicator, dict) else {}
    total = _safe_int(summary.get("matched")) + _safe_int(summary.get("unmatched"))
    pending = _safe_int(summary.get("pending"), total)
    return {"total": total, "approved": max(0, total - pending), "rejected": 0, "pending": pending}
