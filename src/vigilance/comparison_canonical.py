"""Fonctions utilitaires pour le payload canonique UI/export de comparaison utilise par Dash.

Distinction importante :
- ``TableArtifact`` est l'objet de comparaison canonique en memoire.
- Ce module gere la forme du payload canonique UI/export consomme par Dash.

Les anciens noms de fonctions sont conserves comme wrappers pour la
compatibilite, mais les noms preferes dans ce module mentionnent
explicitement le role de payload UI.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any

from vigilance.quarter_utils import format_quarter_label, get_payload_quarter_context

UI_COMPARISON_PAYLOAD_SCHEMA_VERSION = "comparison_canonical_v1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GenAI triage → UI field mapping
# ---------------------------------------------------------------------------

_CATEGORY_TO_RELEVANCE: dict[str, str] = {
    "REGLEMENTAIRE": "Réglementaire",
    "RISQUE": "Risque",
    "CAPITAL": "Capital",
    "STRUCTURE": "Structure",
    "NON_PERTINENT": "Non pertinent",
    "INCONNU": "Inconnu",
}

_RELEVANCE_TO_PRIORITY: dict[str, str] = {
    "ELEVEE": "critique",
    "MOYENNE": "prioritaire",
    "FAIBLE": "normale",
}


def _map_genai_triage_to_ui(triage: dict[str, Any]) -> dict[str, Any]:
    """Expose le triage AMF v2 unifié pour la consommation UI / export.

    Schéma exposé (source de vérité unique) :
    - ``is_relevant`` (bool) — décision pertinence
    - ``themes_amf`` (list[str]) — multi-label, codes AMF
    - ``impact_level`` (str) — MAJEUR / MODERE / MINEUR
    - ``nouvelle_idee`` (bool) — décision OUI/NON IA
    - ``nouvelle_idee_justification`` (str) — 2 phrases ancrées sur le rapport
    - ``action_requise`` (str) — revue_prioritaire / investigation / ...
    - ``exclusion_reason`` (str | None) — si is_relevant=False
    - ``category`` (str) — REGLEMENTAIRE / RISQUE / CAPITAL / STRUCTURE / NON_PERTINENT
      (présent dans le triage brut, propagé tel quel)

    Plus de mapping legacy translated (relevance / risk_level / review_priority).
    Les consommateurs ont été migrés pour lire directement ``category``,
    ``impact_level`` et ``nouvelle_idee``.
    """
    if not triage:
        return {}
    mapped: dict[str, Any] = dict(triage)

    # Normalisation des champs AMF v2 (formes canoniques).
    mapped["is_relevant"] = bool(triage.get("is_relevant", False))
    mapped["themes_amf"] = [
        str(t).upper() for t in (triage.get("themes_amf") or []) if t
    ]
    mapped["impact_level"] = str(triage.get("impact_level") or "MINEUR").upper()
    mapped["nouvelle_idee"] = bool(triage.get("nouvelle_idee", False))
    mapped["nouvelle_idee_justification"] = str(
        triage.get("nouvelle_idee_justification") or ""
    )
    mapped["action_requise"] = str(triage.get("action_requise") or "aucune").lower()
    mapped["exclusion_reason"] = triage.get("exclusion_reason")
    return mapped


# ---------------------------------------------------------------------------
# Changed-tables metrics
# ---------------------------------------------------------------------------


def _is_comparison_changed(c: dict[str, Any]) -> bool:
    """Retourne True lorsqu'une entree de comparaison appariee a un changement detecte.

    Couvre les diffs au niveau des indicateurs (table_status != 'stable')
    et les changements de notes de bas de page qui n'alterent pas le
    table_status.
    """
    if c.get("table_status", "stable") != "stable":
        return True
    fn = c.get("footnotes_counts") or {}
    return bool(fn.get("added", 0) or fn.get("removed", 0) or fn.get("modified", 0))


def compute_changed_tables_t1(result: dict[str, Any]) -> int:
    """Compte les tableaux T1 distincts impliques dans au moins un changement.

    Un tableau T1 est considere comme modifie lorsqu'il participe a l'une des
    situations suivantes :

    - une paire appariee avec des diffs d'indicateurs/notes ou un changement
      de structure,
    - un retrait du T2, alors qu'il etait present en T1.

    Utilise ``table_id_t1`` (paires appariees) et ``table_id``
    (tables_removed) comme cles de deduplication stables.
    """
    changed: set[str] = set()
    for c in result.get("table_comparisons", []):
        if _is_comparison_changed(c):
            tid = c.get("table_id_t1")
            if tid:
                changed.add(tid)
    for t in result.get("tables_removed", []):
        tid = t.get("table_id")
        if tid:
            changed.add(tid)
    return len(changed)


def compute_changed_tables_t2(result: dict[str, Any]) -> int:
    """Compte les tableaux T2 distincts impliques dans au moins un changement.

    Un tableau T2 est considere comme modifie lorsqu'il participe a l'une des
    situations suivantes :

    - une paire appariee avec des diffs d'indicateurs/notes ou un changement
      de structure,
    - un ajout au T2, alors qu'il etait absent en T1.

    Utilise ``table_id_t2`` (paires appariees) et ``table_id``
    (tables_added) comme cles de deduplication stables.
    """
    changed: set[str] = set()
    for c in result.get("table_comparisons", []):
        if _is_comparison_changed(c):
            tid = c.get("table_id_t2")
            if tid:
                changed.add(tid)
    for t in result.get("tables_added", []):
        tid = t.get("table_id")
        if tid:
            changed.add(tid)
    return len(changed)


def get_meta_value(meta: dict[str, Any] | None, *keys: str) -> Any:
    """Recupere une valeur imbriquee depuis les metadonnees de maniere securisee.

    Parcourt le chemin de cles dans le dict imbrique. Retourne ``None`` si
    ``meta`` est ``None``, si une valeur intermediaire n'est pas un dict,
    ou si une cle est absente.

    Args:
        meta: Dictionnaire de metadonnees, ou ``None`` (traite comme vide).
        *keys: Une ou plusieurs cles pour la recherche imbriquee
            (ex. ``("section", "title")``).

    Returns:
        La valeur au chemin imbrique, ou ``None`` si le chemin ne peut etre
        parcouru.
    """
    cur: Any = meta or {}
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def is_ui_comparison_payload(payload: Any) -> bool:
    """Retourne True lorsque le payload suit le contrat canonique Dash/UI de comparaison."""
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    )


def new_empty_ui_comparison_payload() -> dict[str, Any]:
    """Construit un payload canonique UI de comparaison vide avec la structure par defaut.

    Retourne un dict contenant toutes les cles de premier niveau requises
    par l'application Dash : schema_version, bank_code, champs de trimestre,
    summary (compteurs a zero et status_counts), listes vides pour
    table_comparisons/tables_added/tables_removed et leurs variantes, et
    meta (generated_at, provenance, source_format, executive_summary).
    Le executive_summary est un message en francais indiquant qu'aucune
    comparaison n'est disponible.

    Returns:
        Dictionnaire entierement initialise conforme au schema canonique.
    """
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": UI_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "bank_code": "",
        "quarter_from": "",
        "quarter_to": "",
        "previous_quarter": "",
        "current_quarter": "",
        "comparison_direction": "current_vs_previous",
        "year": datetime.now().year,
        "summary": {
            "tables_t1": 0,
            "tables_t2": 0,
            "tables_extracted_t1": 0,
            "tables_extracted_t2": 0,
            "tables_comparable_t1": 0,
            "tables_comparable_t2": 0,
            "tables_matched": 0,
            "tables_added": 0,
            "tables_removed": 0,
            "tables_added_confirmed": 0,
            "tables_removed_confirmed": 0,
            "tables_added_pending_review": 0,
            "tables_removed_pending_review": 0,
            "artifacts_confirmed_previous": 0,
            "artifacts_confirmed_current": 0,
            "extraction_suspects_previous": 0,
            "extraction_suspects_current": 0,
            "ambiguous_tables": 0,
            "review_candidates": 0,
            "ambiguous_pairs": 0,
            "pairing_coverage": 0.0,
            "indicator_change_pairs": 0,
            "footnote_change_pairs": 0,
            "pairing_low_confidence": False,
            "tables_changed_t1": 0,
            "tables_changed_t2": 0,
            "total_added_indicators": 0,
            "total_removed_indicators": 0,
            "total_renamed_indicators": 0,
            "status_counts": {
                "stable": 0,
                "modifie": 0,
                "renommage_probable": 0,
                "incertain": 0,
                "review_candidate": 0,
                "needs_review": 0,
                "structure_change": 0,
                "ajoute": 0,
                "supprime": 0,
                "ajoute_pending_review": 0,
                "supprime_pending_review": 0,
                "artefact": 0,
                "suspect_extraction": 0,
            },
        },
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
        "artifacts_confirmed_previous": [],
        "artifacts_confirmed_current": [],
        "extraction_suspects_previous": [],
        "extraction_suspects_current": [],
        "tables_added_confirmed": [],
        "tables_removed_confirmed": [],
        "tables_added_pending_review": [],
        "tables_removed_pending_review": [],
        "review_candidates": [],
        "meta": {
            "generated_at": now,
            "provenance": "dash_adapter",
            "source_format": "fallback",
            "executive_summary": {"content": "Aucune comparaison disponible."},
        },
    }


def _quarter_label(quarter: str, year: int) -> str:
    """Formate un libelle de trimestre lisible a partir du code et de l'annee."""
    return format_quarter_label(quarter, year)


def _build_report_comparison_summary_text(
    *,
    current_label: str,
    previous_label: str,
    matched_pairs: int,
    tables_added: int,
    tables_removed: int,
    indicator_changes: int,
    footnote_changes: int,
    high_priority_items: int,
) -> str:
    """Genere le texte du resume executif de la comparaison entre deux trimestres.

    Args:
        current_label: Libelle du trimestre courant.
        previous_label: Libelle du trimestre precedent.
        matched_pairs: Nombre de paires appariees.
        tables_added: Nombre de tableaux ajoutes.
        tables_removed: Nombre de tableaux supprimes.
        indicator_changes: Nombre total de changements d'indicateurs.
        footnote_changes: Nombre total de changements de notes de bas de page.
        high_priority_items: Nombre d'elements a priorite elevee.

    Returns:
        Texte du resume executif en francais.
    """
    parts = [
        f"Comparaison {current_label} vs {previous_label} : {matched_pairs} tableau(x) apparié(s)."
    ]
    parts.append(f" {tables_added} tableau(x) ajouté(s), {tables_removed} supprimé(s).")
    parts.append(
        f" {indicator_changes} changement(s) d'indicateur et {footnote_changes} changement(s) de note de bas de tableau détecté(s)."
    )
    if high_priority_items:
        parts.append(f" {high_priority_items} élément(s) prioritaire(s) à réviser.")
    else:
        parts.append(" Aucun élément prioritaire détecté.")
    return "".join(parts)


def _build_footnotes_diff(
    technical_diff: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Construit le diff structure des notes de bas de page pour le payload UI.

    Args:
        technical_diff: Dictionnaire du diff technique contenant
            ``footnotes_added``, ``footnotes_removed`` et ``footnotes_renamed``.

    Returns:
        Tuple ``(diff_dict, counts_dict)`` ou ``diff_dict`` contient les listes
        ``added``, ``removed``, ``modified`` et ``counts``, et ``counts_dict``
        est le sous-dictionnaire des compteurs.
    """
    added_items = [
        {
            "footnote_ref": str(item.get("id", "") or ""),
            "old_text": "",
            "new_text": str(item.get("text", "") or ""),
            "change_type": "new",
            "reason": str(item.get("reason", "") or ""),
            "analyst_assessment": dict(item.get("analyst_assessment") or {})
            if isinstance(item.get("analyst_assessment"), dict)
            else {},
        }
        for item in list(technical_diff.get("footnotes_added", []) or [])
        if isinstance(item, dict)
    ]
    removed_items = [
        {
            "footnote_ref": str(item.get("id", "") or ""),
            "old_text": str(item.get("text", "") or ""),
            "new_text": "",
            "change_type": "removed",
            "reason": str(item.get("reason", "") or ""),
            "analyst_assessment": dict(item.get("analyst_assessment") or {})
            if isinstance(item.get("analyst_assessment"), dict)
            else {},
        }
        for item in list(technical_diff.get("footnotes_removed", []) or [])
        if isinstance(item, dict)
    ]
    modified_items = [
        {
            "footnote_ref": str(
                item.get("current_id") or item.get("previous_id") or ""
            ),
            "old_text": str(item.get("previous_text", "") or ""),
            "new_text": str(item.get("current_text", "") or ""),
            "change_type": "modified",
            "reason": str(item.get("reason", "") or ""),
            "analyst_assessment": dict(item.get("analyst_assessment") or {})
            if isinstance(item.get("analyst_assessment"), dict)
            else {},
        }
        for item in list(technical_diff.get("footnotes_renamed", []) or [])
        if isinstance(item, dict)
    ]
    counts = {
        "added": len(added_items),
        "removed": len(removed_items),
        "modified": len(modified_items),
    }
    return (
        {
            "added": added_items,
            "removed": removed_items,
            "modified": modified_items,
            "counts": counts,
        },
        counts,
    )


def _warn_missing_visual_context(
    *,
    previous_table_id: str,
    current_table_id: str,
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
) -> None:
    """Journalise un avertissement si le contexte visuel (page/bbox) est absent."""
    missing_page = []
    missing_bbox = []

    if previous_table.get("page") is None:
        missing_page.append("t1")
    if current_table.get("page") is None:
        missing_page.append("t2")
    if previous_table.get("bbox") is None:
        missing_bbox.append("t1")
    if current_table.get("bbox") is None:
        missing_bbox.append("t2")

    if not missing_page and not missing_bbox:
        return

    logger.warning(
        "[comparison_canonical] missing_visual_context previous_table_id=%s current_table_id=%s missing_page=%s missing_bbox=%s",
        previous_table_id,
        current_table_id,
        ",".join(missing_page) or "-",
        ",".join(missing_bbox) or "-",
    )


def _report_comparison_to_ui_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convertit un payload ``report_comparison`` brut en payload canonique UI Dash.

    Peuple les comparaisons de tableaux, les tableaux ajoutes/supprimes,
    les artefacts, les suspects d'extraction, le resume et les metadonnees.

    Args:
        payload: Payload brut avec ``artifact_type == "report_comparison"``.

    Returns:
        Dictionnaire conforme au schema canonique UI de comparaison.
    """
    ui_payload = new_empty_ui_comparison_payload()
    bank_code = str(payload.get("bank_code", "") or "").lower()
    year_previous = int(payload.get("year_previous", 0) or 0)
    year_current = int(payload.get("year_current", 0) or 0)
    quarter_previous = str(payload.get("quarter_previous", "") or "").lower()
    quarter_current = str(payload.get("quarter_current", "") or "").lower()
    current_label = _quarter_label(quarter_current, year_current)
    previous_label = _quarter_label(quarter_previous, year_previous)
    pdf_previous = str(
        payload.get("archived_pdf_previous") or payload.get("source_pdf_previous") or ""
    ).strip()
    pdf_current = str(
        payload.get("archived_pdf_current") or payload.get("source_pdf_current") or ""
    ).strip()

    matched_lookup = {
        str(item.get("previous_table_id", "") or ""): item
        for item in list((payload.get("matching") or {}).get("matched_pairs", []) or [])
        if isinstance(item, dict)
    }

    ui_payload["bank_code"] = bank_code
    ui_payload["year"] = year_current or ui_payload["year"]
    ui_payload["quarter_from"] = previous_label
    ui_payload["quarter_to"] = current_label
    ui_payload["previous_quarter"] = previous_label
    ui_payload["current_quarter"] = current_label
    ui_payload["comparison_direction"] = "current_vs_previous"

    table_comparisons: list[dict[str, Any]] = []
    for item in list(payload.get("pair_comparisons", []) or []):
        if not isinstance(item, dict):
            continue
        previous_table_id = str(item.get("previous_table_id", "") or "").strip()
        current_table_id = str(item.get("current_table_id", "") or "").strip()
        previous_table = item.get("previous_table") or {}
        current_table = item.get("current_table") or {}
        technical_diff = item.get("technical_diff", {}) or {}
        _warn_missing_visual_context(
            previous_table_id=previous_table_id,
            current_table_id=current_table_id,
            previous_table=previous_table,
            current_table=current_table,
        )
        footnotes_diff, footnotes_counts = _build_footnotes_diff(technical_diff)
        added_raw = [
            entry
            for entry in list(technical_diff.get("indicators_added", []) or [])
            if isinstance(entry, dict) and str(entry.get("value", "") or "").strip()
        ]
        removed_raw = [
            entry
            for entry in list(technical_diff.get("indicators_removed", []) or [])
            if isinstance(entry, dict) and str(entry.get("value", "") or "").strip()
        ]
        renamed_raw = [
            entry
            for entry in list(technical_diff.get("indicators_renamed", []) or [])
            if isinstance(entry, dict)
            and str(entry.get("previous", "") or "").strip()
            and str(entry.get("current", "") or "").strip()
        ]
        added = [str(entry.get("value", "") or "").strip() for entry in added_raw]
        removed = [str(entry.get("value", "") or "").strip() for entry in removed_raw]
        renamed = [
            {
                "from": str(entry.get("previous", "") or "").strip(),
                "to": str(entry.get("current", "") or "").strip(),
            }
            for entry in renamed_raw
        ]
        match_info = matched_lookup.get(previous_table_id, {})

        all_inds_t1 = list(previous_table.get("indicators", []) or [])
        all_inds_t2 = list(current_table.get("indicators", []) or [])
        drastic_row_drop = False
        if len(all_inds_t1) > 0:
            drop_ratio = (len(all_inds_t1) - len(all_inds_t2)) / len(all_inds_t1)
            # If the row count dropped by >= 80% and the absolute drop is > 5 rows
            if drop_ratio >= 0.8 and (len(all_inds_t1) - len(all_inds_t2)) > 5:
                drastic_row_drop = True

        table_comparisons.append(
            {
                "table_id_t1": previous_table_id,
                "table_id_t2": current_table_id,
                "title_t1": str(previous_table.get("title", "") or ""),
                "title_t2": str(current_table.get("title", "") or ""),
                "page_t1": previous_table.get("page"),
                "page_t2": current_table.get("page"),
                "section": str(
                    current_table.get("section")
                    or previous_table.get("section")
                    or "unknown_section"
                ),
                "bbox_t1": previous_table.get("bbox"),
                "bbox_t2": current_table.get("bbox"),
                "source_pdf_t1": pdf_previous,
                "source_pdf_t2": pdf_current,
                "table_status": str(
                    technical_diff.get("table_level_change", "inchange") or "inchange"
                ),
                "match_score": float(
                    item.get("match_confidence")
                    or match_info.get("match_confidence")
                    or 0.0
                ),
                "match_reason": str(
                    item.get("match_reason") or match_info.get("reason") or ""
                ),
                "added_indicators": added,
                "removed_indicators": removed,
                "renamed_indicators": renamed,
                "added_indicators_raw": added_raw,
                "removed_indicators_raw": removed_raw,
                "renamed_indicators_raw": renamed_raw,
                "all_indicators_t1": list(previous_table.get("indicators", []) or []),
                "all_indicators_t2": list(current_table.get("indicators", []) or []),
                "counts": {
                    "added": len(added),
                    "removed": len(removed),
                    "renamed": len(renamed),
                    "renamed_probable": 0,
                },
                "footnotes_diff": footnotes_diff,
                "footnotes_counts": footnotes_counts,
                "uncertain_diff": False,
                "genai_analysis": {
                    **dict(item.get("analyst_assessment") or {}),
                    **_map_genai_triage_to_ui(item.get("genai_triage") or {}),
                },
                "match_metadata": {
                    "drastic_row_drop": drastic_row_drop,
                    "theme": str(
                        (item.get("analyst_assessment") or {}).get("theme") or ""
                    ),
                    "review_priority": str(
                        (item.get("analyst_assessment") or {}).get("review_priority")
                        or ""
                    ),
                    "change_significance": str(
                        (item.get("analyst_assessment") or {}).get(
                            "change_significance"
                        )
                        or ""
                    ),
                    "analyst_summary": str(
                        (item.get("analyst_assessment") or {}).get("analyst_summary")
                        or ""
                    ),
                },
            }
        )

    def _table_side_items(items: list[Any], side: str) -> list[dict[str, Any]]:
        """Construit les lignes UI pour les tableaux d'un cote (previous/current)."""
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            table_id = str(item.get("table_id", "") or "").strip()
            out.append(
                {
                    "table_id": table_id,
                    "title": str(item.get("title", "") or ""),
                    "page": item.get("page"),
                    "section": str(item.get("section", "") or "unknown_section"),
                    "bbox_t1": item.get("bbox") if side == "previous" else None,
                    "bbox_t2": item.get("bbox") if side == "current" else None,
                    "source_pdf_t1": pdf_previous if side == "previous" else "",
                    "source_pdf_t2": pdf_current if side == "current" else "",
                    "all_indicators_t1": list(item.get("indicators", []) or [])
                    if side == "previous"
                    else [],
                    "all_indicators_t2": list(item.get("indicators", []) or [])
                    if side == "current"
                    else [],
                    "first_column_indicators_raw": list(
                        item.get("indicators", []) or []
                    ),
                    "first_column_indicators": list(item.get("indicators", []) or []),
                    "genai_analysis": {
                        **dict(item.get("analyst_assessment") or {}),
                        **_map_genai_triage_to_ui(item.get("genai_triage") or {}),
                    },
                    "semantic_judge": str(item.get("reason", "") or ""),
                }
            )
        return out

    matching = payload.get("matching") or {}
    ui_payload["table_comparisons"] = table_comparisons
    ui_payload["tables_added"] = _table_side_items(
        list(matching.get("tables_added", []) or []),
        "current",
    )
    ui_payload["tables_removed"] = _table_side_items(
        list(matching.get("tables_removed", []) or []),
        "previous",
    )
    ui_payload["artifacts_confirmed_previous"] = _table_side_items(
        list(matching.get("artifacts_confirmed_previous", []) or []),
        "previous",
    )
    ui_payload["artifacts_confirmed_current"] = _table_side_items(
        list(matching.get("artifacts_confirmed_current", []) or []),
        "current",
    )
    ui_payload["extraction_suspects_previous"] = _table_side_items(
        list(matching.get("extraction_suspects_previous", []) or []),
        "previous",
    )
    ui_payload["extraction_suspects_current"] = _table_side_items(
        list(matching.get("extraction_suspects_current", []) or []),
        "current",
    )
    ui_payload["tables_added_confirmed"] = list(ui_payload["tables_added"])
    ui_payload["tables_removed_confirmed"] = list(ui_payload["tables_removed"])
    ui_payload["tables_added_pending_review"] = list(
        ui_payload["extraction_suspects_current"]
    )
    ui_payload["tables_removed_pending_review"] = list(
        ui_payload["extraction_suspects_previous"]
    )
    ui_payload["review_candidates"] = list(
        ui_payload["extraction_suspects_current"]
    ) + list(ui_payload["extraction_suspects_previous"])

    indicator_change_pairs = sum(
        1
        for item in table_comparisons
        if item["counts"]["added"]
        or item["counts"]["removed"]
        or item["counts"]["renamed"]
    )
    footnote_change_pairs = sum(
        1
        for item in table_comparisons
        if any((item.get("footnotes_counts") or {}).values())
    )
    matched_pairs_total = int(
        (payload.get("summary") or {}).get("matched_pairs_total", 0) or 0
    )
    tables_extracted_t1 = (
        matched_pairs_total
        + len(ui_payload["tables_removed"])
        + len(ui_payload["artifacts_confirmed_previous"])
        # NOTE: extraction_suspects excluded from totals – they are unverified
    )
    tables_extracted_t2 = (
        matched_pairs_total
        + len(ui_payload["tables_added"])
        + len(ui_payload["artifacts_confirmed_current"])
        # NOTE: extraction_suspects excluded from totals – they are unverified
    )
    tables_t1 = tables_extracted_t1
    tables_t2 = tables_extracted_t2
    tables_comparable_t1 = matched_pairs_total + len(ui_payload["tables_removed"])
    tables_comparable_t2 = matched_pairs_total + len(ui_payload["tables_added"])
    pairing_coverage = (
        matched_pairs_total / max(tables_comparable_t1, tables_comparable_t2, 1)
        if tables_comparable_t1 or tables_comparable_t2
        else 0.0
    )
    summary = ui_payload["summary"]
    summary.update(
        {
            "tables_t1": tables_t1,
            "tables_t2": tables_t2,
            "tables_extracted_t1": tables_extracted_t1,
            "tables_extracted_t2": tables_extracted_t2,
            "tables_comparable_t1": tables_comparable_t1,
            "tables_comparable_t2": tables_comparable_t2,
            "tables_matched": matched_pairs_total,
            "tables_added": len(ui_payload["tables_added"]),
            "tables_removed": len(ui_payload["tables_removed"]),
            "tables_added_confirmed": len(ui_payload["tables_added"]),
            "tables_removed_confirmed": len(ui_payload["tables_removed"]),
            "tables_added_pending_review": len(
                ui_payload["tables_added_pending_review"]
            ),
            "tables_removed_pending_review": len(
                ui_payload["tables_removed_pending_review"]
            ),
            "artifacts_confirmed_previous": len(
                ui_payload["artifacts_confirmed_previous"]
            ),
            "artifacts_confirmed_current": len(
                ui_payload["artifacts_confirmed_current"]
            ),
            "extraction_suspects_previous": len(
                ui_payload["extraction_suspects_previous"]
            ),
            "extraction_suspects_current": len(
                ui_payload["extraction_suspects_current"]
            ),
            "ambiguous_tables": 0,
            "review_candidates": len(ui_payload["review_candidates"]),
            "ambiguous_pairs": 0,
            "pairing_coverage": pairing_coverage,
            "indicator_change_pairs": indicator_change_pairs,
            "footnote_change_pairs": footnote_change_pairs,
            "pairing_low_confidence": any(
                float(item.get("match_score", 0.0) or 0.0) < 0.75
                for item in table_comparisons
            ),
            "total_added_indicators": sum(
                len(item.get("added_indicators", []) or [])
                for item in table_comparisons
            ),
            "total_removed_indicators": sum(
                len(item.get("removed_indicators", []) or [])
                for item in table_comparisons
            ),
            "total_renamed_indicators": sum(
                len(item.get("renamed_indicators", []) or [])
                for item in table_comparisons
            ),
        }
    )
    summary["status_counts"].update(
        {
            "stable": sum(
                1
                for item in table_comparisons
                if item.get("table_status") == "inchange"
            ),
            "modifie": sum(
                1 for item in table_comparisons if item.get("table_status") == "modifie"
            ),
            "ajoute": len(ui_payload["tables_added"]),
            "supprime": len(ui_payload["tables_removed"]),
            "ajoute_pending_review": len(ui_payload["tables_added_pending_review"]),
            "supprime_pending_review": len(ui_payload["tables_removed_pending_review"]),
            "artefact": len(ui_payload["artifacts_confirmed_previous"])
            + len(ui_payload["artifacts_confirmed_current"]),
            "suspect_extraction": len(ui_payload["extraction_suspects_previous"])
            + len(ui_payload["extraction_suspects_current"]),
        }
    )
    summary["tables_changed_t1"] = compute_changed_tables_t1(ui_payload)
    summary["tables_changed_t2"] = compute_changed_tables_t2(ui_payload)

    raw_summary = payload.get("summary") or {}
    quarter_context = {
        "current": {
            "label": current_label,
            "code": quarter_current,
            "quarter": int(quarter_current[1])
            if quarter_current.startswith("t")
            else None,
            "year": year_current,
        },
        "previous": {
            "label": previous_label,
            "code": quarter_previous,
            "quarter": int(quarter_previous[1])
            if quarter_previous.startswith("t")
            else None,
            "year": year_previous,
        },
        "comparison_direction": "current_vs_previous",
        "comparison_label": f"{current_label} vs {previous_label}",
    }
    ui_payload["meta"].update(
        {
            "source_format": "report_comparison",
            "model_version": str(payload.get("model_version", "") or ""),
            "prompt_version_match": str(payload.get("prompt_version_match", "") or ""),
            "prompt_version_diff": str(payload.get("prompt_version_diff", "") or ""),
            "quarter_context": quarter_context,
            "run_id": str(payload.get("run_id", "") or ""),
            "reference_resolution": dict(payload.get("reference_resolution") or {}),
            "run_metrics": dict(payload.get("run_metrics") or {}),
            "pdf_paths": {
                "pdf_t1": pdf_previous,
                "pdf_t2": pdf_current,
                "pdf_previous": pdf_previous,
                "pdf_current": pdf_current,
            },
            "archived_pdf_previous": str(
                payload.get("archived_pdf_previous", "") or ""
            ),
            "archived_pdf_current": str(payload.get("archived_pdf_current", "") or ""),
            "source_pdf_previous": str(payload.get("source_pdf_previous", "") or ""),
            "source_pdf_current": str(payload.get("source_pdf_current", "") or ""),
            "global_summary": dict(payload.get("global_summary") or {}),
            "executive_summary": {
                "content": _build_report_comparison_summary_text(
                    current_label=current_label,
                    previous_label=previous_label,
                    matched_pairs=matched_pairs_total,
                    tables_added=len(ui_payload["tables_added"]),
                    tables_removed=len(ui_payload["tables_removed"]),
                    indicator_changes=int(
                        raw_summary.get("indicator_changes_total", 0) or 0
                    ),
                    footnote_changes=int(
                        raw_summary.get("footnote_changes_total", 0) or 0
                    ),
                    high_priority_items=int(
                        raw_summary.get("high_priority_items_total", 0) or 0
                    ),
                )
            },
        }
    )
    return ui_payload


def to_ui_comparison_payload(payload: Any) -> dict[str, Any]:
    """Conversion best-effort vers le payload canonique utilise par l'application Dash.

    Si le payload est deja canonique (schema_version correspondant a
    ``UI_COMPARISON_PAYLOAD_SCHEMA_VERSION``), retourne une copie profonde.
    S'il s'agit d'un resultat legacy ``metier_tableaux``, convertit vers la
    forme canonique et peuple table_comparisons depuis la liste ``changes``.
    Sinon, retourne un payload canonique vide avec les champs de base
    remplis a partir de l'entree lorsque possible.

    Args:
        payload: Donnees de comparaison brutes. Peut etre un dict canonique,
            un dict legacy metier avec result_type ``metier_tableaux``, ou
            tout autre dict/objet. Les entrees non-dict produisent un payload
            canonique vide.

    Returns:
        Dictionnaire conforme au schema canonique UI Dash de comparaison.
    """
    if is_ui_comparison_payload(payload):
        return deepcopy(payload)

    ui_payload = new_empty_ui_comparison_payload()
    if not isinstance(payload, dict):
        return ui_payload

    if payload.get("result_type") == "metier_tableaux":
        ui_payload["bank_code"] = str(payload.get("bank_code", ""))
        ui_payload["year"] = int(payload.get("year") or ui_payload["year"])
        changes = payload.get("changes") or []
        table_comparisons: list[dict[str, Any]] = []
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                continue
            ctype = str(change.get("change_type", ""))
            added = (
                [str(change.get("indicator_name"))]
                if ctype in {"added", "table_added"} and change.get("indicator_name")
                else []
            )
            removed = (
                [str(change.get("indicator_name"))]
                if ctype in {"removed", "table_removed"}
                and change.get("indicator_name")
                else []
            )
            table_comparisons.append(
                {
                    "table_id_t1": f"legacy_t1_{index}",
                    "table_id_t2": f"legacy_t2_{index}",
                    "title_t1": change.get("table_title", ""),
                    "title_t2": change.get("table_title", ""),
                    "page_t1": change.get("page_t1"),
                    "page_t2": change.get("page_t2"),
                    "section": change.get("section", "unknown_section"),
                    "added_indicators": added,
                    "removed_indicators": removed,
                    "renamed_indicators": [],
                    "counts": {
                        "added": len(added),
                        "removed": len(removed),
                        "renamed": 0,
                        "renamed_probable": 0,
                    },
                    "uncertain_diff": False,
                    "table_status": "modifie" if added or removed else "stable",
                }
            )
        ui_payload["table_comparisons"] = table_comparisons
        ui_payload["summary"]["tables_matched"] = len(table_comparisons)
        ui_payload["summary"]["total_added_indicators"] = sum(
            len(item.get("added_indicators", [])) for item in table_comparisons
        )
        ui_payload["summary"]["total_removed_indicators"] = sum(
            len(item.get("removed_indicators", [])) for item in table_comparisons
        )
        ui_payload["summary"]["status_counts"]["modifie"] = sum(
            1 for item in table_comparisons if item.get("table_status") == "modifie"
        )
        ui_payload["summary"]["tables_changed_t1"] = compute_changed_tables_t1(
            ui_payload
        )
        ui_payload["summary"]["tables_changed_t2"] = compute_changed_tables_t2(
            ui_payload
        )
        ui_payload["meta"]["source_format"] = "legacy_metier"
        ui_payload["meta"]["executive_summary"] = {
            "content": "Conversion depuis un format metier legacy."
        }
        return ui_payload

    if payload.get("artifact_type") == "report_comparison":
        return _report_comparison_to_ui_payload(payload)

    ui_payload["bank_code"] = str(payload.get("bank_code", ""))
    ui_payload["quarter_from"] = str(payload.get("quarter_from", ""))
    ui_payload["quarter_to"] = str(payload.get("quarter_to", ""))
    ui_payload["previous_quarter"] = str(
        payload.get("previous_quarter", ui_payload["quarter_from"])
    )
    ui_payload["current_quarter"] = str(
        payload.get("current_quarter", ui_payload["quarter_to"])
    )
    ui_payload["comparison_direction"] = str(
        payload.get("comparison_direction", "current_vs_previous")
    )
    try:
        ui_payload["year"] = int(payload.get("year") or ui_payload["year"])
    except (TypeError, ValueError):
        pass

    ui_payload["meta"]["source_format"] = "unknown"
    ui_payload["meta"]["quarter_context"] = get_payload_quarter_context(payload)
    ui_payload["meta"]["executive_summary"] = {
        "content": "Format de comparaison non reconnu. Resultat vide genere."
    }
    return ui_payload


def is_canonical_comparison(payload: Any) -> bool:
    """Alias retro-compatible pour :func:`is_ui_comparison_payload`."""
    return is_ui_comparison_payload(payload)


def _empty_canonical() -> dict[str, Any]:
    """Alias retro-compatible pour :func:`new_empty_ui_comparison_payload`."""
    return new_empty_ui_comparison_payload()


def to_canonical_payload(payload: Any) -> dict[str, Any]:
    """Alias retro-compatible pour :func:`to_ui_comparison_payload`."""
    return to_ui_comparison_payload(payload)
