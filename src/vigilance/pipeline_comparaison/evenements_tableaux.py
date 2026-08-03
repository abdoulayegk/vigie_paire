"""Traitement des ajouts, retraits et exclusions de tableaux."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from vigilance.comparison_analyst import build_analyst_assessment
from vigilance.comparison_io import (
    _is_boundary_inventory_candidate,
    _merge_extraction_suspect_side,
    _table_card,
)
from vigilance.pipeline_comparaison.ancrages_visuels import (
    _infer_opposite_page_from_matched_pairs,
    _resolve_visual_table_anchor,
    _visual_sanity_meta,
)


logger = logging.getLogger(__name__)


def _exclure_candidats_bordure(
    items: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    *,
    side: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Garder les candidats de bordure pour le matching, pas comme changements."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in items:
        table_id = str(item.get("table_id", "") or "").strip()
        snapshot = snapshots.get(table_id, {})
        if not _is_boundary_inventory_candidate(snapshot):
            kept.append(item)
            continue
        excluded.append(
            {
                **item,
                **snapshot,
                "scope_side": side,
                "exclusion_reason": (
                    "Tableau detecte uniquement sur une page limitrophe ajoutee "
                    "pour l'appariement; son absence de correspondance ne constitue "
                    "pas un ajout ou un retrait dans le perimetre compare."
                ),
            }
        )
    return kept, excluded


def construire_evenements_non_apparies(
    match_result: dict[str, Any],
    *,
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
    previous_lookup: dict[str, dict[str, Any]],
    current_lookup: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Construire les ajouts/retraits metier et isoler les pages limitrophes."""
    match_result["tables_added"], boundary_scope_exclusions_current = (
        _exclure_candidats_bordure(
            list(match_result.get("tables_added", []) or []),
            current_snapshots,
            side="current",
        )
    )
    match_result["tables_removed"], boundary_scope_exclusions_previous = (
        _exclure_candidats_bordure(
            list(match_result.get("tables_removed", []) or []),
            previous_snapshots,
            side="previous",
        )
    )

    tables_added: list[dict[str, Any]] = []
    for item in match_result["tables_added"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "ajoute",
        }
        tables_added.append(
            {
                **item,
                **current_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="ajoute",
                ),
            }
        )

    tables_removed: list[dict[str, Any]] = []
    for item in match_result["tables_removed"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "supprime",
        }
        tables_removed.append(
            {
                **item,
                **previous_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=previous_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="supprime",
                ),
            }
        )

    return (
        tables_added,
        tables_removed,
        boundary_scope_exclusions_previous,
        boundary_scope_exclusions_current,
    )


def appliquer_revue_avocat_diable(
    match_result: dict[str, Any],
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
    *,
    previous_business_tables: list[dict[str, Any]],
    current_business_tables: list[dict[str, Any]],
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
    hybrid_recovery_enabled: bool,
    model_name: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_records: list[dict[str, Any]],
    reviewer: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Faire une seconde revue des non-apparies et paires peu fiables."""
    low_confidence_pairs = [
        pair
        for pair in match_result.get("matched_pairs", [])
        if float(pair.get("match_confidence", 1.0)) < 0.90
    ]
    added_ids = {
        item.get("table_id") for item in match_result.get("tables_added", [])
    }
    removed_ids = {
        item.get("table_id") for item in match_result.get("tables_removed", [])
    }
    da_added_cards = [
        _table_card(entry)
        for entry in current_business_tables
        if entry.get("table_id") in added_ids
    ]
    da_removed_cards = [
        _table_card(entry)
        for entry in previous_business_tables
        if entry.get("table_id") in removed_ids
    ]

    if hybrid_recovery_enabled:
        da_result: dict[str, Any] = {
            "new_matches": [],
            "contested_pairs": [],
            "warnings": [],
        }
    else:
        da_result = reviewer(
            da_added_cards,
            da_removed_cards,
            low_confidence_pairs,
            model=model_name,
            call_openai_json=call_openai_json,
            usage_recorder=usage_records,
        )

    for new_match in da_result.get("new_matches", []):
        prev_id = str(new_match.get("previous_table_id", "") or "").strip()
        cur_id = str(new_match.get("current_table_id", "") or "").strip()
        if not prev_id or not cur_id:
            continue
        if prev_id not in previous_snapshots or cur_id not in current_snapshots:
            logger.warning("Devil's Advocate: skipping invalid match %s <-> %s", prev_id, cur_id)
            continue
        match_result["matched_pairs"].append(
            {
                "previous_table_id": prev_id,
                "current_table_id": cur_id,
                "match_confidence": float(new_match.get("match_confidence", 0.75)),
                "reason": str(new_match.get("reason", "")),
                "source": "devil_advocate",
            }
        )
        tables_added = [item for item in tables_added if item.get("table_id") != cur_id]
        tables_removed = [
            item for item in tables_removed if item.get("table_id") != prev_id
        ]
        logger.info(
            "Devil's Advocate promoted match: %s <-> %s (conf=%.2f)",
            prev_id,
            cur_id,
            float(new_match.get("match_confidence", 0.75)),
        )

    for contested in da_result.get("contested_pairs", []):
        prev_id = str(contested.get("previous_table_id", "") or "").strip()
        cur_id = str(contested.get("current_table_id", "") or "").strip()
        for pair in match_result["matched_pairs"]:
            if pair.get("previous_table_id") == prev_id and pair.get("current_table_id") == cur_id:
                pair["review_required"] = True
                pair["devil_advocate_reason"] = str(contested.get("reason", ""))
                logger.info("Devil's Advocate contested pair: %s <-> %s", prev_id, cur_id)

    return tables_added, tables_removed


def construire_etats_extraction(
    *,
    previous_tables: list[dict[str, Any]],
    current_tables: list[dict[str, Any]],
    previous_artifact_refs: list[dict[str, Any]],
    current_artifact_refs: list[dict[str, Any]],
    previous_suspect_refs: list[dict[str, Any]],
    current_suspect_refs: list[dict[str, Any]],
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Assembler les artefacts confirmes et les extractions suspectes."""
    artifacts_confirmed_previous = [
        {**item, **previous_snapshots[item["table_id"]]}
        for item in previous_artifact_refs
    ]
    artifacts_confirmed_current = [
        {**item, **current_snapshots[item["table_id"]]}
        for item in current_artifact_refs
    ]
    extraction_suspects_previous = _merge_extraction_suspect_side(
        previous_tables,
        previous_suspect_refs,
        previous_snapshots,
    )
    extraction_suspects_current = _merge_extraction_suspect_side(
        current_tables,
        current_suspect_refs,
        current_snapshots,
    )
    return (
        artifacts_confirmed_previous,
        artifacts_confirmed_current,
        extraction_suspects_previous,
        extraction_suspects_current,
    )


def _pire_statut_rendu(statuses: list[str]) -> str:
    """Retourner le pire statut de rendu visuel, avec priorite aux erreurs."""
    for candidate in (
        "skipped_missing_pdf",
        "skipped_missing_anchor",
        "skipped_missing_bbox",
        "skipped_render_failed",
    ):
        if candidate in statuses:
            return candidate
    return "ok"


def _rendre_preuves_evenement(
    *,
    event_type: str,
    event_snapshot: dict[str, Any],
    match_result: dict[str, Any],
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
    source_pdf_previous: str | None,
    source_pdf_current: str | None,
    renderer: Callable[..., tuple[bytes | None, str]],
) -> tuple[bytes | None, bytes | None, str, str]:
    """Rendre les deux preuves visuelles d'un ajout ou retrait de tableau."""
    normalized_event_type = str(event_type or "").strip().lower()
    render_mode = "full"
    if normalized_event_type == "table_added":
        opposite_anchor = _resolve_visual_table_anchor(event_snapshot, previous_snapshots)
        if opposite_anchor is None:
            opposite_page = _infer_opposite_page_from_matched_pairs(
                event_snapshot,
                match_result["matched_pairs"],
                current_snapshots,
                previous_snapshots,
                event_side="current",
            )
            if opposite_page is None:
                return None, None, "skipped_missing_anchor", render_mode
            render_mode = "full_page_context_fallback"
            previous_render, previous_status = renderer(
                source_pdf_previous,
                page=opposite_page,
                bbox=None,
                allow_full_page_fallback=True,
            )
        else:
            previous_render, previous_status = renderer(
                source_pdf_previous,
                page=opposite_anchor.get("page"),
                bbox=opposite_anchor.get("bbox"),
            )
        current_render, current_status = renderer(
            source_pdf_current,
            page=event_snapshot.get("page"),
            bbox=event_snapshot.get("bbox"),
        )
    else:
        opposite_anchor = _resolve_visual_table_anchor(event_snapshot, current_snapshots)
        if opposite_anchor is None:
            opposite_page = _infer_opposite_page_from_matched_pairs(
                event_snapshot,
                match_result["matched_pairs"],
                previous_snapshots,
                current_snapshots,
                event_side="previous",
            )
            if opposite_page is None:
                return None, None, "skipped_missing_anchor", render_mode
            render_mode = "full_page_context_fallback"
        else:
            opposite_page = opposite_anchor.get("page")
        previous_render, previous_status = renderer(
            source_pdf_previous,
            page=event_snapshot.get("page"),
            bbox=event_snapshot.get("bbox"),
        )
        current_render, current_status = renderer(
            source_pdf_current,
            page=opposite_page,
            bbox=None if opposite_anchor is None else opposite_anchor.get("bbox"),
            allow_full_page_fallback=opposite_anchor is None,
        )
    return (
        previous_render,
        current_render,
        _pire_statut_rendu([previous_status, current_status]),
        render_mode,
    )


def valider_evenements_visuellement(
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
    *,
    match_result: dict[str, Any],
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
    source_pdf_previous: str | None,
    source_pdf_current: str | None,
    model_name: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_records: list[dict[str, Any]],
    renderer: Callable[..., tuple[bytes | None, str]],
    verifier: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Valider visuellement les ajouts et retraits avant publication."""
    filtered: dict[str, list[dict[str, Any]]] = {
        "table_added": [],
        "table_removed": [],
    }
    for event_type, items in (
        ("table_added", tables_added),
        ("table_removed", tables_removed),
    ):
        for item in items:
            previous_render, current_render, render_status, render_mode = (
                _rendre_preuves_evenement(
                    event_type=event_type,
                    event_snapshot=item,
                    match_result=match_result,
                    previous_snapshots=previous_snapshots,
                    current_snapshots=current_snapshots,
                    source_pdf_previous=source_pdf_previous,
                    source_pdf_current=source_pdf_current,
                    renderer=renderer,
                )
            )
            if render_status != "ok":
                item.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                        render_mode=render_mode,
                    )
                )
                filtered[event_type].append(item)
                continue
            verdict = verifier(
                previous_render,
                current_render,
                event_type=event_type,
                table_id=str(item.get("table_id", "") or ""),
                table_title=str(item.get("title", "") or ""),
                model=model_name,
                call_openai_json=call_openai_json,
                usage_recorder=usage_records,
            )
            item.update({key: value for key, value in verdict.items() if key != "confirmed"})
            item["visual_sanity_render_mode"] = render_mode
            if verdict.get("confirmed", True):
                filtered[event_type].append(item)
    return filtered["table_added"], filtered["table_removed"]
