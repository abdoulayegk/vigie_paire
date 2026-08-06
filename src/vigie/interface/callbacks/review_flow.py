"""Callbacks de navigation, filtrage et validation de la file de revue."""

from __future__ import annotations

import json
import logging

from dash import ALL, Input, Output, State, callback, ctx, html
from dash.exceptions import PreventUpdate

from vigie.interface.components.review_queue_v2 import build_review_queue_v2
from vigie.interface.layouts.page_results import build_analyst_kpi_card
from vigie.interface.services.export_helpers import _review_items_from_v2_queue
from vigie.interface.services.review_navigation import (
    _get_table_by_review_id,
    _normalize_last_positions,
    _remember_selection,
    _resolve_change_id,
    _resolve_selection,
    _review_id,
    _table_decision_bucket,
    _visible_review_ids,
)
from vigie.interface.services.review_persistence import _persist_review_state
from vigie.support.i18n import t

logger = logging.getLogger(__name__)


def _has_final_indicator_decision(change: dict) -> bool:
    """Retourne True si le changement indicateur a une decision finale."""
    return str(change.get("validation_status", "pending")) in {"approved", "rejected"}


def _indicator_review_summary(changes: list[dict]) -> dict[str, int]:
    """Recalcule les compteurs de revue pour un tableau indicateur."""
    finalized = sum(1 for c in changes if _has_final_indicator_decision(c))
    pending = max(0, len(changes) - finalized)
    return {
        "total_changes": len(changes),
        "indicators_added": sum(1 for c in changes if c.get("change_type") == "indicator_added"),
        "indicators_removed": sum(1 for c in changes if c.get("change_type") == "indicator_removed"),
        "indicators_renamed": sum(1 for c in changes if c.get("change_type") == "indicator_renamed"),
        "footnotes_changed": sum(1 for c in changes if "footnote" in c.get("change_type", "")),
        "validated": finalized,
        "pending": pending,
    }


def _apply_indicator_table_status(table: dict, changes: list[dict]) -> None:
    """Met a jour le statut du tableau selon les decisions finales."""
    summary = _indicator_review_summary(changes)
    if summary["pending"] == 0:
        table["table_status"] = "completed"
    elif summary["validated"] > 0:
        table["table_status"] = "partial"
    else:
        table["table_status"] = "pending"
    table["summary"] = summary


@callback(
    Output("review-queue-container", "children"),
    Output("kpi-queue-total", "children"),
    Output("kpi-queue-approved", "children"),
    Output("kpi-queue-rejected", "children"),
    Output("kpi-queue-pending", "children"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    Input("store-review-filters", "data"),
    prevent_initial_call=True,
)
def update_review_queue(
    review_queue_data,
    selection,
    indicator_result,
    show_results,
    filters,
):
    """Mettre a jour la file de revue laterale gauche et les KPIs en haut de page."""
    if not show_results:
        raise PreventUpdate

    if review_queue_data is None:
        return (
            html.Div(
                [
                    html.I(className="bi bi-hourglass-split me-2"),
                    "Chargement de la file de revue...",
                ],
                className="text-muted p-3",
            ),
            build_analyst_kpi_card(t("file_review_total"), "-", color="white"),
            build_analyst_kpi_card(t("validated"), "-", color="white"),
            build_analyst_kpi_card(t("rejected"), "-", color="white"),
            build_analyst_kpi_card(t("pending"), "-", color="white"),
        )
    if len(review_queue_data) == 0:
        empty_msg = t("no_changes_review", "Aucun changement a revoir.")
        return (
            html.Div(
                empty_msg,
                className="text-muted p-3",
            ),
            build_analyst_kpi_card(t("file_review_total"), "0", color="white"),
            build_analyst_kpi_card(t("validated"), "0", color="white"),
            build_analyst_kpi_card(t("rejected"), "0", color="white"),
            build_analyst_kpi_card(t("pending"), "0", color="white"),
        )

    queue = review_queue_data or []
    resolved_selection, _, _ = _resolve_selection(queue, selection, filters)
    current_review_id = resolved_selection.get("review_id")

    total = len(queue)
    approved = sum(1 for t in queue if _table_decision_bucket(t) == "approved")
    rejected = sum(1 for t in queue if _table_decision_bucket(t) == "rejected")
    pending = max(0, total - approved - rejected)

    queue_component = build_review_queue_v2(
        queue,
        current_review_id,
        resolved_selection.get("change_id"),
        filters,
    )

    return (
        queue_component,
        build_analyst_kpi_card(t("file_review_total"), str(total), color="white"),
        build_analyst_kpi_card(t("validated"), str(approved), color="white"),
        build_analyst_kpi_card(t("rejected"), str(rejected), color="white"),
        build_analyst_kpi_card(t("pending"), str(pending), color="white"),
    )


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("store-review-queue", "data"),
    Input("store-review-filters", "data"),
    State("store-review-selection", "data"),
    State("store-review-last-positions", "data"),
    prevent_initial_call=True,
)
def sync_review_selection(queue, filters, selection, last_positions):
    """Garder la selection stable lors du rafraichissement de la file ou du changement de filtres."""
    if queue is None:
        raise PreventUpdate
    resolved_selection, _, change_idx = _resolve_selection(queue or [], selection, filters, last_positions)
    if resolved_selection == (selection or {}):
        raise PreventUpdate
    return resolved_selection, change_idx


@callback(
    Output("store-review-last-positions", "data", allow_duplicate=True),
    Input("store-review-selection", "data"),
    State("store-review-last-positions", "data"),
    prevent_initial_call=True,
)
def remember_review_position_v2(selection, last_positions):
    """Memoriser le dernier changement visite pour chaque tableau dans la revue active."""
    updated = _remember_selection(last_positions, selection)
    if updated == _normalize_last_positions(last_positions):
        raise PreventUpdate
    return updated


@callback(
    Output("store-review-filters", "data"),
    Input({"type": "filter-section-v2", "value": ALL}, "n_clicks"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_filter_section(n_clicks, current_filters):
    """Mettre a jour le filtre de section quand un bouton de filtre est clique."""
    if not ctx.triggered_id:
        raise PreventUpdate

    section_value = ctx.triggered_id.get("value")
    new_filters = dict(current_filters or {})
    new_filters["section"] = section_value
    return new_filters


# =============================================================================
# V2 CALLBACKS: Per-Change Validation in Grouped Review Queue
# =============================================================================


@callback(
    Output("store-review-queue", "data", allow_duplicate=True),
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("btn-approve-change-v2", "n_clicks"),
    Input("btn-reject-change-v2", "n_clicks"),
    Input("btn-skip-change-v2", "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("store-review-last-positions", "data"),
    State("validation-notes-v2", "value"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def on_validate_change_v2(
    approve,
    reject,
    skip,
    queue,
    selection,
    filters,
    last_positions,
    notes,
    indicator_meta,
):
    """Appliquer la validation au changement courant de la file V2 et avancer automatiquement."""
    from datetime import datetime, timezone

    if not ctx.triggered_id or not queue:
        raise PreventUpdate

    action_map = {
        "btn-approve-change-v2": "approved",
        "btn-reject-change-v2": "rejected",
        "btn-skip-change-v2": "skipped",
    }
    decision = action_map.get(ctx.triggered_id)
    if not decision:
        raise PreventUpdate

    resolved_selection, table_idx, change_idx = _resolve_selection(queue, selection, filters, last_positions)
    if table_idx < 0:
        raise PreventUpdate

    new_queue = json.loads(json.dumps(queue))
    table = new_queue[table_idx]
    changes = table.get("changes", [])
    selected_change_id = str(resolved_selection.get("change_id") or "")
    if selected_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == selected_change_id:
                change_idx = idx
                break
    if change_idx >= len(changes) or change_idx < 0:
        raise PreventUpdate

    changes[change_idx]["validation_status"] = decision
    changes[change_idx]["validation_decision"] = decision
    changes[change_idx]["validation_notes"] = notes or ""
    changes[change_idx]["validated_at"] = datetime.now(timezone.utc).isoformat()
    changes[change_idx]["validated_by"] = "analyst"

    table["changes"] = changes
    _apply_indicator_table_status(table, changes)

    next_selection = dict(resolved_selection)
    remembered_positions = _remember_selection(last_positions, resolved_selection)

    next_selection, new_table_idx, new_change_idx = _resolve_selection(
        new_queue, next_selection, filters, remembered_positions
    )

    logger.info(
        "[on_validate_change_v2] review_id=%s change=%s decision=%s -> review_id=%s change=%s",
        resolved_selection.get("review_id"),
        resolved_selection.get("change_id"),
        decision,
        next_selection.get("review_id"),
        next_selection.get("change_id"),
    )

    _persist_review_state(
        indicator_meta=indicator_meta,
        review_items=[it.to_dict() for it in _review_items_from_v2_queue(new_queue)],
        review_queue=new_queue,
        review_selection=next_selection,
        review_current_idx=new_table_idx,
        current_change_idx=new_change_idx,
        preferred_store="review_queue",
        source="on_validate_change_v2",
    )
    return new_queue, next_selection, new_change_idx


@callback(
    Output("store-review-queue", "data", allow_duplicate=True),
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input({"type": "btn-reset-change-v2", "change_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("store-review-last-positions", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def on_reset_change_decision(n_clicks, queue, selection, filters, last_positions, indicator_meta):
    """Reinitialiser un changement valide a l'etat en attente pour permettre a l'analyste de re-decider."""
    if not ctx.triggered_id or not queue:
        raise PreventUpdate
    if not any(nc for nc in (n_clicks or [])):
        raise PreventUpdate
    if not isinstance(ctx.triggered_id, dict) or ctx.triggered_id.get("type") != "btn-reset-change-v2":
        raise PreventUpdate

    change_id = str(ctx.triggered_id.get("change_id") or "")
    if not change_id:
        raise PreventUpdate

    new_queue = json.loads(json.dumps(queue))
    found = False
    for table in new_queue:
        for change in table.get("changes", []) or []:
            if str(change.get("change_id", "")) == change_id:
                change["validation_status"] = "pending"
                change.pop("validation_decision", None)
                change.pop("validation_notes", None)
                change.pop("validated_at", None)
                change.pop("validated_by", None)
                found = True
                break
        if found:
            changes = table.get("changes", [])
            _apply_indicator_table_status(table, changes)
            break

    if not found:
        raise PreventUpdate

    new_selection = dict(selection or {})
    for table in new_queue:
        for idx, change in enumerate(table.get("changes", []) or []):
            if str(change.get("change_id", "")) == change_id:
                new_selection = {"review_id": _review_id(table), "change_id": change_id}
                break

    new_selection, _, new_change_idx = _resolve_selection(new_queue, new_selection, filters, last_positions)

    _persist_review_state(
        indicator_meta=indicator_meta,
        review_items=[it.to_dict() for it in _review_items_from_v2_queue(new_queue)],
        review_queue=new_queue,
        review_selection=new_selection,
        review_current_idx=None,
        current_change_idx=new_change_idx,
        preferred_store="review_queue",
        source="on_reset_change_decision",
    )

    logger.info("[on_reset_change_decision] change_id=%s reset to pending", change_id)
    return new_queue, new_selection, new_change_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("btn-prev-change-v2", "n_clicks"),
    Input("btn-next-change-v2", "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("store-review-last-positions", "data"),
    prevent_initial_call=True,
)
def on_navigate_change_v2(prev, next_c, queue, selection, filters, last_positions):
    """Naviguer precedent/suivant parmi les changements du tableau courant (V2)."""
    if not ctx.triggered_id or not queue:
        raise PreventUpdate

    resolved_selection, table_idx, change_idx = _resolve_selection(queue, selection, filters, last_positions)
    if table_idx < 0:
        raise PreventUpdate

    table = queue[table_idx]
    changes = table.get("changes", [])
    n_changes = len(changes)
    selected_change_id = str(resolved_selection.get("change_id") or "")
    if selected_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == selected_change_id:
                change_idx = idx
                break

    if ctx.triggered_id == "btn-prev-change-v2":
        new_idx = max(0, change_idx - 1)
    elif ctx.triggered_id == "btn-next-change-v2":
        new_idx = min(n_changes - 1, change_idx + 1) if n_changes > 0 else 0
    else:
        raise PreventUpdate

    new_change_id = str(changes[new_idx].get("change_id", "")) if 0 <= new_idx < len(changes) else None
    new_selection = {
        "review_id": resolved_selection.get("review_id"),
        "change_id": new_change_id,
    }
    new_selection, _, new_change_idx = _resolve_selection(
        queue,
        new_selection,
        filters,
        last_positions,
    )
    return new_selection, new_change_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input({"type": "change-row-v2", "change_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("store-review-last-positions", "data"),
    prevent_initial_call=True,
)
def on_change_row_click(n_clicks, queue, selection, filters, last_positions):
    """Selectionner une ligne de changement specifique par son change_id stable."""
    if not ctx.triggered_id or not queue:
        raise PreventUpdate
    if not any(nc for nc in (n_clicks or [])):
        raise PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or trig.get("type") != "change-row-v2":
        raise PreventUpdate
    change_id = str(trig.get("change_id") or "")
    if not change_id:
        raise PreventUpdate

    owner_table = None
    # Variable nommee 'table' et non 't' : 't' est la fonction de traduction i18n
    # importee en tete de module, que la boucle masquait dans cette fonction.
    for table in queue:
        if any(str(c.get("change_id", "")) == change_id for c in (table.get("changes", []) or [])):
            owner_table = table
            break
    if owner_table is None:
        raise PreventUpdate

    new_selection = {
        "review_id": _review_id(owner_table),
        "change_id": change_id,
    }
    new_selection, _, new_change_idx = _resolve_selection(
        queue,
        new_selection,
        filters,
        last_positions,
    )
    return new_selection, new_change_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("btn-prev-table-v2", "n_clicks"),
    Input("btn-next-table-v2", "n_clicks"),
    Input({"type": "queue-table-item-v2", "review_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("store-review-last-positions", "data"),
    prevent_initial_call=True,
)
def on_navigate_table_v2(prev, next_t, clicks, queue, selection, filters, last_positions):
    """Naviguer entre les tableaux ou sauter directement depuis la file de revue."""
    if not queue:
        raise PreventUpdate

    resolved_selection, table_idx, _ = _resolve_selection(
        queue,
        selection,
        filters,
        last_positions,
    )
    visible_ids = _visible_review_ids(queue, filters) or [_review_id(t) for t in queue if _review_id(t)]
    if not visible_ids:
        raise PreventUpdate

    current_review_id = str(resolved_selection.get("review_id") or "")
    if current_review_id not in visible_ids:
        current_review_id = visible_ids[0]
    pos = visible_ids.index(current_review_id)

    target_review_id = current_review_id
    if ctx.triggered_id == "btn-prev-table-v2":
        target_review_id = visible_ids[max(0, pos - 1)]
    elif ctx.triggered_id == "btn-next-table-v2":
        target_review_id = visible_ids[min(len(visible_ids) - 1, pos + 1)]
    elif isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "queue-table-item-v2":
        clicked_review_id = str(ctx.triggered_id.get("review_id") or "")
        if clicked_review_id:
            target_review_id = clicked_review_id
    else:
        raise PreventUpdate

    _, target_table = _get_table_by_review_id(queue, target_review_id)
    new_selection = {
        "review_id": target_review_id,
        "change_id": _resolve_change_id(target_table or {}, None, last_positions),
    }
    new_selection, new_table_idx, new_change_idx = _resolve_selection(
        queue,
        new_selection,
        filters,
        last_positions,
    )
    logger.info(
        "[on_navigate_table_v2] trig=%s from=%r to=%s visible_total=%s",
        ctx.triggered_id,
        (selection or {}).get("review_id"),
        new_selection.get("review_id"),
        len(visible_ids),
    )
    return new_selection, new_change_idx


@callback(
    Output("btn-prev-table-v2", "disabled"),
    Output("btn-next-table-v2", "disabled"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-review-filters", "data"),
)
def block_table_navigation_v2(queue, selection, filters):
    """Desactiver les boutons de navigation entre tableaux aux limites visibles."""
    if not queue:
        return True, True

    visible_ids = _visible_review_ids(queue, filters) or [_review_id(table) for table in queue if _review_id(table)]
    if not visible_ids:
        return True, True

    resolved_selection, _, _ = _resolve_selection(queue, selection, filters)
    current_review_id = str(resolved_selection.get("review_id") or "")
    if current_review_id not in visible_ids:
        current_review_id = visible_ids[0]

    pos = visible_ids.index(current_review_id)
    prev_disabled = pos <= 0
    next_disabled = pos >= len(visible_ids) - 1

    return prev_disabled, next_disabled
