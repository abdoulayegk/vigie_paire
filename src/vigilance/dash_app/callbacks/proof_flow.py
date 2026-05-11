"""Callbacks d'affichage des preuves, metadonnees de revue et banniere de progression."""

from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from vigilance.dash_app.components.review_detail_v2 import build_review_detail_v2
from vigilance.dash_app.components.review_display_shared import build_proofs_section
from vigilance.dash_app.services.comparison_context import _normalize_pdf_paths_store
from vigilance.dash_app.services.pdf_rendering import (
    _get_proof_render_result_for_item,
)
from vigilance.dash_app.services.review_navigation import (
    _effective_proof_display_mode,
    _get_table_by_review_id,
    _resolve_selection,
    _table_to_proof_item,
)

logger = logging.getLogger(__name__)


@callback(
    Output("review-proof-container", "children"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-pdf-paths", "data"),
    Input("store-show-results-page", "data"),
    Input("store-proof-display-mode", "data"),
    prevent_initial_call=True,
)
def update_review_proofs(
    review_queue_data, selection, paths, show_results, proof_display_mode
):
    """Mettre a jour la section des preuves visuelles (portee tableau ; stable lors de la navigation entre indicateurs)."""
    if not show_results:
        raise PreventUpdate
    if not review_queue_data:
        return html.Div(
            "Veuillez lancer une analyse pour voir les détails.",
            className="text-center text-muted mt-5",
        )

    resolved_selection, _, _ = _resolve_selection(review_queue_data, selection)
    _, table = _get_table_by_review_id(
        review_queue_data, resolved_selection.get("review_id")
    )
    if table is None:
        return html.Div(
            [
                html.H6(
                    "Preuves visuelles : courant vs précédent", className="mb-1"
                ),
                html.P(
                    "Aucune preuve visuelle disponible pour la sélection courante.",
                    className="text-muted small mb-0",
                ),
            ],
            className="p-3 bg-light rounded border",
        )

    item = _table_to_proof_item(table, resolved_selection)
    mode = _effective_proof_display_mode(
        table,
        resolved_selection,
        proof_display_mode,
    )

    normalized_paths = _normalize_pdf_paths_store(paths or {})
    pdf_path_t1 = normalized_paths.get("pdf_t1", "") or item.get("source_ref_t1", "")
    pdf_path_t2 = normalized_paths.get("pdf_t2", "") or item.get("source_ref_t2", "")

    # primary (magenta) = active change being validated
    # secondary (yellow) = all other changes in the table (context)
    primary_rects_t1: list[list[float]] = []
    primary_rects_t2: list[list[float]] = []
    secondary_rects_t1: list[list[float]] = []
    secondary_rects_t2: list[list[float]] = []

    active_change_id = resolved_selection.get("change_id")
    all_changes = table.get("changes") or []

    if all_changes and (pdf_path_t1 or pdf_path_t2):
        from vigilance.utils.pdf_highlight import find_text_bboxes_in_region

        _FOOTNOTE_CHANGE_TYPES = {
            "footnote_added",
            "FOOTNOTE_ADDED",
            "footnote_removed",
            "FOOTNOTE_REMOVED",
            "footnote_modified",
            "FOOTNOTE_MODIFIED",
        }

        def _extended_bbox(bbox: list[float]) -> list[float]:
            """Etendre la bbox vers le bas pour couvrir les notes de bas de tableau."""
            return [bbox[0], bbox[1], bbox[2], min(1.0, bbox[3] + 0.25)]

        def _texts_for_change(change: dict) -> tuple[str, str]:
            """Extraire les textes de recherche T1/T2 depuis un changement."""
            payload = change.get("payload") or {}
            ct = str(change.get("change_type", "") or "")
            if ct in ("indicator_renamed", "INDICATOR_RENAMED"):
                return payload.get("from", ""), payload.get("to", "")
            if ct in ("footnote_added", "FOOTNOTE_ADDED"):
                return "", (payload.get("new_text", "") or "").split("\n")[0].strip()
            if ct in ("footnote_removed", "FOOTNOTE_REMOVED"):
                return (payload.get("old_text", "") or "").split("\n")[0].strip(), ""
            if ct in ("footnote_modified", "FOOTNOTE_MODIFIED"):
                return (
                    (payload.get("old_text", "") or "").split("\n")[0].strip(),
                    (payload.get("new_text", "") or "").split("\n")[0].strip(),
                )
            if ct in ("indicator_added", "INDICATOR_ADDED"):
                return "", payload.get("indicator_name", "")
            if ct in ("indicator_removed", "INDICATOR_REMOVED"):
                return payload.get("indicator_name", ""), ""
            name = payload.get("indicator_name", "")
            return name, name

        for change in all_changes:
            is_active = str(change.get("change_id", "")) == str(active_change_id or "")
            s_t1, s_t2 = _texts_for_change(change)
            is_fn = str(change.get("change_type", "") or "") in _FOOTNOTE_CHANGE_TYPES

            if (
                s_t1
                and pdf_path_t1
                and item.get("page_t1") is not None
                and item.get("bbox_t1")
            ):
                try:
                    bbox = _extended_bbox(item["bbox_t1"]) if is_fn else item["bbox_t1"]
                    rects = find_text_bboxes_in_region(
                        str(pdf_path_t1), max(1, int(item["page_t1"])), s_t1, bbox
                    )
                    (primary_rects_t1 if is_active else secondary_rects_t1).extend(
                        rects
                    )
                except Exception as e:
                    logger.warning(
                        f"Highlight t1 failed for {change.get('change_id')}: {e}"
                    )

            if (
                s_t2
                and pdf_path_t2
                and item.get("page_t2") is not None
                and item.get("bbox_t2")
            ):
                try:
                    bbox = _extended_bbox(item["bbox_t2"]) if is_fn else item["bbox_t2"]
                    rects = find_text_bboxes_in_region(
                        str(pdf_path_t2), max(1, int(item["page_t2"])), s_t2, bbox
                    )
                    (primary_rects_t2 if is_active else secondary_rects_t2).extend(
                        rects
                    )
                except Exception as e:
                    logger.warning(
                        f"Highlight t2 failed for {change.get('change_id')}: {e}"
                    )

    proof_t1 = _get_proof_render_result_for_item(
        item,
        "t1",
        paths or {},
        proof_display_mode=mode,
        highlight_rects=primary_rects_t1 or None,
        secondary_highlight_rects=secondary_rects_t1 or None,
    )
    proof_t2 = _get_proof_render_result_for_item(
        item,
        "t2",
        paths or {},
        proof_display_mode=mode,
        highlight_rects=primary_rects_t2 or None,
        secondary_highlight_rects=secondary_rects_t2 or None,
    )
    return build_proofs_section(
        item=item,
        img_t1_b64=str(proof_t1.get("image_b64") or "") or None,
        img_t2_b64=str(proof_t2.get("image_b64") or "") or None,
        proof_display_mode=mode,
        proof_result_t1=proof_t1,
        proof_result_t2=proof_t2,
    )


@callback(
    Output("review-meta-container", "children"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def update_review_meta(review_queue_data, selection, show_results):
    """Mettre a jour les metadonnees V2 et la section de validation par changement."""
    if not show_results:
        raise PreventUpdate
    if not review_queue_data:
        return html.Div(
            "Veuillez lancer une analyse pour voir les détails.",
            className="text-center text-muted mt-5",
        )

    resolved_selection, _, change_idx = _resolve_selection(review_queue_data, selection)
    _, table = _get_table_by_review_id(
        review_queue_data, resolved_selection.get("review_id")
    )
    if table is None:
        return html.Div(
            "Aucun tableau sélectionné.",
            className="text-center text-muted mt-5",
        )

    return build_review_detail_v2(
        table=table,
        current_change_idx=change_idx,
        proof_image_t1_b64="",
        proof_image_t2_b64="",
        show_proofs=False,
    )


@callback(
    Output("review-progress-banner", "children"),
    Input("store-review-queue", "data"),
    Input("store-show-results-page", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def update_progress_banner(review_queue_data, show_results, indicator_meta):
    """Afficher la banniere de progression globale au-dessus des preuves visuelles."""
    if not show_results:
        return html.Div()
    if not review_queue_data:
        return html.Div()

    total_tables = len(review_queue_data)
    completed_tables = sum(
        1 for t in review_queue_data if t.get("table_status") == "completed"
    )
    total_changes = sum(len(t.get("changes", []) or []) for t in review_queue_data)
    validated_changes = sum(
        1
        for t in review_queue_data
        for c in (t.get("changes", []) or [])
        if c.get("validation_status", "pending") in ("approved", "rejected", "skipped")
    )
    pct = int(validated_changes / total_changes * 100) if total_changes else 0

    # Bank + period label from meta
    meta = indicator_meta or {}
    bank = str(meta.get("bank_code", "") or "").upper()
    q_current = str(meta.get("quarter_current", "") or "").upper()
    year_current = str(meta.get("year_current", "") or "")
    q_previous = str(meta.get("quarter_previous", "") or "").upper()
    year_previous = str(meta.get("year_previous", "") or "")
    period_label = ""
    if bank and q_current:
        period_label = (
            f"{bank} — {q_current} {year_current} vs {q_previous} {year_previous}"
        )
    source_label = str(meta.get("source_label", "") or "").strip()

    return html.Div(
        [
            html.Div(
                [
                    html.Span(period_label, className="fw-semibold me-3")
                    if period_label
                    else None,
                    dbc.Badge(source_label, color="light", text_color="dark")
                    if source_label
                    else None,
                    html.Span(
                        f"{completed_tables}/{total_tables} tables complètes",
                        className="me-3 text-muted small",
                    ),
                    html.Span(
                        f"{validated_changes}/{total_changes} changements validés",
                        className="me-3 text-muted small",
                    ),
                    dbc.Badge(f"{pct}%", color="primary" if pct < 100 else "success"),
                ],
                className="d-flex align-items-center flex-wrap gap-1 mb-1",
            ),
            dbc.Progress(
                value=pct,
                color="success" if pct == 100 else "primary",
                style={"height": "6px"},
                className="review-progress-bar",
            ),
        ],
        className="review-progress-banner px-3 py-2 rounded border mb-1",
    )
