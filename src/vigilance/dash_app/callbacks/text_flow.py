"""Callbacks de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import ALL, Input, Output, State, callback, ctx, dcc
from dash.exceptions import PreventUpdate

from vigilance.dash_app.services.text_review import (
    apply_text_review_decision,
    write_text_review_to_disk,
)
from vigilance.dash_app.layouts.page_text_analysis import (
    build_filtered_text_cards,
    build_text_analysis_tab,
)
from vigilance.quarter_utils import quarter_label_from_payload


def _filename_period(label: str) -> str:
    """Normalise un libelle de trimestre pour un nom de fichier."""
    return "_".join(str(label or "").strip().upper().split())


@callback(
    Output("text-analysis-tab-content", "children"),
    Input("store-text-comparison", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_text_analysis(text_data, show_results):
    """Reconstruit le layout du tab quand les données arrivent."""
    if not show_results:
        raise PreventUpdate
    return build_text_analysis_tab(text_data)


@callback(
    Output("text-cards-container", "children"),
    Output("text-filter-count", "children"),
    Input("store-text-comparison", "data"),
    Input("text-filter-section", "value"),
    Input("text-filter-category", "value"),
    Input("text-filter-new-idea", "value"),
    Input("text-filter-review", "value"),
    Input("text-filter-search", "value"),
    prevent_initial_call=True,
)
def filter_text_cards(
    text_data,
    filter_section=None,
    filter_category=None,
    filter_nouvelle_idee=None,
    filter_review=None,
    filter_search=None,
):
    """Filtre et trie les cartes analytiques selon les dropdowns."""
    if not text_data:
        raise PreventUpdate
    return build_filtered_text_cards(
        text_data,
        filter_section,
        filter_category,
        filter_nouvelle_idee,
        filter_review,
        filter_search,
    )


@callback(
    Output("store-text-comparison", "data", allow_duplicate=True),
    Input({"type": "text-review-action", "change_id": ALL, "action": ALL}, "n_clicks"),
    State({"type": "text-review-action", "change_id": ALL, "action": ALL}, "id"),
    State({"type": "text-review-comment", "change_id": ALL}, "value"),
    State({"type": "text-review-comment", "change_id": ALL}, "id"),
    State("store-text-comparison", "data"),
    prevent_initial_call=True,
)
def review_text_change(action_clicks, action_ids, comments, comment_ids, text_data):
    """Applique et persiste une decision analyste sur un changement texte."""
    if not text_data:
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate
    if not any(int(value or 0) > 0 for value in (action_clicks or [])):
        raise PreventUpdate

    change_id = str(triggered.get("change_id") or "").strip()
    action = str(triggered.get("action") or "").strip().lower()
    if not change_id or action not in {"approved", "rejected", "skipped"}:
        raise PreventUpdate

    comment = ""
    for value, id_value in zip(comments or [], comment_ids or [], strict=False):
        if isinstance(id_value, dict) and str(id_value.get("change_id") or "") == change_id:
            comment = str(value or "")
            break

    updated, found = apply_text_review_decision(
        text_data,
        change_id=change_id,
        status=action,
        comment=comment,
    )
    if not found:
        raise PreventUpdate
    write_text_review_to_disk(updated, regenerate_excel=action != "skipped")
    return updated


@callback(
    Output("download-text-excel", "data"),
    Input("btn-download-text-excel", "n_clicks"),
    State("store-text-comparison", "data"),
    prevent_initial_call=True,
)
def download_text_excel(n_clicks, text_data):
    """Génère et envoie le fichier Excel analyste."""
    if not n_clicks or not text_data:
        raise PreventUpdate

    from vigilance.dash_app.services.text_comparison_store import load_text_comparison_for_dash
    from vigilance.text_comparison import generate_text_comparison_excel

    bank_code = str(text_data.get("bank_code", "banque")).lower()
    quarter_current = str(text_data.get("quarter_current", "")).lower()
    quarter_previous = str(text_data.get("quarter_previous", "")).lower()
    latest_text_data = load_text_comparison_for_dash(
        bank_code=bank_code,
        quarter_current=quarter_current,
        quarter_previous=quarter_previous,
    )
    payload = latest_text_data or text_data

    excel_bytes = generate_text_comparison_excel(payload, output_path=None)
    bank = str(payload.get("bank_code", "banque")).upper()
    q_cur = _filename_period(quarter_label_from_payload(payload, "current"))
    filename = f"veille_textuelle_{bank}_{q_cur}.xlsx"
    return dcc.send_bytes(excel_bytes, filename)
