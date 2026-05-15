"""Callbacks de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import Input, Output, State, callback, dcc
from dash.exceptions import PreventUpdate

from vigilance.dash_app.layouts.page_text_analysis import (
    build_filtered_text_cards,
    build_text_analysis_tab,
)


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
    Input("text-filter-impact", "value"),
    Input("text-filter-action", "value"),
    prevent_initial_call=True,
)
def filter_text_cards(text_data, filter_section, filter_impact, filter_action):
    """Filtre et trie les cartes analytiques selon les dropdowns."""
    if not text_data:
        raise PreventUpdate
    return build_filtered_text_cards(text_data, filter_section, filter_impact, filter_action)


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
    q_cur = str(payload.get("quarter_current", "")).replace("_", "")
    filename = f"veille_textuelle_{bank}_{q_cur}.xlsx"
    return dcc.send_bytes(excel_bytes, filename)
