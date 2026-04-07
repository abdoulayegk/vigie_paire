"""Callbacks de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import Input, Output, State, callback, dcc
from dash.exceptions import PreventUpdate

from vigilance.dash_app.layouts.page_text_analysis import (
    _IMPACT_ORDER,
    _SECTION_LABELS,
    _build_change_card,
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

    # 1. Collecter tous les changements pertinents
    items: list[tuple[int, dict, str]] = []  # (sort_key, change, section_title)
    for sec in text_data.get("section_comparisons") or []:
        key = sec.get("section_key", "")
        title = sec.get("section_title") or _SECTION_LABELS.get(key, key)

        # Filtre section
        if filter_section and key != filter_section:
            continue

        for change in sec.get("block_comparisons") or []:
            diff_type = change.get("diff_type", "")
            if diff_type == "unchanged":
                continue
            triage = change.get("genai_triage") or {}
            if not triage.get("is_relevant", False):
                continue

            impact = (triage.get("impact_level") or "MINEUR").upper()
            action = (triage.get("action_requise") or "aucune").lower()
            signals = triage.get("signals") or {}
            keep_change = impact == "MAJEUR" or (
                impact == "MODERE"
                and (
                    triage.get("nouvelle_idee", False)
                    or signals.get("regulatory_reference_added", False)
                    or signals.get("methodology_change", False)
                )
            )
            if not keep_change:
                continue

            # Filtre impact
            if filter_impact and impact != filter_impact.upper():
                continue
            # Filtre action
            if filter_action and action != filter_action.lower():
                continue

            sort_key = _IMPACT_ORDER.get(impact, 99)
            items.append((sort_key, change, title))

    # 2. Trier MAJEUR → MODERE → MINEUR
    items.sort(key=lambda x: x[0])

    # 3. Construire les cartes
    cards = []
    for _, change, title in items:
        card = _build_change_card(change, title)
        if card is not None:
            cards.append(card)

    count_text = f"{len(cards)} changement(s) affiché(s)"
    return cards or _empty_state(), count_text


def _empty_state():
    from dash import html

    return [
        html.Div(
            "Aucun changement pertinent correspondant aux filtres sélectionnés.",
            className="text-muted text-center py-4",
        )
    ]


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

    from vigilance.text_comparison import generate_text_comparison_excel

    excel_bytes = generate_text_comparison_excel(text_data, output_path=None)
    bank = str(text_data.get("bank_code", "banque")).upper()
    q_cur = text_data.get("quarter_current", "").replace("_", "")
    filename = f"veille_textuelle_{bank}_{q_cur}.xlsx"
    return dcc.send_bytes(excel_bytes, filename)
