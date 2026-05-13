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
from vigilance.text_comparison.text_comparison_excel import _should_exclude


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

    # 1. Collecter tous les changements détectés (hors unchanged)
    items: list[tuple[tuple[int, int, str, str, str], dict, str]] = []  # (sort_key, change, section_title)
    for sec in text_data.get("section_comparisons") or []:
        key = sec.get("section_key", "")
        title = sec.get("section_title") or _SECTION_LABELS.get(key, key)

        # Filtre section
        if filter_section and key != filter_section:
            continue

        for change in sec.get("all_block_comparisons") or []:
            diff_type = change.get("diff_type", "")
            if diff_type == "unchanged":
                continue
            if _should_exclude(change):
                continue
            triage = change.get("genai_triage") or {}

            impact = (triage.get("impact_level") or "MINEUR").upper()
            action = (triage.get("action_requise") or "aucune").lower()
            nouvelle_idee = bool(triage.get("nouvelle_idee", False))
            pages = change.get("pages_t2") or change.get("pages_t1") or []
            page_sort = ""
            if pages:
                try:
                    page_sort = f"{int(pages[0]):06d}"
                except (TypeError, ValueError):
                    page_sort = str(pages[0])

            # Filtre impact
            if filter_impact and impact != filter_impact.upper():
                continue
            # Filtre action
            if filter_action and action != filter_action.lower():
                continue

            sort_key = (
                0 if nouvelle_idee else 1,
                _IMPACT_ORDER.get(impact, 99),
                title,
                page_sort,
                diff_type,
            )
            items.append((sort_key, change, title))

    # 2. Trier nouvelle idée → impact → section/page/type
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
    """Composant Dash affiché lorsque aucun changement ne passe les filtres."""
    from dash import html

    return [
        html.Div(
            "Aucun changement détecté correspondant aux filtres sélectionnés.",
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
