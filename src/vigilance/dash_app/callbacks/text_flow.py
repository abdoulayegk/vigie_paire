"""Callback de rendu de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import Input, Output, callback
from dash.exceptions import PreventUpdate

from vigilance.dash_app.layouts.page_text_analysis import build_text_analysis_tab


@callback(
    Output("text-analysis-tab-content", "children"),
    Input("store-text-comparison", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_text_analysis(text_data, show_results):
    """Reconstruit l'onglet analyse textuelle quand les données arrivent.

    Dash Inputs:
        store-text-comparison.data: Contenu de text_comparison.json.
        store-show-results-page.data: Booléen déclencheur de la page résultats.

    Dash Output:
        text-analysis-tab-content.children: Composants de l'onglet.
    """
    if not show_results:
        raise PreventUpdate
    return build_text_analysis_tab(text_data)
