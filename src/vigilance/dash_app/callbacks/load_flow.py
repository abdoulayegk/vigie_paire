"""Load comparison from file system callbacks."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from vigilance.comparison_canonical import (
    is_canonical_comparison,
    to_canonical_payload,
)
from vigilance.dash_app.layouts import build_page_results
from vigilance.dash_app.services.comparison_store import (
    build_file_comparison_store,
)
from vigilance.ui_config import INDICATOR_COMPARISON_DIR


@callback(
    Output("load-comparison-dropdown", "options"),
    Input("store-detection", "data"),
)
def populate_load_options(_detection):
    """Remplir le menu des analyses enregistrées disponibles."""
    store = build_file_comparison_store(root_dir=INDICATOR_COMPARISON_DIR)
    return store.list_comparison_options()


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-pdf-paths", "data", allow_duplicate=True),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("main-content", "children", allow_duplicate=True),
    Output("notification", "children", allow_duplicate=True),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Input("btn-load-comparison", "n_clicks"),
    State("load-comparison-dropdown", "value"),
    prevent_initial_call=True,
)
def on_load_comparison(n_clicks, filename):
    """Charger une analyse enregistrée (canonique ou métier)."""
    if not n_clicks or not filename:
        raise PreventUpdate

    store = build_file_comparison_store(root_dir=INDICATOR_COMPARISON_DIR)
    loaded = store.load_dash_payload(
        filename,
        source="analyse_enregistree",
        source_label="Analyse enregistrée",
    )
    if not loaded:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dbc.Alert(
                f"Impossible de charger l'analyse enregistrée {filename}",
                color="danger",
            ),
            no_update,
        )

    data = loaded["raw_data"]
    indicator_result = loaded["indicator_result"]
    indicator_meta = dict(loaded["indicator_meta"])
    pdf_paths = dict(loaded["pdf_paths"])
    warning = str(loaded["warning"] or "")

    if data.get("result_type") == "metier_tableaux":
        return (
            None,
            data,
            indicator_meta,
            pdf_paths,
            True,
            build_page_results(),
            dbc.Alert(
                warning or f"Analyse enregistrée chargée: {filename}",
                color="warning" if warning else "success",
            ),
            True,
        )

    canonical = to_canonical_payload(data)
    if not is_canonical_comparison(canonical):
        canonical = indicator_result

    return (
        canonical,
        indicator_result,
        indicator_meta,
        pdf_paths,
        True,
        build_page_results(),
        dbc.Alert(
            warning or f"Analyse enregistrée chargée: {filename}",
            color="warning" if warning else "success",
        ),
        True,
    )
