"""Callbacks de chargement de comparaison depuis le systeme de fichiers."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from vigie.comparaison.canonical import (
    is_canonical_comparison,
    to_canonical_payload,
)
from vigie.interface.layouts import build_page_results
from vigie.interface.services import text_comparison_store
from vigie.interface.services.comparison_store import (
    build_file_comparison_store,
)
from vigie.interface.ui_config import INDICATOR_COMPARISON_DIR


@callback(
    Output("load-comparison-dropdown", "options"),
    Input("store-detection", "data"),
)
def populate_load_options(_detection):
    """Rafraîchit la liste des analyses enregistrées après une détection.

    L'événement Dash sert de déclencheur; la sortie contient les options et la
    sélection initiale du menu de chargement.
    """
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
    Output("store-text-comparison", "data", allow_duplicate=True),
    Input("btn-load-comparison", "n_clicks"),
    State("load-comparison-dropdown", "value"),
    prevent_initial_call=True,
)
def on_load_comparison(n_clicks, filename):
    """Recharge une analyse canonique ou métier choisie par l'utilisateur.

    Le clic et le nom de fichier produisent les stores de résultats, les chemins
    associés et le message d'état nécessaires pour reprendre la revue.
    """
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
            no_update,
        )

    data = loaded["raw_data"]
    indicator_result = loaded["indicator_result"]
    indicator_meta = dict(loaded["indicator_meta"])
    pdf_paths = dict(loaded["pdf_paths"])
    warning = str(loaded["warning"] or "")

    # Charger le text_comparison.json correspondant (silencieux si absent)
    canonical_for_text = to_canonical_payload(data) if data else {}
    text_comparison_data = text_comparison_store.resolve_text_comparison_from_payload(canonical_for_text)

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
            text_comparison_data,
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
        text_comparison_data,
    )
