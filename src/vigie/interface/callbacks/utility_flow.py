"""Callbacks utilitaires : barre laterale, options, minuterie, reinitialisation, mode d'affichage des preuves."""

from __future__ import annotations

from dash import (
    Input,
    Output,
    State,
    callback,
    ctx,
    no_update,
)
from dash.exceptions import PreventUpdate

from vigie.interface.layouts import build_page_upload

# -- Sidebar ------------------------------------------------------------------


@callback(
    Output("store-sidebar-collapsed", "data"),
    Input("btn-toggle-sidebar", "n_clicks"),
    Input("store-show-results-page", "data"),
    State("store-sidebar-collapsed", "data"),
    prevent_initial_call=True,
)
def update_sidebar_collapsed_state(toggle_clicks, show_results, is_collapsed):
    """Détermine l'état replié de la barre latérale de revue.

    L'affichage des résultats applique le défaut de revue, tandis que le bouton
    permet de l'inverser; la sortie alimente le store d'état de la barre.
    """
    if ctx.triggered_id == "btn-toggle-sidebar":
        return not bool(is_collapsed)
    if ctx.triggered_id == "store-show-results-page" and bool(show_results):
        return True
    raise PreventUpdate


@callback(
    Output("analysis-sidebar", "className"),
    Output("analysis-sidebar-body", "className"),
    Output("analysis-sidebar-title", "className"),
    Output("analysis-sidebar-toggle-icon", "className"),
    Output("main-content-col", "className"),
    Input("store-sidebar-collapsed", "data"),
)
def sync_sidebar_layout(is_collapsed):
    """Synchronise la mise en page avec l'état de la barre latérale.

    Le booléen stocké produit les classes CSS, l'icône et le libellé accessibles
    du bouton de basculement.
    """
    collapsed = bool(is_collapsed)
    sidebar_class = "analysis-sidebar bg-light border-end p-3"
    body_class = "analysis-sidebar-body"
    title_class = "analysis-sidebar-title mb-0 text-primary"
    icon_class = "bi bi-layout-sidebar-inset"
    main_class = "analysis-main-content p-4 bg-light"
    if collapsed:
        sidebar_class += " is-collapsed"
        body_class += " is-collapsed"
        title_class += " is-collapsed"
        icon_class = "bi bi-layout-sidebar"
        main_class += " is-expanded"
    return sidebar_class, body_class, title_class, icon_class, main_class


# -- Proof display mode -------------------------------------------------------


@callback(
    Output("store-proof-display-mode", "data"),
    Input("proof-display-mode", "value"),
)
def on_proof_display_mode_change(value):
    """Persiste le mode d'affichage choisi pour les preuves PDF.

    La valeur du contrôle produit le store utilisé pour choisir entre recadrage
    et page entière annotée lors des prochains rendus.
    """
    if value in ("crop", "full", "footnote"):
        return value
    return no_update


# -- Reset / toggles ----------------------------------------------------------


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("store-review-items", "data", allow_duplicate=True),
    Output("store-review-queue", "data", allow_duplicate=True),
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-review-last-positions", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Output("main-content", "children", allow_duplicate=True),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Output("store-review-filters", "data", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def on_reset(n_clicks):
    """Réinitialise les stores et l'interface pour une nouvelle analyse.

    Le clic retourne les valeurs initiales de la navigation, des résultats et
    des panneaux; il ne supprime pas les artéfacts déjà enregistrés.
    """
    if n_clicks:
        return (
            None,
            None,
            None,
            False,
            None,
            None,
            {"review_id": None, "change_id": None},
            {},
            0,
            build_page_upload(),
            False,
            {"section": "all", "status": "all"},
        )
    raise PreventUpdate


@callback(
    Output("collapse-options", "is_open"),
    Input("btn-toggle-options", "n_clicks"),
    State("collapse-options", "is_open"),
    prevent_initial_call=True,
)
def toggle_options(n, is_open):
    """Inverse l'état ouvert du panneau d'options après un clic."""
    if n:
        return not is_open
    return is_open


@callback(
    Output("collapse-stats", "is_open"),
    Input("btn-toggle-stats", "n_clicks"),
    State("collapse-stats", "is_open"),
    prevent_initial_call=True,
)
def toggle_stats(n, is_open):
    """Inverse l'état ouvert du panneau de statistiques après un clic."""
    if n:
        return not is_open
    return is_open
