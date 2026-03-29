"""Page chargement d'une comparaison existante."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def build_page_load(options: list[dict]) -> html.Div:
    """Contenu pour charger une comparaison existante."""
    return html.Div(
        id="load-content",
        children=[
            html.H4("Charger une comparaison existante"),
            html.P(
                "Sélectionnez un fichier JSON de comparaison.", className="text-muted"
            ),
            html.Hr(),
            dcc.Dropdown(
                id="load-comparison-dropdown",
                options=options,
                value=None,
                placeholder="Sélectionner...",
                className="mb-3",
            ),
            dbc.Button(
                "Charger",
                id="btn-load-comparison",
                color="primary",
            ),
        ],
        className="p-4",
    )
