"""Page chargement d'une comparaison existante."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def build_page_load(options: list[dict]) -> html.Div:
    """Construit le layout de la page de chargement d'une analyse enregistree.

    Args:
        options: Liste de dictionnaires ``{label, value}`` pour le menu deroulant
            de selection des comparaisons sauvegardees.

    Returns:
        Composant ``html.Div`` contenant le formulaire de chargement.
    """
    return html.Div(
        id="load-content",
        children=[
            html.H4("Charger une analyse enregistrée"),
            html.P(
                "Sélectionnez un fichier JSON de comparaison enregistré.",
                className="text-muted",
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
