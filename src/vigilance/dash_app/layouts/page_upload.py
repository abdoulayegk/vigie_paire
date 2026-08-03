"""Page initiale: analyses enregistrees et instruction."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def build_page_upload() -> html.Div:
    """Construit le layout de la page d'accueil lorsqu'aucune analyse n'est chargee.

    Returns:
        Composant ``html.Div`` affichant les instructions de chargement
        et un message d'information.
    """
    return html.Div(
        [
            html.H4("Comparateur de Rapports Bancaires"),
            html.P(
                "Consultez d'abord les analyses enregistrées dans la barre latérale. "
                "Sélectionnez une analyse disponible, puis chargez-la pour consulter les indicateurs, "
                "l'analyse textuelle et les changements communs entre banques.",
                className="text-muted",
            ),
            dbc.Alert(
                "Aucune analyse affichée pour le moment.",
                color="info",
                className="mt-4",
            ),
        ],
        className="p-4",
    )
