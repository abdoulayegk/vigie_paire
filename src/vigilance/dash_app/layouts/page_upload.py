"""Page initiale: upload et instruction."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def build_page_upload() -> html.Div:
    """Construit le layout de la page d'accueil lorsqu'aucun PDF n'est detecte.

    Returns:
        Composant ``html.Div`` affichant les instructions de telechargement
        et un message d'information.
    """
    return html.Div(
        [
            html.H4("Comparateur de Rapports Bancaires"),
            html.P(
                "Consultez d'abord les analyses enregistrées dans la barre latérale. "
                "Si aucune analyse n'est disponible, téléversez ensuite les rapports "
                "PDF du trimestre courant et du trimestre précédent pour lancer une nouvelle exécution.",
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
