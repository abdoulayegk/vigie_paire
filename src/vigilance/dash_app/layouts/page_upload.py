"""Page initiale: upload et instruction."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def build_page_upload() -> html.Div:
    """Contenu quand aucun PDF detecte."""
    return html.Div(
        [
            html.H4("Comparateur de Rapports Bancaires"),
            html.P(
                "Téléversez les deux fichiers PDF du trimestre courant et du trimestre précédent "
                "dans la barre latérale, puis lancez l'analyse depuis le panneau de gauche.",
                className="text-muted",
            ),
            dbc.Alert(
                "En attente des rapports courant et précédent.",
                color="info",
                className="mt-4",
            ),
        ],
        className="p-4",
    )
