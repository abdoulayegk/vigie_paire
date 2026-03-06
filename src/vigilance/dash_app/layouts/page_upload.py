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
                "Uploadez les deux fichiers PDF du trimestre courant et du trimestre precedent "
                "dans la barre laterale, puis cliquez sur '1. Detecter Sections' pour demarrer l'analyse.",
                className="text-muted",
            ),
            dbc.Alert(
                "En attente d'upload des rapports courant et precedent.",
                color="info",
                className="mt-4",
            ),
        ],
        className="p-4",
    )
