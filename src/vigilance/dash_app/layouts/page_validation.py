"""Page validation des sections detectees."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def build_page_validation() -> html.Div:
    """Contenu pour la validation des sections (placeholder dynamique)."""
    return html.Div(
        id="validation-content",
        children=[
            html.H4("Validation des Sections"),
            html.P(
                "Sections detectees. Ajustez si necessaire puis validez.", className="text-muted"
            ),
            dbc.Spinner(html.Div(id="validation-sections-container"), color="primary"),
            html.Hr(),
            html.Div(id="validation-adjusted-indicator", className="mb-2"),
            dbc.Button(
                "2. Valider et Analyser",
                id="btn-analyze",
                color="primary",
                className="mt-2",
            ),
        ],
        className="p-4",
    )
