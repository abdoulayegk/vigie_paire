"""Module spécialisé dans la génération des filtres pour la page d'analyse textuelle Dash."""

from __future__ import annotations

from typing import Any
import dash_bootstrap_components as dbc
from dash import dcc, html


def build_text_analysis_filters(bank_codes: list[str] | None = None) -> html.Div:
    """Génère la barre de filtres interactifs pour la page d'analyse textuelle."""
    banks = bank_codes or ["rbc", "bmo", "cibc", "bns", "td", "bnc"]
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Banque :", className="fw-bold mb-1"),
                            dcc.Dropdown(
                                id="text-filter-bank",
                                options=[{"label": b.upper(), "value": b} for b in banks],
                                value=banks[0] if banks else "rbc",
                                clearable=False,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Niveau d'impact :", className="fw-bold mb-1"),
                            dcc.Dropdown(
                                id="text-filter-impact",
                                options=[
                                    {"label": "Tous les impacts", "value": "ALL"},
                                    {"label": "Majeur", "value": "MAJEUR"},
                                    {"label": "Modéré", "value": "MODERE"},
                                    {"label": "Mineur", "value": "MINEUR"},
                                ],
                                value="ALL",
                                clearable=False,
                            ),
                        ],
                        width=3,
                    ),
                ],
                className="g-2 mb-3",
            ),
        ],
        id="text-analysis-filters-container",
    )
