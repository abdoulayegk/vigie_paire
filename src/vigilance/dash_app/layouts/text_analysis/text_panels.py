"""Module spécialisé dans le rendu des volets latéraux de détails pour la page d'analyse textuelle."""

from __future__ import annotations

from typing import Any
import dash_bootstrap_components as dbc
from dash import html


def build_text_analysis_detail_panel(selected_data: dict[str, Any] | None = None) -> html.Div:
    """Génère le volet latéral de détails (explications IA, thèmes AMF v2, posture)."""
    if not selected_data:
        return html.Div(
            [
                html.P("Sélectionnez une section textuelle pour afficher l'analyse détaillée de l'IA.", className="text-muted fst-italic p-3"),
            ],
            id="text-analysis-detail-panel",
            className="border rounded p-3 bg-light",
        )

    themes = selected_data.get("themes_amf") or []
    posture = str(selected_data.get("posture_change") or "Neutre")
    explanation = str(selected_data.get("explanation") or "Aucune explication disponible.")

    return html.Div(
        [
            html.H5("Analyse IA Détaillée", className="mb-3 text-primary"),
            html.Div(
                [
                    html.H6("Posture de la banque :", className="fw-bold mb-1"),
                    dbc.Badge(posture, color="info", className="mb-3 fs-6"),
                ]
            ),
            html.Div(
                [
                    html.H6("Thèmes AMF v2 :", className="fw-bold mb-1"),
                    html.Div([dbc.Badge(t, color="secondary", className="me-1 mb-1") for t in themes]),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.H6("Synthèse de l'explication :", className="fw-bold mb-1"),
                    html.P(explanation, className="text-dark bg-white p-2 border rounded"),
                ]
            ),
        ],
        id="text-analysis-detail-panel",
        className="border rounded p-3 bg-light shadow-sm",
    )
