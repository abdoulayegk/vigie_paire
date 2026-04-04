"""Composant cartes metriques pour le tableau de bord Dash."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def metric_card(title: str, value: str | int, delta: str | None = None) -> dbc.Card:
    """Construit une carte metrique compacte avec titre, valeur et delta optionnel.

    Args:
        title: Libelle de la metrique affiche sous la valeur.
        value: Valeur principale (chiffre ou texte).
        delta: Badge optionnel indiquant la variation.

    Returns:
        Une carte Bootstrap affichant la metrique.
    """
    body = [
        html.H5(str(value), className="card-title"),
        html.P(title, className="card-text small"),
    ]
    if delta:
        body.append(html.Span(delta, className="badge bg-secondary"))

    return dbc.Card(
        dbc.CardBody(body),
        className="mb-2",
    )


def metrics_row(metrics: list[dict]) -> html.Div:
    """Construit une rangee horizontale de cartes metriques.

    Args:
        metrics: Liste de dictionnaires ``{"title": ..., "value": ...,
            "delta": ...}`` decrivant chaque carte.

    Returns:
        Une ``Row`` Bootstrap contenant les cartes.
    """
    cols = []
    for m in metrics:
        cols.append(
            dbc.Col(
                metric_card(
                    m.get("title", ""),
                    m.get("value", "-"),
                    m.get("delta"),
                ),
                md=2,
            )
        )
    return dbc.Row(cols, className="mb-3")
