"""Composant cartes metriques."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def metric_card(title: str, value: str | int, delta: str | None = None) -> dbc.Card:
    """Carte metrique simple."""
    body = [html.H5(str(value), className="card-title"), html.P(title, className="card-text small")]
    if delta:
        body.append(html.Span(delta, className="badge bg-secondary"))

    return dbc.Card(
        dbc.CardBody(body),
        className="mb-2",
    )


def metrics_row(metrics: list[dict]) -> html.Div:
    """
    Ligne de metriques.

    metrics: [{"title": "...", "value": ..., "delta": "..."}]
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
