"""Composants graphiques du cockpit : cartes, donuts, barres et tableaux de priorite.

Extrait de ``vigie_dashboard_flow.py`` sans modification.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from vigilance.dash_app.services.export_helpers import _is_high_priority_item
from vigilance.dash_app.services.review_navigation import _table_decision_bucket

from .formatting import (
    _badge_class,
    _change_label,
    _change_total,
    _confidence_label,
    _format_number,
    _impact_label,
    _indicator_confidence,
    _low_confidence,
    _plot_layout,
    _safe_int,
    _table_title,
)

def _kpi(icon: str, label: str, value: int | str, helper: str, tone: str = "neutral") -> html.Div:
    """Construit une carte KPI (icône + libellé + valeur + helper) pour le cockpit."""
    return html.Div(
        [
            html.Div(html.I(className=f"bi {icon}"), className=f"vigie-cockpit-kpi-icon is-{tone}"),
            html.Div(
                [
                    html.Div(label, className="vigie-cockpit-kpi-label"),
                    html.Div(_format_number(value), className="vigie-cockpit-kpi-value"),
                    html.Div(helper, className="vigie-cockpit-kpi-helper"),
                ],
                className="min-w-0",
            ),
        ],
        className="vigie-cockpit-kpi-card",
    )


def _chart_card(title: str, figure: go.Figure, *, theme: str) -> html.Div:
    """Encapsule un graphique Plotly dans le composant carte cockpit."""
    figure.update_layout(**_plot_layout(theme))
    return html.Div(
        [
            html.Div(title, className="vigie-cockpit-panel-title"),
            dcc.Graph(figure=figure, config={"displayModeBar": False}, className="vigie-cockpit-graph"),
        ],
        className="vigie-cockpit-panel",
    )


def _donut_chart(text_total: int, indicator_total: int, *, theme: str) -> html.Div:
    """Graphique donut combiné texte + indicateurs avec total au centre."""
    total = text_total + indicator_total
    if total <= 0:
        return html.Div("Aucun changement disponible", className="vigie-cockpit-empty")
    is_light = theme == "light"
    main_color = "#172033" if is_light else "#f4f7fb"
    muted_color = "#64748b" if is_light else "#c8d2e0"
    fig = go.Figure(
        go.Pie(
            labels=["Changements textuels", "Changements indicateurs"],
            values=[text_total, indicator_total],
            hole=0.62,
            marker={"colors": ["#e45142", "#4b74f2"]},
            textinfo="none",
        )
    )
    fig.add_annotation(text=str(total), x=0.5, y=0.55, showarrow=False, font={"size": 28, "color": main_color})
    fig.add_annotation(
        text="Total des<br>changements", x=0.5, y=0.42, showarrow=False, font={"size": 11, "color": muted_color}
    )
    return _chart_card("APERÇU COMBINÉ", fig, theme=theme)


def _bar_chart(values: dict[str, int], *, theme: str) -> html.Div:
    """Histogramme horizontal de la répartition par nature de changement."""
    grid_color = "#d8e0ea" if theme == "light" else "#223248"
    fig = go.Figure(
        go.Bar(
            x=list(values.values()),
            y=list(values.keys()),
            orientation="h",
            marker={"color": ["#4b74f2", "#e45142", "#f3b23c", "#68b976"]},
            text=[str(v) for v in values.values()],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(xaxis={"gridcolor": grid_color}, yaxis={"gridcolor": "rgba(0,0,0,0)"})
    return _chart_card("RÉPARTITION PAR NATURE", fig, theme=theme)


def _review_status_chart(counts: dict[str, int], *, theme: str) -> html.Div:
    """Donut du statut de la file de revue (validés / en attente / rejetés)."""
    total = _safe_int(counts.get("total"))
    if total <= 0:
        return html.Div("Aucune revue disponible", className="vigie-cockpit-empty")
    is_light = theme == "light"
    main_color = "#172033" if is_light else "#f4f7fb"
    muted_color = "#64748b" if is_light else "#c8d2e0"
    values = [
        _safe_int(counts.get("approved")),
        _safe_int(counts.get("pending")),
        _safe_int(counts.get("rejected")),
    ]
    labels = ["Validés", "En attente", "Rejetés"]
    marker_colors = ["#68b976", "#f3b23c", "#e45142"]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker={"colors": marker_colors},
            textinfo="none",
            sort=False,
            domain={"x": [0.0, 0.46], "y": [0.0, 1.0]},
        )
    )
    fig.add_annotation(text=str(total), x=0.23, y=0.56, showarrow=False, font={"size": 28, "color": main_color})
    fig.add_annotation(text="Total", x=0.23, y=0.41, showarrow=False, font={"size": 12, "color": muted_color})
    for idx, (label, value, color) in enumerate(zip(labels, values, marker_colors, strict=False)):
        y = 0.75 - idx * 0.24
        pct = value / total if total else 0
        fig.add_annotation(text="●", x=0.55, y=y, showarrow=False, font={"size": 18, "color": color}, xanchor="left")
        fig.add_annotation(
            text=label,
            x=0.62,
            y=y,
            showarrow=False,
            font={"size": 12, "color": main_color},
            xanchor="left",
        )
        fig.add_annotation(
            text=f"{value} ({pct:.0%})",
            x=0.98,
            y=y,
            showarrow=False,
            font={"size": 12, "color": main_color},
            xanchor="right",
        )
    fig.update_layout(showlegend=False)
    return _chart_card("STATUT DE LA FILE DE REVUE", fig, theme=theme)


def _top_text(text_metrics: dict[str, Any]) -> html.Div:
    """Construit la liste des 5 principaux changements textuels avec badge d'impact."""
    rows = []
    for item in text_metrics.get("top") or []:
        label = {
            "MAJEUR": "Majeur",
            "MODERE": "Modéré",
            "MINEUR": "Faible",
            "NON_PERTINENT": "Non pertinent",
        }.get(item["impact"], item["impact"].title())
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(item["summary"], className="vigie-cockpit-change-title"),
                            html.Div(item["section"], className="vigie-cockpit-change-meta"),
                        ],
                        className="min-w-0",
                    ),
                    html.Span(label, className=_badge_class(label)),
                ],
                className="vigie-cockpit-change-row",
            )
        )
    return html.Div(rows or ["Aucun changement textuel prioritaire."], className="vigie-cockpit-change-list")


def _priority_table(indicator_metrics: dict[str, Any], review_queue: list | None) -> html.Div:
    """Construit la table « Top tableaux à prioriser » du cockpit."""
    rows = _priority_rows(indicator_metrics, review_queue, limit=6)
    body = []
    for idx, row in enumerate(rows, start=1):
        body.append(
            html.Tr(
                [
                    html.Td(str(idx)),
                    html.Td(row["title"], className="vigie-cockpit-table-name"),
                    html.Td(row["change"]),
                    html.Td("Indicateurs"),
                    html.Td(html.Span(row["impact"], className=_badge_class(row["impact"]))),
                    html.Td(html.Span(row["confidence"], className=_badge_class(row["confidence"]))),
                    html.Td(html.Span(row["status_label"], className=_badge_class(row["status_label"]))),
                ]
            )
        )
    return html.Div(
        [
            html.Div("TOP TABLEAUX À PRIORISER", className="vigie-cockpit-panel-title"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("#"),
                                html.Th("Nom du tableau"),
                                html.Th("Type"),
                                html.Th("Pipeline"),
                                html.Th("Impact"),
                                html.Th("Confiance"),
                                html.Th("Statut"),
                            ]
                        )
                    ),
                    html.Tbody(body or [html.Tr(html.Td("Aucun tableau prioritaire détecté.", colSpan=7))]),
                ],
                borderless=True,
                responsive=True,
                size="sm",
                className="vigie-cockpit-table mb-0",
            ),
        ],
        className="vigie-cockpit-panel",
    )


def _priority_rows(indicator_metrics: dict[str, Any], review_queue: list | None, *, limit: int = 6) -> list[dict[str, Any]]:
    """Calcule les lignes priorisées (haute priorité, faible confiance) pour le top tableaux."""
    queue_lookup = {}
    if isinstance(review_queue, list):
        for table in review_queue:
            if isinstance(table, dict):
                queue_lookup[str(table.get("table_name") or table.get("table_title") or "")] = _table_decision_bucket(
                    table
                )
    rows = []
    for comp in indicator_metrics.get("comparisons") or []:
        if _change_total(comp) <= 0:
            continue
        title = _table_title(comp)
        score = _indicator_confidence(comp)
        rows.append(
            {
                "title": title,
                "change": _change_label(comp),
                "impact": _impact_label(comp),
                "confidence": _confidence_label(score),
                "status": queue_lookup.get(title, "pending"),
                "rank": (0 if _is_high_priority_item(comp) else 1, 0 if _low_confidence(comp) else 1, -(score or 0)),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    result: list[dict[str, Any]] = []
    for row in rows[:limit]:
        status = {"approved": "Validé", "rejected": "Rejeté", "pending": "En attente"}.get(row["status"], "En attente")
        result.append({**row, "status_label": status})
    return result
