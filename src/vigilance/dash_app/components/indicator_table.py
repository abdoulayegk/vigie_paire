"""Composant tableau indicateurs."""

from __future__ import annotations

from dash import dash_table, html


def indicator_diff_table(data: list[dict]) -> dash_table.DataTable | html.Div:
    """
    Tableau des changements d'indicateurs.

    data: Liste de dicts avec colonnes Section, Type, Indicateur, etc.
    """
    if not data:
        return html.Div("Aucun changement", className="text-muted")

    columns = [{"name": k, "id": k} for k in (data[0].keys() if data else [])]
    return dash_table.DataTable(
        data=data,
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold"},
    )
