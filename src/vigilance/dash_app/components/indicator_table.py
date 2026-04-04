"""Composant tableau des indicateurs pour le tableau de bord Dash."""

from __future__ import annotations

from dash import dash_table, html


def indicator_diff_table(data: list[dict]) -> dash_table.DataTable | html.Div:
    """Construit un tableau interactif des changements d'indicateurs.

    Args:
        data: Liste de dictionnaires avec les colonnes Section, Type,
            Indicateur, etc.

    Returns:
        Un ``DataTable`` paginable et triable, ou un ``Div`` de repli
        si *data* est vide.
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
