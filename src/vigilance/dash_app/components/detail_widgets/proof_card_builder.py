"""Composant spécialisé dans la génération des cartes de comparaison Dash."""

from __future__ import annotations

from typing import Any
from dash import html
from vigilance.dash_app.components.detail_widgets.proof_badge_builder import build_proof_badges


def build_comparison_proof_card(
    item_id: str,
    change_item: dict[str, Any],
    title: str = "",
) -> html.Div:
    """Construit une carte HTML de preuve visuelle pour l'interface Dash."""
    badges_widget = build_proof_badges(change_item)
    return html.Div(
        [
            html.H6(title or "Preuve Visuelle", className="card-title"),
            badges_widget,
            html.Div(
                str(change_item.get("description") or change_item.get("text") or ""),
                className="card-body text-muted",
            ),
        ],
        id=f"proof-card-{item_id}",
        className="card mb-2 p-2 shadow-sm",
    )
