"""Composant spécialisé dans la génération des badges et drapeaux de preuves visuelles Dash."""

from __future__ import annotations

from typing import Any
from dash import html


def build_proof_badges(change_item: dict[str, Any], year_t1: str = "", year_t2: str = "") -> html.Div:
    """Génère les badges HTML pour marquer les évolutions (T1 vs T2, note ajoutée, etc.)."""
    change_type = str(change_item.get("change_type") or "").lower()
    badges = []

    if "added" in change_type:
        label = f"{year_t2 or 'T2'} - Ajouté"
        badges.append(html.Span(label, className="badge badge-success me-1"))
    elif "removed" in change_type:
        label = f"{year_t1 or 'T1'} - Retiré"
        badges.append(html.Span(label, className="badge badge-danger me-1"))
    elif "modified" in change_type or "renamed" in change_type:
        label = "Modification"
        badges.append(html.Span(label, className="badge badge-warning me-1"))

    return html.Div(badges, className="d-inline-block")
