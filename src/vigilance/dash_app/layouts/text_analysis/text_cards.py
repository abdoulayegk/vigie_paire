"""Module spécialisé dans le rendu des cartes de sections et paragraphes textuels comparés."""

from __future__ import annotations

from typing import Any
from dash import html


def build_text_section_card(
    section_id: str,
    section_title: str,
    change_data: dict[str, Any],
) -> html.Div:
    """Construit une carte HTML affichant une section textuelle et ses évolutions."""
    impact = str(change_data.get("impact_level") or "MINEUR").upper()
    badge_color = "danger" if impact == "MAJEUR" else "warning" if impact == "MODERE" else "info"

    return html.Div(
        [
            html.Div(
                [
                    html.Span(section_title, className="fw-bold me-2"),
                    html.Span(impact, className=f"badge bg-{badge_color}"),
                ],
                className="card-header bg-light d-flex justify-content-between align-items-center",
            ),
            html.Div(
                [
                    html.P(str(change_data.get("explanation") or change_data.get("summary") or ""), className="card-text mb-0"),
                ],
                className="card-body",
            ),
        ],
        id=f"text-card-{section_id}",
        className="card mb-3 shadow-sm",
    )
