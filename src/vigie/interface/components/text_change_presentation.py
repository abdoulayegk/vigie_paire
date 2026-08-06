"""Composants Dash dédiés à la présentation des changements textuels."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def build_source_evidence_details(text_block: html.Div) -> html.Details:
    """Place les extraits exacts au second niveau de lecture de la carte."""
    return html.Details(
        [
            html.Summary(
                "Voir la preuve source",
                className="fw-semibold small text-primary",
                style={"cursor": "pointer"},
            ),
            html.Div(text_block, className="pt-3"),
        ],
        open=False,
        className="mb-3 border rounded bg-light px-2 pt-2",
    )


def atomic_parent_key(
    change: dict[str, Any],
    *,
    section_title: str,
) -> tuple[str, str, str] | None:
    """Retourne une clé de groupe stable pour les enfants d'une même liste."""
    unit_roles = {
        str(change.get("unit_role_t1") or "").strip().lower(),
        str(change.get("unit_role_t2") or "").strip().lower(),
    }
    if "item" not in unit_roles:
        return None

    parent_t1 = str(change.get("parent_chunk_id_t1") or "").strip()
    parent_t2 = str(change.get("parent_chunk_id_t2") or "").strip()
    if not parent_t1 and not parent_t2:
        return None
    return section_title, parent_t1, parent_t2


def atomic_parent_context(change: dict[str, Any]) -> str:
    """Retourne le contexte parent courant, puis précédent en repli."""
    context = str(change.get("parent_context_t2") or change.get("parent_context_t1") or "").strip()
    context = " ".join(context.split())
    if len(context) <= 280:
        return context
    window = context[:280]
    word_end = window.rfind(" ")
    return (window[:word_end] if word_end > 80 else window).rstrip(" ,;:") + "."


def build_atomic_change_group(
    *,
    parent_context: str,
    cards: list[dbc.Card],
) -> html.Div:
    """Regroupe visuellement les idées atomiques sans les fusionner."""
    count = len(cards)
    count_label = f"{count} {'idée modifiée' if count == 1 else 'idées modifiées'}"
    header_children: list[Any] = [
        html.Strong("Bloc de liste analysé", className="small"),
        dbc.Badge(
            count_label,
            color="primary",
            className="ms-2",
        ),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        header_children,
                        className="d-flex align-items-center flex-wrap mb-1",
                    ),
                    html.P(parent_context, className="small text-muted mb-0") if parent_context else None,
                ],
                className="px-3 py-2 border-bottom bg-light",
            ),
            html.Div(cards, className="p-2"),
        ],
        className="mb-3 border rounded bg-white",
    )
