"""Composant de file de revue V2 -- tableaux groupes et dedupliques.

Ce composant genere le panneau gauche de l'interface de revue, affichant
un element par tableau (et non par changement) avec indicateurs de
progression.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from vigie.interface.components.review_display_shared import section_display_label
from vigie.interface.components.table_title_resolver import resolve_display_table_title
from vigie.support.i18n import t
from vigie.interface.review_models_v2 import ChangeType


_ACTION_DISPLAY = {
    "revue_prioritaire": ("Revue prioritaire", "danger"),
    "investigation": ("Investigation", "warning"),
    "confirmation": ("Confirmation", "info"),
    "information": ("Information", "secondary"),
    "aucune": None,
}

_IMPACT_LEVEL_BADGE: dict[str, tuple[str, str]] = {
    "MAJEUR": ("MAJEUR", "danger"),
    "MODERE": ("MODÉRÉ", "warning"),
    "MINEUR": ("MINEUR", "info"),
}


def _format_section(section: str) -> str:
    """Formate le nom de section pour l'affichage."""
    return section_display_label(section)


def _queue_page_summary_v2(table: dict) -> str:
    """Retourne un resume concis du contexte de pages pour les elements de la file."""
    page_t1 = table.get("page_t1")
    page_t2 = table.get("page_t2")

    # Check if this is an added or removed table
    changes = table.get("changes", [])
    change_types = {c.get("change_type", "") for c in changes}

    if ChangeType.TABLE_ADDED.value in change_types or "table_added" in change_types:
        return f"p.{page_t2}" if page_t2 is not None else ""
    if ChangeType.TABLE_REMOVED.value in change_types or "table_removed" in change_types:
        return f"p.{page_t1}" if page_t1 is not None else ""
    if page_t1 is not None and page_t2 is not None:
        return f"Préc. p.{page_t1} / Cour. p.{page_t2}"
    if page_t2 is not None:
        return f"p.{page_t2}"
    if page_t1 is not None:
        return f"p.{page_t1}"
    return ""


def _build_genai_summary_row(table: dict) -> html.Div | None:
    """Synthèse GenAI compacte affichée sur chaque carte de la file de revue.

    Aligné sur la taxonomie AMF unifiée :
    - ✨ Nouvelle idée (si ``nouvelle_idee=True``)
    - Badge impact_level coloré (MAJEUR/MODÉRÉ/MINEUR)
    - Badge action_requise si revue prioritaire/investigation
    - Justification AMF tronquée (≤ 90 chars)

    Cartes filtrées : si is_relevant=False ou genai_analysis absent → None
    (la carte n'affiche rien — l'analyste se concentre sur le pertinent).
    """
    ga = table.get("genai_analysis")
    if not isinstance(ga, dict) or not ga:
        return None
    if not bool(ga.get("is_relevant", False)):
        return None

    nouvelle_idee = bool(ga.get("nouvelle_idee", False))
    impact_level = str(ga.get("impact_level", "") or "").strip().upper()
    action = str(ga.get("action_requise", "") or "").strip().lower()
    # Schéma AMF v2 strict (plus de fallback legacy).
    justification = str(ga.get("nouvelle_idee_justification", "") or "").strip()

    chips: list = []
    if nouvelle_idee:
        chips.append(
            dbc.Badge(
                "✨ Nouvelle idée",
                color="primary",
                className="me-1",
                style={"fontSize": "0.65rem"},
            )
        )
    impact_info = _IMPACT_LEVEL_BADGE.get(impact_level)
    if impact_info:
        label, color = impact_info
        chips.append(dbc.Badge(label, color=color, className="me-1", style={"fontSize": "0.65rem"}))
    action_info = _ACTION_DISPLAY.get(action)
    if action_info and action in {"revue_prioritaire", "investigation"}:
        a_label, a_color = action_info
        chips.append(
            dbc.Badge(
                a_label,
                color=a_color,
                className="me-1",
                style={"fontSize": "0.65rem"},
            )
        )

    if not chips and not justification:
        return None

    parts: list = []
    if chips:
        parts.append(html.Div(chips, className="d-flex flex-wrap gap-1 mb-1"))
    if justification:
        display = justification if len(justification) <= 90 else justification[:87] + "…"
        parts.append(
            html.Div(
                display,
                className="review-queue-narrative",
                title=justification,
            )
        )
    return html.Div(parts, className="review-queue-genai-summary mt-1")


def _build_table_metric_badges(summary: dict) -> list[html.Span]:
    """Construit les badges metriques (ajoutes, supprimes, renommes, notes) pour un tableau."""
    badges: list[html.Span] = []
    n_added = int(summary.get("indicators_added", 0) or 0)
    n_removed = int(summary.get("indicators_removed", 0) or 0)
    n_renamed = int(summary.get("indicators_renamed", 0) or 0)
    n_footnotes = int(summary.get("footnotes_changed", 0) or 0)

    if n_added:
        badges.append(html.Span(f"+{n_added}", className="review-queue-stat-chip is-added"))
    if n_removed:
        badges.append(html.Span(f"-{n_removed}", className="review-queue-stat-chip is-removed"))
    if n_renamed:
        badges.append(html.Span(f"RN {n_renamed}", className="review-queue-stat-chip is-renamed"))
    if n_footnotes:
        badges.append(
            html.Span(
                f"FN {n_footnotes}",
                className="review-queue-stat-chip is-footnotes",
            )
        )

    return badges


def _build_progress_pill(validated: int, total: int, table_status: str, is_active: bool) -> html.Span | None:
    """Construit la pastille de progression valides/total pour un tableau."""
    if total <= 0:
        return None

    class_name = "review-queue-progress-pill"
    if table_status == "completed":
        class_name += " is-completed"
    elif table_status == "partial":
        class_name += " is-partial"
    else:
        class_name += " is-pending"
    if is_active:
        class_name += " is-active"

    return html.Span(f"{validated}/{total}", className=class_name)


def build_review_queue_v2(
    tables: list[dict],
    current_review_id: str | None,
    current_change_id: str | None = None,
    active_filters: dict | None = None,
) -> html.Div:
    """Construit le panneau de file de revue V2 avec tableaux groupes.

    Args:
        tables: Liste de dictionnaires ``ReviewTableItem``
            (provenant de ``store-review-queue``).
        current_review_id: Identifiant du tableau actuellement selectionne.
        current_change_id: Identifiant du changement actuellement selectionne.
        active_filters: Filtres optionnels (section, statut).

    Returns:
        Un ``Div`` contenant le panneau complet de la file de revue.
    """
    if not tables:
        return html.Div(
            [
                html.H5(t("file_review")),
                html.P("Aucun élément à réviser.", className="text-muted"),
            ]
        )

    # Extract all sections for filter
    all_sections = sorted(set(t.get("section", "Autre") for t in tables))

    active_section = (active_filters or {}).get("section", "all")
    active_status = (active_filters or {}).get("status", "all")

    def _matches_filters(table: dict) -> bool:
        """Verifie si un tableau correspond aux filtres actifs."""
        if active_section and active_section != "all":
            if table.get("section") != active_section:
                return False
        if active_status and active_status != "all":
            if table.get("table_status") != active_status:
                return False
        return True

    # Filter tables
    filtered_with_idx = [(idx, table) for idx, table in enumerate(tables) if _matches_filters(table)]
    filtered_tables = [t for _, t in filtered_with_idx]

    # Compute stats
    total = len(filtered_tables)
    completed = sum(1 for t in filtered_tables if t.get("table_status") == "completed")
    partial = sum(1 for t in filtered_tables if t.get("table_status") == "partial")
    pending = sum(1 for t in filtered_tables if t.get("table_status") == "pending")

    # Build filter buttons
    filter_buttons: list = [
        dbc.Button(
            [
                html.I(className="bi bi-funnel me-2"),
                f"{t('all_sections')} ({len(tables)})",
            ],
            id={"type": "filter-section-v2", "value": "all"},
            color="light",
            size="sm",
            className=(
                "review-queue-filter-button w-100 text-start mb-1" + (" is-active" if active_section == "all" else "")
            ),
        )
    ]
    for section in all_sections:
        section_count = sum(1 for tb in tables if tb.get("section") == section)
        section_label = _format_section(section)
        filter_buttons.append(
            dbc.Button(
                [
                    html.I(className="bi bi-folder me-2"),
                    f"{section_label} ({section_count})",
                ],
                id={"type": "filter-section-v2", "value": section},
                color="light",
                size="sm",
                className=(
                    "review-queue-filter-button w-100 text-start mb-1"
                    + (" is-active" if active_section == section else "")
                ),
            )
        )

    filter_bar = html.Div(filter_buttons, className="review-queue-filter-bar mb-3 p-2 rounded border")

    queue_items = []
    for _, table in filtered_with_idx:
        summary = table.get("summary", {})
        n_total = summary.get("total_changes", 0)
        n_validated = summary.get("validated", 0)
        table_status = table.get("table_status", "pending")
        if table_status == "completed":
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif table_status == "partial":
            icon = html.I(className="bi bi-pie-chart-fill text-info me-2")
        else:
            icon = html.I(className="bi bi-circle text-warning me-2")

        review_id = str(table.get("review_id") or table.get("table_key") or "")
        is_active = bool(review_id) and review_id == str(current_review_id or "")

        # Table display info
        section = _format_section(table.get("section", ""))
        table_name = resolve_display_table_title(table)
        page_summary = _queue_page_summary_v2(table)
        context_text = f"{section} - {page_summary}" if page_summary else section
        stat_badges = _build_table_metric_badges(summary)
        progress_badge = _build_progress_pill(
            int(n_validated or 0),
            int(n_total or 0),
            str(table_status or "pending"),
            is_active,
        )
        genai_summary = _build_genai_summary_row(table)

        row_class = "review-queue-table-row"
        if is_active:
            row_class += " is-active border-primary"
        if table_status == "completed":
            row_class += " is-completed"
        elif table_status == "partial":
            row_class += " is-partial"
        else:
            row_class += " is-pending"

        queue_items.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(icon, className="review-queue-table-icon"),
                                    html.Div(
                                        table_name,
                                        className="review-queue-table-title",
                                        title=table_name,
                                    ),
                                ],
                                className="review-queue-table-head-main",
                            ),
                            progress_badge,
                        ],
                        className="review-queue-table-head",
                    ),
                    html.Div(
                        context_text,
                        className="review-queue-table-context",
                    ),
                    genai_summary,
                    html.Div(
                        [
                            html.Span(
                                f"{n_total} changement(s)",
                                className="review-queue-table-count",
                            ),
                            html.Div(
                                stat_badges,
                                className="review-queue-table-stats",
                            )
                            if stat_badges
                            else None,
                        ],
                        className="review-queue-table-foot",
                    ),
                ],
                id={"type": "queue-table-item-v2", "review_id": review_id},
                className=row_class,
                n_clicks=0,
            )
        )

    return html.Div(
        [
            html.H5(t("file_review"), className="mb-3"),
            html.Div(
                [
                    html.Span(f"Total: {total}", className="me-3 fw-bold"),
                    html.Span(
                        [
                            html.I(className="bi bi-check-circle-fill text-success me-1"),
                            f"{completed}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-pie-chart-fill text-info me-1"),
                            f"{partial}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-circle text-warning me-1"),
                            f"{pending}",
                        ],
                        className="small",
                    ),
                ],
                className="mb-3 small text-muted",
            ),
            filter_bar,
            html.Div(
                queue_items,
                className="review-queue-v2-list overflow-auto",
                style={"maxHeight": "calc(100vh - 340px)"},
            ),
        ],
        className="h-100",
    )
