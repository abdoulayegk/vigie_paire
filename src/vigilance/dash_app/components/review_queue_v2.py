"""Review Queue Component V2 - Deduplicated grouped tables.

This component renders the left panel of the review UI showing
one item per table (not per change), with progress indicators.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from app.i18n import t
from app.review_models_v2 import ChangeType

_RELEVANCE_DISPLAY = {
    "REGLEMENTAIRE": "Reglementaire",
    "NOUVELLE_DIVULGATION": "Nouvelle divulgation",
    "STRUCTUREL": "Structurel",
    "NON_SIGNIFICATIF": "Non significatif",
    "NON_CLASSIFIE": "Non classifie",
}

_RELEVANCE_COLORS = {
    "REGLEMENTAIRE": "danger",
    "NOUVELLE_DIVULGATION": "info",
    "STRUCTUREL": "primary",
    "NON_SIGNIFICATIF": "secondary",
    "NON_CLASSIFIE": "light",
}

_RISK_DISPLAY = {
    "ELEVE": "Eleve",
    "MODERE": "Modere",
    "FAIBLE": "Faible",
}

_RISK_COLORS = {
    "ELEVE": "danger",
    "MODERE": "warning",
    "FAIBLE": "success",
}


def _format_section(section: str) -> str:
    """Format section name for display."""
    if not section:
        return "Autre"
    # Capitalize first letter of each word
    return " ".join(w.capitalize() for w in section.replace("_", " ").split())


def _queue_page_summary_v2(table: dict) -> str:
    """Return concise page context string for queue items."""
    page_t1 = table.get("page_t1")
    page_t2 = table.get("page_t2")

    # Check if this is an added or removed table
    changes = table.get("changes", [])
    change_types = {c.get("change_type", "") for c in changes}

    if ChangeType.TABLE_ADDED.value in change_types or "table_added" in change_types:
        return f"p.{page_t2}" if page_t2 is not None else ""
    if (
        ChangeType.TABLE_REMOVED.value in change_types
        or "table_removed" in change_types
    ):
        return f"p.{page_t1}" if page_t1 is not None else ""
    if page_t1 is not None and page_t2 is not None:
        return f"T1 p.{page_t1} / T2 p.{page_t2}"
    if page_t2 is not None:
        return f"p.{page_t2}"
    if page_t1 is not None:
        return f"p.{page_t1}"
    return ""


def build_review_queue_v2(
    tables: list[dict],
    current_idx: int,
    active_filters: dict | None = None,
) -> html.Div:
    """Build the left-side review queue panel V2 with grouped tables.

    Args:
        tables: List of ReviewTableItem dicts (from store-review-queue)
        current_idx: Currently selected table index
        active_filters: Optional filters (section, status)

    Returns:
        Div containing the queue panel
    """
    if not tables:
        return html.Div(
            [
                html.H5(t("file_review")),
                html.P("Aucun element a reviser.", className="text-muted"),
            ]
        )

    # Extract all sections for filter
    all_sections = sorted(set(t.get("section", "Autre") for t in tables))

    active_section = (active_filters or {}).get("section", "all")
    active_status = (active_filters or {}).get("status", "all")

    def _matches_filters(table: dict) -> bool:
        if active_section and active_section != "all":
            if table.get("section") != active_section:
                return False
        if active_status and active_status != "all":
            if table.get("table_status") != active_status:
                return False
        return True

    # Filter tables
    filtered_with_idx = [
        (idx, table) for idx, table in enumerate(tables) if _matches_filters(table)
    ]
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
            color="primary" if active_section == "all" else "light",
            size="sm",
            className="w-100 text-start mb-1",
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
                color="primary" if active_section == section else "light",
                size="sm",
                className="w-100 text-start mb-1",
            )
        )

    filter_bar = html.Div(filter_buttons, className="mb-3 p-2 bg-white rounded border")

    # Build list items
    list_items = []
    for full_idx, table in filtered_with_idx:
        summary = table.get("summary", {})
        n_total = summary.get("total_changes", 0)
        n_validated = summary.get("validated", 0)
        n_pending = summary.get("pending", n_total)
        table_status = table.get("table_status", "pending")

        # Status icon
        if table_status == "completed":
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif table_status == "partial":
            icon = html.I(className="bi bi-pie-chart-fill text-info me-2")
        else:
            icon = html.I(className="bi bi-circle text-warning me-2")

        # Active class
        active_class = (
            "bg-light border-start border-4 border-primary"
            if full_idx == current_idx
            else "border-bottom"
        )

        # Table display info
        section = _format_section(table.get("section", ""))
        table_name = (
            table.get("table_name")
            or table.get("table_title_raw")
            or table.get("table_id_t2")
            or table.get("table_id_t1")
            or "Tableau"
        )
        page_summary = _queue_page_summary_v2(table)

        # Change badges
        badge_children = []
        n_added = summary.get("indicators_added", 0)
        n_removed = summary.get("indicators_removed", 0)
        n_renamed = summary.get("indicators_renamed", 0)
        n_footnotes = summary.get("footnotes_changed", 0)

        if n_added:
            badge_children.append(
                dbc.Badge(f"+{n_added}", color="success", className="me-1")
            )
        if n_removed:
            badge_children.append(
                dbc.Badge(f"-{n_removed}", color="danger", className="me-1")
            )
        if n_renamed:
            badge_children.append(
                dbc.Badge(
                    f"~{n_renamed}",
                    color="warning",
                    text_color="dark",
                    className="me-1",
                )
            )
        if n_footnotes:
            badge_children.append(
                dbc.Badge(f"FN {n_footnotes}", color="info", className="me-1")
            )

        # Priority badges (relevance, risk)
        relevance = table.get("relevance", "")
        risk_level = table.get("risk_level", "")
        if relevance:
            badge_children.append(
                dbc.Badge(
                    _RELEVANCE_DISPLAY.get(relevance.upper(), relevance),
                    color=_RELEVANCE_COLORS.get(relevance.upper(), "secondary"),
                    className="me-1",
                )
            )
        if risk_level:
            badge_children.append(
                dbc.Badge(
                    f"Risque {_RISK_DISPLAY.get(risk_level.upper(), risk_level)}",
                    color=_RISK_COLORS.get(risk_level.upper(), "secondary"),
                    className="me-1",
                )
            )

        # Progress badge
        progress_badge = (
            dbc.Badge(
                f"{n_validated}/{n_total}",
                color="success"
                if table_status == "completed"
                else ("primary" if table_status == "partial" else "secondary"),
                className="ms-auto",
                pill=True,
            )
            if n_total > 0
            else html.Span()
        )

        item_content = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                icon,
                                html.Span(
                                    table_name,
                                    className="fw-semibold small",
                                    title=table_name,
                                    style={
                                        "whiteSpace": "normal",
                                        "overflowWrap": "anywhere",
                                        "flex": "1",
                                    },
                                ),
                                progress_badge,
                            ],
                            className="d-flex align-items-center",
                        ),
                        html.Small(
                            f"{section} - {page_summary}" if page_summary else section,
                            className="text-muted d-block ms-4",
                        ),
                        html.Div(
                            [
                                html.Small(
                                    f"{n_total} changement(s)",
                                    className="text-muted me-2",
                                ),
                                *badge_children,
                            ],
                            className="ms-4 mt-1",
                        ),
                    ],
                    className="w-100",
                )
            ],
            action=True,
            id={"type": "queue-table-item-v2", "index": full_idx},
            className=f"p-2 {active_class}",
            style={"cursor": "pointer"},
        )
        list_items.append(item_content)

    return html.Div(
        [
            html.H5(t("file_review"), className="mb-3"),
            html.Div(
                [
                    html.Span(f"Total: {total}", className="me-3 fw-bold"),
                    html.Span(
                        [
                            html.I(
                                className="bi bi-check-circle-fill text-success me-1"
                            ),
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
            dbc.ListGroup(
                list_items,
                flush=True,
                className="overflow-auto",
                style={"maxHeight": "calc(100vh - 400px)"},
            ),
        ],
        className="h-100",
    )
