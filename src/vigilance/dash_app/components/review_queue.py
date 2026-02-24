"""Review Queue Component."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from app.i18n import t
from app.review_models import (
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
)


def build_review_queue(
    items: list[dict],
    current_idx: int,
    active_filters: dict | None = None,
) -> html.Div:
    """Build the left-side review queue panel with dynamic section filters."""

    all_sections = sorted(set(item.get("section", "Autre") for item in items))

    active_section = (active_filters or {}).get("section", "all")
    active_status = (active_filters or {}).get("status", "all")

    filtered_items = items
    if active_section and active_section != "all":
        filtered_items = [i for i in filtered_items if i.get("section") == active_section]
    if active_status and active_status != "all":
        filtered_items = [i for i in filtered_items if i.get("review_status") == active_status]

    total = len(filtered_items)
    approved = sum(1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_APPROVED)
    rejected = sum(1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_REJECTED)
    pending = sum(1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_PENDING)

    filter_buttons: list = [
        dbc.Button(
            [html.I(className="bi bi-funnel me-2"), f"{t('all_sections')} ({len(items)})"],
            id={"type": "filter-section", "value": "all"},
            color="primary" if active_section == "all" else "light",
            size="sm",
            className="w-100 text-start mb-1",
        )
    ]
    for section in all_sections:
        section_count = sum(1 for i in items if i.get("section") == section)
        filter_buttons.append(
            dbc.Button(
                [html.I(className="bi bi-folder me-2"), f"{section} ({section_count})"],
                id={"type": "filter-section", "value": section},
                color="primary" if active_section == section else "light",
                size="sm",
                className="w-100 text-start mb-1",
            )
        )

    filter_bar = html.Div(filter_buttons, className="mb-3 p-2 bg-white rounded border")

    list_items = []
    for idx, item in enumerate(filtered_items):
        status = item.get("review_status", REVIEW_STATUS_PENDING)

        if status == REVIEW_STATUS_APPROVED:
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif status == REVIEW_STATUS_REJECTED:
            icon = html.I(className="bi bi-x-circle-fill text-danger me-2")
        else:
            icon = html.I(className="bi bi-circle text-warning me-2")

        active_class = "bg-light border-start border-4 border-primary" if idx == current_idx else "border-bottom"

        table_name = item.get("table_name", "Unknown Table")
        section = item.get("section", "Unknown Section")
        indicators = item.get("indicators", [])
        n_indicators = len(indicators)
        change_type = item.get("change_type", "")

        if len(table_name) > 45:
            table_name = table_name[:42] + "..."

        n_added = sum(1 for ind in indicators if ind.get("type") == "added")
        n_removed = sum(1 for ind in indicators if ind.get("type") == "removed")
        n_renamed = sum(1 for ind in indicators if ind.get("type") == "renamed")

        badge_children = []
        if change_type == CHANGE_TYPE_TABLE_ADDED:
            badge_children.append(dbc.Badge(t("table_added").upper(), color="info", className="me-1"))
        elif change_type == CHANGE_TYPE_TABLE_REMOVED:
            badge_children.append(dbc.Badge(t("table_removed").upper(), color="dark", text_color="white", className="me-1"))
        else:
            if n_added:
                badge_children.append(dbc.Badge(f"+{n_added}", color="success", className="me-1"))
            if n_removed:
                badge_children.append(dbc.Badge(f"-{n_removed}", color="danger", className="me-1"))
            if n_renamed:
                badge_children.append(dbc.Badge(f"~{n_renamed}", color="warning", text_color="dark", className="me-1"))

        item_content = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.Div(
                            [icon, html.Span(table_name, className="fw-semibold small")],
                            className="d-flex align-items-center",
                        ),
                        html.Small(section, className="text-muted d-block ms-4"),
                        html.Div(
                            [
                                html.Small(
                                    f"{n_indicators} indicateur(s)" if n_indicators > 0 else "Tableau entier",
                                    className="text-muted me-2"
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
            id={"type": "review-item", "index": idx},
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
                    html.Span([html.I(className="bi bi-circle-fill text-success me-1"), f"{approved}"], className="me-3 small"),
                    html.Span([html.I(className="bi bi-circle-fill text-warning me-1"), f"{pending}"], className="me-3 small"),
                    html.Span([html.I(className="bi bi-circle-fill text-danger me-1"), f"{rejected}"], className="small"),
                ],
                className="mb-3 small text-muted",
            ),
            filter_bar,
            dbc.ListGroup(list_items, flush=True, className="overflow-auto", style={"maxHeight": "calc(100vh - 400px)"}),
        ],
        className="h-100",
    )
