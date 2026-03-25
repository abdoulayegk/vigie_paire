"""Review Queue Component."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from app.i18n import t
from app.review_priority import get_priority_signals
from app.review_models import (
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
)
from vigilance.dash_app.components.review_detail import (
    section_display_label,
    table_display_label,
)

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


def _queue_page_summary(item: dict) -> str:
    """Return concise page context string for queue items."""
    change_type = (item.get("change_type") or "").strip()
    page_t1 = item.get("page_t1")
    page_t2 = item.get("page_t2")

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        return f"p.{page_t2}" if page_t2 is not None else ""
    if change_type == CHANGE_TYPE_TABLE_REMOVED:
        return f"p.{page_t1}" if page_t1 is not None else ""
    if page_t1 is not None and page_t2 is not None:
        return f"Préc. p.{page_t1} / Cour. p.{page_t2}"
    if page_t2 is not None:
        return f"p.{page_t2}"
    if page_t1 is not None:
        return f"p.{page_t1}"
    return ""


def build_review_queue(
    items: list[dict],
    current_idx: int,
    active_filters: dict | None = None,
) -> html.Div:
    """Build the left-side review queue panel with dynamic section filters."""

    all_sections = sorted(set(item.get("section", "Autre") for item in items))

    active_section = (active_filters or {}).get("section", "all")
    active_status = (active_filters or {}).get("status", "all")

    def _matches_filters(item: dict) -> bool:
        if active_section and active_section != "all":
            if item.get("section") != active_section:
                return False
        if active_status and active_status != "all":
            if item.get("review_status") != active_status:
                return False
        return True

    filtered_with_full_idx = [
        (idx, item) for idx, item in enumerate(items) if _matches_filters(item)
    ]
    filtered_items = [it for _, it in filtered_with_full_idx]

    total = len(filtered_items)
    approved = sum(
        1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_APPROVED
    )
    rejected = sum(
        1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_REJECTED
    )
    pending = sum(
        1 for i in filtered_items if i.get("review_status") == REVIEW_STATUS_PENDING
    )

    filter_buttons: list = [
        dbc.Button(
            [
                html.I(className="bi bi-funnel me-2"),
                f"{t('all_sections')} ({len(items)})",
            ],
            id={"type": "filter-section", "value": "all"},
            color="primary" if active_section == "all" else "light",
            size="sm",
            className="w-100 text-start mb-1",
        )
    ]
    for section in all_sections:
        section_count = sum(1 for i in items if i.get("section") == section)
        section_label = section_display_label(section)
        filter_buttons.append(
            dbc.Button(
                [
                    html.I(className="bi bi-folder me-2"),
                    f"{section_label} ({section_count})",
                ],
                id={"type": "filter-section", "value": section},
                color="primary" if active_section == section else "light",
                size="sm",
                className="w-100 text-start mb-1",
            )
        )

    filter_bar = html.Div(filter_buttons, className="mb-3 p-2 bg-white rounded border")

    list_items = []
    for full_idx, item in filtered_with_full_idx:
        indicators = item.get("indicators", [])
        n_indicators = len(indicators)
        n_decided = sum(
            1
            for ind in indicators
            if ind.get("review_status", REVIEW_STATUS_PENDING)
            in (REVIEW_STATUS_APPROVED, REVIEW_STATUS_REJECTED)
        )
        all_decided = n_decided == n_indicators and n_indicators > 0
        n_rejected = sum(
            1
            for ind in indicators
            if ind.get("review_status") == REVIEW_STATUS_REJECTED
        )

        if all_decided and n_rejected == 0:
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif all_decided and n_rejected == n_indicators:
            icon = html.I(className="bi bi-x-circle-fill text-danger me-2")
        elif all_decided:
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif n_decided > 0:
            icon = html.I(className="bi bi-pie-chart-fill text-info me-2")
        else:
            status = item.get("review_status", REVIEW_STATUS_PENDING)
            if status == REVIEW_STATUS_APPROVED:
                icon = html.I(className="bi bi-check-circle-fill text-success me-2")
            elif status == REVIEW_STATUS_REJECTED:
                icon = html.I(className="bi bi-x-circle-fill text-danger me-2")
            else:
                icon = html.I(className="bi bi-circle text-warning me-2")

        active_class = (
            "bg-light border-start border-4 border-primary"
            if full_idx == current_idx
            else "border-bottom"
        )

        section = section_display_label(item.get("section", "Unknown Section"))
        try:
            table_display = table_display_label(item)
        except Exception:
            table_display = (
                item.get("table_name")
                or item.get("table_title_raw")
                or item.get("table_id_t2")
                or item.get("table_id_t1")
                or "Tableau"
            )
        indicators = item.get("indicators", [])
        n_indicators = len(indicators)
        page_summary = _queue_page_summary(item)

        n_added = sum(1 for ind in indicators if ind.get("type") == "added")
        n_removed = sum(1 for ind in indicators if ind.get("type") == "removed")
        n_renamed = sum(1 for ind in indicators if ind.get("type") == "renamed")

        item_type = item.get("item_type", "indicator")
        change_type = item.get("change_type", "")
        badge_children = []
        if item_type == "footnote":
            n_fn = len(indicators)
            badge_children.append(
                dbc.Badge(f"FN {n_fn}", color="info", className="me-1")
            )
        elif change_type == CHANGE_TYPE_TABLE_ADDED:
            badge_children.append(
                dbc.Badge(t("table_added").upper(), color="info", className="me-1")
            )
        elif change_type == CHANGE_TYPE_TABLE_REMOVED:
            badge_children.append(
                dbc.Badge(
                    t("table_removed").upper(),
                    color="dark",
                    text_color="white",
                    className="me-1",
                )
            )
        else:
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

        relevance, risk, conf_f = get_priority_signals(item)
        if relevance:
            badge_children.append(
                dbc.Badge(
                    _RELEVANCE_DISPLAY.get(relevance, relevance),
                    color=_RELEVANCE_COLORS.get(relevance, "secondary"),
                    className="me-1",
                )
            )
        if risk:
            badge_children.append(
                dbc.Badge(
                    f"Risque {_RISK_DISPLAY.get(risk, risk)}",
                    color=_RISK_COLORS.get(risk, "secondary"),
                    className="me-1",
                )
            )
        if conf_f >= 0:
            badge_children.append(
                dbc.Badge(
                    f"Conf {round(conf_f * 100):.0f}%",
                    color="light",
                    text_color="dark",
                    className="me-1",
                )
            )

        progress_badge = (
            dbc.Badge(
                f"{n_decided}/{n_indicators}",
                color="primary" if all_decided else "secondary",
                className="ms-auto",
                pill=True,
            )
            if n_indicators > 0
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
                                    table_display,
                                    className="fw-semibold small",
                                    title=table_display,
                                    style={
                                        "whiteSpace": "normal",
                                        "overflowWrap": "anywhere",
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
                                    f"{n_indicators} note(s)"
                                    if item_type == "footnote"
                                    else (
                                        f"{n_indicators} indicateur(s)"
                                        if n_indicators > 0
                                        else "Tableau entier"
                                    ),
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
            id={"type": "review-item", "index": full_idx},
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
                            html.I(className="bi bi-circle-fill text-success me-1"),
                            f"{approved}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-circle-fill text-warning me-1"),
                            f"{pending}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-circle-fill text-danger me-1"),
                            f"{rejected}",
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
