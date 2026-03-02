"""Page resultats de comparaison - Modern Design."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from app.i18n import t


def build_analyst_kpi_card(
    title: str, value: str | int, delta_icon: str | None = None, color: str = "light"
) -> dbc.Card:
    """Build a single KPI card for the analyst panel."""
    body_children = [
        html.P(title, className="text-muted mb-1 small"),
        html.H3(str(value), className="mb-0 fw-bold"),
    ]
    if delta_icon:
        body_children.append(html.Span(delta_icon, className="text-muted small"))
    return dbc.Card(
        dbc.CardBody(body_children, className="text-center py-3"),
        className=f"shadow-sm border-0 bg-{color}",
    )


def build_section_accordion_item(
    section_name: str,
    tables_with_changes: list,
    tables_added: list,
    tables_removed: list,
    item_id: str,
) -> dbc.AccordionItem:
    """Build one accordion item for the section-based changes tab."""
    parts = []
    if tables_with_changes:
        n = len(tables_with_changes)
        items = []
        for comp in tables_with_changes[:10]:
            title = comp.get("table_title") or comp.get("title_t1") or "Sans titre"
            fn_c = comp.get("footnotes_counts", {})
            fn_total = sum(fn_c.get(k, 0) for k in ("added", "removed", "modified"))
            badges = []
            if comp.get("table_status") == "structure_change":
                badges.append(
                    dbc.Badge(t("fusion_split"), color="warning", className="ms-2")
                )
            if comp.get("table_status") == "incertain":
                badges.append(
                    dbc.Badge("INCERTAIN", color="secondary", className="ms-2")
                )
            if fn_total:
                badges.append(
                    dbc.Badge(
                        f"FN +{fn_c.get('added', 0)}/-{fn_c.get('removed', 0)}/~{fn_c.get('modified', 0)}",
                        color="info",
                        className="ms-2",
                    )
                )
            ga = comp.get("genai_analysis", {})
            ga_rel = ga.get("relevance", "")
            if ga_rel:
                _rel_display = {
                    "REGLEMENTAIRE": "Reglementaire",
                    "NON_SIGNIFICATIF": "Non significatif",
                    "STRUCTUREL": "Structurel",
                    "NOUVELLE_DIVULGATION": "Nouvelle divulgation",
                    "NON_CLASSIFIE": "Non classifie",
                }
                _rel_colors = {
                    "REGLEMENTAIRE": "danger",
                    "NON_SIGNIFICATIF": "secondary",
                    "STRUCTUREL": "primary",
                    "NOUVELLE_DIVULGATION": "info",
                    "NON_CLASSIFIE": "light",
                }
                badges.append(
                    dbc.Badge(
                        _rel_display.get(ga_rel, ga_rel),
                        color=_rel_colors.get(ga_rel, "secondary"),
                        className="ms-2",
                    )
                )
            items.append(html.Li([title, *badges], className="mb-0 small text-muted"))
        parts.append(
            html.Div(
                [
                    html.Strong(f"{n} tableau(x) avec changements"),
                    html.Ul(items, className="mb-0"),
                ]
                if n <= 10
                else [html.Strong(f"{n} tableau(x) avec changements")],
                className="mb-2",
            )
        )
    if tables_added:
        parts.append(
            html.Div(
                [
                    html.Strong(f"{t('table_added_plural')}: {len(tables_added)}"),
                    html.Ul(
                        [
                            html.Li(
                                tbl.get("table_title")
                                or tbl.get("title")
                                or "Sans titre"
                            )
                            for tbl in tables_added[:5]
                        ],
                        className="mb-0 small text-muted",
                    ),
                ]
                if len(tables_added) <= 5
                else [html.Strong(f"{t('table_added_plural')}: {len(tables_added)}")],
                className="mb-2",
            )
        )
    if tables_removed:
        parts.append(
            html.Div(
                [
                    html.Strong(f"{t('table_removed_plural')}: {len(tables_removed)}"),
                    html.Ul(
                        [
                            html.Li(
                                tbl.get("table_title")
                                or tbl.get("title")
                                or "Sans titre"
                            )
                            for tbl in tables_removed[:5]
                        ],
                        className="mb-0 small text-muted",
                    ),
                ]
                if len(tables_removed) <= 5
                else [
                    html.Strong(f"{t('table_removed_plural')}: {len(tables_removed)}")
                ],
                className="mb-2",
            )
        )
    body = (
        html.Div(parts, className="p-2")
        if parts
        else html.Div("Aucun detail.", className="text-muted p-2")
    )
    return dbc.AccordionItem(
        body,
        title=section_name,
        id=item_id,
    )


def build_page_results() -> html.Div:
    """Contenu pour les resultats avec design moderne et onglets par section."""
    return html.Div(
        id="results-content",
        children=[
            # Header Section
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H2(t("analyse_comparative"), className="mb-1"),
                                    html.Div(
                                        id="results-header", className="text-muted mb-3"
                                    ),
                                    html.Div(
                                        id="results-executive-summary", className="mb-3"
                                    ),
                                    html.Div(id="results-kpis"),
                                ],
                                width=9,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-arrow-clockwise me-2"),
                                        t("nouvelle_analyse"),
                                    ],
                                    id="btn-reset",
                                    color="outline-secondary",
                                    size="sm",
                                    className="float-end mt-2",
                                ),
                                width=3,
                            ),
                        ]
                    )
                ],
                className="mb-4",
            ),
            # KPI Row (File de Revue summary)
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                build_analyst_kpi_card(
                                    t("file_review_total"), "0", color="white"
                                ),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-total",
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(
                                    t("validated"), "0", color="white"
                                ),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-approved",
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(
                                    t("rejected"), "0", color="white"
                                ),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-rejected",
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(
                                    t("pending"), "0", color="white"
                                ),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-pending",
                            ),
                        ],
                        className="g-3 mb-4",
                    ),
                ],
            ),
            # Main Split Pane (Review Dashboard)
            html.Div(
                dbc.Row(
                    [
                        # Left Panel: Review Queue
                        dbc.Col(
                            html.Div(
                                id="review-queue-container",
                                className="bg-white p-3 shadow-sm rounded h-100",
                                style={"overflowY": "auto"},
                            ),
                            md=4,
                            className="h-100",
                        ),
                        # Right Panel: Detail View + Nav (buttons in layout so callbacks fire)
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(id="review-proof-container", className="mb-3"),
                                        html.Div(id="review-meta-container"),
                                    ],
                                    id="review-detail-container",
                                    className="bg-white p-4 shadow-sm rounded",
                                    style={"overflowY": "auto", "minHeight": "400px"},
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            [
                                                html.I(className="bi bi-chevron-left"),
                                                f" {t('btn_prev', 'Precedent')}",
                                            ],
                                            id="btn-prev",
                                            color="light",
                                            size="sm",
                                            className="me-2",
                                        ),
                                        dbc.Button(
                                            [
                                                f"{t('btn_next', 'Suivant')} ",
                                                html.I(className="bi bi-chevron-right"),
                                            ],
                                            id="btn-next",
                                            color="light",
                                            size="sm",
                                        ),
                                    ],
                                    className="d-flex justify-content-end mt-3 pt-3 border-top",
                                ),
                            ],
                            md=8,
                            className="h-100 d-flex flex-column",
                        ),
                    ],
                    className="g-4 h-100",
                    style={"minHeight": "600px", "height": "calc(100vh - 280px)"},
                ),
                className="mb-5",
            ),
            # Footer: Validation Client Statistics
            html.Div(
                [
                    html.H5(t("statistiques_validation"), className="mb-3"),
                    dbc.Progress(
                        [
                            dbc.Progress(
                                value=0,
                                color="success",
                                bar=True,
                                id="progress-approved",
                            ),
                            dbc.Progress(
                                value=0,
                                color="danger",
                                bar=True,
                                id="progress-rejected",
                            ),
                            dbc.Progress(
                                value=100,
                                color="warning",
                                bar=True,
                                id="progress-pending",
                            ),
                        ],
                        className="mb-3",
                        style={"height": "20px"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="stats-validation-time",
                                    className="text-muted small",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(
                                    id="stats-export-status",
                                    className="text-end text-muted small",
                                ),
                                width=6,
                            ),
                        ]
                    ),
                    html.Hr(className="my-4"),
                    html.Div(id="results-export-tab"),
                ],
                className="p-4 bg-light rounded shadow-sm",
            ),
            # Hidden div for storing review state
            dcc.Store(id="store-review-data", data=[]),
            dcc.Store(id="store-current-review-index", data=0),
            # Legacy tabs container (hidden or repurposed if needed later, but keeping structure clean)
            html.Div(id="results-sections-tab", style={"display": "none"}),
            html.Div(id="results-review-tab", style={"display": "none"}),
            html.Div(id="results-table-tab", style={"display": "none"}),
            # Nav debug panel (instrumentation: triggered_id, current_idx, last writer)
            html.Div(
                id="nav-debug-panel",
                className="mt-2 p-2 small font-monospace bg-dark text-light rounded",
                style={"maxHeight": "180px", "overflowY": "auto"},
            ),
        ],
        className="p-4",
    )
