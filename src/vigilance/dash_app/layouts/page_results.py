"""Page resultats de comparaison - Modern Design."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from app.i18n import t
from app.dash_app.components.review_queue import build_review_queue
from app.dash_app.components.review_detail import build_review_detail


def build_analyst_kpi_card(title: str, value: str | int, delta_icon: str | None = None, color: str = "light") -> dbc.Card:
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
            if comp.get("table_status") == "structure_change":
                items.append(
                    html.Li(
                        [title, dbc.Badge(t("fusion_split"), color="warning", className="ms-2")],
                        className="mb-0 small text-muted",
                    )
                )
            else:
                items.append(html.Li(title, className="mb-0 small text-muted"))
        parts.append(
            html.Div(
                [html.Strong(f"{n} tableau(x) avec changements"), html.Ul(items, className="mb-0")]
                if n <= 10
                else [html.Strong(f"{n} tableau(x) avec changements")],
                className="mb-2",
            )
        )
    if tables_added:
        parts.append(
            html.Div(
                [html.Strong(f"{t('table_added_plural')}: {len(tables_added)}"), html.Ul(
                    [html.Li(tbl.get("table_title") or tbl.get("title") or "Sans titre") for tbl in tables_added[:5]],
                    className="mb-0 small text-muted"
                )] if len(tables_added) <= 5 else [html.Strong(f"{t('table_added_plural')}: {len(tables_added)}")],
                className="mb-2",
            )
        )
    if tables_removed:
        parts.append(
            html.Div(
                [html.Strong(f"{t('table_removed_plural')}: {len(tables_removed)}"), html.Ul(
                    [html.Li(tbl.get("table_title") or tbl.get("title") or "Sans titre") for tbl in tables_removed[:5]],
                    className="mb-0 small text-muted"
                )] if len(tables_removed) <= 5 else [html.Strong(f"{t('table_removed_plural')}: {len(tables_removed)}")],
                className="mb-2",
            )
        )
    body = html.Div(parts, className="p-2") if parts else html.Div("Aucun detail.", className="text-muted p-2")
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
                                    html.Div(id="results-header", className="text-muted mb-3"),
                                    html.Div(id="results-executive-summary", className="mb-3"),
                                    html.Div(id="results-kpis"),
                                ],
                                width=9
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-clockwise me-2"), t("nouvelle_analyse")],
                                    id="btn-reset",
                                    color="outline-secondary",
                                    size="sm",
                                    className="float-end mt-2"
                                ),
                                width=3
                            )
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
                                build_analyst_kpi_card(t("file_review_total"), "0", color="white"),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-total"
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(t("validated"), "0", color="white"),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-approved"
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(t("rejected"), "0", color="white"),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-rejected"
                            ),
                            dbc.Col(
                                build_analyst_kpi_card(t("pending"), "0", color="white"),
                                width=3,
                                className="mb-3",
                                id="kpi-queue-pending"
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
                            html.Div(id="review-queue-container", className="bg-white p-3 shadow-sm rounded h-100", style={"overflowY": "auto"}),
                            md=4,
                            className="h-100"
                        ),
                        # Right Panel: Detail View
                        dbc.Col(
                            html.Div(id="review-detail-container", className="bg-white p-4 shadow-sm rounded h-100", style={"overflowY": "auto"}),
                            md=8,
                            className="h-100"
                        ),
                    ],
                    className="g-4 h-100",
                    style={"minHeight": "600px", "height": "calc(100vh - 280px)"}
                ),
                className="mb-5"
            ),

            # Footer: Validation Client Statistics
            html.Div(
                [
                    html.H5(t("statistiques_validation"), className="mb-3"),
                     dbc.Progress(
                        [
                            dbc.Progress(value=0, color="success", bar=True, id="progress-approved"),
                            dbc.Progress(value=0, color="danger", bar=True, id="progress-rejected"),
                            dbc.Progress(value=100, color="warning", bar=True, id="progress-pending"),
                        ],
                        className="mb-3",
                        style={"height": "20px"}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Div(id="stats-validation-time", className="text-muted small"), width=6),
                            dbc.Col(html.Div(id="stats-export-status", className="text-end text-muted small"), width=6),
                        ]
                    ),
                    html.Hr(className="my-4"),
                    html.Div(id="results-export-tab"),
                ],
                className="p-4 bg-light rounded shadow-sm"
            ),
            
            # Hidden div for storing review state
            dcc.Store(id="store-review-data", data=[]),
            dcc.Store(id="store-current-review-index", data=0),
            
            # Legacy tabs container (hidden or repurposed if needed later, but keeping structure clean)
            html.Div(id="results-sections-tab", style={"display": "none"}),
            html.Div(id="results-review-tab", style={"display": "none"}),
            html.Div(id="results-table-tab", style={"display": "none"}),
        ],
        className="p-4",
    )
