"""Export and download callbacks: Excel, TXT, export tab rendering."""

from __future__ import annotations

import base64

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from vigilance.dash_app.services.export_helpers import _resolve_export_review_items
from vigilance.quarter_utils import quarter_label_from_payload
from vigilance.review_export import (
    generate_validation_excel,
    generate_validation_txt,
)


@callback(
    Output("results-export-tab", "children"),
    Input("store-review-items", "data"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_export_tab(review_items_data, indicator_result, show_results):
    """Rendre l'onglet Export avec boutons de telechargement."""
    if not show_results:
        raise PreventUpdate
    if not review_items_data and not indicator_result:
        return html.Div("Aucun resultat a exporter.", className="text-muted")

    ir = indicator_result or {}
    if not review_items_data and not ir:
        return html.Div("Aucun resultat a exporter.", className="text-muted")

    card_body = [
        html.Div(
            [
                html.H5("Exporter la revue", className="mb-1"),
                html.P(
                    "Téléchargez la revue expert au format Excel ou TXT.",
                    className="text-muted mb-0",
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Télécharger le fichier Excel expert",
                        id="btn-download-review-excel",
                        color="primary",
                        className="w-100 py-3 fw-semibold",
                    ),
                    xs=12,
                    md=6,
                ),
                dbc.Col(
                    dbc.Button(
                        "Télécharger le rapport TXT",
                        id="btn-download-review-txt",
                        color="secondary",
                        outline=True,
                        className="w-100 py-3 fw-semibold",
                    ),
                    xs=12,
                    md=6,
                    className="mt-2 mt-md-0",
                ),
            ],
            className="g-3 align-items-stretch",
        ),
    ]
    if not review_items_data and ir:
        card_body.append(
            html.P(
                "Aucun changement détecté à exporter pour cette comparaison.",
                className="small text-muted mt-4 mb-0",
            )
        )

    return dbc.Card(
        dbc.CardBody(card_body, className="p-4 p-lg-5"),
        className="shadow-sm border-0 bg-white",
    )


@callback(
    Output("download-review-excel", "data"),
    Input("btn-download-review-excel", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-queue", "data"),
    State("store-indicator-result", "data"),
    State("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def on_download_excel(
    n_clicks, review_items_data, review_queue_data, indicator_result, paths
):
    """Telecharger le fichier Excel de validation (.xlsx)."""
    if not n_clicks:
        raise PreventUpdate
    ir = indicator_result or {}
    items = _resolve_export_review_items(
        review_items_data, review_queue_data, indicator_result, paths
    )
    bank = str(ir.get("bank_code", "bank")).upper()
    q_from = quarter_label_from_payload(ir, "previous").upper()
    q_to = quarter_label_from_payload(ir, "current").upper()
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_to}_vs_{q_from}_{year_val}.xlsx"
    excel_bytes = generate_validation_excel(items, ir)
    b64 = base64.b64encode(excel_bytes).decode("ascii")
    return dict(content=b64, filename=filename, base64=True)


@callback(
    Output("download-review-txt", "data"),
    Input("btn-download-review-txt", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-queue", "data"),
    State("store-indicator-result", "data"),
    State("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def on_download_txt(
    n_clicks, review_items_data, review_queue_data, indicator_result, paths
):
    """Telecharger le rapport TXT de revue expert."""
    if not n_clicks:
        raise PreventUpdate
    ir = indicator_result or {}
    items = _resolve_export_review_items(
        review_items_data, review_queue_data, indicator_result, paths
    )
    bank = str(ir.get("bank_code", "bank")).upper()
    q_from = quarter_label_from_payload(ir, "previous").upper()
    q_to = quarter_label_from_payload(ir, "current").upper()
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_to}_vs_{q_from}_{year_val}.txt"
    txt_content = generate_validation_txt(items, ir)
    return dict(content=txt_content, filename=filename)
