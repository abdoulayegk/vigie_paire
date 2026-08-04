"""Callbacks d'export et de telechargement Excel."""

from __future__ import annotations

import base64

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from vigie.interface.services.export_helpers import _resolve_export_review_items
from vigie.support.quarter_utils import quarter_label_from_payload
from vigie.interface.review_export import generate_validation_excel


def _filename_period(label: str) -> str:
    """Normalise un libelle de trimestre pour un nom de fichier."""
    return "_".join(str(label or "").strip().upper().split())


@callback(
    Output("results-export-tab", "children"),
    Input("store-review-items", "data"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_export_tab(review_items_data, indicator_result, show_results):
    """Rendre le panneau compact d'export Excel."""
    if not show_results:
        raise PreventUpdate
    if not review_items_data and not indicator_result:
        return html.Div("Aucun resultat a exporter.", className="text-muted")

    ir = indicator_result or {}
    if not review_items_data and not ir:
        return html.Div("Aucun resultat a exporter.", className="text-muted")

    content = [
        html.Div(
            [
                html.Div("Export", className="small text-uppercase text-muted fw-semibold mb-1"),
                html.H5("Revue expert Excel", className="mb-1"),
                html.P(
                    "Fichier structuré pour validation, suivi et archivage.",
                    className="text-muted small mb-0",
                ),
            ],
            className="mb-3",
        ),
        dbc.Button(
            [
                html.I(className="bi bi-file-earmark-excel me-2"),
                "Télécharger Excel",
            ],
            id="btn-download-review-excel",
            color="primary",
            className="w-100 py-2 fw-semibold",
        ),
    ]
    if not review_items_data and ir:
        content.append(
            html.P(
                "Aucun changement détecté à exporter pour cette comparaison.",
                className="small text-muted mt-3 mb-0",
            )
        )

    return html.Div(
        content,
        className="h-100 p-3 bg-white rounded border",
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
    q_from = _filename_period(quarter_label_from_payload(ir, "previous"))
    q_to = _filename_period(quarter_label_from_payload(ir, "current"))
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_to}_vs_{q_from}_{year_val}.xlsx"
    excel_bytes = generate_validation_excel(items, ir)
    b64 = base64.b64encode(excel_bytes).decode("ascii")
    return dict(content=b64, filename=filename, base64=True)
