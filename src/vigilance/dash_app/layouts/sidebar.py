"""Sidebar layout pour la configuration."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from app.i18n import t
from app.ui_config import AVAILABLE_BANKS


def build_sidebar() -> dbc.Col:
    """Construire la sidebar de configuration."""
    bank_options = [
        {"label": f"{k.upper()} - {v}", "value": k} for k, v in AVAILABLE_BANKS.items()
    ]

    # Generate years from the current year through 2050.
    current_year = 2025
    year_options = [
        {"label": str(y), "value": str(y)} for y in range(2050, 2020, -1)
    ]

    quarter_options = [
        {"label": "Q1", "value": "Q1"},
        {"label": "Q2", "value": "Q2"},
        {"label": "Q3", "value": "Q3"},
    ]

    return dbc.Col(
        [
            html.Div(
                [
                    html.H4(
                        "Contexte d'Analyse",
                        id="analysis-sidebar-title",
                        className="analysis-sidebar-title mb-0 text-primary",
                    ),
                    dbc.Button(
                        html.I(
                            id="analysis-sidebar-toggle-icon",
                            className="bi bi-layout-sidebar-inset",
                        ),
                        id="btn-toggle-sidebar",
                        color="light",
                        size="sm",
                        className="analysis-sidebar-toggle shadow-sm",
                        n_clicks=0,
                    ),
                ],
                className="d-flex align-items-center justify-content-between mb-3",
            ),
            html.Div(
                [
                    html.Hr(),
                    # 1. Analyst Info
                    html.Label("Nom de l'Analyste", className="fw-bold small"),
                    dbc.Input(
                        id="analyst-name",
                        placeholder="ex: Jean Dupont",
                        className="mb-3",
                    ),
                    # 2. Bank & Year
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Banque", className="fw-bold small"),
                                    dcc.Dropdown(
                                        id="bank-code",
                                        options=bank_options,
                                        value=bank_options[0]["value"]
                                        if bank_options
                                        else "bnc",
                                        clearable=False,
                                        className="small",
                                    ),
                                ],
                                width=7,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Année", className="fw-bold small"),
                                    dcc.Dropdown(
                                        id="analysis-year",
                                        options=year_options,
                                        value=str(current_year),
                                        clearable=False,
                                        className="small",
                                    ),
                                ],
                                width=5,
                            ),
                        ],
                        className="mb-3 g-2",
                    ),
                    # 3. Quarter Selection for Comparison (Global)
                    html.Label(
                        "Trimestre courant sélectionné", className="fw-bold small"
                    ),
                    dcc.Dropdown(
                        id="current-quarter",
                        options=quarter_options,
                        value="Q2",
                        clearable=False,
                        className="small mb-2",
                    ),
                    html.Label(
                        "Trimestre précédent comparé", className="fw-bold small"
                    ),
                    html.Div(
                        id="previous-quarter-display",
                        className="small border rounded bg-light px-2 py-2 mb-2 text-muted",
                    ),
                    html.Div(
                        id="comparison-pair-display",
                        className="small text-muted mb-3",
                    ),
                    html.Hr(),
                    # 4. Source Selection
                    html.Label("Source des Données", className="fw-bold small mt-2"),
                    dbc.RadioItems(
                        id="data-source-type",
                        options=[
                            {"label": "Upload PDF", "value": "upload"},
                            {"label": "Base de Données", "value": "db"},
                        ],
                        value="upload",
                        className="mb-3 small",
                    ),
                    # 4b. DB Run Selector (hidden by default)
                    html.Div(
                        [
                            html.Label(
                                "Comparaison disponible",
                                className="fw-bold small",
                            ),
                            dcc.Dropdown(
                                id="db-run-selector",
                                options=[],
                                placeholder="Sélectionner un run...",
                                className="small mb-2",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="bi bi-database me-2"),
                                    "Charger depuis DB",
                                ],
                                id="btn-load-db",
                                color="success",
                                className="w-100 mb-3 shadow-sm",
                            ),
                        ],
                        id="db-source-container",
                        style={"display": "none"},
                    ),
                    html.Hr(),
                    # 4. Upload Zones (togglable)
                    html.Div(
                        [
                            html.H5("Rapports PDF", className="mb-2 fs-6"),
                            html.Label(
                                "Rapport trimestre courant",
                                id="upload-current-label",
                                className="small text-muted",
                            ),
                            dcc.Upload(
                                id="upload-t2",
                                children=html.Div(
                                    [
                                        html.I(className="bi bi-file-earmark-pdf me-2"),
                                        html.Span("Glisser ou cliquer", className="small"),
                                    ],
                                    className="d-flex align-items-center justify-content-center",
                                ),
                                style={
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "padding": "10px",
                                    "borderColor": "#dee2e6",
                                },
                                multiple=False,
                                className="mb-1",
                            ),
                            html.Div(
                                id="upload-t2-name",
                                className="small text-success mb-2 fst-italic",
                            ),
                            html.Label(
                                "Rapport trimestre precedent",
                                id="upload-previous-label",
                                className="small text-muted",
                            ),
                            dcc.Upload(
                                id="upload-t1",
                                children=html.Div(
                                    [
                                        html.I(className="bi bi-file-earmark-pdf me-2"),
                                        html.Span("Glisser ou cliquer", className="small"),
                                    ],
                                    className="d-flex align-items-center justify-content-center",
                                ),
                                style={
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "padding": "10px",
                                    "borderColor": "#dee2e6",
                                },
                                multiple=False,
                                className="mb-1",
                            ),
                            html.Div(
                                id="upload-t1-name",
                                className="small text-success mb-2 fst-italic",
                            ),
                            html.Hr(),
                            # 6. Action Button
                            dbc.Button(
                                [html.I(className="bi bi-search me-2"), "Lancer l'Analyse"],
                                id="btn-detect",
                                color="primary",
                                className="w-100 mb-3 shadow-sm",
                            ),
                        ],
                        id="upload-source-container",
                    ),
                    # 7. Options (Collapsed)
                    dbc.Button(
                        "Options Avancées",
                        id="btn-toggle-options",
                        color="link",
                        size="sm",
                        className="p-0 mb-2 text-decoration-none",
                    ),
                    dbc.Collapse(
                        [
                            dbc.Checklist(
                                id="option-footnotes",
                                options=[
                                    {
                                        "label": "Notes de bas de tableau (footnotes)",
                                        "value": "footnotes",
                                    }
                                ],
                                value=["footnotes"],
                                switch=True,
                                className="small mb-1",
                            ),
                            dbc.Checklist(
                                id="option-genai-classification",
                                options=[
                                    {
                                        "label": "Classifier les changements avec GenAI (GPT-4o)",
                                        "value": "classify",
                                    }
                                ],
                                value=["classify"],
                                switch=True,
                                className="small mb-1",
                            ),
                            dbc.Checklist(
                                id="option-force-reextract",
                                options=[
                                    {
                                        "label": t("option_force_reextract"),
                                        "value": "reextract",
                                    }
                                ],
                                value=[],
                                switch=True,
                                className="small mb-1",
                            ),
                        ],
                        id="collapse-options",
                        is_open=False,
                    ),
                ],
                id="analysis-sidebar-body",
                className="analysis-sidebar-body",
            ),
        ],
        id="analysis-sidebar",
        className="analysis-sidebar bg-light border-end p-3",
        style={"minHeight": "100vh", "height": "100%", "overflowY": "auto"},
    )
