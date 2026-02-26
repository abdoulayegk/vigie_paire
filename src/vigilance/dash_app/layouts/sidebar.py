"""Sidebar layout pour la configuration."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from app.ui_config import AVAILABLE_BANKS


def build_sidebar() -> dbc.Col:
    """Construire la sidebar de configuration."""
    bank_options = [{"label": f"{k.upper()} - {v}", "value": k} for k, v in AVAILABLE_BANKS.items()]
    
    # Generate years (current + previous 5)
    current_year = 2026
    year_options = [{"label": str(y), "value": str(y)} for y in range(current_year, current_year - 6, -1)]
    
    quarter_options = [
        {"label": "T1 (Q1)", "value": "Q1"},
        {"label": "T2 (Q2)", "value": "Q2"},
        {"label": "T3 (Q3)", "value": "Q3"},
        {"label": "T4 (Q4)", "value": "Q4"},
    ]

    return dbc.Col(
        [
            html.H4("Contexte d'Analyse", className="mb-3 text-primary"),
            html.Hr(),
            
            # 1. Analyst Info
            html.Label("Nom de l'Analyste", className="fw-bold small"),
            dbc.Input(id="analyst-name", placeholder="ex: Jean Dupont", className="mb-3"),
            
            # 2. Bank & Year
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Banque", className="fw-bold small"),
                            dcc.Dropdown(
                                id="bank-code",
                                options=bank_options,
                                value=bank_options[0]["value"] if bank_options else "bnc",
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
            
            # 3. Source Selection
            html.Label("Source des Données", className="fw-bold small mt-2"),
            dbc.RadioItems(
                id="data-source-type",
                options=[
                    {"label": "Upload PDF", "value": "upload"},
                    {"label": "Base de Données", "value": "db", "disabled": True},
                ],
                value="upload",
                className="mb-3 small",
            ),
            
            html.Hr(),
            
            # 4. Upload Zones (Dynamic based on logic, but for now static 3 slots)
            html.H5("Rapports PDF", className="mb-2 fs-6"),
            
            # T1 Upload
            html.Label("Rapport T1 (Référence)", className="small text-muted"),
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
            html.Div(id="upload-t1-name", className="small text-success mb-2 fst-italic"),

            # T2 Upload
            html.Label("Rapport T2 (Comparaison)", className="small text-muted"),
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
            html.Div(id="upload-t2-name", className="small text-success mb-2 fst-italic"),
            
            # T3 Upload (Optional)
            html.Label("Rapport T3 (Optionnel)", className="small text-muted"),
            dcc.Upload(
                id="upload-t3",
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
            html.Div(id="upload-t3-name", className="small text-success mb-3 fst-italic"),

            # 5. Quarter Selection for Comparison
            html.Label("Comparaison à effectuer", className="fw-bold small"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="quarter-1",
                            options=quarter_options,
                            value="Q1",
                            clearable=False,
                            className="small",
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        html.Div("VS", className="text-center pt-2 fw-bold small"),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="quarter-2",
                            options=quarter_options,
                            value="Q2",
                            clearable=False,
                            className="small",
                        ),
                        width=5,
                    ),
                ],
                className="mb-3 g-1",
            ),

            html.Hr(),
            
            # 6. Action Button
            dbc.Button(
                [html.I(className="bi bi-search me-2"), "Lancer l'Analyse"],
                id="btn-detect",
                color="primary",
                className="w-100 mb-3 shadow-sm",
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
                        id="option-visual-proofs",
                        options=[{"label": "Preuves Visuelles", "value": "proofs"}],
                        value=["proofs"],
                        switch=True,
                        className="small mb-1",
                    ),
                    dbc.Checklist(
                        id="option-vision",
                        options=[{"label": "Fallback Vision pour indicateurs", "value": "vision"}],
                        value=["vision"],
                        switch=True,
                        className="small mb-1",
                    ),
                    dbc.Checklist(
                        id="option-auto-indicator",
                        options=[{"label": "Auto-Comparaison", "value": "compare"}],
                        value=["compare"],
                        switch=True,
                        className="small mb-1",
                    ),
                    dbc.Checklist(
                        id="option-footnotes",
                        options=[{"label": "Notes de bas de tableau (footnotes)", "value": "footnotes"}],
                        value=[],
                        switch=True,
                        className="small mb-1",
                    ),
                    dbc.Checklist(
                        id="option-genai-classification",
                        options=[{"label": "Classifier les changements avec GenAI (GPT-4o)", "value": "classify"}],
                        value=[],
                        switch=True,
                        className="small mb-1",
                    ),
                ],
                id="collapse-options",
                is_open=False,
            ),
        ],
        md=2,
        className="bg-light border-end p-3",
        style={"minHeight": "100vh", "height": "100%", "overflowY": "auto"},
    )
