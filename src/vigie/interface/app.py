"""Application Dash - Comparateur de Rapports Bancaires.

Pour lancer:
    uv run python -m vigie.interface.app
    ou: uv run dash run vigie.interface.app --port 8050
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import dash_bootstrap_components as dbc
from dash import (
    Dash,
    Input,
    Output,
    State,
    clientside_callback,
    dcc,
    html,
)

from vigie.interface.layouts import (
    build_page_upload,
    build_sidebar,
)
from vigie.support.quarter_utils import build_quarter_context

# Theme Bootstrap
APP_THEME = dbc.themes.FLATLY

app = Dash(
    __name__,
    external_stylesheets=[APP_THEME, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder="assets",
)

# Stores pour l'etat
stores = [
    dcc.Store(id="store-upload-t1", data=None),
    dcc.Store(id="store-upload-t2", data=None),
    dcc.Store(
        id="store-quarter-context",
        data=build_quarter_context("T2", year=2025),
    ),
    dcc.Store(id="store-pdf-paths", data=None),
    dcc.Store(id="store-temp-dir", data=None),
    dcc.Store(id="store-detection", data=None),
    dcc.Store(id="store-adjusted-sections", data=None),
    dcc.Store(id="store-sections-validated", data=False),
    dcc.Store(id="store-comparison-result", data=None),
    dcc.Store(id="store-indicator-result", data=None),
    dcc.Store(id="store-indicator-meta", data=None),
    dcc.Store(id="store-review-items", data=None),
    dcc.Store(id="store-review-queue", data=None),  # V2: deduplicated grouped queue
    dcc.Store(id="store-review-selection", data={"review_id": None, "change_id": None}),
    dcc.Store(id="store-review-last-positions", data={}),
    dcc.Store(id="store-current-change-idx", data=0),  # V2: index within table's changes
    dcc.Store(id="store-analysis-running", data=False),
    dcc.Store(id="store-analysis-start-ms", data=None),
    dcc.Store(id="store-validation-start-ms", data=None),
    dcc.Store(id="store-validation-duration-sec", data=None),
    dcc.Store(id="store-show-results-page", data=False),
    dcc.Store(id="store-text-comparison", data=None),
    dcc.Store(id="store-text-review-filters", data={}),
    dcc.Store(id="store-review-filters", data={"section": "all", "status": "all"}),
    dcc.Store(id="store-proof-display-mode", data="crop"),
    dcc.Store(id="store-sidebar-collapsed", data=False),
    dcc.Interval(id="analysis-timer-interval", interval=1000, n_intervals=0, disabled=True),
]


# Layout
app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand("Comparateur Bancaire", href="/", className="fw-bold"),
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Accueil", href="#")),
                            dbc.NavItem(dbc.NavLink("Historique", href="#")),
                            dbc.NavItem(dbc.NavLink("Aide", href="#")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                ],
                fluid=True,
            ),
            color="primary",
            dark=True,
            className="mb-0 shadow-sm",
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        build_sidebar(),
                        dbc.Col(
                            [
                                html.Div(id="main-content", children=build_page_upload()),
                                html.Div(
                                    id="analysis-progress-container",
                                    children=[
                                        html.Div(
                                            "Sections validees. Analyse en cours...",
                                            className="small fw-semibold",
                                        ),
                                        dbc.Progress(
                                            value=100,
                                            striped=True,
                                            animated=True,
                                            color="info",
                                            className="mb-1",
                                        ),
                                        html.Div(
                                            id="analysis-progress-text",
                                            children="Analyse en attente.",
                                            className="small text-muted",
                                        ),
                                    ],
                                    style={"display": "none"},
                                    className="mb-3",
                                ),
                                html.Div(id="notification", className="mt-3"),
                            ],
                            id="main-content-col",
                            className="analysis-main-content p-4 bg-light",
                            style={"minHeight": "100vh"},
                        ),
                    ],
                    id="analysis-workspace",
                    className="g-0 analysis-workspace",
                ),
            ],
            className="container-fluid p-0",
        ),
    ]
    + stores
    + [
        dcc.Download(id="download-review-excel"),
        dcc.Download(id="download-text-excel"),
    ],
)


# =============================================================================
# Callbacks
# =============================================================================

# Clientside callbacks (pure JS, must live in the module that owns `app`)
clientside_callback(
    """
    function(running) {
        return running ? {"display": "block"} : {"display": "none"};
    }
    """,
    Output("analysis-progress-container", "style"),
    Input("store-analysis-running", "data"),
)


clientside_callback(
    """
    function(running, currentStart) {
        if (running) {
            if (!currentStart) {
                return [Date.now(), false];
            }
            return [currentStart, false];
        }
        return [null, true];
    }
    """,
    Output("store-analysis-start-ms", "data"),
    Output("analysis-timer-interval", "disabled"),
    Input("store-analysis-running", "data"),
    State("store-analysis-start-ms", "data"),
)


clientside_callback(
    """
    function(running, n_intervals, startMs) {
        if (!running) {
            return "Analyse en attente.";
        }
        if (!startMs) {
            return "Analyse en cours... 00:00";
        }
        const elapsed = Math.max(0, Math.floor((Date.now() - startMs) / 1000));
        const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const ss = String(elapsed % 60).padStart(2, "0");
        return `Analyse en cours... ${mm}:${ss}`;
    }
    """,
    Output("analysis-progress-text", "children"),
    Input("store-analysis-running", "data"),
    Input("analysis-timer-interval", "n_intervals"),
    State("store-analysis-start-ms", "data"),
)


# Register all server-side callbacks from the callbacks/ sub-package.
# This import must happen AFTER `app = Dash(...)` and layout assignment above.
from vigie.interface.callbacks import register_all_callbacks  # noqa: E402

register_all_callbacks()


if __name__ == "__main__":
    import os

    debug = os.getenv("DASH_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    port = int(os.getenv("DASH_PORT", "8050"))
    app.run(debug=debug, use_reloader=debug, port=port)
