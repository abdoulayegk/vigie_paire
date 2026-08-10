"""Application Dash unifiée du Comparateur Bancaire.

Pour lancer:
    uv run python -m vigie.interface.app
    python -m vigie.interface.app --revue --analyste NOM
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

from vigie.interface import review_runtime

DEFAULT_PORT = 8050


class _FrenchArgumentParser(argparse.ArgumentParser):
    """Présente l'aide et les erreurs de ligne de commande en français."""

    def format_help(self) -> str:
        """Traduit les en-têtes fixes produits par ``argparse``."""
        return super().format_help().replace("usage:", "Utilisation :").replace("options:", "Options :")

    def format_usage(self) -> str:
        """Traduit l'en-tête de la ligne d'utilisation."""
        return super().format_usage().replace("usage:", "Utilisation :")

    def error(self, message: str) -> None:
        """Affiche une erreur française et termine avec le code standard 2."""
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: erreur : {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Construit les options françaises du point d'entrée unique."""
    parser = _FrenchArgumentParser(
        description="Lancer le Comparateur Bancaire.",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        "--aide",
        action="help",
        help="Afficher cette aide et quitter",
    )
    parser.add_argument(
        "--revue",
        action="store_true",
        help="Activer la revue analyste sans extraction ni appel LLM",
    )
    parser.add_argument(
        "--resultats",
        help="Dossier racine contenant les résultats existants",
    )
    parser.add_argument(
        "--analyste",
        help="Identifiant utilisé pour le fichier individuel de revue",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASH_PORT", DEFAULT_PORT)),
        help=f"Port de l'interface (défaut : {DEFAULT_PORT})",
    )
    return parser


def _configure_startup(args: argparse.Namespace) -> None:
    """Configure les résultats et la revue avant l'import des composants Dash."""
    raw_resultats = str(args.resultats or os.environ.get("VIGIE_RESULTATS_DIR", "")).strip()
    if raw_resultats:
        resultats_dir = Path(raw_resultats).expanduser()
        if not resultats_dir.is_dir():
            raise ValueError(f"Dossier de résultats introuvable : {resultats_dir}")
        os.environ["VIGIE_RESULTATS_DIR"] = str(resultats_dir.resolve())

    analyst = str(args.analyste or os.environ.get("VIGIE_ANALYSTE", "")).strip()
    if analyst:
        os.environ["VIGIE_ANALYSTE"] = analyst

    raw_review_mode = os.environ.get("VIGIE_MODE_REVUE", "").strip().lower()
    review_mode = bool(args.revue or args.resultats or analyst) or raw_review_mode in {"1", "true", "yes", "on"}
    os.environ["VIGIE_MODE_REVUE"] = "1" if review_mode else "0"
    review_runtime.set_review_mode(review_mode)
    review_runtime.set_analyst(analyst or None)


_STARTUP_OPTIONS: argparse.Namespace | None = None
if __name__ == "__main__":
    _parser = build_parser()
    _STARTUP_OPTIONS = _parser.parse_args()
    try:
        _configure_startup(_STARTUP_OPTIONS)
    except ValueError as exc:
        _parser.error(str(exc))
else:
    review_runtime.configure_from_environment()

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
from vigie.interface.ui_config import RESULTATS_DIR
from vigie.support.quarter_utils import build_quarter_context

# Theme Bootstrap
APP_THEME = dbc.themes.FLATLY

app = Dash(
    __name__,
    title="Comparateur Bancaire",
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


def _run_application() -> None:
    """Démarre l'application et annonce clairement les résultats détectés."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    debug = os.getenv("DASH_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    options = _STARTUP_OPTIONS or build_parser().parse_args([])
    result_count = sum(1 for _path in RESULTATS_DIR.glob("*/*/comparison.json"))
    logger.info("Application : Comparateur Bancaire")
    logger.info("Dossier de résultats : %s", RESULTATS_DIR)
    logger.info("Analyses détectées : %d", result_count)
    if review_runtime.is_review_mode():
        logger.info("Mode revue analyste : %s", review_runtime.current_analyst())
    if result_count == 0:
        logger.warning("Aucune analyse trouvée. Vérifiez le dossier fourni avec --resultats.")
    app.run(host="127.0.0.1", debug=debug, use_reloader=debug, port=options.port)


if __name__ == "__main__":
    _run_application()
