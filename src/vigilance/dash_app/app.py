"""
Application Dash - Comparateur de Rapports Bancaires.

Pour lancer:
    uv run python -m app.dash_app.app
    ou: uv run dash run app.dash_app.app --port 8050
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _cached_render_or_crop(
    pdf_path: str, page: int, scale: float, bbox_key: str, display_mode: str = "crop"
) -> bytes:
    """Return PNG bytes based on display_mode: crop, full (page + bbox highlight), or footnote."""
    from vigilance.utils.pdf_crop import (
        crop_footnote_region_to_bytes,
        crop_table_region_to_bytes,
        render_page_with_bbox_highlight_to_bytes,
    )

    if not bbox_key:
        raw = get_pdf_preview(pdf_path, page, scale=scale)
        return raw if raw else b""

    try:
        bbox = json.loads(bbox_key)
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raw = get_pdf_preview(pdf_path, page, scale=scale)
            return raw if raw else b""
    except (json.JSONDecodeError, TypeError):
        raw = get_pdf_preview(pdf_path, page, scale=scale)
        return raw if raw else b""

    try:
        if display_mode == "full":
            return render_page_with_bbox_highlight_to_bytes(
                pdf_path, page, bbox, scale=scale
            )
        elif display_mode == "footnote":
            return crop_footnote_region_to_bytes(pdf_path, page, bbox, scale=scale)
        else:  # "crop" (default)
            return crop_table_region_to_bytes(pdf_path, page, bbox, scale=scale)
    except Exception:
        pass

    raw = get_pdf_preview(pdf_path, page, scale=scale)
    return raw if raw else b""


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Assurer que src/ est dans le path
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dash_bootstrap_components as dbc
from dash import (
    ALL,
    MATCH,
    Dash,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from app.comparison_canonical import (
    get_meta_value,
    is_canonical_comparison,
    to_canonical_payload,
)
from app.comparison_runner import run_comparison_with_sections
from app.dash_app.components.pdf_preview import pdf_images_from_base64
from app.dash_app.components.review_detail import (
    build_proofs_section,
    section_display_label,
)
from app.dash_app.layouts import (
    build_page_results,
    build_page_upload,
    build_page_validation,
    build_sidebar,
)
from app.dash_app.layouts.page_results import build_analyst_kpi_card
from app.i18n import t
from app.quarter_utils import (
    build_quarter_context,
    get_payload_quarter_context,
    quarter_label_from_payload,
)
from app.review_adapters import build_review_items_from_indicator_result
from app.review_export import (
    export_review_items_json_fr,
    generate_validation_csv,
    generate_validation_excel,
)
from app.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    ReviewItem,
)
from app.review_models_v2 import ChangeType
from app.review_priority import sort_review_items_by_priority
from app.review_queue_normalizer import (
    build_normalized_review_queue,
)
from app.review_state import set_review_status
from app.ui_config import INDICATOR_COMPARISON_DIR
from app.ui_detection import (
    _detect_sections_core,
    get_pdf_preview,
    get_section_preview_images,
)
from app.ui_indicators import build_indicator_change_rows
from app.ui_io import (
    get_available_indicator_comparison_options,
    load_comparison_result,
    save_pdfs_to_temp,
)
from vigilance.dash_app.components.review_detail_v2 import build_review_detail_v2
from vigilance.dash_app.components.review_queue_v2 import build_review_queue_v2
from vigilance.extraction.table_annotator import annotate_table_with_changes
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison

# Theme Bootstrap
APP_THEME = dbc.themes.FLATLY
REVIEW_QUEUE_V2_ACTIVE = os.getenv("REVIEW_QUEUE_V2_ACTIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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
        data=build_quarter_context("Q2", year=2025),
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
    dcc.Store(id="store-review-current-idx", data=0),
    dcc.Store(id="store-review-current-indicator-idx", data=0),
    dcc.Store(
        id="store-current-change-idx", data=0
    ),  # V2: index within table's changes
    dcc.Store(id="store-analysis-running", data=False),
    dcc.Store(id="store-analysis-start-ms", data=None),
    dcc.Store(id="store-validation-start-ms", data=None),
    dcc.Store(id="store-validation-duration-sec", data=None),
    dcc.Store(id="store-show-results-page", data=False),
    dcc.Store(id="store-review-filters", data={"section": "all", "status": "all"}),
    dcc.Store(id="store-proof-display-mode", data="crop"),
    dcc.Store(id="store-nav-debug", data=None),
    dcc.Store(id="store-sidebar-collapsed", data=False),
    dcc.Interval(
        id="analysis-timer-interval", interval=1000, n_intervals=0, disabled=True
    ),
]


def _quarter_context_from_store(data: dict | None) -> dict:
    if isinstance(data, dict):
        current = data.get("current")
        previous = data.get("previous")
        if isinstance(current, dict) and isinstance(previous, dict):
            return data
    return build_quarter_context("Q2", year=2025)


def _comparison_export_base_name(payload: dict | None, suffix: str) -> str:
    ctx = get_payload_quarter_context(payload)
    bank = str((payload or {}).get("bank_code", "bank")).strip().lower() or "bank"
    current_label = str(ctx.get("current", {}).get("label") or "current").lower()
    previous_label = str(ctx.get("previous", {}).get("label") or "previous").lower()
    year_val = str((payload or {}).get("year", "2025"))
    current_slug = current_label.replace(" ", "_").replace("-", "_")
    previous_slug = previous_label.replace(" ", "_").replace("-", "_")
    return f"{bank}_{current_slug}_vs_{previous_slug}_{year_val}_{suffix}"


# Layout
app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand(
                        "Comparateur Bancaire", href="/", className="fw-bold"
                    ),
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
                                html.Div(
                                    id="main-content", children=build_page_upload()
                                ),
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
        dcc.Download(id="download-review-csv"),
        dcc.Download(id="download-review-json"),
        dcc.Download(id="download-review-excel"),
        dcc.Download(id="download-indicator-json-brut"),
        # Placeholder elements for callbacks that reference dynamically-rendered
        # button IDs. Dash 2.x validates Input IDs client-side even when
        # suppress_callback_exceptions=True is set server-side. These elements
        # are always hidden and never interact with the user.
        html.Div(
            [
                dbc.Button(id="btn-approve", n_clicks=0),
                dbc.Button(id="btn-reject", n_clicks=0),
                dbc.Button(id="btn-apply", n_clicks=0),
                dcc.Textarea(id="review-comment", value=""),
            ],
            style={"display": "none"},
            id="_callback-placeholders",
        ),
    ],
)


# =============================================================================
# Callbacks
# =============================================================================


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


@callback(
    Output("store-sidebar-collapsed", "data"),
    Input("btn-toggle-sidebar", "n_clicks"),
    Input("store-show-results-page", "data"),
    State("store-sidebar-collapsed", "data"),
    prevent_initial_call=True,
)
def update_sidebar_collapsed_state(toggle_clicks, show_results, is_collapsed):
    """Collapse the sidebar for review by default while keeping manual toggle control."""
    if ctx.triggered_id == "btn-toggle-sidebar":
        return not bool(is_collapsed)
    if ctx.triggered_id == "store-show-results-page" and bool(show_results):
        return True
    raise PreventUpdate


@callback(
    Output("analysis-sidebar", "className"),
    Output("analysis-sidebar-body", "className"),
    Output("analysis-sidebar-title", "className"),
    Output("analysis-sidebar-toggle-icon", "className"),
    Output("main-content-col", "className"),
    Input("store-sidebar-collapsed", "data"),
)
def sync_sidebar_layout(is_collapsed):
    collapsed = bool(is_collapsed)
    sidebar_class = "analysis-sidebar bg-light border-end p-3"
    body_class = "analysis-sidebar-body"
    title_class = "analysis-sidebar-title mb-0 text-primary"
    icon_class = "bi bi-layout-sidebar-inset"
    main_class = "analysis-main-content p-4 bg-light"
    if collapsed:
        sidebar_class += " is-collapsed"
        body_class += " is-collapsed"
        title_class += " is-collapsed"
        icon_class = "bi bi-layout-sidebar"
        main_class += " is-expanded"
    return sidebar_class, body_class, title_class, icon_class, main_class


@callback(
    Output("store-quarter-context", "data"),
    Output("previous-quarter-display", "children"),
    Output("comparison-pair-display", "children"),
    Output("upload-previous-label", "children"),
    Output("upload-current-label", "children"),
    Input("analysis-year", "value"),
    Input("current-quarter", "value"),
)
def sync_quarter_context(year_value, current_quarter):
    """Derive the previous quarter from the selected current quarter."""
    ctx = build_quarter_context(current_quarter or "Q2", year=year_value or 2025)
    previous_label = str(ctx["previous"]["label"])
    current_label = str(ctx["current"]["label"])
    return (
        ctx,
        previous_label,
        f"Comparaison exécutée: {current_label} vs {previous_label}",
        f"Rapport trimestre précédent ({previous_label})",
        f"Rapport trimestre courant ({current_label})",
    )


@callback(
    Output("store-upload-t1", "data"),
    Output("upload-t1-name", "children"),
    Input("upload-t1", "contents"),
    State("upload-t1", "filename"),
)
def on_upload_t1(content, filename):
    """Stocker l'upload T1."""
    if not content:
        return None, ""
    return {
        "content": content,
        "filename": filename or "previous.pdf",
    }, f"Précédent: {filename or 'previous.pdf'}"


@callback(
    Output("store-upload-t2", "data"),
    Output("upload-t2-name", "children"),
    Input("upload-t2", "contents"),
    State("upload-t2", "filename"),
)
def on_upload_t2(content, filename):
    """Stocker l'upload T2."""
    if not content:
        return None, ""
    return {
        "content": content,
        "filename": filename or "current.pdf",
    }, f"Courant: {filename or 'current.pdf'}"


@callback(
    Output("store-pdf-paths", "data"),
    Output("store-temp-dir", "data"),
    Output("store-detection", "data"),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("main-content", "children"),
    Output("notification", "children", allow_duplicate=True),
    Output("store-validation-start-ms", "data"),
    Output("store-show-results-page", "data"),
    Input("btn-detect", "n_clicks"),
    State("store-upload-t1", "data"),
    State("store-upload-t2", "data"),
    State("store-quarter-context", "data"),
    State("bank-code", "value"),
    prevent_initial_call=True,
)
def on_detect(n_clicks, upl_t1, upl_t2, quarter_context, bank_code):
    """Détecter les sections sur le couple courant/précédent."""
    import base64
    import tempfile

    if not n_clicks:
        raise PreventUpdate

    quarter_context = _quarter_context_from_store(quarter_context)
    previous_label = str(quarter_context["previous"]["label"])
    current_label = str(quarter_context["current"]["label"])

    if not upl_t1 or not upl_t2 or not bank_code:
        return (
            None,
            None,
            None,
            False,
            build_page_upload(),
            dbc.Alert(
                f"Veuillez uploader les rapports {previous_label} et {current_label}.",
                color="warning",
            ),
            None,
            False,
        )

    def decode(content):
        if content and "," in content:
            return base64.b64decode(content.split(",")[1])
        return base64.b64decode(content) if content else b""

    try:
        b1 = decode(upl_t1.get("content"))
        b2 = decode(upl_t2.get("content"))
        temp_dir = tempfile.mkdtemp()
        path_t1, path_t2 = save_pdfs_to_temp(b1, b2, temp_dir=Path(temp_dir))
        paths = {
            "pdf_t1": path_t1,
            "pdf_t2": path_t2,
            "pdf_previous": path_t1,
            "pdf_current": path_t2,
        }
    except ValueError as e:
        return (
            None,
            None,
            None,
            False,
            build_page_upload(),
            dbc.Alert(str(e), color="danger"),
            None,
            False,
        )

    try:
        mapping_t1 = _detect_sections_core(path_t1, bank_code)
        mapping_t2 = _detect_sections_core(path_t2, bank_code)
    except Exception as e:
        return (
            None,
            temp_dir,
            None,
            False,
            build_page_upload(),
            dbc.Alert(f"Erreur detection: {e}", color="danger"),
            None,
            False,
        )

    detection = {"detection_t1": mapping_t1, "detection_t2": mapping_t2}
    import time

    validation_start_ms = int(time.time() * 1000)
    return (
        paths,
        temp_dir,
        detection,
        False,
        build_page_validation(),
        dbc.Alert(
            (
                "Sections détectées: "
                f"{previous_label}={len(mapping_t1.get('sections', []))}, "
                f"{current_label}={len(mapping_t2.get('sections', []))}"
            ),
            color="success",
        ),
        validation_start_ms,
        False,
    )


@callback(
    Output("validation-sections-container", "children"),
    Input("store-detection", "data"),
    Input("store-pdf-paths", "data"),
    Input("store-adjusted-sections", "data"),
)
def render_validation_sections(detection, paths, adjusted_sections):
    """Afficher les sections detectees avec inputs d'ajustement et preview PDF."""
    if not detection:
        return html.Div("Aucune detection.")

    t1 = detection.get("detection_t1", {})
    t2 = detection.get("detection_t2", {})
    sections_t1 = t1.get("sections", [])
    sections_t2 = t2.get("sections", [])
    total_pages_t1 = t1.get("total_pages", 1)
    total_pages_t2 = t2.get("total_pages", 1)

    path_t1 = paths.get("pdf_t1") if paths else None
    path_t2 = paths.get("pdf_t2") if paths else None

    if (
        adjusted_sections
        and adjusted_sections.get("sections_t1")
        and adjusted_sections.get("sections_t2")
    ):
        use_t1 = adjusted_sections["sections_t1"]
        use_t2 = adjusted_sections["sections_t2"]
    else:
        use_t1 = sections_t1
        use_t2 = sections_t2

    rows = []
    for i, (s1, s2, o1, o2) in enumerate(zip(use_t1, use_t2, sections_t1, sections_t2)):
        label1 = s1.get("label", s1.get("type", ""))
        label2 = s2.get("label", s2.get("type", ""))

        def _section_col(section, orig_section, label, total_pages, path, doc_key, idx):
            start, end = section.get("start_page", 1), section.get("end_page", 1)
            orig_start = orig_section.get("start_page", 1)
            orig_end = orig_section.get("end_page", 1)
            is_adjusted = start != orig_start or end != orig_end
            range_text = (
                f"Ajuste: {start}-{end} (original: {orig_start}-{orig_end})"
                if is_adjusted
                else f"Plage: {start}-{end}"
            )
            captions = [f"Page {p}" for p in range(start, min(end + 1, start + 6))]
            preview_imgs = []
            if path:
                try:
                    imgs = get_section_preview_images(path, section, max_pages=5)
                    preview_imgs = pdf_images_from_base64(imgs, captions[: len(imgs)])
                except Exception:
                    preview_imgs = html.Div(
                        "Preview indisponible", className="text-muted"
                    )
            else:
                preview_imgs = html.Div("Chemin PDF manquant", className="text-muted")

            return dbc.Col(
                [
                    html.H6(label, className="mb-2"),
                    html.P(range_text, className="small text-muted mb-1"),
                    html.Label("Debut", className="form-label small"),
                    dbc.Input(
                        type="number",
                        id={"type": "section-start", "index": idx, "doc": doc_key},
                        min=1,
                        max=total_pages,
                        value=start,
                        className="mb-1",
                    ),
                    html.Label("Fin", className="form-label small"),
                    dbc.Input(
                        type="number",
                        id={"type": "section-end", "index": idx, "doc": doc_key},
                        min=1,
                        max=total_pages,
                        value=end,
                        className="mb-2",
                    ),
                    dbc.Collapse(
                        preview_imgs,
                        id={"type": "section-preview", "index": idx, "doc": doc_key},
                        is_open=False,
                    ),
                    dbc.Button(
                        "Voir preview",
                        id={
                            "type": "section-preview-btn",
                            "index": idx,
                            "doc": doc_key,
                        },
                        color="link",
                        size="sm",
                        className="p-0 mb-2",
                    ),
                ],
                md=6,
                className="p-2 border rounded",
            )

        col_t1 = _section_col(s1, o1, label1, total_pages_t1, path_t1, "t1", i)
        col_t2 = _section_col(s2, o2, label2, total_pages_t2, path_t2, "t2", i)

        rows.append(
            html.Div(
                [
                    html.H6(f"Section {i + 1}: {label1} / {label2}", className="mb-2"),
                    dbc.Row([col_t1, col_t2]),
                ],
                className="mb-3 p-2 border rounded",
            )
        )

    return html.Div(rows)


@callback(
    Output("validation-adjusted-indicator", "children"),
    Input("store-adjusted-sections", "data"),
    Input("store-detection", "data"),
)
def update_adjusted_indicator(adjusted_sections, detection):
    """Afficher le nombre de sections ajustees pres du bouton Analyser."""
    if not adjusted_sections or not detection:
        return None
    t1_orig = detection.get("detection_t1", {}).get("sections", [])
    t2_orig = detection.get("detection_t2", {}).get("sections", [])
    t1_adj = adjusted_sections.get("sections_t1", [])
    t2_adj = adjusted_sections.get("sections_t2", [])
    count = 0
    for o1, o2, a1, a2 in zip(t1_orig, t2_orig, t1_adj, t2_adj):
        if (
            a1.get("start_page") != o1.get("start_page")
            or a1.get("end_page") != o1.get("end_page")
            or a2.get("start_page") != o2.get("start_page")
            or a2.get("end_page") != o2.get("end_page")
        ):
            count += 1
    if count == 0:
        return None
    return html.Span(
        f"{count} section(s) ajustee(s)",
        className="badge bg-secondary",
    )


@callback(
    Output({"type": "section-preview", "index": MATCH, "doc": MATCH}, "is_open"),
    Input({"type": "section-preview-btn", "index": MATCH, "doc": MATCH}, "n_clicks"),
    State({"type": "section-preview", "index": MATCH, "doc": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_section_preview(n_clicks, is_open):
    """Basculer l'affichage du preview PDF d'une section."""
    if n_clicks:
        return not is_open
    return no_update


@callback(
    Output("store-adjusted-sections", "data"),
    Input({"type": "section-start", "index": ALL, "doc": ALL}, "value"),
    Input({"type": "section-end", "index": ALL, "doc": ALL}, "value"),
    State({"type": "section-start", "index": ALL, "doc": ALL}, "id"),
    State({"type": "section-end", "index": ALL, "doc": ALL}, "id"),
    State("store-detection", "data"),
    prevent_initial_call=True,
)
def compile_adjusted_sections(starts, ends, ids_start, ids_end, detection):
    """Compiler les valeurs Debut/Fin dans store-adjusted-sections."""
    if not detection:
        raise PreventUpdate

    t1 = detection.get("detection_t1", {})
    t2 = detection.get("detection_t2", {})
    sections_t1 = list(t1.get("sections", []))
    sections_t2 = list(t2.get("sections", []))

    if not ids_start or not ids_end:
        raise PreventUpdate

    by_key = {}
    for sid, val in zip(ids_start, starts):
        key = (sid.get("index"), sid.get("doc"))
        if key not in by_key:
            by_key[key] = {}
        by_key[key]["start"] = val
    for sid, val in zip(ids_end, ends):
        key = (sid.get("index"), sid.get("doc"))
        if key not in by_key:
            by_key[key] = {}
        by_key[key]["end"] = val

    for (idx, doc), vals in by_key.items():
        start_val = vals.get("start")
        end_val = vals.get("end")
        if start_val is None or end_val is None:
            continue
        try:
            sp, ep = int(start_val), int(end_val)
        except (TypeError, ValueError):
            continue
        if doc == "t1" and idx < len(sections_t1):
            sections_t1[idx] = {**sections_t1[idx], "start_page": sp, "end_page": ep}
        elif doc == "t2" and idx < len(sections_t2):
            sections_t2[idx] = {**sections_t2[idx], "start_page": sp, "end_page": ep}

    return {"sections_t1": sections_t1, "sections_t2": sections_t2}


@callback(
    Output("store-comparison-result", "data"),
    Output("store-indicator-result", "data"),
    Output("store-indicator-meta", "data"),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("main-content", "children", allow_duplicate=True),
    Output("notification", "children", allow_duplicate=True),
    Output("store-validation-duration-sec", "data"),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Input("btn-analyze", "n_clicks"),
    State("store-detection", "data"),
    State("store-adjusted-sections", "data"),
    State("store-pdf-paths", "data"),
    State("store-quarter-context", "data"),
    State("bank-code", "value"),
    State("option-footnotes", "value"),
    State("option-genai-classification", "value"),
    State("option-force-reextract", "value"),
    State("store-validation-start-ms", "data"),
    prevent_initial_call=True,
    running=[
        (
            Output("analysis-progress-container", "style"),
            {"display": "block"},
            {"display": "none"},
        ),
        (Output("store-analysis-running", "data"), True, False),
    ],
)
def on_analyze(
    n_clicks,
    detection,
    adjusted_sections,
    paths,
    quarter_context,
    bank_code,
    footnotes_opt,
    genai_classification_opt,
    force_reextract_opt,
    validation_start_ms,
):
    """Valider et lancer l'analyse."""
    import os

    if not n_clicks or not detection or not paths or not bank_code:
        return None, None, None, False, build_page_validation(), None, None, False

    quarter_context = _quarter_context_from_store(quarter_context)

    if (
        adjusted_sections
        and adjusted_sections.get("sections_t1")
        and adjusted_sections.get("sections_t2")
    ):
        sections_t1 = adjusted_sections["sections_t1"]
        sections_t2 = adjusted_sections["sections_t2"]
    else:
        sections_t1 = detection.get("detection_t1", {}).get("sections", [])
        sections_t2 = detection.get("detection_t2", {}).get("sections", [])
    path_t1 = paths.get("pdf_previous") or paths.get("pdf_t1")
    path_t2 = paths.get("pdf_current") or paths.get("pdf_t2")

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
    use_genai = bool(api_key)

    try:
        from vigilance.config import get_vision_extraction_config

        cfg = get_vision_extraction_config(bank_code=bank_code) or {}
        use_vision_extraction = bool(cfg.get("enabled", False))
    except Exception:
        use_vision_extraction = False
    include_footnotes = bool(footnotes_opt and "footnotes" in footnotes_opt)
    include_genai_classification = bool(
        genai_classification_opt and "classify" in genai_classification_opt and api_key
    )
    # Stored extraction disabled by default; re-enable by uncommenting and removing the line below.
    use_stored_extraction = False
    # use_stored_extraction = not bool(
    #     force_reextract_opt and "reextract" in (force_reextract_opt or [])
    # )

    try:
        result = run_comparison_with_sections(
            pdf_path_previous=path_t1,
            pdf_path_current=path_t2,
            bank_code=bank_code,
            sections_previous=sections_t1,
            sections_current=sections_t2,
            current_quarter=str(quarter_context["current"]["label"]),
            previous_quarter=str(quarter_context["previous"]["label"]),
            current_year=int(quarter_context["current"]["year"]),
            previous_year=int(quarter_context["previous"]["year"]),
            use_genai=use_genai,
            api_key=api_key,
            use_vision_extraction=use_vision_extraction,
            include_footnotes=include_footnotes,
            include_genai_classification=include_genai_classification,
            use_stored_extraction_if_available=use_stored_extraction,
        )
    except Exception as e:
        err_text = str(e)
        if "Vision schema contract invalid" in err_text:
            err_text = (
                "Run interrompu: contrat de schema Vision invalide. "
                "Corriger l'extracteur avant relance."
            )
        return (
            None,
            None,
            None,
            False,
            build_page_validation(),
            dbc.Alert(err_text, color="danger"),
            None,
            False,
        )

    indicator_result = (
        result if is_canonical_comparison(result) else to_canonical_payload(result)
    )
    indicator_meta = (
        indicator_result.get("meta", {}) if isinstance(indicator_result, dict) else {}
    )
    analyze_notification = dbc.Alert(
        "Analyse terminée. Indicateurs comparés.", color="success"
    )

    # Calculate validation duration
    import time

    validation_duration_sec = None
    if validation_start_ms:
        validation_end_ms = int(time.time() * 1000)
        validation_duration_sec = max(
            0, (validation_end_ms - validation_start_ms) // 1000
        )

    return (
        result,
        indicator_result,
        indicator_meta,
        True,
        build_page_results(),
        analyze_notification,
        validation_duration_sec,
        True,
    )


# =============================================================================
# Modern Review Dashboard Callbacks
# =============================================================================


@callback(
    Output("review-queue-container", "children"),
    Output("kpi-queue-total", "children"),
    Output("kpi-queue-approved", "children"),
    Output("kpi-queue-rejected", "children"),
    Output("kpi-queue-pending", "children"),
    Output("progress-approved", "value"),
    Output("progress-rejected", "value"),
    Output("progress-pending", "value"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    Input("store-review-filters", "data"),
    prevent_initial_call=True,
)
def update_review_queue(
    review_queue_data,
    selection,
    indicator_result,
    show_results,
    filters,
):
    """Update the left-side review queue and top KPIs."""
    if not show_results:
        raise PreventUpdate
    if not REVIEW_QUEUE_V2_ACTIVE:
        raise PreventUpdate

    if review_queue_data is None:
        return (
            html.Div(
                [
                    html.I(className="bi bi-hourglass-split me-2"),
                    "Chargement de la file de revue...",
                ],
                className="text-muted p-3",
            ),
            build_analyst_kpi_card(t("file_review_total"), "-", color="white"),
            build_analyst_kpi_card(t("validated"), "-", color="white"),
            build_analyst_kpi_card(t("rejected"), "-", color="white"),
            build_analyst_kpi_card(t("pending"), "-", color="white"),
            0,
            0,
            0,
        )
    if len(review_queue_data) == 0:
        empty_msg = t("no_changes_review", "Aucun changement a revoir.")
        return (
            html.Div(
                empty_msg,
                className="text-muted p-3",
            ),
            build_analyst_kpi_card(t("file_review_total"), "0", color="white"),
            build_analyst_kpi_card(t("validated"), "0", color="white"),
            build_analyst_kpi_card(t("rejected"), "0", color="white"),
            build_analyst_kpi_card(t("pending"), "0", color="white"),
            0,
            0,
            0,
        )

    queue = review_queue_data or []
    resolved_selection, _, _ = _resolve_selection(queue, selection, filters)
    current_review_id = resolved_selection.get("review_id")

    total = len(queue)
    approved = sum(1 for t in queue if _table_decision_bucket(t) == "approved")
    rejected = sum(1 for t in queue if _table_decision_bucket(t) == "rejected")
    pending = max(0, total - approved - rejected)
    pct_approved = (approved / total) * 100 if total else 0
    pct_rejected = (rejected / total) * 100 if total else 0
    pct_pending = (pending / total) * 100 if total else 0

    queue_component = build_review_queue_v2(queue, current_review_id, filters)

    return (
        queue_component,
        build_analyst_kpi_card(t("file_review_total"), str(total), color="white"),
        build_analyst_kpi_card(t("validated"), str(approved), color="white"),
        build_analyst_kpi_card(t("rejected"), str(rejected), color="white"),
        build_analyst_kpi_card(t("pending"), str(pending), color="white"),
        pct_approved,
        pct_rejected,
        pct_pending,
    )


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("store-review-queue", "data"),
    Input("store-review-filters", "data"),
    State("store-review-selection", "data"),
    prevent_initial_call=True,
)
def sync_review_selection(queue, filters, selection):
    """Keep selection stable on queue refresh/filter changes, with safe fallback."""
    if not REVIEW_QUEUE_V2_ACTIVE:
        raise PreventUpdate
    if queue is None:
        raise PreventUpdate
    resolved_selection, table_idx, change_idx = _resolve_selection(
        queue or [], selection, filters
    )
    if resolved_selection == (selection or {}):
        raise PreventUpdate
    selection_miss = str((selection or {}).get("review_id") or "") not in {
        _review_id(t) for t in (queue or [])
    }
    dbg = {
        "writer": "sync_review_selection",
        "trigger": "queue_or_filter_refresh",
        "from": (selection or {}).get("review_id"),
        "to": resolved_selection.get("review_id"),
        "total": len(queue or []),
        "selection_miss": selection_miss,
    }
    return resolved_selection, table_idx, change_idx, dbg


@callback(
    Output("store-review-filters", "data"),
    Input({"type": "filter-section-v2", "value": ALL}, "n_clicks"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_filter_section(n_clicks, current_filters):
    """Update section filter when a filter button is clicked."""
    if not ctx.triggered_id:
        raise PreventUpdate

    section_value = ctx.triggered_id.get("value")
    new_filters = dict(current_filters or {})
    new_filters["section"] = section_value
    return new_filters


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input({"type": "queue-table-item-v2", "review_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_queue_item_click(n_clicks, queue, current_selection, filters):
    """Handle click on a table item in the V2 queue."""
    if REVIEW_QUEUE_V2_ACTIVE:
        raise PreventUpdate
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered_id
    if not button_id or not isinstance(button_id, dict):
        raise PreventUpdate

    clicked_review_id = str(button_id.get("review_id") or "")
    if not clicked_review_id:
        raise PreventUpdate

    if not queue:
        raise PreventUpdate
    total = len(queue)
    resolved_selection, table_idx, change_idx = _resolve_selection(
        queue,
        {"review_id": clicked_review_id, "change_id": None},
        filters,
    )

    dbg = {
        "writer": "on_queue_item_click",
        "trigger": str(button_id),
        "from": (current_selection or {}).get("review_id"),
        "to": resolved_selection.get("review_id"),
        "total": total,
    }
    logger.info(
        "[on_queue_item_click] trig=%s current_review_id=%r total=%s -> review_id=%s",
        button_id,
        (current_selection or {}).get("review_id"),
        total,
        resolved_selection.get("review_id"),
    )
    return resolved_selection, table_idx, change_idx, dbg


@callback(
    Output("review-proof-container", "children"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-pdf-paths", "data"),
    Input("store-show-results-page", "data"),
    Input("store-proof-display-mode", "data"),
    prevent_initial_call=True,
)
def update_review_proofs(
    review_queue_data, selection, paths, show_results, proof_display_mode
):
    """Update proof images section only (table-scoped; stable across indicator navigation)."""
    if not show_results:
        raise PreventUpdate
    if not REVIEW_QUEUE_V2_ACTIVE:
        raise PreventUpdate
    if not review_queue_data:
        return html.Div(
            "Veuillez lancer une analyse pour voir les details.",
            className="text-center text-muted mt-5",
        )

    resolved_selection, _, _ = _resolve_selection(review_queue_data, selection)
    _, table = _get_table_by_review_id(
        review_queue_data, resolved_selection.get("review_id")
    )
    if table is None:
        return html.Div()

    item = _table_to_proof_item(table, resolved_selection)
    mode = (proof_display_mode or "crop").strip().lower()
    if mode not in ("crop", "full", "footnote"):
        mode = "crop"

    img_t1_b64 = _get_proof_image_b64_for_item(
        item, "t1", paths or {}, proof_display_mode=mode
    )
    img_t2_b64 = _get_proof_image_b64_for_item(
        item, "t2", paths or {}, proof_display_mode=mode
    )
    return build_proofs_section(
        item=item, img_t1_b64=img_t1_b64, img_t2_b64=img_t2_b64, proof_display_mode=mode
    )


@callback(
    Output("review-meta-container", "children"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def update_review_meta(review_queue_data, selection, show_results):
    """Update V2 metadata + per-change validation section."""
    if not show_results:
        raise PreventUpdate
    if not REVIEW_QUEUE_V2_ACTIVE:
        raise PreventUpdate
    if not review_queue_data:
        return html.Div(
            "Veuillez lancer une analyse pour voir les details.",
            className="text-center text-muted mt-5",
        )

    resolved_selection, _, change_idx = _resolve_selection(review_queue_data, selection)
    _, table = _get_table_by_review_id(
        review_queue_data, resolved_selection.get("review_id")
    )
    if table is None:
        return html.Div(
            "Aucun tableau sélectionné.",
            className="text-center text-muted mt-5",
        )

    return build_review_detail_v2(
        table=table,
        current_change_idx=change_idx,
        proof_image_t1_b64="",
        proof_image_t2_b64="",
        show_proofs=False,
    )


@callback(
    Output("store-proof-display-mode", "data"),
    Input("proof-display-mode", "value"),
)
def on_proof_display_mode_change(value):
    """Persist proof display mode (crop vs full page + bbox)."""
    if value in ("crop", "full", "footnote"):
        return value
    return no_update


def _derive_table_status(indicators: list[dict]) -> str:
    """Derive table-level review status from per-indicator statuses."""
    if not indicators:
        return REVIEW_STATUS_APPROVED
    statuses = [ind.get("review_status", REVIEW_STATUS_PENDING) for ind in indicators]
    if all(s == REVIEW_STATUS_APPROVED for s in statuses):
        return REVIEW_STATUS_APPROVED
    if all(s == REVIEW_STATUS_REJECTED for s in statuses):
        return REVIEW_STATUS_REJECTED
    if all(s in (REVIEW_STATUS_APPROVED, REVIEW_STATUS_REJECTED) for s in statuses):
        return REVIEW_STATUS_APPROVED
    return REVIEW_STATUS_PENDING


@callback(
    Output("store-review-items", "data", allow_duplicate=True),
    Output("store-review-current-indicator-idx", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Input("btn-approve", "n_clicks"),
    Input("btn-reject", "n_clicks"),
    Input("btn-apply", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-current-idx", "data"),
    State("store-review-current-indicator-idx", "data"),
    State("review-comment", "value"),
    prevent_initial_call=True,
)
def on_review_action_modern(
    btn_approve,
    btn_reject,
    btn_apply,
    review_items,
    current_idx,
    indicator_idx,
    comment,
):
    """Handle review actions per-indicator: Approve/Reject applies to the current indicator only.

    Auto-advance rules:
    - After decision, advance to next indicator in same table (images stable).
    - When last indicator of table is decided, auto-advance to next table + reset indicator_idx.
    """
    import json

    from dash import ctx

    if not ctx.triggered_id or not review_items:
        raise PreventUpdate

    idx = max(0, min(int(current_idx or 0), len(review_items) - 1))
    item_dict = review_items[idx]
    indicators = item_dict.get("indicators", [])
    ind_idx = int(indicator_idx or 0)

    triggered = ctx.triggered_id

    if triggered == "btn-approve":
        if not btn_approve:
            raise PreventUpdate
        new_ind_status = REVIEW_STATUS_APPROVED
    elif triggered == "btn-reject":
        if not btn_reject:
            raise PreventUpdate
        new_ind_status = REVIEW_STATUS_REJECTED
    elif triggered == "btn-apply":
        if not btn_apply:
            raise PreventUpdate
        updated_item = json.loads(json.dumps(item_dict))
        if comment is not None:
            updated_item["comment"] = comment
        new_items = json.loads(json.dumps(review_items))
        new_items[idx] = updated_item
        return new_items, no_update, no_update
    else:
        raise PreventUpdate

    updated_item = json.loads(json.dumps(item_dict))
    updated_indicators = updated_item.get("indicators", [])
    event_type = (updated_item.get("event_type") or "").strip()

    if updated_indicators and 0 <= ind_idx < len(updated_indicators):
        updated_indicators[ind_idx]["review_status"] = new_ind_status
    elif not updated_indicators:
        updated_item["review_status"] = new_ind_status

    # Whole-table events: keep item-level status from button. Matched/footnote: derive from indicators.
    if event_type not in ("table_added", "table_removed"):
        updated_item["review_status"] = _derive_table_status(updated_indicators)
    if comment is not None:
        updated_item["comment"] = comment

    new_items = json.loads(json.dumps(review_items))
    new_items[idx] = updated_item

    next_ind_idx = ind_idx
    next_table_idx = idx

    if not updated_indicators:
        return new_items, next_ind_idx, no_update

    def _is_decided(s: str | None) -> bool:
        return s in (REVIEW_STATUS_APPROVED, REVIEW_STATUS_REJECTED)

    all_decided = all(
        _is_decided(ind.get("review_status")) for ind in updated_indicators
    )

    if ind_idx < len(updated_indicators) - 1:
        next_ind_idx = ind_idx + 1
        return new_items, next_ind_idx, no_update
    if all_decided and idx < len(new_items) - 1:
        next_table_idx = idx + 1
        next_ind_idx = 0
        return new_items, next_ind_idx, next_table_idx
    return new_items, next_ind_idx, no_update


@callback(
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-review-current-indicator-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("store-review-current-idx", "data"),
    State("store-review-items", "data"),
    prevent_initial_call=True,
)
def on_modern_nav(prev_clicks, next_clicks, current_idx, items):
    """Handle Previous/Next table buttons. Resets indicator_idx to 0 on table change."""
    logger.info(
        "[on_modern_nav] ENTER trig=%s current_idx=%r items_len=%s",
        ctx.triggered_id,
        current_idx,
        len(items) if items else 0,
    )

    if not items or not isinstance(items, list):
        logger.warning("[on_modern_nav] PreventUpdate: no items")
        raise PreventUpdate

    total = len(items)
    if total == 0:
        logger.warning("[on_modern_nav] PreventUpdate: total=0")
        raise PreventUpdate

    try:
        idx = int(current_idx) if current_idx is not None else 0
    except (TypeError, ValueError):
        idx = 0

    triggered = ctx.triggered_id
    if triggered == "btn-prev":
        idx = idx - 1
    elif triggered == "btn-next":
        idx = idx + 1
    else:
        logger.warning(
            "[on_modern_nav] PreventUpdate: trig=%r (not btn-prev/btn-next)", triggered
        )
        raise PreventUpdate

    idx = max(0, min(idx, total - 1))
    dbg = {
        "writer": "on_modern_nav",
        "trigger": triggered,
        "from": current_idx,
        "to": idx,
        "total": total,
    }
    logger.info("[on_modern_nav] EXIT -> new_idx=%s total=%s", idx, total)
    return idx, 0, dbg


@callback(
    Output("store-review-current-indicator-idx", "data", allow_duplicate=True),
    Input({"type": "indicator-item", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_indicator_item_click(n_clicks):
    """Handle click on an individual indicator row in the detail panel."""
    if not ctx.triggered:
        raise PreventUpdate
    button_id = ctx.triggered_id
    if not button_id or not isinstance(button_id, dict):
        raise PreventUpdate
    clicked_index = button_id.get("index")
    if clicked_index is None:
        raise PreventUpdate
    return int(clicked_index)


@callback(
    Output("nav-debug-panel", "children"),
    Input("store-review-current-idx", "data"),
    Input("store-nav-debug", "data"),
    State("store-review-items", "data"),
)
def render_nav_debug(idx, dbg, items):
    """Debug panel: show current idx, total items, last writer (to detect nav / reset)."""
    total = len(items) if items and isinstance(items, list) else None
    payload = {"idx": idx, "total": total, "dbg": dbg}
    return html.Pre(
        json.dumps(payload, ensure_ascii=False, indent=2), className="mb-0 small"
    )


@callback(
    Output("btn-prev", "disabled"),
    Output("btn-next", "disabled"),
    Input("store-review-current-idx", "data"),
    Input("store-review-items", "data"),
    Input("store-show-results-page", "data"),
)
def update_review_nav_disabled(current_idx, items, show_results):
    """Disable Prev/Next when at first/last item.
    btn-next also disabled if current table has pending indicators."""
    if not show_results or not items:
        return True, True
    idx = max(0, min(int(current_idx or 0), len(items) - 1))
    n = len(items)
    at_first = idx <= 0
    at_last = idx >= n - 1

    current_item = items[idx]
    indicators = current_item.get("indicators", [])
    has_pending = any(
        ind.get("review_status", REVIEW_STATUS_PENDING) == REVIEW_STATUS_PENDING
        for ind in indicators
    )
    next_disabled = at_last or has_pending
    return at_first, next_disabled


@callback(
    Output("stats-validation-time", "children"),
    Input("store-validation-duration-sec", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def update_validation_time_footer(duration_sec, show_results):
    if not show_results:
        raise PreventUpdate
    if duration_sec is None:
        return f"{t('validation_time')}: --:--"
    return f"{t('validation_time')}: {_format_duration(duration_sec)}"


# =============================================================================
# Legacy / Existing Callbacks (Kept for compatibility where needed)
# =============================================================================


@callback(
    Output("results-header", "children"),
    Output("results-executive-summary", "children"),
    Output("results-kpis", "children"),
    Input("store-comparison-result", "data"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_results(comparison, indicator, show_results):
    """Afficher les resultats."""
    if not show_results:
        raise PreventUpdate
    if not comparison and not indicator:
        return html.Div(), html.Div(), html.Div()

    bank = "N/A"
    title = "Comparaison"
    data = indicator if indicator else comparison
    if comparison:
        bank = comparison.get("bank_code", "N/A")
        title = comparison.get(
            "comparison", comparison.get("comparison_date", "Comparaison")
        )
    elif indicator:
        bank = indicator.get("bank_code", "N/A")
        title = "Indicateurs"
    quarter_context = get_payload_quarter_context(
        data if isinstance(data, dict) else {}
    )
    previous_label = str(quarter_context["previous"]["label"])
    current_label = str(quarter_context["current"]["label"])
    header = html.H5(
        f"{str(bank).upper()} - {title} - {current_label} vs {previous_label}"
    )

    executive_summary = html.Div()
    if indicator and isinstance(indicator, dict):
        kpi = indicator.get("summary", indicator.get("kpi_metier", {})) or {}
        status_counts = kpi.get("status_counts", {}) or {}
        tables_t1 = kpi.get("tables_t1", 0)
        tables_t2 = kpi.get("tables_t2", 0)
        tables_comparable_t1 = int(kpi.get("tables_comparable_t1", 0) or 0)
        tables_comparable_t2 = int(kpi.get("tables_comparable_t2", 0) or 0)
        tables_matched = kpi.get("tables_matched", 0)
        ambiguous_tables = kpi.get("ambiguous_tables", 0)
        vision_rescued_pairs = kpi.get("vision_rescued_pairs", 0)
        cross_section_rescued_pairs = kpi.get("cross_section_rescued_pairs", 0)
        structure_change = status_counts.get("structure_change", 0)
        incertain = status_counts.get("incertain", 0)
        added = kpi.get("total_added_indicators", 0)
        removed = kpi.get("total_removed_indicators", 0)
        renamed = kpi.get("total_renamed_indicators", 0)

        comparisons = indicator.get("table_comparisons", []) or []
        tables_added = indicator.get("tables_added", []) or []
        tables_removed = indicator.get("tables_removed", []) or []
        section_counts: dict[str, int] = {}
        for comp in comparisons:
            n_changes = (
                len(comp.get("added_indicators", []))
                + len(comp.get("removed_indicators", []))
                + len(comp.get("renamed_indicators", []))
            )
            fn = comp.get("footnotes_counts", {}) or {}
            n_changes += sum(fn.get(k, 0) for k in ("added", "removed", "modified"))
            if n_changes > 0:
                sec = comp.get("section", "Autres")
                section_counts[sec] = section_counts.get(sec, 0) + 1
        for tab in tables_added:
            sec = tab.get("section", "Autres")
            section_counts[sec] = section_counts.get(sec, 0) + 1
        for tab in tables_removed:
            sec = tab.get("section", "Autres")
            section_counts[sec] = section_counts.get(sec, 0) + 1

        notes_total = 0
        for comp in comparisons:
            fn = comp.get("footnotes_counts", {}) or {}
            notes_total += sum(fn.get(k, 0) for k in ("added", "removed", "modified"))

        parts = [
            f"{tables_t1} {t('tables')} au trimestre précédent ({previous_label}), "
            f"{tables_t2} au trimestre courant ({current_label}). "
            f"{tables_matched} appariés",
        ]
        if structure_change:
            parts.append(f", {structure_change} fusion/split")
        if incertain:
            parts.append(f", {incertain} incertain(s)")
        if ambiguous_tables:
            parts.append(f", {ambiguous_tables} ambigu(s)")
        parts.append(". ")
        if added or removed or renamed:
            sub = []
            if added:
                sub.append(f"{added} ajout(s)")
            if removed:
                sub.append(f"{removed} retrait(s)")
            if renamed:
                sub.append(f"{renamed} renommage(s)")
            parts.append(", ".join(sub))
            parts.append(". ")
        if section_counts:
            sections_str = ", ".join(
                f"{section_display_label(s)} ({n} tableaux)"
                for s, n in sorted(section_counts.items())
            )
            parts.append(f"Sections impactees : {sections_str}. ")
        if notes_total:
            parts.append(f"{notes_total} note(s) de bas de tableau modifiees.")
        else:
            parts.append("Aucune note de bas de tableau modifiee.")
        if (
            tables_t1
            and tables_t2
            and tables_comparable_t1 == 0
            and tables_comparable_t2 == 0
        ):
            parts.append(
                " Aucun tableau comparable: l'extraction Vision de ce run n'a "
                "renvoye aucun indicateur exploitable."
            )
        if vision_rescued_pairs:
            parts.append(f" {vision_rescued_pairs} tableau(x) recuperes par Vision.")
        if cross_section_rescued_pairs:
            parts.append(
                f" {cross_section_rescued_pairs} tableau(x) recuperes par rescue cross-section."
            )

        summary_text = "".join(parts)
        executive_summary = dbc.Alert(
            html.P(summary_text, className="mb-0 small"),
            color="info",
            className="mb-3",
        )
        meta = indicator.get("meta", {}) or {}
        genai_text = get_meta_value(meta, "executive_summary", "content") or ""
        if genai_text:
            executive_summary = html.Div(
                [
                    executive_summary,
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                html.P(genai_text, className="mb-0 small text-muted"),
                                title="Résumé GenAI (cliquer pour dérouler)",
                            )
                        ],
                        start_collapsed=True,
                        className="mb-3 shadow-sm",
                    ),
                ]
            )

    kpis = []
    if indicator:
        kpi = indicator.get("summary", indicator.get("kpi_metier", {}))
        status_counts = kpi.get("status_counts", {}) or {}
        structure_change = status_counts.get("structure_change", 0)
        changed_t1 = kpi.get("tables_changed_t1", 0)
        changed_t2 = kpi.get("tables_changed_t2", 0)
        ambiguous_tables = int(kpi.get("ambiguous_tables", 0) or 0)
        ambiguous_pairs = int(kpi.get("ambiguous_pairs", 0) or 0)
        vision_rescued_pairs = int(kpi.get("vision_rescued_pairs", 0) or 0)
        cross_section_rescued_pairs = int(
            kpi.get("cross_section_rescued_pairs", 0) or 0
        )
        tables_comparable_t1 = int(kpi.get("tables_comparable_t1", 0) or 0)
        tables_comparable_t2 = int(kpi.get("tables_comparable_t2", 0) or 0)
        pairing_coverage = float(kpi.get("pairing_coverage", 0.0) or 0.0)
        pairing_low_confidence = bool(
            kpi.get("pairing_low_confidence", False)
            or pairing_coverage < 0.75
            or ambiguous_tables > 0
            or ambiguous_pairs > 0
        )
        tables_added_confirmed = int(kpi.get("tables_added_confirmed", 0) or 0)
        tables_removed_confirmed = int(kpi.get("tables_removed_confirmed", 0) or 0)
        cols = [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                f"{t('tables')} ({previous_label})",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(kpi.get("tables_t1", 0)), className="mb-0 fw-bold"
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                f"{t('tables')} ({current_label})",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(kpi.get("tables_t2", 0)), className="mb-0 fw-bold"
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(t("matched"), className="small text-muted mb-0"),
                            html.H4(
                                str(kpi.get("tables_matched", 0)),
                                className="mb-0 fw-bold",
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                t("fusion_split"), className="small text-muted mb-0"
                            ),
                            html.H4(str(structure_change), className="mb-0 fw-bold"),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                t("kpi_changed_t1"),
                                className="small text-muted mb-0",
                            ),
                            html.H4(str(changed_t1), className="mb-0 fw-bold"),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                t("kpi_changed_t2"),
                                className="small text-muted mb-0",
                            ),
                            html.H4(str(changed_t2), className="mb-0 fw-bold"),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
        ]
        kpis.append(dbc.Row(cols, className="mb-3"))
        resolution_cols = [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Récupérés par Vision",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(vision_rescued_pairs), className="mb-0 fw-bold"
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Cross-section",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(cross_section_rescued_pairs),
                                className="mb-0 fw-bold",
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P("Ambigus", className="small text-muted mb-0"),
                            html.H4(str(ambiguous_tables), className="mb-0 fw-bold"),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Ajoutes confirms"
                                if not pairing_low_confidence
                                else "Ajoutes non apparies",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(tables_added_confirmed),
                                className="mb-0 fw-bold",
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Supprimes confirms"
                                if not pairing_low_confidence
                                else "Supprimes non apparies",
                                className="small text-muted mb-0",
                            ),
                            html.H4(
                                str(tables_removed_confirmed),
                                className="mb-0 fw-bold",
                            ),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=3,
            ),
        ]
        kpis.append(dbc.Row(resolution_cols, className="mb-3"))

        pairing_context = (
            f"Comparables: {tables_comparable_t1} ({previous_label}) / "
            f"{tables_comparable_t2} ({current_label})"
            f" | Couverture pairing: {pairing_coverage * 100:.0f}%"
        )
        if tables_comparable_t1 == 0 and tables_comparable_t2 == 0:
            pairing_context += (
                " | Extraction vide: Vision n'a produit aucun indicateur comparable."
            )
        if pairing_low_confidence:
            pairing_context += (
                " | Pairing faible: les tableaux non apparies restent a confirmer."
            )
        kpis.append(
            dbc.Alert(
                pairing_context,
                color="light",
                className="mb-3 small shadow-sm border-0",
            )
        )

        meta = indicator.get("meta", {}) or {}
        validation_summary = (
            meta.get("validation_summary", {}) if isinstance(meta, dict) else {}
        )
        if isinstance(validation_summary, dict):
            vp = validation_summary.get("vision_pair", {}) or {}
            vur = validation_summary.get("vision_unmatched_rescue", {}) or {}
            rv = validation_summary.get("rename_validator", {}) or {}
            atv = validation_summary.get("added_table_validator", {}) or {}
            iv = validation_summary.get("indicator_validator", {}) or {}
            if any(
                bool(block.get("enabled", False))
                for block in (vp, vur, rv, atv, iv)
                if isinstance(block, dict)
            ):
                validation_cols = [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Validation paires (Vision)",
                                        className="small text-muted mb-0",
                                    ),
                                    html.H5(
                                        f"{int(vp.get('accepted', 0))}/{int(vp.get('rejected', 0))}",
                                        className="mb-0 fw-bold",
                                    ),
                                    html.Small(
                                        (
                                            f"calls={int(vp.get('calls', 0))} "
                                            f"rescue={int(vur.get('rescued_pairs', 0))} "
                                            f"err={int(vp.get('errors', 0)) + int(vur.get('errors', 0))}"
                                        ),
                                        className="text-muted",
                                    ),
                                ],
                                className="p-2 text-center",
                            ),
                            className="shadow-sm border-0",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Validation renommages",
                                        className="small text-muted mb-0",
                                    ),
                                    html.H5(
                                        f"{int(rv.get('accepted', 0))}/{int(rv.get('rejected', 0))}",
                                        className="mb-0 fw-bold",
                                    ),
                                    html.Small(
                                        (
                                            f"band={int(rv.get('candidates_in_band', 0))} "
                                            f"skip={int(rv.get('auto_accepted_out_of_band', 0))}"
                                        ),
                                        className="text-muted",
                                    ),
                                ],
                                className="p-2 text-center",
                            ),
                            className="shadow-sm border-0",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Validation tableaux ajoutés",
                                        className="small text-muted mb-0",
                                    ),
                                    html.H5(
                                        f"{int(atv.get('accepted', 0))}/{int(atv.get('rejected', 0))}",
                                        className="mb-0 fw-bold",
                                    ),
                                    html.Small(
                                        f"calls={int(atv.get('calls', 0))} err={int(atv.get('errors', 0))}",
                                        className="text-muted",
                                    ),
                                ],
                                className="p-2 text-center",
                            ),
                            className="shadow-sm border-0",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Validation indicateurs",
                                        className="small text-muted mb-0",
                                    ),
                                    html.H5(
                                        f"-{int(iv.get('filtered_added', 0))}/-{int(iv.get('filtered_removed', 0))}",
                                        className="mb-0 fw-bold",
                                    ),
                                    html.Small(
                                        f"calls={int(iv.get('calls', 0))} err={int(iv.get('errors', 0))}",
                                        className="text-muted",
                                    ),
                                ],
                                className="p-2 text-center",
                            ),
                            className="shadow-sm border-0",
                        ),
                        width=3,
                    ),
                ]
                kpis.append(dbc.Row(validation_cols, className="mb-3"))
    elif comparison:
        summary = comparison.get("summary", {})
        total_changes = summary.get("total_changes")
        if total_changes is None:
            total_changes = int(summary.get("total_added_indicators", 0)) + int(
                summary.get("total_removed_indicators", 0)
            )
        kpis.append(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("Changements"),
                                        html.P(str(total_changes)),
                                    ]
                                )
                            ]
                        ),
                        md=2,
                    ),
                ],
                className="mb-3",
            )
        )

    return header, executive_summary, html.Div(kpis)


def _build_kpi_card(
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


def _format_duration(seconds: int | None) -> str:
    """Format seconds as MM:SS."""
    if seconds is None:
        return "--:--"
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"


def _review_id(table: dict) -> str:
    return str(table.get("review_id") or table.get("table_key") or "")


def _table_matches_filters(table: dict, filters: dict | None) -> bool:
    active_section = (filters or {}).get("section", "all")
    active_status = (filters or {}).get("status", "all")
    if active_section and active_section != "all":
        if table.get("section") != active_section:
            return False
    if active_status and active_status != "all":
        if table.get("table_status") != active_status:
            return False
    return True


def _visible_review_ids(queue: list[dict], filters: dict | None) -> list[str]:
    return [
        _review_id(table)
        for table in queue
        if _review_id(table) and _table_matches_filters(table, filters)
    ]


def _get_table_by_review_id(
    queue: list[dict], review_id: str | None
) -> tuple[int, dict | None]:
    if not queue:
        return -1, None
    rid = str(review_id or "")
    if rid:
        for idx, table in enumerate(queue):
            if _review_id(table) == rid:
                return idx, table
    return -1, None


def _resolve_change_id(
    table: dict, preferred_change_id: str | None = None
) -> str | None:
    changes = table.get("changes", []) or []
    preferred = str(preferred_change_id or "")
    if preferred and any(str(c.get("change_id", "")) == preferred for c in changes):
        return preferred
    for change in changes:
        status = str(change.get("validation_status", "pending"))
        if change.get("is_required", True) and status == "pending":
            return str(change.get("change_id", "")) or None
    if changes:
        return str(changes[0].get("change_id", "")) or None
    return None


def _resolve_selection(
    queue: list[dict],
    selection: dict | None,
    filters: dict | None = None,
) -> tuple[dict, int, int]:
    if not queue:
        return {"review_id": None, "change_id": None}, 0, 0

    visible_ids = _visible_review_ids(queue, filters) or [
        _review_id(t) for t in queue if _review_id(t)
    ]
    preferred_review_id = str((selection or {}).get("review_id") or "")
    if preferred_review_id not in visible_ids:
        preferred_review_id = visible_ids[0] if visible_ids else _review_id(queue[0])

    table_idx, table = _get_table_by_review_id(queue, preferred_review_id)
    if table is None:
        table_idx = 0
        table = queue[0]
        preferred_review_id = _review_id(table)

    resolved_change_id = _resolve_change_id(
        table, preferred_change_id=(selection or {}).get("change_id")
    )
    changes = table.get("changes", []) or []
    change_idx = 0
    if resolved_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == resolved_change_id:
                change_idx = idx
                break
    return (
        {"review_id": preferred_review_id, "change_id": resolved_change_id},
        table_idx,
        change_idx,
    )


def _legacy_change_type_from_v2(change_type: str) -> str:
    ctype = str(change_type or "")
    if ctype in (ChangeType.TABLE_ADDED.value, "table_added"):
        return CHANGE_TYPE_TABLE_ADDED
    if ctype in (ChangeType.TABLE_REMOVED.value, "table_removed"):
        return CHANGE_TYPE_TABLE_REMOVED
    if ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
        return CHANGE_TYPE_ADDED
    if ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
        return CHANGE_TYPE_REMOVED
    if ctype in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
        return CHANGE_TYPE_RENAMED
    if "footnote" in ctype:
        return CHANGE_TYPE_MODIFIED
    return CHANGE_TYPE_MODIFIED


def _table_to_proof_item(table: dict, selection: dict | None) -> dict:
    changes = table.get("changes", []) or []
    selected_change_id = str((selection or {}).get("change_id") or "")
    selected_change = None
    for change in changes:
        if str(change.get("change_id", "")) == selected_change_id:
            selected_change = change
            break
    if selected_change is None and changes:
        selected_change = changes[0]
    selected_change_type = (
        selected_change.get("change_type", "")
        if isinstance(selected_change, dict)
        else ""
    )
    primary_type = _legacy_change_type_from_v2(selected_change_type)
    added_indicators: list[str] = []
    removed_indicators: list[str] = []
    for change in changes:
        ctype = str(change.get("change_type", ""))
        payload = change.get("payload", {}) or {}
        indicator_name = str(payload.get("indicator_name", "")).strip()
        if (
            ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added")
            and indicator_name
        ):
            added_indicators.append(indicator_name)
        if (
            ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed")
            and indicator_name
        ):
            removed_indicators.append(indicator_name)

    return {
        "change_type": primary_type,
        "table_name": table.get("table_name", ""),
        "table_title_raw": table.get("table_title", "") or table.get("table_name", ""),
        "table_number": table.get("table_number", ""),
        "section": table.get("section", ""),
        "page_t1": table.get("page_t1"),
        "page_t2": table.get("page_t2"),
        "bbox_t1": table.get("bbox_t1"),
        "bbox_t2": table.get("bbox_t2"),
        "source_ref_t1": table.get("source_pdf_t1", ""),
        "source_ref_t2": table.get("source_pdf_t2", ""),
        "confidence": float(table.get("confidence", 0.0) or 0.0),
        "genai_analysis": table.get("genai_analysis") or {},
        "added_indicators": added_indicators,
        "removed_indicators": removed_indicators,
    }


def _table_decision_bucket(table: dict) -> str:
    changes = table.get("changes", []) or []
    if not changes:
        return "pending"
    required = [c for c in changes if c.get("is_required", True)]
    if not required:
        required = changes
    statuses = [str(c.get("validation_status", "pending")) for c in required]
    if all(s in ("approved", "skipped") for s in statuses):
        return "approved"
    if all(s == "rejected" for s in statuses):
        return "rejected"
    return "pending"


def _review_items_from_v2_queue(queue: list[dict]) -> list[ReviewItem]:
    """Convert canonical V2 queue to legacy ReviewItem list for exports."""
    items: list[ReviewItem] = []
    for idx, table in enumerate(queue or [], start=1):
        review_id = _review_id(table) or f"tbl_{idx:04d}"
        section = str(table.get("section", ""))
        table_name = str(table.get("table_name") or table.get("table_title") or "")
        table_number = str(table.get("table_number") or "")
        table_id_t1 = str(table.get("table_id_t1") or "")
        table_id_t2 = str(table.get("table_id_t2") or "")
        page_t1 = table.get("page_t1")
        page_t2 = table.get("page_t2")
        changes = table.get("changes", []) or []

        def _legacy_status(status: str) -> str:
            s = str(status or "pending")
            if s == "approved" or s == "skipped":
                return REVIEW_STATUS_APPROVED
            if s == "rejected":
                return REVIEW_STATUS_REJECTED
            return REVIEW_STATUS_PENDING

        has_table_added = any(
            str(c.get("change_type", ""))
            in (ChangeType.TABLE_ADDED.value, "table_added")
            for c in changes
        )
        has_table_removed = any(
            str(c.get("change_type", ""))
            in (ChangeType.TABLE_REMOVED.value, "table_removed")
            for c in changes
        )
        table_bucket = _table_decision_bucket(table)
        review_status = (
            REVIEW_STATUS_APPROVED
            if table_bucket == "approved"
            else REVIEW_STATUS_REJECTED
            if table_bucket == "rejected"
            else REVIEW_STATUS_PENDING
        )

        indicators: list[dict[str, str]] = []
        item_type = "indicator"
        event_type = "matched_pair"
        if has_table_added or has_table_removed:
            change_type = (
                CHANGE_TYPE_TABLE_ADDED
                if has_table_added
                else CHANGE_TYPE_TABLE_REMOVED
            )
            event_type = (
                EVENT_TYPE_TABLE_ADDED if has_table_added else EVENT_TYPE_TABLE_REMOVED
            )
            summary_indicator = (
                "Tableau entier ajouté" if has_table_added else "Tableau entier retiré"
            )
        else:
            n_added = n_removed = n_renamed = 0
            for change in changes:
                ctype = str(change.get("change_type", ""))
                payload = change.get("payload", {}) or {}
                c_status = _legacy_status(change.get("validation_status", "pending"))
                if ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
                    n_added += 1
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", "")),
                            "type": CHANGE_TYPE_ADDED,
                            "review_status": c_status,
                        }
                    )
                elif ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
                    n_removed += 1
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", "")),
                            "type": CHANGE_TYPE_REMOVED,
                            "review_status": c_status,
                        }
                    )
                elif ctype in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
                    n_renamed += 1
                    from_val = str(payload.get("from", ""))
                    to_val = str(payload.get("to", ""))
                    indicators.append(
                        {
                            "name": f"{from_val} -> {to_val}".strip(" ->"),
                            "type": CHANGE_TYPE_RENAMED,
                            "from": from_val,
                            "to": to_val,
                            "review_status": c_status,
                        }
                    )
                elif "footnote" in ctype:
                    item_type = "footnote"
                    event_type = EVENT_TYPE_FOOTNOTE_ONLY
                    indicators.append(
                        {
                            "name": str(payload.get("indicator_name", ""))
                            or str(payload.get("new_text", "")),
                            "type": CHANGE_TYPE_MODIFIED,
                            "review_status": c_status,
                        }
                    )
            if n_removed >= n_added and n_removed >= n_renamed:
                change_type = CHANGE_TYPE_REMOVED
            elif n_added >= n_renamed:
                change_type = CHANGE_TYPE_ADDED
            else:
                change_type = CHANGE_TYPE_RENAMED
            summary_parts = []
            if n_added:
                summary_parts.append(f"{n_added} ajouté(s)")
            if n_removed:
                summary_parts.append(f"{n_removed} retiré(s)")
            if n_renamed:
                summary_parts.append(f"{n_renamed} renommé(s)")
            summary_indicator = ", ".join(summary_parts)

        items.append(
            ReviewItem(
                change_id=review_id,
                change_type=change_type,
                indicator=summary_indicator,
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t1=table_id_t1,
                table_id_t2=table_id_t2,
                page_t1=page_t1,
                page_t2=page_t2,
                source_ref_t1=str(table.get("source_pdf_t1", "")),
                source_ref_t2=str(table.get("source_pdf_t2", "")),
                review_status=review_status,
                confidence=float(table.get("confidence", 0.0) or 0.0),
                table_title_raw=str(table.get("table_title") or table_name),
                table_status=str(table.get("table_status", "")),
                indicators=indicators,
                match_method=str(table.get("match_method", "")),
                bbox_t1=table.get("bbox_t1"),
                bbox_t2=table.get("bbox_t2"),
                genai_analysis=table.get("genai_analysis") or {},
                match_metadata=table.get("match_metadata") or {},
                item_type=item_type,
                event_type=event_type,
            )
        )
    return items


@callback(
    Output("kpi-tables-matched", "children"),
    Output("kpi-added-indicators", "children"),
    Output("kpi-removed-indicators", "children"),
    Output("kpi-renamed-indicators", "children"),
    Input("store-indicator-result", "data"),
)
def render_main_kpis(indicator_result):
    """Render the main KPI cards."""
    if not indicator_result:
        return (
            _build_kpi_card(t("kpi_matched"), 0),
            _build_kpi_card(t("kpi_added"), 0),
            _build_kpi_card(t("kpi_removed"), 0),
            _build_kpi_card(t("kpi_renamed"), 0),
        )

    summary = indicator_result.get("summary", indicator_result.get("kpi_metier", {}))
    tables_matched = summary.get("tables_matched", 0)
    added = summary.get("total_added_indicators", 0)
    removed = summary.get("total_removed_indicators", 0)
    renamed = summary.get("total_renamed_indicators", 0)

    return (
        _build_kpi_card(t("kpi_matched"), tables_matched),
        _build_kpi_card(t("kpi_added"), added),
        _build_kpi_card(t("kpi_removed"), removed),
        _build_kpi_card(t("kpi_renamed"), renamed),
    )


@callback(
    Output("section-changes-header", "children"),
    Output("kpi-indicators-removed-detail", "children"),
    Output("kpi-indicators-added-detail", "children"),
    Output("kpi-validation-time", "children"),
    Input("store-indicator-result", "data"),
    Input("store-validation-duration-sec", "data"),
)
def render_secondary_kpis(indicator_result, validation_duration_sec):
    """Render the secondary KPI row with validation time."""
    if not indicator_result:
        return (
            f"Differences d'indicateurs (0 {t('tables')} avec changements)",
            _build_kpi_card(t("kpi_removed"), 0, delta_icon=None),
            _build_kpi_card(t("kpi_added"), 0, delta_icon=None),
            _build_kpi_card(t("validation_time"), _format_duration(None)),
        )

    # Count tables with changes
    comparisons = indicator_result.get("table_comparisons", [])
    tables_with_changes = [
        c
        for c in comparisons
        if (
            len(c.get("added_indicators", []))
            + len(c.get("removed_indicators", []))
            + len(c.get("renamed_indicators", []))
        )
        > 0
    ]
    n_tables = len(tables_with_changes)

    # Sum indicators
    total_added = sum(len(c.get("added_indicators", [])) for c in tables_with_changes)
    total_removed = sum(
        len(c.get("removed_indicators", [])) for c in tables_with_changes
    )

    header_text = (
        f"Differences d'indicateurs ({n_tables} {t('tables')} avec changements)"
    )

    return (
        header_text,
        _build_kpi_card(t("kpi_removed"), total_removed, delta_icon=None),
        _build_kpi_card(t("kpi_added"), total_added, delta_icon=None),
        _build_kpi_card(
            t("validation_time"), _format_duration(validation_duration_sec)
        ),
    )


@callback(
    Output("results-sections-tab", "children"),
    Input("store-indicator-result", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_sections_tab(indicator_result, show_results):
    """Render the section-based changes tab with accordion."""
    if not show_results:
        raise PreventUpdate
    from app.dash_app.layouts.page_results import build_section_accordion_item

    if not indicator_result:
        return html.Div("Aucun resultat disponible.", className="text-muted")

    comparisons = indicator_result.get("table_comparisons", [])
    tables_added = indicator_result.get("tables_added", [])
    tables_removed = indicator_result.get("tables_removed", [])

    # Group by section
    sections: dict[str, dict] = {}

    for comp in comparisons:
        section = comp.get("section", "Autres")
        if section not in sections:
            sections[section] = {"changes": [], "added": [], "removed": []}
        fn_counts = comp.get("footnotes_counts", {})
        fn_total = sum(fn_counts.get(k, 0) for k in ("added", "removed", "modified"))
        n_changes = (
            len(comp.get("added_indicators", []))
            + len(comp.get("removed_indicators", []))
            + len(comp.get("renamed_indicators", []))
            + fn_total
        )
        if n_changes > 0:
            sections[section]["changes"].append(comp)

    for t in tables_added:
        section = t.get("section", "Autres")
        if section not in sections:
            sections[section] = {"changes": [], "added": [], "removed": []}
        sections[section]["added"].append(t)

    for t in tables_removed:
        section = t.get("section", "Autres")
        if section not in sections:
            sections[section] = {"changes": [], "added": [], "removed": []}
        sections[section]["removed"].append(t)

    if not sections:
        return html.Div("Aucun changement detecte.", className="text-muted")

    # Build accordion items
    accordion_items = []
    for i, (section_name, data) in enumerate(sorted(sections.items())):
        item = build_section_accordion_item(
            section_name=section_display_label(section_name),
            tables_with_changes=data["changes"],
            tables_added=data["added"],
            tables_removed=data["removed"],
            item_id=f"section-{i}",
        )
        accordion_items.append(item)

    # Determine which sections to expand by default (those with changes)
    active_items = [
        f"section-{i}"
        for i, (_, data) in enumerate(sorted(sections.items()))
        if data["changes"] or data["added"] or data["removed"]
    ]

    return dbc.Accordion(
        accordion_items,
        id="sections-accordion",
        active_item=active_items[:3]
        if active_items
        else None,  # Expand first 3 with changes
        always_open=True,
    )


@callback(
    Output("store-review-items", "data"),
    Output("store-review-queue", "data"),  # V2: deduplicated grouped queue
    Output("store-review-selection", "data"),
    Output("store-review-current-idx", "data"),
    Output("store-current-change-idx", "data"),  # V2: reset change index
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("store-indicator-result", "data"),
    Input("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def init_review_items(indicator_result, paths):
    """Construire les ReviewItems depuis indicator_result pour la revue.

    Also builds the V2 deduplicated review queue.
    Quality gate is not used for blocking; review items are always built from comparison results.
    """
    if not indicator_result or not paths:
        raise PreventUpdate

    path_t1 = paths.get("pdf_previous", "") or paths.get("pdf_t1", "")
    path_t2 = paths.get("pdf_current", "") or paths.get("pdf_t2", "")
    bank_code = str(indicator_result.get("bank_code", ""))
    quarter_from = quarter_label_from_payload(indicator_result, "previous")
    quarter_to = quarter_label_from_payload(indicator_result, "current")

    items = build_review_items_from_indicator_result(
        indicator_result,
        bank_code=bank_code,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
        pdf_path_t1=path_t1,
        pdf_path_t2=path_t2,
    )
    serialized = sort_review_items_by_priority([it.to_dict() for it in items])

    # V2: Build deduplicated review queue
    grouped_tables = build_normalized_review_queue(
        indicator_result,
        serialized,
        pdf_path_t1=path_t1,
        pdf_path_t2=path_t2,
    )
    serialized_v2 = [t.to_dict() for t in grouped_tables]

    total = len(serialized)
    total_v2 = len(serialized_v2)
    dedup_merged = max(0, total - total_v2)
    resolved_selection, sel_table_idx, sel_change_idx = _resolve_selection(
        serialized_v2, {"review_id": None, "change_id": None}
    )
    dbg = {
        "writer": "init_review_items",
        "trigger": "init",
        "from": None,
        "to": sel_table_idx,
        "total": total,
        "total_v2": total_v2,
        "dedup_merged": dedup_merged,
    }
    logger.info(
        "[init_review_items] v1_count=%s v2_count=%s dedup_merged=%s review_queue_v2_active=%s",
        total,
        total_v2,
        dedup_merged,
        REVIEW_QUEUE_V2_ACTIVE,
    )
    return (
        serialized,
        serialized_v2,
        resolved_selection,
        sel_table_idx,
        sel_change_idx,
        dbg,
    )


def _build_comparison_statement(item: ReviewItem) -> str:
    """Phrase d'interpretation metier pour un changement."""
    table = item.table_name or "(tableau inconnu)"
    table_id = (item.table_id_t2 or item.table_id_t1 or "").strip()
    table_label = f"Tableau n°{table_id}: {table}" if table_id else f"Tableau: {table}"
    if item.indicators:
        return f"{table_label} -- {item.indicator}"
    indicator = item.indicator or "(indicateur non disponible)"
    if item.change_type == CHANGE_TYPE_TABLE_ADDED:
        return f"Tableau entier ajouté au trimestre courant: {table_label}"
    if item.change_type == CHANGE_TYPE_TABLE_REMOVED:
        return (
            f"Tableau entier supprimé depuis le trimestre précédent: {table_label} "
            "(présent au trimestre précédent)"
        )
    if item.change_type == CHANGE_TYPE_ADDED:
        return (
            f"Ajout au trimestre courant: {indicator} -- absent au trimestre précédent."
        )
    if item.change_type == CHANGE_TYPE_REMOVED:
        return f"Suppression au trimestre courant: {indicator} -- présent au trimestre précédent."
    if item.change_type == CHANGE_TYPE_RENAMED:
        return f"Renommage entre trimestre précédent et trimestre courant: {indicator}."
    return f"Changement detecte: {indicator}"


def _filter_noise(items: list[str]) -> list[str]:
    """Filter out noise lines (dates, units, footnotes) using normalize_indicator_for_comparison."""
    return [
        x for x in items if x and normalize_indicator_for_comparison(str(x).strip())
    ]


def _get_proof_image_b64_for_item(
    item_dict: dict, side: str, paths: dict, *, proof_display_mode: str = "crop"
) -> str | None:
    """Get proof image base64. With proof_display_mode='full' skip crop (full page); with 'crop' use bbox; with 'footnote' show only footnote region."""
    display_mode = (proof_display_mode or "crop").strip().lower()

    table_status = (item_dict.get("table_status") or "").strip().lower()
    if table_status == "stable" and display_mode == "crop":
        return _get_proof_image_b64(item_dict, side, paths)

    proof_image_path = item_dict.get("proof_image_path", "") or ""
    if side == "t2" and proof_image_path and display_mode == "crop":
        return None

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    path_t1 = paths.get("pdf_t1", "") if paths else ""
    path_t2 = paths.get("pdf_t2", "") if paths else ""
    pdf_path = path_t1 if side == "t1" else path_t2
    if not pdf_path:
        pdf_path = ref

    # For "footnote" or "full" modes, always render from PDF (ignore pre-existing images)
    if display_mode in ("footnote", "full"):
        if not pdf_path or page is None:
            return _get_proof_image_b64(item_dict, side, paths)

        page_effective = max(1, int(page))
        bbox = item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2")
        bbox_key = ""
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            bbox_key = json.dumps(bbox)
        try:
            raw_bytes = _cached_render_or_crop(
                str(pdf_path), page_effective, 1.5, bbox_key, display_mode
            )
            if raw_bytes:
                return base64.b64encode(raw_bytes).decode("ascii")
        except Exception as e:
            logger.warning("Render failed for mode %s: %s", display_mode, e)
        return _get_proof_image_b64(item_dict, side, paths)

    # For "crop" mode, use existing images if available
    base_img_b64: str | None = None
    if (
        side == "t1"
        and proof_image_path
        and Path(proof_image_path).exists()
        and Path(proof_image_path).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(proof_image_path, "rb") as f:
                raw = f.read()
            base_img_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
    if (
        base_img_b64 is None
        and ref
        and Path(ref).exists()
        and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            base_img_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    if base_img_b64 is None:
        if not pdf_path or page is None:
            return _get_proof_image_b64(item_dict, side, paths)

        page_effective = max(1, int(page))
        bbox = item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2")
        bbox_key = ""
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            bbox_key = json.dumps(bbox)
        try:
            raw_bytes = _cached_render_or_crop(
                str(pdf_path), page_effective, 1.5, bbox_key, display_mode
            )
            base_img_b64 = (
                base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else None
            )
        except Exception as e:
            logger.warning("Render/crop failed: %s", e)
            base_img_b64 = None

    if not base_img_b64:
        return _get_proof_image_b64(item_dict, side, paths)

    all_t1 = _filter_noise(list(item_dict.get("all_indicators_t1") or []))
    all_t2 = _filter_noise(list(item_dict.get("all_indicators_t2") or []))
    rows_added = _filter_noise(list(item_dict.get("added_indicators") or []))
    rows_removed = _filter_noise(list(item_dict.get("removed_indicators") or []))

    try:
        annotated = annotate_table_with_changes(
            image_base64=base_img_b64,
            rows_added=rows_added,
            rows_removed=rows_removed,
            all_indicators_t1=all_t1,
            all_indicators_t2=all_t2,
            is_for_t1=(side == "t1"),
            row_bboxes_t1=None,
            row_bboxes_t2=None,
        )
        if annotated:
            return base64.b64encode(annotated).decode("ascii")
    except Exception as e:
        logger.warning("Annotation failed, returning non-annotated image: %s", e)

    return base_img_b64


def _get_proof_image_b64(item_dict: dict, side: str, paths: dict) -> str | None:
    """Obtenir l'image base64 pour T1 ou T2 (PDF page ou fichier PNG)."""
    import base64

    proof_image_path = item_dict.get("proof_image_path", "") or ""
    if (
        side == "t1"
        and proof_image_path
        and Path(proof_image_path).exists()
        and Path(proof_image_path).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(proof_image_path, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
    if side == "t2" and proof_image_path:
        # Une preuve tableau entier couvre les deux periodes.
        return None

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    path_t1 = paths.get("pdf_t1", "") if paths else ""
    path_t2 = paths.get("pdf_t2", "") if paths else ""

    pdf_path = path_t1 if side == "t1" else path_t2
    if not pdf_path:
        pdf_path = ref

    if (
        ref
        and Path(ref).exists()
        and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    if pdf_path and page is not None:
        page_effective = max(1, int(page))
        raw = get_pdf_preview(pdf_path, page_effective, scale=1.5)
        if raw:
            return base64.b64encode(raw).decode("ascii")
    return None


@callback(
    Output("results-review-tab", "children"),
    Input("store-review-items", "data"),
    Input("store-review-current-idx", "data"),
    Input("store-pdf-paths", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_review_tab(review_items_data, current_idx, paths, show_results):
    """Rendre l'onglet Revue et validation."""
    if not show_results:
        raise PreventUpdate
    if not review_items_data:
        return html.Div(
            "Aucun changement a revoir. La revue est disponible lorsque la comparaison "
            "indicateurs produit des changements.",
            className="text-muted",
        )

    items = [ReviewItem.from_dict(d) for d in review_items_data]
    idx = max(0, min(int(current_idx or 0), len(items) - 1))
    current = items[idx]

    approved = sum(1 for it in items if it.review_status == REVIEW_STATUS_APPROVED)
    rejected = sum(1 for it in items if it.review_status == REVIEW_STATUS_REJECTED)
    pending = sum(1 for it in items if it.review_status == REVIEW_STATUS_PENDING)

    current_dict = review_items_data[idx]
    proof_image_path = current_dict.get("proof_image_path", "") or ""
    proof_mode = current_dict.get("proof_mode", "") or ""

    img_t1_b64 = _get_proof_image_b64_for_item(
        current_dict, "t1", paths or {}, proof_display_mode="crop"
    )
    img_t2_b64 = _get_proof_image_b64_for_item(
        current_dict, "t2", paths or {}, proof_display_mode="crop"
    )

    def _img_div(b64, label, caption):
        if b64:
            src = f"data:image/png;base64,{b64}"
            return html.Div(
                [
                    html.P(label, className="small text-muted mb-1"),
                    html.Img(
                        src=src, style={"maxWidth": "100%", "border": "1px solid #ddd"}
                    ),
                    html.P(caption, className="small text-muted mt-1")
                    if caption
                    else None,
                ],
                className="mb-2",
            )
        return html.Div(
            [
                html.P(label, className="small text-muted"),
                html.P("Preview non disponible", className="text-muted"),
            ]
        )

    if proof_image_path:
        col1_content = _img_div(
            img_t1_b64,
            "Preuve tableau entier courant/précédent",
            f"Mode {proof_mode}" if proof_mode else None,
        )
        col2_content = html.Div(
            [
                html.P("Trimestre précédent", className="small text-muted"),
                html.P("Inclus dans la preuve tableau entier."),
            ]
        )
    else:
        col1_content = _img_div(
            img_t2_b64,
            "Trimestre courant",
            f"Page {current.page_t2}" if current.page_t2 else None,
        )
        col2_content = _img_div(
            img_t1_b64,
            "Trimestre précédent",
            f"Page {current.page_t1}" if current.page_t1 else None,
        )
    unit_context_t1 = current_dict.get("unit_context_t1", "") or ""
    unit_context_t2 = current_dict.get("unit_context_t2", "") or ""
    title_method_t1 = current_dict.get("title_resolution_method_t1", "") or ""
    title_method_t2 = current_dict.get("title_resolution_method_t2", "") or ""
    table_title_raw = current_dict.get("table_title_raw", "") or ""

    meta_lines = []
    if proof_image_path:
        meta_lines.append(
            html.P(
                f"Preuve image: {proof_image_path}",
                className="small text-muted mb-1",
            )
        )
    if unit_context_t1 or unit_context_t2:
        meta_lines.append(
            html.P(
                "Contexte unité précédent/courant: "
                f"{unit_context_t1 or '-'} | {unit_context_t2 or '-'}",
                className="small text-muted mb-1",
            )
        )
    if title_method_t1 or title_method_t2:
        meta_lines.append(
            html.P(
                "Méthode titre précédent/courant: "
                f"{title_method_t1 or '-'} | {title_method_t2 or '-'}",
                className="small text-muted mb-1",
            )
        )
    if table_title_raw:
        meta_lines.append(
            html.P(f"Titre brut: {table_title_raw}", className="small text-muted mb-1")
        )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Precedent",
                            id="btn-review-prev",
                            color="secondary",
                            size="sm",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Suivant",
                            id="btn-review-next",
                            color="secondary",
                            size="sm",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(
                            f"Changement {idx + 1} / {len(items)}",
                            className="text-muted",
                        ),
                        width="auto",
                    ),
                ],
                className="mb-2 align-items-center g-2",
            ),
            html.P(
                f"{approved} valides | {rejected} rejetes | {pending} en attente",
                className="small text-muted mb-2",
            ),
            html.Hr(),
            dbc.Alert(_build_comparison_statement(current), color="success"),
            html.Div(meta_lines, className="mb-2") if meta_lines else html.Div(),
            dbc.Row(
                [
                    dbc.Col(col1_content, md=6),
                    dbc.Col(col2_content, md=6),
                ],
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Valider",
                            id="btn-review-approve",
                            color="success",
                            size="sm",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Rejeter", id="btn-review-reject", color="danger", size="sm"
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Passer", id="btn-review-pass", color="secondary", size="sm"
                        ),
                        width="auto",
                    ),
                ],
                className="g-2",
            ),
        ],
    )


@callback(
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("btn-review-prev", "n_clicks"),
    Input("btn-review-next", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-current-idx", "data"),
    prevent_initial_call=True,
)
def on_review_navigate(prev_clicks, next_clicks, review_items, current_idx):
    """Navigation Precedent/Suivant dans la revue (legacy buttons)."""
    from dash import ctx

    if not ctx.triggered_id or not review_items:
        raise PreventUpdate
    n = len(review_items)
    idx = int(current_idx or 0)
    if ctx.triggered_id == "btn-review-prev":
        idx = max(0, idx - 1)
    elif ctx.triggered_id == "btn-review-next":
        idx = min(n - 1, idx + 1)
    else:
        raise PreventUpdate
    dbg = {
        "writer": "on_review_navigate",
        "trigger": ctx.triggered_id,
        "from": current_idx,
        "to": idx,
        "total": n,
    }
    logger.info(
        "[on_review_navigate] trig=%s current_idx=%r total=%s -> new_idx=%s",
        ctx.triggered_id,
        current_idx,
        n,
        idx,
    )
    return idx, dbg


@callback(
    Output("store-review-items", "data", allow_duplicate=True),
    Input("btn-review-approve", "n_clicks"),
    Input("btn-review-reject", "n_clicks"),
    Input("btn-review-pass", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-current-idx", "data"),
    prevent_initial_call=True,
)
def on_review_status(
    approve_clicks, reject_clicks, pass_clicks, review_items, current_idx
):
    """Appliquer Valider/Rejeter/Passer sur l'item courant."""
    import json

    from dash import ctx

    if not ctx.triggered_id or not review_items:
        raise PreventUpdate

    idx = max(0, min(int(current_idx or 0), len(review_items) - 1))
    status_map = {
        "btn-review-approve": REVIEW_STATUS_APPROVED,
        "btn-review-reject": REVIEW_STATUS_REJECTED,
        "btn-review-pass": REVIEW_STATUS_PENDING,
    }
    status = status_map.get(ctx.triggered_id)
    if not status:
        raise PreventUpdate

    item = ReviewItem.from_dict(review_items[idx])
    updated = set_review_status(item, status)

    # Deep copy to ensure Dash detects the change
    new_items = json.loads(json.dumps(review_items))
    new_items[idx] = updated.to_dict()
    return new_items


@callback(
    Output("results-table-tab", "children"),
    Input("store-indicator-result", "data"),
    Input("store-comparison-result", "data"),
    Input("store-show-results-page", "data"),
    prevent_initial_call=True,
)
def render_table_tab(indicator_result, comparison_result, show_results):
    """Rendre l'onglet Tableau Analyse avec les changements."""
    if not show_results:
        raise PreventUpdate
    include_uncertain = False
    include_review_status = False
    if indicator_result:
        rows = build_indicator_change_rows(
            indicator_result,
            include_uncertain=include_uncertain,
            include_review_status=include_review_status,
        )
        if not rows:
            return html.Div("Aucun changement a afficher.", className="text-muted")
        headers = list(rows[0].keys()) if rows else []
        header_row = html.Tr([html.Th(h) for h in headers])
        body_rows = [
            html.Tr([html.Td(str(row.get(h, ""))) for h in headers]) for row in rows
        ]
        return html.Div(
            [
                html.P(f"{len(rows)} changement(s) detecte(s)", className="mb-2"),
                dbc.Table(
                    [html.Thead(header_row), html.Tbody(body_rows)],
                    bordered=True,
                    striped=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )
    if comparison_result and is_canonical_comparison(comparison_result):
        rows = build_indicator_change_rows(
            comparison_result,
            include_uncertain=include_uncertain,
            include_review_status=include_review_status,
        )
        if not rows:
            return html.Div("Aucun changement a afficher.", className="text-muted")
        headers = list(rows[0].keys()) if rows else []
        header_row = html.Tr([html.Th(h) for h in headers])
        body_rows = [
            html.Tr([html.Td(str(row.get(h, ""))) for h in headers]) for row in rows
        ]
        return html.Div(
            [
                html.P(f"{len(rows)} changement(s) detecte(s)", className="mb-2"),
                dbc.Table(
                    [html.Thead(header_row), html.Tbody(body_rows)],
                    bordered=True,
                    striped=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )
    if comparison_result:
        changes = comparison_result.get("changes", [])
        if not changes:
            return html.Div(
                "Aucun changement structurel detecte.", className="text-muted"
            )
        flat = []
        for c in changes:
            for ind in c.get("rows_added", []):
                flat.append(
                    {
                        "Type": "Ajout",
                        "Phrase": ind[:80] + "..." if len(ind) > 80 else ind,
                        "Page": c.get("page_t2"),
                        "Tableau": c.get("table_title", ""),
                    }
                )
            for ind in c.get("rows_removed", []):
                flat.append(
                    {
                        "Type": "Suppression",
                        "Phrase": ind[:80] + "..." if len(ind) > 80 else ind,
                        "Page": c.get("page_t1"),
                        "Tableau": c.get("table_title", ""),
                    }
                )
        if not flat:
            flat = [
                {
                    "Titre": c.get("titre", c.get("table_title", "")),
                    "Page": c.get("page", ""),
                    "Phrase": str(c.get("phrase", ""))[:80],
                }
                for c in changes[:50]
            ]
        headers = list(flat[0].keys()) if flat else []
        header_row = html.Tr([html.Th(h) for h in headers])
        body_rows = [
            html.Tr([html.Td(str(r.get(h, ""))) for h in headers]) for r in flat
        ]
        return html.Div(
            [
                html.P(f"{len(changes)} changement(s) structurel(s)", className="mb-2"),
                dbc.Table(
                    [html.Thead(header_row), html.Tbody(body_rows)],
                    bordered=True,
                    striped=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )
    return html.Div("Aucun resultat a afficher.", className="text-muted")


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
    base_name = _comparison_export_base_name(ir, "review")

    content = [html.H5("Exporter les resultats")]
    if review_items_data or ir:
        content.append(
            html.Div(
                [
                    html.P("Export revue (avec statuts validation)", className="mb-2"),
                    dbc.Button(
                        "Telecharger CSV (avec statuts validation)",
                        id="btn-download-review-csv",
                        color="primary",
                        className="me-2 mb-2",
                    ),
                    dbc.Button(
                        "Telecharger JSON (comparaison + statuts validation)",
                        id="btn-download-review-json",
                        color="primary",
                        outline=True,
                        className="me-2 mb-2",
                    ),
                    dbc.Button(
                        "Telecharger Excel (avec statuts validation)",
                        id="btn-download-review-excel",
                        color="primary",
                        outline=True,
                        className="me-2 mb-2",
                    ),
                ],
                className="mb-4",
            )
        )
        content.append(
            html.P(
                "Les exports contiennent les changements avec leurs statuts de validation (valide, rejete, en attente).",
                className="small text-muted",
            )
        )
    if ir:
        content.append(
            html.Div(
                [
                    html.P(
                        "Export payload canonical complet (summary, status_counts, structure_change_detected)",
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Telecharger JSON brut",
                        id="btn-download-indicator-json-brut",
                        color="secondary",
                        outline=True,
                        className="mb-2",
                    ),
                ],
                className="mb-4",
            )
        )
        if not review_items_data:
            content.append(
                html.P(
                    "Aucun changement a exporter. La revue genere des exports lorsque des changements sont detectes.",
                    className="text-muted",
                )
            )
    else:
        content.append(
            html.P(
                "Export revue disponible pour le mode indicateurs. Chargez une comparaison indicateurs ou lancez une analyse.",
                className="text-muted",
            )
        )
    return html.Div(content)


@callback(
    Output("download-review-csv", "data"),
    Input("btn-download-review-csv", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-queue", "data"),
    State("store-indicator-result", "data"),
    State("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def on_download_csv(
    n_clicks, review_items_data, review_queue_data, indicator_result, paths
):
    """Telecharger le CSV de validation (Excel FR, UTF-8 BOM, separateur ;)."""
    if not n_clicks:
        raise PreventUpdate
    ir = indicator_result or {}
    items = []
    if review_queue_data:
        items = _review_items_from_v2_queue(review_queue_data)
    elif review_items_data:
        try:
            items = [ReviewItem.from_dict(d) for d in review_items_data]
        except Exception:
            pass
    if not items and ir:
        # Reconstruire depuis indicator_result si store desynchronise
        paths = paths or {}
        path_t1 = (
            paths.get("pdf_previous", "") or paths.get("pdf_t1", "")
            if isinstance(paths, dict)
            else ""
        )
        path_t2 = (
            paths.get("pdf_current", "") or paths.get("pdf_t2", "")
            if isinstance(paths, dict)
            else ""
        )
        items = build_review_items_from_indicator_result(
            ir,
            bank_code=str(ir.get("bank_code", "")),
            quarter_from=quarter_label_from_payload(ir, "previous"),
            quarter_to=quarter_label_from_payload(ir, "current"),
            pdf_path_t1=path_t1 or "",
            pdf_path_t2=path_t2 or "",
        )
    bank = str(ir.get("bank_code", "bank")).upper()
    q_from = quarter_label_from_payload(ir, "previous").upper()
    q_to = quarter_label_from_payload(ir, "current").upper()
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_to}_vs_{q_from}_{year_val}.csv"
    csv_str = generate_validation_csv(items, ir)
    return dict(content=csv_str, filename=filename)


@callback(
    Output("download-review-json", "data"),
    Input("btn-download-review-json", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-queue", "data"),
    State("store-indicator-result", "data"),
    prevent_initial_call=True,
)
def on_download_json(n_clicks, review_items_data, review_queue_data, indicator_result):
    """Telecharger le JSON de revue."""
    if not n_clicks or (not review_items_data and not review_queue_data):
        raise PreventUpdate
    from datetime import datetime

    ir = indicator_result or {}
    if review_queue_data:
        items = _review_items_from_v2_queue(review_queue_data)
    else:
        items = [ReviewItem.from_dict(d) for d in review_items_data]
    bank = str(ir.get("bank_code", "bank"))
    q_from = quarter_label_from_payload(ir, "previous")
    q_to = quarter_label_from_payload(ir, "current")
    year_val = str(ir.get("year", "2025"))
    base_name = _comparison_export_base_name(ir, "review").replace(" ", "_").lower()
    json_str = export_review_items_json_fr(
        items,
        metadata={
            "bank_code": bank,
            "quarter_from": q_from,
            "quarter_to": q_to,
            "previous_quarter": q_from,
            "current_quarter": q_to,
            "comparison_direction": "current_vs_previous",
            "year": year_val,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return dict(content=json_str, filename=f"{base_name}.json")


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
    items = []
    if review_queue_data:
        items = _review_items_from_v2_queue(review_queue_data)
    elif review_items_data:
        try:
            items = [ReviewItem.from_dict(d) for d in review_items_data]
        except Exception:
            pass
    if not items and ir:
        paths = paths or {}
        path_t1 = (
            paths.get("pdf_previous", "") or paths.get("pdf_t1", "")
            if isinstance(paths, dict)
            else ""
        )
        path_t2 = (
            paths.get("pdf_current", "") or paths.get("pdf_t2", "")
            if isinstance(paths, dict)
            else ""
        )
        items = build_review_items_from_indicator_result(
            ir,
            bank_code=str(ir.get("bank_code", "")),
            quarter_from=quarter_label_from_payload(ir, "previous"),
            quarter_to=quarter_label_from_payload(ir, "current"),
            pdf_path_t1=path_t1 or "",
            pdf_path_t2=path_t2 or "",
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
    Output("download-indicator-json-brut", "data"),
    Input("btn-download-indicator-json-brut", "n_clicks"),
    State("store-indicator-result", "data"),
    prevent_initial_call=True,
)
def on_download_indicator_json_brut(n_clicks, indicator_result):
    """Telecharger le payload canonical complet en JSON."""
    if not n_clicks or not indicator_result:
        raise PreventUpdate
    import json

    bank = str(indicator_result.get("bank_code", "bank"))
    base_name = (
        _comparison_export_base_name(indicator_result, "canonical")
        .replace(" ", "_")
        .lower()
    )
    json_str = json.dumps(indicator_result, ensure_ascii=False, indent=2)
    return dict(content=json_str, filename=f"{base_name}.json")


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("store-review-items", "data", allow_duplicate=True),
    Output("store-review-queue", "data", allow_duplicate=True),  # V2
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),  # V2
    Output("main-content", "children", allow_duplicate=True),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Output("store-review-filters", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def on_reset(n_clicks):
    """Reinitialiser pour nouvelle analyse."""
    if n_clicks:
        dbg = {
            "writer": "on_reset",
            "trigger": "btn-reset",
            "from": None,
            "to": 0,
            "total": None,
        }
        logger.info("[on_reset] -> idx=0")
        return (
            None,
            None,
            None,
            False,
            None,
            None,  # store-review-queue
            {"review_id": None, "change_id": None},
            0,
            0,  # store-current-change-idx
            build_page_upload(),
            False,
            {"section": "all", "status": "all"},
            dbg,
        )
    raise PreventUpdate


@callback(
    Output("collapse-options", "is_open"),
    Input("btn-toggle-options", "n_clicks"),
    State("collapse-options", "is_open"),
    prevent_initial_call=True,
)
def toggle_options(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("collapse-stats", "is_open"),
    Input("btn-toggle-stats", "n_clicks"),
    State("collapse-stats", "is_open"),
    prevent_initial_call=True,
)
def toggle_stats(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("load-comparison-dropdown", "options"),
    Input("store-detection", "data"),
)
def populate_load_options(_detection):
    """Remplir le dropdown avec les comparaisons disponibles (canoniques)."""
    return get_available_indicator_comparison_options()


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("main-content", "children", allow_duplicate=True),
    Output("notification", "children", allow_duplicate=True),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Input("btn-load-comparison", "n_clicks"),
    State("load-comparison-dropdown", "value"),
    prevent_initial_call=True,
)
def on_load_comparison(n_clicks, filename):
    """Charger une comparaison existante (canonique ou metier)."""
    if not n_clicks or not filename:
        raise PreventUpdate

    filepath = INDICATOR_COMPARISON_DIR / filename
    data = load_comparison_result(filepath)
    if not data:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dbc.Alert(f"Impossible de charger {filename}", color="danger"),
            no_update,
        )

    if data.get("result_type") == "metier_tableaux":
        return (
            None,
            data,
            {"compare_path": str(filepath), "source": "load"},
            True,
            build_page_results(),
            dbc.Alert(f"Comparaison chargee: {filename}", color="success"),
            True,
        )

    canonical = to_canonical_payload(data)
    if not is_canonical_comparison(canonical):
        canonical = data

    return (
        canonical,
        canonical,
        {"compare_path": str(filepath), "source": "load"},
        True,
        build_page_results(),
        dbc.Alert(f"Comparaison chargee: {filename}", color="success"),
        True,
    )


# =============================================================================
# V2 CALLBACKS: Per-Change Validation in Grouped Review Queue
# =============================================================================


@callback(
    Output("store-review-queue", "data", allow_duplicate=True),
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Input("btn-approve-change-v2", "n_clicks"),
    Input("btn-reject-change-v2", "n_clicks"),
    Input("btn-skip-change-v2", "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    State("validation-notes-v2", "value"),
    prevent_initial_call=True,
)
def on_validate_change_v2(approve, reject, skip, queue, selection, filters, notes):
    """Apply validation to current change in V2 queue, auto-advance.

    Auto-advance rules:
    - After decision, advance to next change in same table
    - When last change of table is decided, auto-advance to next table
    """
    from datetime import datetime

    if not ctx.triggered_id or not queue:
        raise PreventUpdate

    action_map = {
        "btn-approve-change-v2": "approved",
        "btn-reject-change-v2": "rejected",
        "btn-skip-change-v2": "skipped",
    }
    decision = action_map.get(ctx.triggered_id)
    if not decision:
        raise PreventUpdate

    resolved_selection, table_idx, change_idx = _resolve_selection(
        queue, selection, filters
    )
    if table_idx < 0:
        raise PreventUpdate

    new_queue = json.loads(json.dumps(queue))
    table = new_queue[table_idx]
    changes = table.get("changes", [])
    selected_change_id = str(resolved_selection.get("change_id") or "")
    if selected_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == selected_change_id:
                change_idx = idx
                break
    if change_idx >= len(changes) or change_idx < 0:
        raise PreventUpdate

    # Apply validation on selected change
    changes[change_idx]["validation_status"] = decision
    changes[change_idx]["validation_decision"] = decision
    changes[change_idx]["validation_notes"] = notes or ""
    changes[change_idx]["validated_at"] = datetime.utcnow().isoformat()
    changes[change_idx]["validated_by"] = "analyst"

    # Update table status
    table["changes"] = changes
    n_pending = sum(
        1 for c in changes if c.get("validation_status", "pending") == "pending"
    )
    if n_pending == 0:
        table["table_status"] = "completed"
    else:
        n_validated = sum(
            1
            for c in changes
            if c.get("validation_status", "pending")
            in ("approved", "rejected", "skipped")
        )
        table["table_status"] = "partial" if n_validated > 0 else "pending"

    # Recompute summary
    summary = {
        "total_changes": len(changes),
        "indicators_added": sum(
            1 for c in changes if c.get("change_type") == "indicator_added"
        ),
        "indicators_removed": sum(
            1 for c in changes if c.get("change_type") == "indicator_removed"
        ),
        "indicators_renamed": sum(
            1 for c in changes if c.get("change_type") == "indicator_renamed"
        ),
        "footnotes_changed": sum(
            1 for c in changes if "footnote" in c.get("change_type", "")
        ),
        "validated": len(changes) - n_pending,
        "pending": n_pending,
    }
    table["summary"] = summary

    # Auto-advance logic based on filtered visible queue
    next_selection = dict(resolved_selection)
    if change_idx + 1 < len(changes):
        next_selection["change_id"] = str(changes[change_idx + 1].get("change_id", ""))
    else:
        visible_ids = _visible_review_ids(new_queue, filters) or [
            _review_id(t) for t in new_queue if _review_id(t)
        ]
        current_review_id = str(resolved_selection.get("review_id") or "")
        if current_review_id in visible_ids:
            pos = visible_ids.index(current_review_id)
            if pos + 1 < len(visible_ids):
                next_review_id = visible_ids[pos + 1]
                _, next_table = _get_table_by_review_id(new_queue, next_review_id)
                if next_table is not None:
                    next_selection = {
                        "review_id": next_review_id,
                        "change_id": _resolve_change_id(next_table, None),
                    }
            else:
                next_selection["change_id"] = _resolve_change_id(table, None)
        else:
            next_selection["change_id"] = _resolve_change_id(table, None)

    next_selection, new_table_idx, new_change_idx = _resolve_selection(
        new_queue, next_selection, filters
    )

    logger.info(
        "[on_validate_change_v2] review_id=%s change=%s decision=%s -> review_id=%s change=%s",
        resolved_selection.get("review_id"),
        resolved_selection.get("change_id"),
        decision,
        next_selection.get("review_id"),
        next_selection.get("change_id"),
    )
    logger.debug(
        "[on_validate_change_v2] table_idx=%d change_idx=%d -> table_idx=%d change_idx=%d",
        table_idx,
        change_idx,
        new_table_idx,
        new_change_idx,
    )

    return new_queue, next_selection, new_change_idx, new_table_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("btn-prev-change-v2", "n_clicks"),
    Input("btn-next-change-v2", "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_navigate_change_v2(prev, next_c, queue, selection, filters):
    """Navigate prev/next within current table's changes (V2)."""
    if not ctx.triggered_id or not queue:
        raise PreventUpdate

    resolved_selection, table_idx, change_idx = _resolve_selection(
        queue, selection, filters
    )
    if table_idx < 0:
        raise PreventUpdate

    table = queue[table_idx]
    changes = table.get("changes", [])
    n_changes = len(changes)
    selected_change_id = str(resolved_selection.get("change_id") or "")
    if selected_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == selected_change_id:
                change_idx = idx
                break

    if ctx.triggered_id == "btn-prev-change-v2":
        new_idx = max(0, change_idx - 1)
    elif ctx.triggered_id == "btn-next-change-v2":
        new_idx = min(n_changes - 1, change_idx + 1) if n_changes > 0 else 0
    else:
        raise PreventUpdate

    new_change_id = (
        str(changes[new_idx].get("change_id", ""))
        if 0 <= new_idx < len(changes)
        else None
    )
    new_selection = {
        "review_id": resolved_selection.get("review_id"),
        "change_id": new_change_id,
    }
    new_selection, _, new_change_idx = _resolve_selection(queue, new_selection, filters)
    return new_selection, new_change_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input({"type": "change-row-v2", "change_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_change_row_click(n_clicks, queue, selection, filters):
    """Select a specific change row by stable change_id."""
    if not ctx.triggered_id or not queue:
        raise PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or trig.get("type") != "change-row-v2":
        raise PreventUpdate
    change_id = str(trig.get("change_id") or "")
    if not change_id:
        raise PreventUpdate

    resolved_selection, table_idx, _ = _resolve_selection(queue, selection, filters)
    if table_idx < 0:
        raise PreventUpdate
    table = queue[table_idx]
    if not any(
        str(c.get("change_id", "")) == change_id
        for c in (table.get("changes", []) or [])
    ):
        raise PreventUpdate
    new_selection = {
        "review_id": resolved_selection.get("review_id"),
        "change_id": change_id,
    }
    new_selection, _, new_change_idx = _resolve_selection(queue, new_selection, filters)
    return new_selection, new_change_idx


@callback(
    Output("store-review-selection", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-current-change-idx", "data", allow_duplicate=True),
    Input("btn-prev-table-v2", "n_clicks"),
    Input("btn-next-table-v2", "n_clicks"),
    Input({"type": "queue-table-item-v2", "review_id": ALL}, "n_clicks"),
    State("store-review-queue", "data"),
    State("store-review-selection", "data"),
    State("store-review-filters", "data"),
    prevent_initial_call=True,
)
def on_navigate_table_v2(prev, next_t, clicks, queue, selection, filters):
    """Navigate between tables (V2) using stable review IDs."""
    if not queue:
        raise PreventUpdate

    resolved_selection, table_idx, _ = _resolve_selection(queue, selection, filters)
    visible_ids = _visible_review_ids(queue, filters) or [
        _review_id(t) for t in queue if _review_id(t)
    ]
    if not visible_ids:
        raise PreventUpdate

    current_review_id = str(resolved_selection.get("review_id") or "")
    if current_review_id not in visible_ids:
        current_review_id = visible_ids[0]
    pos = visible_ids.index(current_review_id)

    target_review_id = current_review_id
    if ctx.triggered_id == "btn-prev-table-v2":
        target_review_id = visible_ids[max(0, pos - 1)]
    elif ctx.triggered_id == "btn-next-table-v2":
        target_review_id = visible_ids[min(len(visible_ids) - 1, pos + 1)]
    elif (
        isinstance(ctx.triggered_id, dict)
        and ctx.triggered_id.get("type") == "queue-table-item-v2"
    ):
        clicked_review_id = str(ctx.triggered_id.get("review_id") or "")
        if clicked_review_id:
            target_review_id = clicked_review_id
    else:
        raise PreventUpdate

    _, target_table = _get_table_by_review_id(queue, target_review_id)
    new_selection = {
        "review_id": target_review_id,
        "change_id": _resolve_change_id(target_table or {}, None),
    }
    new_selection, new_table_idx, new_change_idx = _resolve_selection(
        queue, new_selection, filters
    )
    return new_selection, new_table_idx, new_change_idx


@callback(
    Output("btn-next-table-v2", "disabled"),
    Input("store-review-queue", "data"),
    Input("store-review-selection", "data"),
)
def block_next_table_until_complete_v2(queue, selection):
    """Disable 'Next Table' if current table has required pending changes."""
    if not queue:
        return False

    _, table = _get_table_by_review_id(queue, (selection or {}).get("review_id"))
    if table is None:
        return False

    changes = table.get("changes", [])

    # Block if any required change is still pending
    for c in changes:
        if (
            c.get("is_required", True)
            and c.get("validation_status", "pending") == "pending"
        ):
            return True

    return False


if __name__ == "__main__":
    app.run(debug=True, port=8050)
