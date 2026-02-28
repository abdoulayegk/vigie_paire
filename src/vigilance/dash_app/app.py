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
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _cached_render_or_crop(pdf_path: str, page: int, scale: float, bbox_key: str) -> bytes:
    """Return PNG bytes: cropped if bbox_key valid, else full page. Cached for UI performance."""
    if not bbox_key:
        raw = get_pdf_preview(pdf_path, page, scale=scale)
        return raw if raw else b""
    try:
        bbox = json.loads(bbox_key)
        if isinstance(bbox, list) and len(bbox) == 4:
            return crop_table_region_to_bytes(pdf_path, page, bbox, scale=scale)
    except (json.JSONDecodeError, TypeError):
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
    dcc,
    html,
    no_update,
    ctx
)
from dash.exceptions import PreventUpdate

from app.i18n import t
from app.comparison_canonical import (
    get_meta_value,
    is_canonical_comparison,
    to_canonical_payload,
)
from app.comparison_runner import run_comparison_with_sections
from app.dash_app.components.pdf_preview import pdf_images_from_base64
from app.dash_app.components.review_queue import build_review_queue
from app.dash_app.components.review_detail import build_review_detail
from app.dash_app.layouts import (
    build_page_results,
    build_page_upload,
    build_page_validation,
    build_sidebar,
)
from app.dash_app.layouts.page_results import build_analyst_kpi_card
from app.review_adapters import build_review_items_from_indicator_result
from app.review_export import (
    export_review_items_json_fr,
    generate_validation_csv,
    generate_validation_excel,
)
from app.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    ReviewItem,
)
from app.review_state import set_review_status
from app.ui_config import INDICATOR_COMPARISON_DIR
from app.ui_detection import (
    _detect_sections_core,
    get_pdf_preview,
    get_section_preview_images,
)
from app.ui_indicators import build_indicator_change_rows, run_indicator_auto_pipeline
from app.ui_io import (
    get_available_indicator_comparison_options,
    load_comparison_result,
    save_pdfs_to_temp,
)
from vigilance.extraction.table_annotator import annotate_table_with_changes
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.pdf_crop import crop_table_region_to_bytes

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
    dcc.Store(id="store-upload-t3", data=None),
    dcc.Store(id="store-pdf-paths", data=None),
    dcc.Store(id="store-temp-dir", data=None),
    dcc.Store(id="store-detection", data=None),
    dcc.Store(id="store-adjusted-sections", data=None),
    dcc.Store(id="store-sections-validated", data=False),
    dcc.Store(id="store-comparison-result", data=None),
    dcc.Store(id="store-indicator-result", data=None),
    dcc.Store(id="store-indicator-meta", data=None),
    dcc.Store(id="store-review-items", data=None),
    dcc.Store(id="store-review-current-idx", data=0),
    dcc.Store(id="store-analysis-running", data=False),
    dcc.Store(id="store-analysis-start-ms", data=None),
    dcc.Store(id="store-validation-start-ms", data=None),
    dcc.Store(id="store-validation-duration-sec", data=None),
    dcc.Store(id="store-show-results-page", data=False),
    dcc.Store(id="store-review-filters", data={"section": "all", "status": "all"}),
    dcc.Store(id="store-proof-display-mode", data="crop"),
    dcc.Store(id="store-nav-debug", data=None),
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
                            md=10,
                            className="p-4 bg-light",
                            style={"minHeight": "100vh"},
                        ),
                    ],
                    className="g-0",
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
    Output("store-upload-t1", "data"),
    Output("upload-t1-name", "children"),
    Input("upload-t1", "contents"),
    State("upload-t1", "filename"),
)
def on_upload_t1(content, filename):
    """Stocker l'upload T1."""
    if not content:
        return None, ""
    return {"content": content, "filename": filename or "t1.pdf"}, f"T1: {filename or 't1.pdf'}"


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
    return {"content": content, "filename": filename or "t2.pdf"}, f"T2: {filename or 't2.pdf'}"


@callback(
    Output("store-upload-t3", "data"),
    Output("upload-t3-name", "children"),
    Input("upload-t3", "contents"),
    State("upload-t3", "filename"),
)
def on_upload_t3(content, filename):
    """Stocker l'upload T3."""
    if not content:
        return None, ""
    return {"content": content, "filename": filename or "t3.pdf"}, f"T3: {filename or 't3.pdf'}"


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
    State("store-upload-t3", "data"),
    State("bank-code", "value"),
    State("quarter-1", "value"),
    State("quarter-2", "value"),
    prevent_initial_call=True,
)
def on_detect(n_clicks, upl_t1, upl_t2, upl_t3, bank_code, q1, q2):
    """Detecter les sections sur les 2 PDFs selectionnes."""
    import base64
    import tempfile

    if not n_clicks:
        raise PreventUpdate

    # Determine which PDFs to use based on Quarter selection logic
    # Logic: T1 corresponds to upl_t1, T2 to upl_t2, T3 to upl_t3
    # If comparison is Q1 vs Q2 -> Use T1 and T2
    # If comparison is T2 vs T3 -> Use T2 and T3
    # For now, we simplify: Always expect T1 and T2 uploaded for the primary flow.
    # But if T3 is present and requested, handle logic.
    
    # Mapping simplistic logic for now:
    # upl_A = upl_t1
    # upl_B = upl_t2
    # If user wants T2 vs T3, they should ideally upload T2 in slot 1 and T3 in slot 2 OR we handle mapping.
    # Let's stick to strict slots: Slot 1 = T1, Slot 2 = T2. 
    # If a user uploads to T3, we can use it if we implement specific logic.
    # Given the constraint "T1 vs T2 or T2 vs T3", let's assume standard flow uses T1 and T2 slots for the active pair.
    
    active_upl_1 = upl_t1
    active_upl_2 = upl_t2
    
    # If T3 is involved (future logic), swap here.
    
    if not active_upl_1 or not active_upl_2 or not bank_code:
        return (
            None,
            None,
            None,
            False,
            build_page_upload(),
            dbc.Alert("Veuillez uploader les rapports T1 et T2.", color="warning"),
            None,
            False,
        )

    def decode(content):
        if content and "," in content:
            return base64.b64decode(content.split(",")[1])
        return base64.b64decode(content) if content else b""

    try:
        b1 = decode(active_upl_1.get("content"))
        b2 = decode(active_upl_2.get("content"))
        temp_dir = tempfile.mkdtemp()
        path_t1, path_t2 = save_pdfs_to_temp(b1, b2, temp_dir=Path(temp_dir))
        paths = {"pdf_t1": path_t1, "pdf_t2": path_t2}
    except ValueError as e:
        return None, None, None, False, build_page_upload(), dbc.Alert(str(e), color="danger"), None, False

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
            f"Sections detectees: T1={len(mapping_t1.get('sections', []))}, T2={len(mapping_t2.get('sections', []))}",
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
                    preview_imgs = html.Div("Preview indisponible", className="text-muted")
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
                        id={"type": "section-preview-btn", "index": idx, "doc": doc_key},
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
    State("bank-code", "value"),
    State("option-visual-proofs", "value"),
    State("option-vision", "value"),
    State("option-auto-indicator", "value"),
    State("option-footnotes", "value"),
    State("option-genai-classification", "value"),
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
    bank_code,
    visual_proofs,
    vision,
    auto_indicator,
    footnotes_opt,
    genai_classification_opt,
    validation_start_ms,
):
    """Valider et lancer l'analyse."""
    import os

    if not n_clicks or not detection or not paths or not bank_code:
        return None, None, None, False, build_page_validation(), None, None, False

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
    path_t1 = paths.get("pdf_t1")
    path_t2 = paths.get("pdf_t2")

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
    use_genai = bool(api_key)

    generate_visual_proofs = visual_proofs and "proofs" in visual_proofs
    use_vision_fallback = vision and "vision" in vision
    include_footnotes = bool(footnotes_opt and "footnotes" in footnotes_opt)
    include_genai_classification = bool(
        genai_classification_opt and "classify" in genai_classification_opt and api_key
    )

    try:
        result = run_comparison_with_sections(
            pdf_path_t1=path_t1,
            pdf_path_t2=path_t2,
            bank_code=bank_code,
            sections_t1=sections_t1,
            sections_t2=sections_t2,
            use_genai=use_genai,
            api_key=api_key,
            generate_visual_proofs=generate_visual_proofs,
            use_vision_fallback=bool(use_vision_fallback),
            include_footnotes=include_footnotes,
            include_genai_classification=include_genai_classification,
        )
    except Exception as e:
        return (
            None,
            None,
            None,
            False,
            build_page_validation(),
            dbc.Alert(f"Erreur analyse: {e}", color="danger"),
            None,
            False,
        )

    indicator_result = result if is_canonical_comparison(result) else to_canonical_payload(result)
    indicator_meta = indicator_result.get("meta", {}) if isinstance(indicator_result, dict) else {}

    # Calculate validation duration
    import time
    validation_duration_sec = None
    if validation_start_ms:
        validation_end_ms = int(time.time() * 1000)
        validation_duration_sec = max(0, (validation_end_ms - validation_start_ms) // 1000)

    return (
        result,
        indicator_result,
        indicator_meta,
        True,
        build_page_results(),
        dbc.Alert("Analyse terminée. Indicateurs comparés.", color="success"),
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
    Input("store-review-items", "data"),
    Input("store-review-current-idx", "data"),
    Input("store-show-results-page", "data"),
    Input("store-review-filters", "data"),
    Input("btn-approve", "n_clicks"),
    Input("btn-reject", "n_clicks"),
    prevent_initial_call=True,
)
def update_review_queue(review_items_data, current_idx, show_results, filters, _btn_approve, _btn_reject):
    """Update the left-side review queue and top KPIs."""
    if not show_results:
        raise PreventUpdate
    # review_items_data=None: init_review_items pas encore execute (course)
    if review_items_data is None:
        return (
            html.Div(
                [html.I(className="bi bi-hourglass-split me-2"), "Chargement de la file de revue..."],
                className="text-muted p-3",
            ),
            build_analyst_kpi_card(t("file_review_total"), "-", color="white"),
            build_analyst_kpi_card(t("validated"), "-", color="white"),
            build_analyst_kpi_card(t("rejected"), "-", color="white"),
            build_analyst_kpi_card(t("pending"), "-", color="white"),
            0, 0, 0
        )
    if len(review_items_data) == 0:
        return (
            html.Div(t("no_changes_review", "Aucun changement a revoir."), className="text-muted p-3"),
            build_analyst_kpi_card(t("file_review_total"), "0", color="white"),
            build_analyst_kpi_card(t("validated"), "0", color="white"),
            build_analyst_kpi_card(t("rejected"), "0", color="white"),
            build_analyst_kpi_card(t("pending"), "0", color="white"),
            0, 0, 0
        )

    items = review_items_data
    total = len(items)
    approved = sum(1 for i in items if i.get("review_status") == REVIEW_STATUS_APPROVED)
    rejected = sum(1 for i in items if i.get("review_status") == REVIEW_STATUS_REJECTED)
    pending = total - approved - rejected  # Tout le reste = en attente (y compris statut manquant/invalide)

    pct_approved = (approved / total) * 100 if total else 0
    pct_rejected = (rejected / total) * 100 if total else 0
    pct_pending = (pending / total) * 100 if total else 0

    queue_component = build_review_queue(items, current_idx, filters)

    return (
        queue_component,
        build_analyst_kpi_card(t("file_review_total"), str(total), color="white"),
        build_analyst_kpi_card(t("validated"), str(approved), color="white"),
        build_analyst_kpi_card(t("rejected"), str(rejected), color="white"),
        build_analyst_kpi_card(t("pending"), str(pending), color="white"),
        pct_approved, pct_rejected, pct_pending
    )


@callback(
    Output("store-review-filters", "data"),
    Input({"type": "filter-section", "value": ALL}, "n_clicks"),
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
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input({"type": "review-item", "index": ALL}, "n_clicks"),
    State("store-review-current-idx", "data"),
    State("store-review-items", "data"),
    prevent_initial_call=True,
)
def on_queue_item_click(n_clicks, current_idx, items):
    """Handle click on a review item in the queue. Clamp index to valid range."""
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered_id
    if not button_id or not isinstance(button_id, dict):
        raise PreventUpdate

    clicked_index = button_id.get("index")
    if clicked_index is None:
        raise PreventUpdate

    try:
        clicked_index = int(clicked_index)
    except (TypeError, ValueError):
        raise PreventUpdate

    total = len(items) if items and isinstance(items, list) else 0
    if items and isinstance(items, list):
        clicked_index = max(0, min(clicked_index, total - 1))

    dbg = {"writer": "on_queue_item_click", "trigger": str(button_id), "from": current_idx, "to": clicked_index, "total": total}
    logger.info("[on_queue_item_click] trig=%s current_idx=%r total=%s -> new_idx=%s", button_id, current_idx, total, clicked_index)
    return clicked_index, dbg


@callback(
    Output("review-detail-container", "children"),
    Input("store-review-items", "data"),
    Input("store-review-current-idx", "data"),
    Input("store-pdf-paths", "data"),
    Input("store-show-results-page", "data"),
    Input("store-proof-display-mode", "data"),
    prevent_initial_call=True,
)
def update_review_detail(review_items_data, current_idx, paths, show_results, proof_display_mode):
    """Update the right-side detail view."""
    if not show_results:
        raise PreventUpdate
    if not review_items_data:
        return html.Div("Veuillez lancer une analyse pour voir les détails.", className="text-center text-muted mt-5")

    idx = max(0, min(int(current_idx or 0), len(review_items_data) - 1))
    item = review_items_data[idx]
    mode = (proof_display_mode or "crop").strip().lower()
    if mode not in ("crop", "full"):
        mode = "crop"

    img_t1_b64 = _get_proof_image_b64_for_item(item, "t1", paths or {}, proof_display_mode=mode)
    img_t2_b64 = _get_proof_image_b64_for_item(item, "t2", paths or {}, proof_display_mode=mode)

    return build_review_detail(
        item=item,
        img_t1_b64=img_t1_b64,
        img_t2_b64=img_t2_b64,
        current_idx=idx,
        total_items=len(review_items_data),
        proof_display_mode=mode,
    )


@callback(
    Output("store-proof-display-mode", "data"),
    Input("proof-display-mode", "value"),
)
def on_proof_display_mode_change(value):
    """Persist proof display mode (crop vs full page + bbox)."""
    if value in ("crop", "full"):
        return value
    return no_update


@callback(
    Output("store-review-items", "data", allow_duplicate=True),
    Input("btn-approve", "n_clicks"),
    Input("btn-reject", "n_clicks"),
    Input("btn-apply", "n_clicks"),
    State("store-review-items", "data"),
    State("store-review-current-idx", "data"),
    State("review-comment", "value"),
    prevent_initial_call=True,
)
def on_review_action_modern(btn_approve, btn_reject, btn_apply, review_items, current_idx, comment):
    """Handle review actions (Approve, Reject, Comment) in the modern UI."""
    from dash import ctx
    import json

    if not ctx.triggered_id or not review_items:
        raise PreventUpdate

    idx = max(0, min(int(current_idx or 0), len(review_items) - 1))
    item_dict = review_items[idx]
    
    # Determine action
    triggered = ctx.triggered_id
    new_status = item_dict.get("review_status")
    
    if triggered == "btn-approve":
        new_status = REVIEW_STATUS_APPROVED
    elif triggered == "btn-reject":
        new_status = REVIEW_STATUS_REJECTED
    elif triggered == "btn-apply":
        # Just saving comment, status remains unless explicitly changed elsewhere
        pass

    # Deep copy to ensure Dash detects the change
    updated_item = json.loads(json.dumps(item_dict))
    updated_item["review_status"] = new_status
    if comment is not None:
        updated_item["comment"] = comment

    # Create a completely new list with deep copies
    new_items = json.loads(json.dumps(review_items))
    new_items[idx] = updated_item
    
    return new_items


@callback(
    Output("store-review-current-idx", "data", allow_duplicate=True),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("store-review-current-idx", "data"),
    State("store-review-items", "data"),
    prevent_initial_call=True,
)
def on_modern_nav(prev_clicks, next_clicks, current_idx, items):
    """Handle Previous/Next buttons in modern UI. Clamp idx to [0, total-1]."""
    logger.info("[on_modern_nav] ENTER trig=%s current_idx=%r items_len=%s", ctx.triggered_id, current_idx, len(items) if items else 0)

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
        logger.warning("[on_modern_nav] PreventUpdate: trig=%r (not btn-prev/btn-next)", triggered)
        raise PreventUpdate

    idx = max(0, min(idx, total - 1))
    dbg = {"writer": "on_modern_nav", "trigger": triggered, "from": current_idx, "to": idx, "total": total}
    logger.info("[on_modern_nav] EXIT -> new_idx=%s total=%s", idx, total)
    return idx, dbg


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
    return html.Pre(json.dumps(payload, ensure_ascii=False, indent=2), className="mb-0 small")


@callback(
    Output("btn-prev", "disabled"),
    Output("btn-next", "disabled"),
    Input("store-review-current-idx", "data"),
    Input("store-review-items", "data"),
    Input("store-show-results-page", "data"),
)
def update_review_nav_disabled(current_idx, items, show_results):
    """Disable Prev/Next when at first/last item; disable both when no results."""
    if not show_results or not items:
        return True, True
    idx = max(0, min(int(current_idx or 0), len(items) - 1))
    n = len(items)
    return idx <= 0, idx >= n - 1


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
        title = comparison.get("comparison", comparison.get("comparison_date", "Comparaison"))
    elif indicator:
        bank = indicator.get("bank_code", "N/A")
        title = "Indicateurs"
    header = html.H5(f"{str(bank).upper()} - {title}")

    executive_summary = None
    if isinstance(data, dict):
        meta = data.get("meta", {}) or {}
        summary_text = get_meta_value(meta, "executive_summary", "content") or ""
        if summary_text:
            executive_summary = dbc.Accordion(
                [
                    dbc.AccordionItem(
                        html.P(summary_text, className="mb-0 small text-muted"),
                        title="Résumé Exécutif (Cliquer pour dérouler)",
                    )
                ],
                start_collapsed=True,
                className="mb-3 shadow-sm",
            )
    if executive_summary is None:
        executive_summary = html.Div()

    kpis = []
    if indicator:
        kpi = indicator.get("summary", indicator.get("kpi_metier", {}))
        status_counts = kpi.get("status_counts", {}) or {}
        structure_change = status_counts.get("structure_change", 0)
        cols = [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(f"{t('tables')} T1", className="small text-muted mb-0"),
                            html.H4(str(kpi.get("tables_t1", 0)), className="mb-0 fw-bold"),
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
                            html.P(f"{t('tables')} T2", className="small text-muted mb-0"),
                            html.H4(str(kpi.get("tables_t2", 0)), className="mb-0 fw-bold"),
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
                            html.H4(str(kpi.get("tables_matched", 0)), className="mb-0 fw-bold"),
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
                            html.P(t("fusion_split"), className="small text-muted mb-0"),
                            html.H4(str(structure_change), className="mb-0 fw-bold"),
                        ],
                        className="p-2 text-center",
                    ),
                    className="shadow-sm border-0",
                ),
                width=2,
            ),
        ]
        kpis.append(dbc.Row(cols, className="mb-3"))
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


def _build_kpi_card(title: str, value: str | int, delta_icon: str | None = None, color: str = "light") -> dbc.Card:
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
        c for c in comparisons
        if (
            len(c.get("added_indicators", []))
            + len(c.get("removed_indicators", []))
            + len(c.get("renamed_indicators", []))
        ) > 0
    ]
    n_tables = len(tables_with_changes)

    # Sum indicators
    total_added = sum(len(c.get("added_indicators", [])) for c in tables_with_changes)
    total_removed = sum(len(c.get("removed_indicators", [])) for c in tables_with_changes)

    header_text = f"Differences d'indicateurs ({n_tables} {t('tables')} avec changements)"

    return (
        header_text,
        _build_kpi_card(t("kpi_removed"), total_removed, delta_icon=None),
        _build_kpi_card(t("kpi_added"), total_added, delta_icon=None),
        _build_kpi_card(t("validation_time"), _format_duration(validation_duration_sec)),
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
            section_name=section_name,
            tables_with_changes=data["changes"],
            tables_added=data["added"],
            tables_removed=data["removed"],
            item_id=f"section-{i}",
        )
        accordion_items.append(item)

    # Determine which sections to expand by default (those with changes)
    active_items = [
        f"section-{i}" for i, (_, data) in enumerate(sorted(sections.items()))
        if data["changes"] or data["added"] or data["removed"]
    ]

    return dbc.Accordion(
        accordion_items,
        id="sections-accordion",
        active_item=active_items[:3] if active_items else None,  # Expand first 3 with changes
        always_open=True,
    )


@callback(
    Output("store-review-items", "data"),
    Output("store-review-current-idx", "data"),
    Output("store-nav-debug", "data", allow_duplicate=True),
    Input("store-indicator-result", "data"),
    Input("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def init_review_items(indicator_result, paths):
    """Construire les ReviewItems depuis indicator_result pour la revue."""
    if not indicator_result or not paths:
        raise PreventUpdate

    path_t1 = paths.get("pdf_t1", "")
    path_t2 = paths.get("pdf_t2", "")
    bank_code = str(indicator_result.get("bank_code", ""))
    quarter_from = str(indicator_result.get("quarter_from", "t1"))
    quarter_to = str(indicator_result.get("quarter_to", "t2"))

    items = build_review_items_from_indicator_result(
        indicator_result,
        bank_code=bank_code,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
        pdf_path_t1=path_t1,
        pdf_path_t2=path_t2,
    )
    serialized = [it.to_dict() for it in items]
    total = len(serialized)
    dbg = {"writer": "init_review_items", "trigger": "init", "from": None, "to": 0, "total": total}
    logger.info("[init_review_items] total=%s -> idx=0", total)
    return serialized, 0, dbg


def _build_comparison_statement(item: ReviewItem) -> str:
    """Phrase d'interpretation metier pour un changement."""
    table = item.table_name or "(tableau inconnu)"
    table_id = (item.table_id_t2 or item.table_id_t1 or "").strip()
    table_label = f"Tableau n°{table_id}: {table}" if table_id else f"Tableau: {table}"
    if item.indicators:
        return f"{table_label} -- {item.indicator}"
    indicator = item.indicator or "(indicateur non disponible)"
    if item.change_type == CHANGE_TYPE_TABLE_ADDED:
        return f"Tableau entier ajoute en T2: {table_label}"
    if item.change_type == CHANGE_TYPE_TABLE_REMOVED:
        return f"Tableau entier supprime en T2: {table_label} (present en T1)"
    if item.change_type == CHANGE_TYPE_ADDED:
        return f"Ajoute T2: {indicator} -- absent en T1."
    if item.change_type == CHANGE_TYPE_REMOVED:
        return f"Supprime en T2: {indicator} -- present en T1."
    if item.change_type == CHANGE_TYPE_RENAMED:
        return f"Renomme entre T1/T2: {indicator}."
    return f"Changement detecte: {indicator}"


def _filter_noise(items: list[str]) -> list[str]:
    """Filter out noise lines (dates, units, footnotes) using normalize_indicator_for_comparison."""
    return [x for x in items if x and normalize_indicator_for_comparison(str(x).strip())]


def _get_proof_image_b64_for_item(
    item_dict: dict, side: str, paths: dict, *, proof_display_mode: str = "crop"
) -> str | None:
    """Get proof image base64. With proof_display_mode='full' skip crop (full page); with 'crop' use bbox."""
    table_status = (item_dict.get("table_status") or "").strip().lower()
    if table_status == "stable":
        return _get_proof_image_b64(item_dict, side, paths)

    proof_image_path = item_dict.get("proof_image_path", "") or ""
    if side == "t2" and proof_image_path:
        return None

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    path_t1 = paths.get("pdf_t1", "") if paths else ""
    path_t2 = paths.get("pdf_t2", "") if paths else ""
    pdf_path = path_t1 if side == "t1" else path_t2
    if not pdf_path:
        pdf_path = ref

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
    if base_img_b64 is None and ref and Path(ref).exists() and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            base_img_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    use_crop = (proof_display_mode or "crop").strip().lower() == "crop"

    if base_img_b64 is None:
        if not pdf_path or page is None:
            return _get_proof_image_b64(item_dict, side, paths)

        page_effective = max(1, int(page))
        bbox = item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2")
        bbox_key = ""
        if use_crop and bbox and isinstance(bbox, list) and len(bbox) == 4:
            bbox_key = json.dumps(bbox)
        try:
            raw_bytes = _cached_render_or_crop(str(pdf_path), page_effective, 1.5, bbox_key)
            base_img_b64 = base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else None
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

    if ref and Path(ref).exists() and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}:
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

    img_t1_b64 = _get_proof_image_b64_for_item(current_dict, "t1", paths or {}, proof_display_mode="crop")
    img_t2_b64 = _get_proof_image_b64_for_item(current_dict, "t2", paths or {}, proof_display_mode="crop")

    def _img_div(b64, label, caption):
        if b64:
            src = f"data:image/png;base64,{b64}"
            return html.Div(
                [
                    html.P(label, className="small text-muted mb-1"),
                    html.Img(src=src, style={"maxWidth": "100%", "border": "1px solid #ddd"}),
                    html.P(caption, className="small text-muted mt-1") if caption else None,
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
            "Preuve tableau entier T1/T2",
            f"Mode {proof_mode}" if proof_mode else None,
        )
        col2_content = html.Div(
            [html.P("Tn+1", className="small text-muted"), html.P("Inclus dans la preuve tableau entier.")]
        )
    else:
        col1_content = _img_div(
            img_t1_b64, "Tn", f"Page {current.page_t1}" if current.page_t1 else None
        )
        col2_content = _img_div(
            img_t2_b64, "Tn+1", f"Page {current.page_t2}" if current.page_t2 else None
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
                f"Contexte unite T1/T2: {unit_context_t1 or '-'} | {unit_context_t2 or '-'}",
                className="small text-muted mb-1",
            )
        )
    if title_method_t1 or title_method_t2:
        meta_lines.append(
            html.P(
                f"Methode titre T1/T2: {title_method_t1 or '-'} | {title_method_t2 or '-'}",
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
                        dbc.Button("Precedent", id="btn-review-prev", color="secondary", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Suivant", id="btn-review-next", color="secondary", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(f"Changement {idx + 1} / {len(items)}", className="text-muted"),
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
                        dbc.Button("Valider", id="btn-review-approve", color="success", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Rejeter", id="btn-review-reject", color="danger", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Passer", id="btn-review-pass", color="secondary", size="sm"),
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
    dbg = {"writer": "on_review_navigate", "trigger": ctx.triggered_id, "from": current_idx, "to": idx, "total": n}
    logger.info("[on_review_navigate] trig=%s current_idx=%r total=%s -> new_idx=%s", ctx.triggered_id, current_idx, n, idx)
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
def on_review_status(approve_clicks, reject_clicks, pass_clicks, review_items, current_idx):
    """Appliquer Valider/Rejeter/Passer sur l'item courant."""
    from dash import ctx
    import json

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
        body_rows = [html.Tr([html.Td(str(row.get(h, ""))) for h in headers]) for row in rows]
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
        body_rows = [html.Tr([html.Td(str(row.get(h, ""))) for h in headers]) for row in rows]
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
            return html.Div("Aucun changement structurel detecte.", className="text-muted")
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
        body_rows = [html.Tr([html.Td(str(r.get(h, ""))) for h in headers]) for r in flat]
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
    bank = str(ir.get("bank_code", "bank")).lower()
    q_from = str(ir.get("quarter_from", "t1"))
    q_to = str(ir.get("quarter_to", "t2"))
    year_val = str(ir.get("year", "2025"))
    base_name = f"{bank}_{q_from}_vs_{q_to}_{year_val}_review"

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
                    html.P("Export payload canonical complet (summary, status_counts, structure_change_detected)", className="mb-2"),
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
    State("store-indicator-result", "data"),
    State("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def on_download_csv(n_clicks, review_items_data, indicator_result, paths):
    """Telecharger le CSV de validation (Excel FR, UTF-8 BOM, separateur ;)."""
    if not n_clicks:
        raise PreventUpdate
    ir = indicator_result or {}
    items = []
    if review_items_data:
        try:
            items = [ReviewItem.from_dict(d) for d in review_items_data]
        except Exception:
            pass
    if not items and ir:
        # Reconstruire depuis indicator_result si store desynchronise
        paths = paths or {}
        path_t1 = paths.get("pdf_t1", "") if isinstance(paths, dict) else ""
        path_t2 = paths.get("pdf_t2", "") if isinstance(paths, dict) else ""
        items = build_review_items_from_indicator_result(
            ir,
            bank_code=str(ir.get("bank_code", "")),
            quarter_from=str(ir.get("quarter_from", "t1")),
            quarter_to=str(ir.get("quarter_to", "t2")),
            pdf_path_t1=path_t1 or "",
            pdf_path_t2=path_t2 or "",
        )
    bank = str(ir.get("bank_code", "bank")).upper()
    q_from = str(ir.get("quarter_from", "t1")).upper()
    q_to = str(ir.get("quarter_to", "t2")).upper()
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_from}_vs_{q_to}_{year_val}.csv"
    csv_str = generate_validation_csv(items, ir)
    return dict(content=csv_str, filename=filename)


@callback(
    Output("download-review-json", "data"),
    Input("btn-download-review-json", "n_clicks"),
    State("store-review-items", "data"),
    State("store-indicator-result", "data"),
    prevent_initial_call=True,
)
def on_download_json(n_clicks, review_items_data, indicator_result):
    """Telecharger le JSON de revue."""
    if not n_clicks or not review_items_data:
        raise PreventUpdate
    from datetime import datetime

    ir = indicator_result or {}
    items = [ReviewItem.from_dict(d) for d in review_items_data]
    bank = str(ir.get("bank_code", "bank"))
    q_from = str(ir.get("quarter_from", "t1"))
    q_to = str(ir.get("quarter_to", "t2"))
    year_val = str(ir.get("year", "2025"))
    base_name = f"{bank}_{q_from}_vs_{q_to}_{year_val}_review".replace(" ", "_").lower()
    json_str = export_review_items_json_fr(
        items,
        metadata={
            "bank_code": bank,
            "quarter_from": q_from,
            "quarter_to": q_to,
            "year": year_val,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return dict(content=json_str, filename=f"{base_name}.json")


@callback(
    Output("download-review-excel", "data"),
    Input("btn-download-review-excel", "n_clicks"),
    State("store-review-items", "data"),
    State("store-indicator-result", "data"),
    State("store-pdf-paths", "data"),
    prevent_initial_call=True,
)
def on_download_excel(n_clicks, review_items_data, indicator_result, paths):
    """Telecharger le fichier Excel de validation (.xlsx)."""
    if not n_clicks:
        raise PreventUpdate
    ir = indicator_result or {}
    items = []
    if review_items_data:
        try:
            items = [ReviewItem.from_dict(d) for d in review_items_data]
        except Exception:
            pass
    if not items and ir:
        paths = paths or {}
        path_t1 = paths.get("pdf_t1", "") if isinstance(paths, dict) else ""
        path_t2 = paths.get("pdf_t2", "") if isinstance(paths, dict) else ""
        items = build_review_items_from_indicator_result(
            ir,
            bank_code=str(ir.get("bank_code", "")),
            quarter_from=str(ir.get("quarter_from", "t1")),
            quarter_to=str(ir.get("quarter_to", "t2")),
            pdf_path_t1=path_t1 or "",
            pdf_path_t2=path_t2 or "",
        )
    bank = str(ir.get("bank_code", "bank")).upper()
    q_from = str(ir.get("quarter_from", "t1")).upper()
    q_to = str(ir.get("quarter_to", "t2")).upper()
    year_val = str(ir.get("year", "2025"))
    filename = f"Vigie_Comparaison_{bank}_{q_from}_vs_{q_to}_{year_val}.xlsx"
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
    q_from = str(indicator_result.get("quarter_from", "t1"))
    q_to = str(indicator_result.get("quarter_to", "t2"))
    year_val = str(indicator_result.get("year", "2025"))
    base_name = f"{bank}_{q_from}_vs_{q_to}_{year_val}_canonical".replace(" ", "_").lower()
    json_str = json.dumps(indicator_result, ensure_ascii=False, indent=2)
    return dict(content=json_str, filename=f"{base_name}.json")


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-sections-validated", "data", allow_duplicate=True),
    Output("store-review-items", "data", allow_duplicate=True),
    Output("store-review-current-idx", "data", allow_duplicate=True),
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
        dbg = {"writer": "on_reset", "trigger": "btn-reset", "from": None, "to": 0, "total": None}
        logger.info("[on_reset] -> idx=0")
        return None, None, None, False, None, 0, build_page_upload(), False, {"section": "all", "status": "all"}, dbg
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


if __name__ == "__main__":
    app.run(debug=True, port=8050)
