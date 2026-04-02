"""Upload, detection, section validation, and analysis callbacks."""

from __future__ import annotations

import logging
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from vigilance.comparison_canonical import (
    is_canonical_comparison,
    to_canonical_payload,
)
from vigilance.comparison_runner import run_comparison_with_sections
from vigilance.dash_app.components.pdf_preview import pdf_images_from_base64
from vigilance.dash_app.layouts import (
    build_page_results,
    build_page_upload,
    build_page_validation,
)
from vigilance.dash_app.services.comparison_store import (
    build_file_comparison_store,
)
from vigilance.dash_app.services.comparison_context import (
    _normalize_pdf_paths_store,
    _quarter_context_from_store,
)
from vigilance.quarter_utils import build_quarter_context
from vigilance.ui_detection import (
    _detect_sections_core,
    get_section_preview_images,
)
from vigilance.ui_io import save_pdfs_to_temp

logger = logging.getLogger(__name__)


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
    ctx = build_quarter_context(current_quarter or "T2", year=year_value or 2025)
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
    default_name = "rapport_precedent.pdf"
    return {
        "content": content,
        "filename": filename or default_name,
    }, f"Précédent : {filename or default_name}"


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
    default_name = "rapport_courant.pdf"
    return {
        "content": content,
        "filename": filename or default_name,
    }, f"Courant : {filename or default_name}"


@callback(
    Output("upload-source-container", "style"),
    Output("db-source-container", "style"),
    Input("data-source-type", "value"),
)
def toggle_data_source(source_type):
    """Afficher soit la bibliothèque locale, soit la section de téléversement."""
    if source_type == "saved":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@callback(
    Output("db-run-selector", "options"),
    Output("db-run-selector", "value"),
    Input("data-source-type", "value"),
    Input("bank-code", "value"),
    Input("analysis-year", "value"),
    Input("current-quarter", "value"),
)
def populate_saved_runs(source_type, bank_code, year, current_quarter):
    """Remplir le menu déroulant des analyses enregistrées locales."""
    if source_type != "saved" or not bank_code or not year or not current_quarter:
        return [], None

    from vigilance.ui_config import INDICATOR_COMPARISON_DIR

    store = build_file_comparison_store(root_dir=INDICATOR_COMPARISON_DIR)
    filtered_options = store.list_saved_run_options(
        bank_code=str(bank_code),
        year=year,
        current_quarter=str(current_quarter),
    )

    if not filtered_options:
        return [
            {
                "label": "Aucune analyse enregistrée trouvée pour cette sélection",
                "value": "",
                "disabled": True,
            }
        ], None

    return filtered_options, filtered_options[0]["value"]


@callback(
    Output("store-comparison-result", "data", allow_duplicate=True),
    Output("store-indicator-result", "data", allow_duplicate=True),
    Output("store-indicator-meta", "data", allow_duplicate=True),
    Output("store-pdf-paths", "data", allow_duplicate=True),
    Output("store-review-items", "data", allow_duplicate=True),
    Output("store-review-queue", "data", allow_duplicate=True),
    Output("store-show-results-page", "data", allow_duplicate=True),
    Output("main-content", "children", allow_duplicate=True),
    Output("notification", "children", allow_duplicate=True),
    Input("btn-load-db", "n_clicks"),
    State("db-run-selector", "value"),
    prevent_initial_call=True,
)
def load_saved_comparison(n_clicks, run_file):
    """Charger une analyse enregistrée depuis la bibliothèque locale."""
    from vigilance.dash_app.services.review_persistence import (
        _persist_review_state,
        _stored_review_items_from_state,
    )
    from vigilance.quarter_utils import quarter_label_from_payload
    from vigilance.review_adapters import build_review_items_from_indicator_result
    from vigilance.review_queue_normalizer import build_normalized_review_queue
    from vigilance.review_storage import (
        is_review_state_compatible,
        is_review_state_stale,
    )
    from vigilance.ui_config import INDICATOR_COMPARISON_DIR

    if not n_clicks or not run_file:
        raise PreventUpdate

    store = build_file_comparison_store(root_dir=INDICATOR_COMPARISON_DIR)
    target_path = store.resolve_path(run_file)
    if not target_path.exists():
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dbc.Alert("Fichier introuvable.", color="danger"),
        )

    try:
        loaded = store.load_dash_payload(
            target_path,
            source="analyse_enregistree",
            source_label="Analyse enregistrée",
        )
        if not loaded:
            raise ValueError("Le fichier de comparaison est vide ou corrompu.")

        data = loaded["indicator_result"]
        meta = dict(loaded["indicator_meta"])
        paths = dict(loaded["pdf_paths"])
        missing_msg = str(loaded["warning"] or "")

        fresh_review_items = build_review_items_from_indicator_result(
            data,
            bank_code=meta.get("bank_code") or "bnc",
            quarter_from=quarter_label_from_payload(data, "previous"),
            quarter_to=quarter_label_from_payload(data, "current"),
            pdf_path_t1=str(paths.get("pdf_previous") or paths.get("pdf_t1") or ""),
            pdf_path_t2=str(paths.get("pdf_current") or paths.get("pdf_t2") or ""),
        )
        fresh_review_items_serialized = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in fresh_review_items
        ]

        state = store.load_review_state(target_path)
        current_run_id = meta.get("run_id", "")
        if state and is_review_state_stale(state, current_run_id):
            logger.info(
                "Review state is stale (stored run_id=%s vs current=%s) — discarding.",
                state.get("comparison_run_id", ""),
                current_run_id,
            )
            state = None
        stored_review_items = _stored_review_items_from_state(state)
        if state and not is_review_state_compatible(
            state,
            review_items=fresh_review_items_serialized,
            stored_review_items=stored_review_items,
        ):
            logger.info(
                "Review state is incompatible with current comparison payload — discarding."
            )
            state = None

        review_items = stored_review_items if state else None
        if not review_items:
            review_items = fresh_review_items

        raw_items_for_queue = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in review_items
        ]
        review_queue = build_normalized_review_queue(
            indicator_result=data,
            raw_items=raw_items_for_queue,
            pdf_path_t1=str(paths.get("pdf_previous") or paths.get("pdf_t1") or ""),
            pdf_path_t2=str(paths.get("pdf_current") or paths.get("pdf_t2") or ""),
        )

        alert = dbc.Alert(
            "Analyse enregistrée chargée depuis la bibliothèque locale.",
            color="success",
            duration=3000,
        )
        if missing_msg:
            alert = dbc.Alert(missing_msg, color="warning")

        review_items_serialized = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in review_items
        ]
        review_queue_serialized = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in review_queue
        ]

        _persist_review_state(
            indicator_meta=meta,
            indicator_result=data,
            review_items=review_items_serialized,
            review_queue=review_queue_serialized,
            source="load_from_saved",
        )

        return (
            data,
            data,
            meta,
            paths,
            review_items_serialized,
            review_queue_serialized,
            True,
            build_page_results(),
            alert,
        )

    except Exception as e:
        logger.exception("Erreur lors du chargement d'une analyse enregistrée : %s", e)
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dbc.Alert(
                f"Erreur technique lors du chargement de l'analyse enregistrée : {e}",
                color="danger",
            ),
        )


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
                f"Veuillez téléverser les rapports {previous_label} et {current_label}.",
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
        paths = _normalize_pdf_paths_store(
            {
                "pdf_t1": path_t1,
                "pdf_t2": path_t2,
                "pdf_previous": path_t1,
                "pdf_current": path_t2,
            }
        )
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
            dbc.Alert(f"Erreur de détection : {e}", color="danger"),
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
        return html.Div("Aucune détection.")

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
                        "Aperçu indisponible", className="text-muted"
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
                        "Voir l'aperçu",
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
        f"{count} section(s) ajustée(s)",
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
    from vigilance.ui_config import INDICATOR_COMPARISON_DIR

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
    normalized_paths = _normalize_pdf_paths_store(paths)
    path_t1 = normalized_paths.get("pdf_previous") or normalized_paths.get("pdf_t1")
    path_t2 = normalized_paths.get("pdf_current") or normalized_paths.get("pdf_t2")
    logger.info(
        "[on_analyze] bank=%s previous_pdf=%r current_pdf=%r",
        bank_code,
        path_t1,
        path_t2,
    )

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
    use_stored_extraction = not bool(
        force_reextract_opt and "reextract" in (force_reextract_opt or [])
    )

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
        logger.exception(
            "[on_analyze] analysis_failed bank=%s previous_pdf=%r current_pdf=%r",
            bank_code,
            path_t1,
            path_t2,
        )
        err_text = str(e)
        if "__fspath__ returns a str, not 'NoneType'" in err_text:
            err_text = (
                "Un chemin PDF ou artefact de comparaison est manquant dans le pipeline. "
                "Consulter le terminal Dash: la stack trace complète est maintenant journalisée."
            )
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

    store = build_file_comparison_store(root_dir=INDICATOR_COMPARISON_DIR)
    compare_path = ""
    if isinstance(result, dict):
        compare_path = str((result.get("meta", {}) or {}).get("compare_path", "") or "")
    loaded = store.load_dash_payload(
        compare_path,
        source="execution_courante",
        source_label="Exécution courante",
    )
    if loaded:
        indicator_result = loaded["indicator_result"]
        indicator_meta = loaded["indicator_meta"]
        analyze_notification = dbc.Alert(
            loaded["warning"] or "Analyse terminée. Source : exécution courante.",
            color="warning" if loaded["warning"] else "success",
        )
        result = indicator_result
    else:
        indicator_result = (
            result if is_canonical_comparison(result) else to_canonical_payload(result)
        )
        indicator_meta = (
            indicator_result.get("meta", {}) if isinstance(indicator_result, dict) else {}
        )
        if isinstance(indicator_meta, dict):
            indicator_meta["source"] = "execution_courante"
            indicator_meta["source_label"] = "Exécution courante"
            indicator_meta["storage_backend"] = "fichier_local"
        analyze_notification = dbc.Alert(
            "Analyse terminée. Source : exécution courante.",
            color="success",
        )

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
