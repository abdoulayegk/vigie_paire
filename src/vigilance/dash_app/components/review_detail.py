"""Review Detail Component -- table-grouped view with per-indicator validation."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from app.i18n import t
from app.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_FOOTNOTE,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
)
from vigilance.utils.indicator_cleaner import strip_dates_from_table_title

_CHANGE_TYPES_WITH_VISUAL_FLAG = frozenset(
    {
        CHANGE_TYPE_TABLE_ADDED,
        CHANGE_TYPE_TABLE_REMOVED,
        CHANGE_TYPE_ADDED,
        CHANGE_TYPE_REMOVED,
        CHANGE_TYPE_RENAMED,
        CHANGE_TYPE_MODIFIED,
        "uncertain",
        "structure_change",
    }
)

_SECTION_LABELS = {
    "gestion_capital": "Gestion du capital",
    "capital_management": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "risk_management": "Gestion des risques",
    "gestion_reglementation": "Reglementation",
    "regulatory_updates": "Reglementation",
    "reglementation": "Reglementation",
}


def section_display_label(section: str | None) -> str:
    value = (section or "").strip()
    if not value:
        return ""
    return _SECTION_LABELS.get(value, value.replace("_", " ").title())


def _clean_title_for_display(raw_title: str) -> str:
    title = (raw_title or "").strip()
    if not title:
        return ""
    cleaned = strip_dates_from_table_title(title).strip(" -:;,")
    lowered = cleaned.lower()
    if lowered in {"au", "aux", "as at"}:
        return ""
    if cleaned:
        return cleaned
    return ""


def compute_flag_state(item: dict) -> dict:
    """Compute visual flag state for proof cards (T1/T2 borders and badges)."""
    change_type = (item.get("change_type") or "").strip()
    added = item.get("added_indicators") or []
    removed = item.get("removed_indicators") or []
    indicators = item.get("indicators") or []
    added_count = len(added) if isinstance(added, list) else 0
    removed_count = len(removed) if isinstance(removed, list) else 0
    renamed_count = sum(
        1
        for ind in indicators
        if isinstance(ind, dict) and ind.get("type") == CHANGE_TYPE_RENAMED
    )

    has_change = (
        change_type in _CHANGE_TYPES_WITH_VISUAL_FLAG
        or added_count + removed_count + renamed_count > 0
    )

    if not has_change:
        return {
            "has_change": False,
            "t1_class": "proof-card",
            "t2_class": "proof-card",
            "badge_t1": t("table", "Tableau") + " T1",
            "badge_t2": t("table", "Tableau") + " T2",
            "badge_class_t1": "neutral",
            "badge_class_t2": "neutral",
        }

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        return {
            "has_change": True,
            "t1_class": "proof-card",
            "t2_class": "proof-card proof-flag-t2",
            "badge_t1": "T1 (Ancien)",
            "badge_t2": "T2 (Nouveau)",
            "badge_class_t1": "neutral",
            "badge_class_t2": "t2",
        }
    if change_type == CHANGE_TYPE_TABLE_REMOVED:
        return {
            "has_change": True,
            "t1_class": "proof-card proof-flag-t1",
            "t2_class": "proof-card",
            "badge_t1": "T1 (Ancien)",
            "badge_t2": "T2 (Nouveau)",
            "badge_class_t1": "t1",
            "badge_class_t2": "neutral",
        }

    return {
        "has_change": True,
        "t1_class": "proof-card proof-flag-t1",
        "t2_class": "proof-card proof-flag-t2",
        "badge_t1": "T1 (Ancien)",
        "badge_t2": "T2 (Nouveau)",
        "badge_class_t1": "t1",
        "badge_class_t2": "t2",
    }


def table_display_label(item: dict) -> str:
    """Format tableau label as in report: number + title, or page + section when no title."""
    raw_title = (item.get("table_name") or item.get("table_title_raw") or "").strip()
    title = _clean_title_for_display(raw_title)
    has_title = bool(title)
    table_num = (item.get("table_number") or "").strip()
    change_type = item.get("change_type", "")
    if change_type == CHANGE_TYPE_TABLE_ADDED:
        page = item.get("page_t2")
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        page = item.get("page_t1")
    else:
        page = item.get("page_t2") or item.get("page_t1")
    section = section_display_label(item.get("section"))
    prefix = t("table", "Tableau")

    if table_num and has_title:
        return f"{prefix} {table_num}: {title}"
    if table_num:
        page_part = f" (p.{page})" if page is not None else ""
        section_part = f" - {section}" if section else ""
        return f"{prefix} {table_num}{page_part}{section_part}" or f"{prefix} {table_num}"
    if has_title:
        return f"{prefix}: {title}"
    if page is not None and section:
        return f"{prefix} (p.{page}) - {section}"
    if page is not None:
        return f"{prefix} (p.{page})"
    if section:
        return f"{prefix} - {section}"
    return f"{prefix} (sans titre)"


def _indicator_badge(change_type: str) -> dbc.Badge:
    mapping = {
        CHANGE_TYPE_ADDED: (t("indicator_add").upper(), "success"),
        CHANGE_TYPE_REMOVED: (t("indicator_removal").upper(), "danger"),
        CHANGE_TYPE_RENAMED: (t("indicator_rename").upper(), "warning"),
        CHANGE_TYPE_TABLE_ADDED: (t("table_added").upper(), "info"),
        CHANGE_TYPE_TABLE_REMOVED: (t("table_removed").upper(), "dark"),
        CHANGE_TYPE_FOOTNOTE: ("NOTE", "info"),
        CHANGE_TYPE_MODIFIED: ("MODIFIE", "warning"),
    }
    label, color = mapping.get(change_type, ("?", "secondary"))
    extra = {"text_color": "dark"} if color == "warning" else {}
    if color == "dark":
        extra["text_color"] = "white"
    return dbc.Badge(label, color=color, className="me-2", **extra)


def _indicator_status_icon(status: str):
    """Small icon reflecting per-indicator review status."""
    if status == REVIEW_STATUS_APPROVED:
        return html.I(className="bi bi-check-circle-fill text-success me-2")
    if status == REVIEW_STATUS_REJECTED:
        return html.I(className="bi bi-x-circle-fill text-danger me-2")
    return html.I(className="bi bi-circle text-warning me-2")


def _bbox_normalized_for_overlay(bbox: list | None) -> list[float] | None:
    """Return [l, top, r, bottom] in 0..1 if valid for CSS % overlay, else None."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        l_, top, r, bottom = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError):
        return None
    if not (0 <= l_ <= 1 and 0 <= top <= 1 and 0 <= r <= 1 and 0 <= bottom <= 1):
        return None
    if r <= l_ or bottom <= top:
        return None
    return [l_, top, r, bottom]


# ---------------------------------------------------------------------------
# Proof images section (table-scoped, stable across indicator navigation)
# ---------------------------------------------------------------------------

def build_proofs_section(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    proof_display_mode: str = "crop",
) -> html.Div:
    """Build proof images section; depends only on table-level data."""
    flag_state = compute_flag_state(item)
    mode_full = (proof_display_mode or "crop").strip().lower() == "full"

    def _img_card(b64, label, placeholder=None, bbox_norm=None, side: str = "t1"):
        if not b64:
            msg = placeholder if placeholder else t("image_unavailable", "Image non disponible")
            return dbc.Card(
                dbc.CardBody(html.P(msg, className="text-muted text-center small")),
                className="h-100 bg-light border-0",
            )
        img_style = {
            "objectFit": "contain",
            "maxHeight": "calc(50vh - 60px)",
            "width": "100%",
            "height": "auto",
        }
        img_el = html.Img(src=f"data:image/png;base64,{b64}", style=img_style)
        if mode_full and bbox_norm:
            l_, top, r, bottom = bbox_norm
            overlay_style = {
                "position": "absolute",
                "left": f"{l_ * 100}%",
                "top": f"{top * 100}%",
                "width": f"{(r - l_) * 100}%",
                "height": f"{(bottom - top) * 100}%",
                "pointerEvents": "none",
            }
            overlay = html.Div(className="bbox-rect", style=overlay_style)
            bbox_side_class = f"proof-bbox-{side}"
            wrapper = html.Div(
                [img_el, overlay],
                style={"position": "relative", "display": "inline-block", "width": "100%"},
                className=f"proof-img-with-bbox {bbox_side_class}",
            )
            return dbc.Card(
                [
                    dbc.CardBody(wrapper),
                    dbc.CardFooter(
                        html.Small(label, className="text-muted"),
                        className="border-0 bg-white p-1",
                    ),
                ],
                className="h-100 border shadow-sm overflow-hidden",
            )
        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/png;base64,{b64}",
                    top=True,
                    style={"objectFit": "contain", "maxHeight": "calc(50vh - 60px)"},
                ),
                dbc.CardFooter(
                    html.Small(label, className="text-muted"),
                    className="border-0 bg-white p-1",
                ),
            ],
            className="h-100 border shadow-sm overflow-hidden",
        )

    def _proof_wrapper(card, card_class: str, badge_text: str, badge_class: str):
        badge = html.Span(badge_text, className=f"proof-badge {badge_class}")
        return html.Div([badge, card], className=card_class)

    change_type = item.get("change_type", "")
    bbox_t1_norm = _bbox_normalized_for_overlay(item.get("bbox_t1")) if mode_full else None
    bbox_t2_norm = _bbox_normalized_for_overlay(item.get("bbox_t2")) if mode_full else None

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        card_t1 = _img_card(None, "T1 (Ancien)", placeholder=t("no_table_added_t2", "Aucun tableau (ajoute en T2)"))
        card_t2 = _img_card(img_t2_b64, "T2 (Nouveau)", bbox_norm=bbox_t2_norm, side="t2")
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        card_t1 = _img_card(img_t1_b64, "T1 (Ancien)", bbox_norm=bbox_t1_norm, side="t1")
        card_t2 = _img_card(None, "T2 (Nouveau)", placeholder=t("no_table_removed_t2", "Aucun tableau (supprime en T2)"))
    else:
        card_t1 = _img_card(img_t1_b64, "T1 (Ancien)", bbox_norm=bbox_t1_norm, side="t1")
        card_t2 = _img_card(img_t2_b64, "T2 (Nouveau)", bbox_norm=bbox_t2_norm, side="t2")

    col_t1 = dbc.Col(
        _proof_wrapper(card_t1, flag_state["t1_class"], flag_state["badge_t1"], flag_state["badge_class_t1"]),
        width=6,
    )
    col_t2 = dbc.Col(
        _proof_wrapper(card_t2, flag_state["t2_class"], flag_state["badge_t2"], flag_state["badge_class_t2"]),
        width=6,
    )

    proof_mode_toggle = dbc.Row(
        dbc.Col(
            [
                html.Label("Affichage preuve", className="small text-muted me-2"),
                dbc.RadioItems(
                    id="proof-display-mode",
                    options=[
                        {"label": "Crop (focus)", "value": "crop"},
                        {"label": "Page complete + bbox (contexte)", "value": "full"},
                    ],
                    value=(proof_display_mode or "crop"),
                    inline=True,
                    className="small",
                ),
            ],
            width=12,
        ),
        className="mb-2",
    )
    proofs_row = dbc.Row(
        [col_t1, col_t2],
        className="mb-4 g-2",
        style={"height": "50vh", "minHeight": "400px"},
    )

    return html.Div([proof_mode_toggle, proofs_row])


# ---------------------------------------------------------------------------
# Main detail panel (indicator-scoped)
# ---------------------------------------------------------------------------

def build_review_detail(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    current_idx: int,
    total_items: int,
    proof_display_mode: str = "crop",
    indicator_idx: int = 0,
    show_proofs: bool = True,
) -> html.Div:
    """Build the right-side review detail panel with per-indicator validation."""

    table_display = table_display_label(item)
    page_t1 = item.get("page_t1")
    page_t2 = item.get("page_t2")
    if page_t1 is None and page_t2 is not None:
        page_text = f"Page T2: p.{page_t2}"
    elif page_t2 is None and page_t1 is not None:
        page_text = f"Page T1: p.{page_t1}"
    else:
        page_t1_str = str(page_t1) if page_t1 is not None else "-"
        page_t2_str = str(page_t2) if page_t2 is not None else "-"
        page_text = f"Pages: T1: p.{page_t1_str}, T2: p.{page_t2_str}"
    confidence = item.get("confidence", 0.0)
    comment = item.get("comment", "")
    indicators = item.get("indicators", [])
    n_indicators = len(indicators)

    n_decided = sum(1 for ind in indicators if ind.get("review_status", "pending") != "pending")
    all_decided = n_decided == n_indicators if n_indicators else True

    if n_indicators:
        indicator_idx = max(0, min(indicator_idx, n_indicators - 1))
        current_ind = indicators[indicator_idx]
        current_ind_status = current_ind.get("review_status", "pending")
    else:
        current_ind = None
        current_ind_status = "pending"

    progress_text = f"Indicateur {indicator_idx + 1}/{n_indicators}" if n_indicators else "Aucun indicateur"

    header_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5(t("detail_changement", "Detail du Changement"), className="mb-1"),
                    html.Div(
                        [
                            html.Span(f"Tableau {current_idx + 1}/{total_items}", className="me-3 fw-bold"),
                            html.Span(
                                f" | {progress_text} ({n_decided}/{n_indicators} decides)",
                                className="text-primary fw-semibold",
                            ) if n_indicators else html.Span(),
                        ],
                        className="small text-muted mb-2",
                    ),
                    html.Div(
                        [
                            html.Span(table_display, className="d-block fw-semibold"),
                            html.Small(page_text, className="text-muted"),
                        ],
                        className="p-2 bg-light rounded",
                    ),
                ],
                width=10,
            ),
            dbc.Col(
                html.H3(f"{confidence:.2f}", className="text-end text-muted"),
                width=2,
                className="d-flex align-items-center justify-content-end",
            ),
        ],
        className="mb-3",
    )

    indicator_rows = []
    for i, ind in enumerate(indicators):
        name = ind.get("name", "")
        change_type = ind.get("type", "")
        ind_status = ind.get("review_status", "pending")
        is_current = (i == indicator_idx)

        row_class = "d-flex align-items-center py-1 border-bottom px-2 rounded"
        if is_current:
            row_class += " bg-primary bg-opacity-10 border border-primary"

        indicator_rows.append(
            html.Div(
                [
                    _indicator_status_icon(ind_status),
                    _indicator_badge(change_type),
                    html.Span(name, className="small flex-grow-1"),
                    html.I(className="bi bi-chevron-right text-primary") if is_current else html.Span(),
                ],
                id={"type": "indicator-item", "index": i},
                className=row_class,
                style={"cursor": "pointer"},
            )
        )

    item_type = item.get("item_type", "indicator")
    if item_type == "footnote":
        section_title = f"Notes de bas de tableau ({n_indicators})"
    else:
        section_title = f"{t('indicators')} ({n_indicators})"

    table_done_banner = html.Div()
    if all_decided and n_indicators > 0:
        table_done_banner = dbc.Alert(
            [
                html.I(className="bi bi-check2-all me-2"),
                f"Tous les indicateurs ont ete revises ({n_decided}/{n_indicators}). "
                "Cliquez Suivant pour passer au tableau suivant.",
            ],
            color="success",
            className="py-2 mb-2",
        )

    indicators_section = html.Div(
        [
            html.H6(section_title, className="text-muted small mb-2"),
            table_done_banner,
            html.Div(
                indicator_rows if indicator_rows else [
                    html.P(t("no_indicators", "Aucun indicateur"), className="text-muted small")
                ],
                className="mb-3",
                style={"maxHeight": "220px", "overflowY": "auto"},
            ),
        ],
    )

    footnote_detail_rows = []
    if item_type == "footnote":
        for fc in item.get("footnote_changes", []):
            ref = fc.get("footnote_ref", "")
            ctype = fc.get("change_type", "")
            old_text = fc.get("old_text") or ""
            new_text = fc.get("new_text") or ""
            significance = fc.get("significance", "")
            category = fc.get("category", "")

            if "new" in ctype:
                badge_color = "success"
                badge_label = "AJOUT"
            elif "removed" in ctype:
                badge_color = "danger"
                badge_label = "SUPPRIME"
            else:
                badge_color = "warning"
                badge_label = "MODIFIE"

            detail_children = [
                html.Div(
                    [
                        dbc.Badge(badge_label, color=badge_color, className="me-2"),
                        html.Strong(f"[{ref}]", className="me-2"),
                        dbc.Badge(significance, color="secondary", className="me-1") if significance else None,
                        dbc.Badge(category, color="secondary", className="me-1") if category else None,
                    ],
                    className="d-flex align-items-center mb-1",
                ),
            ]
            if old_text:
                detail_children.append(
                    html.Div(
                        [html.Small("T1: ", className="fw-bold text-danger"), html.Small(old_text[:300])],
                        className="ms-3 mb-1 bg-light p-1 rounded",
                    )
                )
            if new_text:
                detail_children.append(
                    html.Div(
                        [html.Small("T2: ", className="fw-bold text-success"), html.Small(new_text[:300])],
                        className="ms-3 mb-1 bg-light p-1 rounded",
                    )
                )
            footnote_detail_rows.append(
                html.Div(detail_children, className="py-2 border-bottom")
            )

    footnote_detail_section = (
        html.Div(
            [
                html.H6("Detail des notes", className="text-muted small mb-2"),
                html.Div(footnote_detail_rows, style={"maxHeight": "200px", "overflowY": "auto"}),
            ],
            className="mb-3",
        )
        if footnote_detail_rows
        else html.Div()
    )

    proofs_section = build_proofs_section(
        item=item, img_t1_b64=img_t1_b64, img_t2_b64=img_t2_b64, proof_display_mode=proof_display_mode,
    ) if show_proofs else html.Div()

    decision_indicator_label = html.Div()
    if current_ind is not None:
        ind_name = current_ind.get("name", "")
        decision_indicator_label = html.Div(
            [
                html.Small("Indicateur en cours : ", className="text-muted fw-bold"),
                html.Small(ind_name, className="text-primary"),
            ],
            className="mb-2 p-2 bg-white rounded border",
        )

    decision_section = dbc.Card(
        dbc.CardBody(
            [
                html.H6(t("decision_analyst", "Decision de l'Analyste"), className="mb-3"),
                decision_indicator_label,
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="bi bi-check-lg me-2"), t("btn_approve", "Valider")],
                                    id="btn-approve",
                                    color="success" if current_ind_status == REVIEW_STATUS_APPROVED else "outline-success",
                                    className="me-2 mb-2 w-100 text-start",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-x-lg me-2"), t("btn_reject", "Rejeter")],
                                    id="btn-reject",
                                    color="danger" if current_ind_status == REVIEW_STATUS_REJECTED else "outline-danger",
                                    className="w-100 text-start",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Textarea(
                                    id="review-comment",
                                    placeholder=t("comment_optional", "Commentaire (Optionnel)"),
                                    value=comment,
                                    rows=3,
                                    className="mb-2",
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            t("btn_apply", "Appliquer"),
                                            id="btn-apply",
                                            color="primary",
                                            size="sm",
                                            className="me-2",
                                        ),
                                    ],
                                    className="text-end",
                                ),
                            ],
                            width=9,
                        ),
                    ]
                ),
            ]
        ),
        className="bg-light border-0",
    )

    # -- Match metadata block (overlap, fragmentation, suspicious, semantic judge) --
    match_meta = item.get("match_metadata") or {}
    match_metadata_section = html.Div()
    if match_meta:
        meta_pills: list = []

        ind_ov = match_meta.get("indicator_overlap")
        if ind_ov is not None:
            meta_pills.append(
                dbc.Badge(f"Overlap: {ind_ov:.2%}", color="info", className="me-1 mb-1")
            )
        eff_ov = match_meta.get("effective_label_overlap")
        if eff_ov is not None and eff_ov != ind_ov:
            meta_pills.append(
                dbc.Badge(f"Eff. overlap: {eff_ov:.2%}", color="info", className="me-1 mb-1")
            )

        frag_t1 = match_meta.get("fragmentation_detected_t1")
        frag_t2 = match_meta.get("fragmentation_detected_t2")
        if frag_t1:
            meta_pills.append(dbc.Badge("Frag. T1", color="warning", text_color="dark", className="me-1 mb-1"))
        if frag_t2:
            meta_pills.append(dbc.Badge("Frag. T2", color="warning", text_color="dark", className="me-1 mb-1"))

        if match_meta.get("suspicious_low_overlap"):
            reason = match_meta.get("suspicious_reason") or ""
            meta_pills.append(
                dbc.Badge("Suspicieux", color="danger", className="me-1 mb-1", title=reason)
            )

        sj = match_meta.get("semantic_judge")
        if isinstance(sj, dict) and sj.get("final_decision"):
            sj_decision = sj.get("final_decision", "")
            sj_conf = sj.get("original_gpt_decision", {})
            sj_confidence = float(sj_conf.get("confidence", 0.0) or 0.0) if isinstance(sj_conf, dict) else 0.0
            sj_guard = sj.get("guard_action", "")
            sj_color = {
                "match": "success", "no_match": "danger", "review": "warning",
                "structural_kept": "primary", "structural_fallback": "secondary",
            }.get(sj_decision, "secondary")
            sj_label = f"Judge: {sj_decision}"
            if sj_confidence:
                sj_label += f" ({sj_confidence:.0%})"
            meta_pills.append(dbc.Badge(sj_label, color=sj_color, className="me-1 mb-1"))
            if sj_guard and sj_guard != "none":
                meta_pills.append(
                    dbc.Badge(f"Rail: {sj_guard}", color="dark", text_color="white", className="me-1 mb-1")
                )

        if meta_pills:
            match_metadata_section = html.Div(
                [
                    html.H6("Signaux de qualite", className="text-muted small mb-2"),
                    html.Div(meta_pills, className="d-flex flex-wrap"),
                ],
                className="mb-3 p-2 border rounded",
            )

    _REL_DISPLAY = {
        "REGLEMENTAIRE": "Reglementaire",
        "NON_SIGNIFICATIF": "Non significatif",
        "STRUCTUREL": "Structurel",
        "NOUVELLE_DIVULGATION": "Nouvelle divulgation",
        "NON_CLASSIFIE": "Non classifie",
    }
    _REL_COLORS = {
        "REGLEMENTAIRE": "danger",
        "NON_SIGNIFICATIF": "secondary",
        "STRUCTUREL": "primary",
        "NOUVELLE_DIVULGATION": "info",
        "NON_CLASSIFIE": "light",
    }
    _RISK_DISPLAY = {
        "ELEVE": "Eleve",
        "MODERE": "Modere",
        "FAIBLE": "Faible",
    }
    _RISK_COLORS = {
        "ELEVE": "danger",
        "MODERE": "warning",
        "FAIBLE": "success",
    }

    genai_analysis_section = html.Div()
    ga = item.get("genai_analysis") or {}
    if ga.get("relevance"):
        rel = ga.get("relevance", "")
        risk = ga.get("risk_level", "")
        conf = ga.get("confidence", 0.0)
        just = ga.get("justification", "")
        rel_label = _REL_DISPLAY.get(rel, rel)
        risk_label = _RISK_DISPLAY.get(risk, risk)
        genai_analysis_section = html.Div(
            [
                html.H6("Analyse GenAI", className="text-muted small mb-2"),
                html.Div(
                    [
                        dbc.Badge(rel_label, color=_REL_COLORS.get(rel, "secondary"), className="me-2"),
                        dbc.Badge(f"Risque : {risk_label}", color=_RISK_COLORS.get(risk, "secondary"), className="me-2"),
                        html.Small(f"Confiance : {conf:.0%}", className="text-muted"),
                    ],
                    className="d-flex align-items-center mb-2",
                ),
                html.Div(
                    html.Small(just, className="text-muted fst-italic"),
                    className="p-2 bg-light rounded",
                ) if just else html.Div(),
            ],
            className="mb-3 p-2 border rounded",
        )

    return html.Div(
        [
            header_row,
            indicators_section,
            match_metadata_section,
            genai_analysis_section,
            footnote_detail_section,
            proofs_section,
            decision_section,
        ],
        className="h-100 d-flex flex-column",
    )
