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
    CHANGE_TYPE_STRUCTURE,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    CHANGE_TYPE_UNCERTAIN,
    EVENT_TYPE_MATCHED_PAIR,
    EVENT_TYPE_TABLE_ADDED,
    EVENT_TYPE_TABLE_REMOVED,
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
        return "Autre section"
    lowered = value.lower()
    if lowered in _SECTION_LABELS:
        return _SECTION_LABELS[lowered]
    if "capital" in lowered or "fonds propres" in lowered:
        return "Gestion du capital"
    if "risque" in lowered or "risk" in lowered:
        return "Gestion des risques"
    if "reglement" in lowered or "regulatory" in lowered:
        return "Reglementation"
    return "Autre section"


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
    """Compute visual flag state for proof cards (previous/current borders and badges)."""
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
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "neutral",
            "badge_class_t2": "neutral",
        }

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        return {
            "has_change": True,
            "t1_class": "proof-card",
            "t2_class": "proof-card proof-flag-t2",
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "neutral",
            "badge_class_t2": "t2",
        }
    if change_type == CHANGE_TYPE_TABLE_REMOVED:
        return {
            "has_change": True,
            "t1_class": "proof-card proof-flag-t1",
            "t2_class": "proof-card",
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "t1",
            "badge_class_t2": "neutral",
        }

    return {
        "has_change": True,
        "t1_class": "proof-card proof-flag-t1",
        "t2_class": "proof-card proof-flag-t2",
        "badge_t1": "Trimestre précédent",
        "badge_t2": "Trimestre courant",
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
        return (
            f"{prefix} {table_num}{page_part}{section_part}" or f"{prefix} {table_num}"
        )
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
        CHANGE_TYPE_UNCERTAIN: ("INCERTAIN", "secondary"),
        CHANGE_TYPE_STRUCTURE: ("FUSION/SPLIT", "primary"),
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


def _indicator_row_content(ind: dict, change_type: str) -> html.Div | html.Span:
    """Build the label content for an indicator row with type-specific styling."""
    name = ind.get("name", "")
    base_class = "small flex-grow-1"

    if change_type == CHANGE_TYPE_RENAMED:
        old_val = ind.get("from", "")
        new_val = ind.get("to", "")
        if old_val or new_val:
            return html.Div(
                [
                    html.Div(
                        [
                            html.Small("Précédent: ", className="fw-bold text-danger"),
                            html.Small(old_val or "-"),
                        ],
                        className="mb-0",
                    ),
                    html.Div(
                        [
                            html.Small("Courant: ", className="fw-bold text-success"),
                            html.Small(new_val or "-"),
                        ],
                        className="mb-0",
                    ),
                ],
                className=base_class,
            )
        return html.Span(name, className=f"{base_class} text-warning")

    if change_type == CHANGE_TYPE_ADDED:
        return html.Div(
            [
                html.I(className="bi bi-plus-circle-fill text-success me-1 small"),
                html.Span(name, className="text-success"),
            ],
            className=f"{base_class} d-flex align-items-center",
        )

    if change_type == CHANGE_TYPE_REMOVED:
        return html.Div(
            [
                html.I(className="bi bi-dash-circle-fill text-danger me-1 small"),
                html.Span(name, className="text-danger"),
            ],
            className=f"{base_class} d-flex align-items-center",
        )

    if change_type == CHANGE_TYPE_FOOTNOTE:
        return html.Span(name, className=f"{base_class} text-info")

    if change_type == CHANGE_TYPE_MODIFIED:
        return html.Span(name, className=f"{base_class} text-warning")

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        return html.Span(name, className=f"{base_class} text-info")
    if change_type == CHANGE_TYPE_TABLE_REMOVED:
        return html.Span(name, className=f"{base_class} text-dark")

    return html.Span(name, className=base_class)


def _bbox_normalized_for_overlay(bbox: list | None) -> list[float] | None:
    """Return [l, top, r, bottom] in 0..1 if valid for CSS % overlay, else None."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        l_, top, r, bottom = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
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
    proof_result_t1: dict | None = None,
    proof_result_t2: dict | None = None,
) -> html.Div:
    """Build proof images section; depends only on table-level data."""
    flag_state = compute_flag_state(item)
    normalized_mode = (proof_display_mode or "crop").strip().lower()
    mode_label_map = {
        "crop": "Mode focus tableau",
        "full": "Mode page complète + bbox",
        "footnote": "Mode note de bas de tableau",
        "full_without_bbox": "Mode page complète sans bbox",
    }

    def _mode_label(mode_value: str | None) -> str:
        return mode_label_map.get(
            (mode_value or normalized_mode).strip().lower(),
            "Mode focus tableau",
        )

    def _proof_caption(
        base_label: str, page: int | None, mode_value: str | None
    ) -> str:
        page_label = f"Page {page}" if page is not None else "Page indisponible"
        return f"{base_label} · {page_label} · {_mode_label(mode_value)}"

    def _proof_placeholder(
        placeholder: str | None, render_result: dict | None, mode_value: str
    ) -> str:
        if placeholder:
            return placeholder
        status = str((render_result or {}).get("status") or "").strip().lower()
        if status == "bbox_missing":
            if mode_value == "footnote":
                return "Zone footnote indisponible: bbox absente pour ce tableau."
            return "Crop indisponible: bbox absente pour ce tableau."
        if status == "page_missing":
            return "Page indisponible pour cette preuve."
        if status == "render_failed":
            return "Rendu impossible pour cette preuve."
        return t("image_unavailable", "Image non disponible")

    def _img_card(
        b64,
        label,
        *,
        mode_value: str,
        placeholder=None,
        bbox_norm=None,
        render_result=None,
        side: str = "t1",
    ):
        if not b64:
            msg = _proof_placeholder(placeholder, render_result, mode_value)
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
        if mode_value == "full" and bbox_norm:
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
                style={
                    "position": "relative",
                    "display": "inline-block",
                    "width": "100%",
                },
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
    mode_t1 = str((proof_result_t1 or {}).get("mode_effective") or normalized_mode)
    mode_t2 = str((proof_result_t2 or {}).get("mode_effective") or normalized_mode)
    bbox_t1_norm = (
        _bbox_normalized_for_overlay(item.get("bbox_t1")) if mode_t1 == "full" else None
    )
    bbox_t2_norm = (
        _bbox_normalized_for_overlay(item.get("bbox_t2")) if mode_t2 == "full" else None
    )

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        card_t1 = _img_card(
            None,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            placeholder=t(
                "no_table_added_t2",
                "Aucun tableau au trimestre précédent (ajout au trimestre courant)",
            ),
            render_result=proof_result_t1,
        )
        card_t2 = _img_card(
            img_t2_b64,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            bbox_norm=bbox_t2_norm,
            render_result=proof_result_t2,
            side="t2",
        )
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        card_t1 = _img_card(
            img_t1_b64,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            bbox_norm=bbox_t1_norm,
            render_result=proof_result_t1,
            side="t1",
        )
        card_t2 = _img_card(
            None,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            placeholder=t(
                "no_table_removed_t2",
                "Aucun tableau au trimestre courant (supprimé depuis le trimestre précédent)",
            ),
            render_result=proof_result_t2,
        )
    else:
        card_t1 = _img_card(
            img_t1_b64,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            bbox_norm=bbox_t1_norm,
            render_result=proof_result_t1,
            side="t1",
        )
        card_t2 = _img_card(
            img_t2_b64,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            bbox_norm=bbox_t2_norm,
            render_result=proof_result_t2,
            side="t2",
        )

    col_t1 = dbc.Col(
        _proof_wrapper(
            card_t1,
            flag_state["t1_class"],
            flag_state["badge_t1"],
            flag_state["badge_class_t1"],
        ),
        width=6,
    )
    col_t2 = dbc.Col(
        _proof_wrapper(
            card_t2,
            flag_state["t2_class"],
            flag_state["badge_t2"],
            flag_state["badge_class_t2"],
        ),
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
                        {"label": "Page complète + bbox", "value": "full"},
                        {"label": "Note de bas de table", "value": "footnote"},
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
        [col_t2, col_t1],
        className="mb-4 g-2",
        style={"height": "50vh", "minHeight": "400px"},
    )

    header = html.Div(
        [
            html.H6("Preuves visuelles T1/T2", className="mb-1"),
            html.P(
                "Référence visuelle pour valider rapidement le changement courant.",
                className="text-muted small mb-2",
            ),
        ]
    )

    return html.Div([header, proof_mode_toggle, proofs_row])


# ---------------------------------------------------------------------------
# GenAI analysis block (shared by table-only and indicator-diff views)
# ---------------------------------------------------------------------------


def _build_genai_analysis_section(item: dict) -> html.Div:
    """Build the GenAI analysis block for display in the detail panel.
    Always shows the section: with data when genai_analysis has relevance,
    or a placeholder when classification was not run or failed.
    """
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
    _RISK_DISPLAY = {"ELEVE": "Eleve", "MODERE": "Modere", "FAIBLE": "Faible"}
    _RISK_COLORS = {"ELEVE": "danger", "MODERE": "warning", "FAIBLE": "success"}
    ga = item.get("genai_analysis") or {}
    if not ga.get("relevance"):
        placeholder = (
            "Classification non exécutée. Activez l'option "
            "'Classer les changements avec l'IA générative (GPT-4o)' dans les options et relancez l'analyse."
        )
        return html.Div(
            [
                html.H6("Analyse IA générative", className="text-muted small mb-2"),
                html.Div(
                    html.Small(placeholder, className="text-muted fst-italic"),
                    className="p-2 bg-light rounded",
                ),
            ],
            className="mb-3 p-2 border rounded",
        )
    rel = ga.get("relevance", "")
    risk = ga.get("risk_level", "")
    conf = ga.get("confidence", 0.0)
    just = ga.get("justification", "")
    rel_label = _REL_DISPLAY.get(rel, rel)
    risk_label = _RISK_DISPLAY.get(risk, risk)
    return html.Div(
        [
            html.H6("Analyse IA générative", className="text-muted small mb-2"),
            html.Div(
                [
                    dbc.Badge(
                        rel_label,
                        color=_REL_COLORS.get(rel, "secondary"),
                        className="me-2",
                    ),
                    dbc.Badge(
                        f"Risque : {risk_label}",
                        color=_RISK_COLORS.get(risk, "secondary"),
                        className="me-2",
                    ),
                    html.Small(f"Confiance : {conf:.0%}", className="text-muted"),
                ],
                className="d-flex align-items-center mb-2",
            ),
            html.Div(
                html.Small(just, className="text-muted fst-italic"),
                className="p-2 bg-light rounded",
            )
            if just
            else html.Div(),
        ],
        className="mb-3 p-2 border rounded",
    )


# ---------------------------------------------------------------------------
# WHOLE-TABLE view: proof + GenAI + single "Validate table" action. No indicator list.
# ---------------------------------------------------------------------------


def render_table_only_view(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    current_idx: int,
    total_items: int,
    proof_display_mode: str = "crop",
    show_proofs: bool = True,
) -> html.Div:
    """Detail panel for table_added / table_removed. Unit of validation is the TABLE.
    Displays: table proof, GenAI analysis, single Validate/Reject table action.
    Must NOT display: Indicateurs (N), indicator rows (AJOUT/SUPPRESSION)."""
    table_display = table_display_label(item)
    page_t1 = item.get("page_t1")
    page_t2 = item.get("page_t2")
    if page_t1 is None and page_t2 is not None:
        page_text = f"Page courante: p.{page_t2}"
    elif page_t2 is None and page_t1 is not None:
        page_text = f"Page précédente: p.{page_t1}"
    else:
        page_t1_str = str(page_t1) if page_t1 is not None else "-"
        page_t2_str = str(page_t2) if page_t2 is not None else "-"
        page_text = f"Pages: précédent p.{page_t1_str}, courant p.{page_t2_str}"
    confidence = item.get("confidence", 0.0)
    review_status = item.get("review_status", REVIEW_STATUS_PENDING)

    header_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5(
                        t("detail_changement", "Detail du Changement"), className="mb-1"
                    ),
                    html.Div(
                        html.Span(
                            f"Tableau {current_idx + 1}/{total_items}",
                            className="me-3 fw-bold",
                        ),
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

    proofs_section = (
        build_proofs_section(
            item=item,
            img_t1_b64=img_t1_b64,
            img_t2_b64=img_t2_b64,
            proof_display_mode=proof_display_mode,
        )
        if show_proofs
        else html.Div()
    )
    genai_section = _build_genai_analysis_section(item)
    decision_section = dbc.Card(
        dbc.CardBody(
            [
                html.H6(
                    t("decision_analyst", "Decision de l'Analyste"), className="mb-3"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-check-lg me-2"),
                                        t("btn_approve", "Valider le tableau"),
                                    ],
                                    id="btn-approve",
                                    color="success"
                                    if review_status == REVIEW_STATUS_APPROVED
                                    else "outline-success",
                                    className="me-2 mb-2 w-100 text-start",
                                ),
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-x-lg me-2"),
                                        t("btn_reject", "Rejeter le tableau"),
                                    ],
                                    id="btn-reject",
                                    color="danger"
                                    if review_status == REVIEW_STATUS_REJECTED
                                    else "outline-danger",
                                    className="w-100 text-start",
                                ),
                            ],
                            width=12,
                        ),
                    ]
                ),
            ]
        ),
        className="bg-light border-0",
    )
    return html.Div(
        [header_row, proofs_section, genai_section, decision_section],
        className="h-100 d-flex flex-column",
    )


# ---------------------------------------------------------------------------
# INDICATOR-DIFF view: indicator list (AJOUT/SUPPRESSION/MODIF) + optional footnotes. Only for matched_pair / footnote_only.
# ---------------------------------------------------------------------------


def render_indicator_diff_view(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    current_idx: int,
    total_items: int,
    proof_display_mode: str = "crop",
    indicator_idx: int = 0,
    show_proofs: bool = True,
) -> html.Div:
    """Detail panel for matched_pair / footnote_only. Unit of validation is INDICATOR.
    Displays: indicator list (AJOUT/SUPPRESSION/MODIF), optional footnotes, per-indicator decisions."""
    table_display = table_display_label(item)
    page_t1 = item.get("page_t1")
    page_t2 = item.get("page_t2")
    if page_t1 is None and page_t2 is not None:
        page_text = f"Page courante: p.{page_t2}"
    elif page_t2 is None and page_t1 is not None:
        page_text = f"Page précédente: p.{page_t1}"
    else:
        page_t1_str = str(page_t1) if page_t1 is not None else "-"
        page_t2_str = str(page_t2) if page_t2 is not None else "-"
        page_text = f"Pages: précédent p.{page_t1_str}, courant p.{page_t2_str}"
    confidence = item.get("confidence", 0.0)
    comment = item.get("comment", "")
    indicators = item.get("indicators", [])

    def _ind_sort_key(ind: dict) -> int:
        assessment = ind.get("analyst_assessment", {})
        level = (
            assessment.get("relevance_level") if isinstance(assessment, dict) else None
        )
        if isinstance(level, int):
            return level
        return 999

    indicators = sorted(indicators, key=_ind_sort_key)
    n_indicators = len(indicators)

    n_decided = sum(
        1 for ind in indicators if ind.get("review_status", "pending") != "pending"
    )
    all_decided = n_decided == n_indicators if n_indicators else True

    if n_indicators:
        indicator_idx = max(0, min(indicator_idx, n_indicators - 1))
        current_ind = indicators[indicator_idx]
        current_ind_status = current_ind.get("review_status", "pending")
    else:
        current_ind = None
        current_ind_status = "pending"

    progress_text = (
        f"Indicateur {indicator_idx + 1}/{n_indicators}"
        if n_indicators
        else "Aucun indicateur"
    )

    header_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5(
                        t("detail_changement", "Detail du Changement"), className="mb-1"
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"Tableau {current_idx + 1}/{total_items}",
                                className="me-3 fw-bold",
                            ),
                            html.Span(
                                f" | {progress_text} ({n_decided}/{n_indicators} decides)",
                                className="text-primary fw-semibold",
                            )
                            if n_indicators
                            else html.Span(),
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

    def _render_assessment(ind_dict: dict) -> html.Div:
        assessment = ind_dict.get("analyst_assessment", {})
        if not assessment:
            return html.Div()
        level = assessment.get("relevance_level")
        just = str(assessment.get("justification", "")).strip()
        if not level and not just:
            return html.Div()

        badge_color = "secondary"
        badge_label = "Info"
        if level == 1:
            badge_color = "danger"
            badge_label = "Critique"
        elif level == 2:
            badge_color = "warning"
            badge_label = "Élevé"
        elif level == 3:
            badge_color = "info"
            badge_label = "Faible"

        return html.Div(
            [
                dbc.Badge(badge_label, color=badge_color, className="me-2"),
                html.Small(just, className="text-muted fst-italic"),
            ],
            className="mt-1 p-1 bg-white rounded border d-flex align-items-center w-100",
            style={"marginLeft": "32px"},
        )

    indicator_rows = []
    for i, ind in enumerate(indicators):
        name = ind.get("name", "")
        change_type = ind.get("type", "")
        ind_status = ind.get("review_status", "pending")
        is_current = i == indicator_idx

        row_class = (
            "d-flex flex-column py-2 border-bottom px-2 rounded align-items-start"
        )
        if is_current:
            row_class += " bg-primary bg-opacity-10 border border-primary"

        row_content = _indicator_row_content(ind, change_type)
        indicator_rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            _indicator_status_icon(ind_status),
                            _indicator_badge(change_type),
                            row_content,
                            html.I(className="bi bi-chevron-right text-primary")
                            if is_current
                            else html.Span(),
                        ],
                        className="d-flex align-items-center w-100",
                    ),
                    _render_assessment(ind),
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
                indicator_rows
                if indicator_rows
                else [
                    html.P(
                        t("no_indicators", "Aucun indicateur"),
                        className="text-muted small",
                    )
                ],
                className="mb-3",
                style={"maxHeight": "220px", "overflowY": "auto"},
            ),
        ],
    )

    footnote_detail_rows = []
    if item_type == "footnote":
        footnote_changes = item.get("footnote_changes", [])

        def _fn_sort_key(fc: dict) -> int:
            assessment = fc.get("analyst_assessment", {})
            level = (
                assessment.get("relevance_level")
                if isinstance(assessment, dict)
                else None
            )
            if isinstance(level, int):
                return level
            return 999

        footnote_changes = sorted(footnote_changes, key=_fn_sort_key)

        for fc in footnote_changes:
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
                        dbc.Badge(significance, color="secondary", className="me-1")
                        if significance
                        else None,
                        dbc.Badge(category, color="secondary", className="me-1")
                        if category
                        else None,
                    ],
                    className="d-flex align-items-center mb-1",
                ),
            ]
            if old_text:
                detail_children.append(
                    html.Div(
                        [
                            html.Small("Précédent: ", className="fw-bold text-danger"),
                            html.Small(old_text[:300]),
                        ],
                        className="ms-3 mb-1 bg-light p-1 rounded",
                    )
                )
            if new_text:
                detail_children.append(
                    html.Div(
                        [
                            html.Small("Courant: ", className="fw-bold text-success"),
                            html.Small(new_text[:300]),
                        ],
                        className="ms-3 mb-1 bg-light p-1 rounded",
                    )
                )
            assessment = fc.get("analyst_assessment", {})
            if assessment:
                level = assessment.get("relevance_level")
                just = str(assessment.get("justification", "")).strip()
                if level or just:
                    if level == 1:
                        badge_color = "danger"
                        badge_label = "Critique"
                    elif level == 2:
                        badge_color = "warning"
                        badge_label = "Élevé"
                    elif level == 3:
                        badge_color = "info"
                        badge_label = "Faible"
                    else:
                        badge_color = "secondary"
                        badge_label = "Info"
                    detail_children.append(
                        html.Div(
                            [
                                dbc.Badge(
                                    badge_label, color=badge_color, className="me-2"
                                ),
                                html.Small(just, className="text-muted fst-italic"),
                            ],
                            className="ms-3 mb-1 mt-1 p-1 bg-white border rounded d-flex align-items-center",
                        )
                    )

            footnote_detail_rows.append(
                html.Div(detail_children, className="py-2 border-bottom")
            )

    footnote_detail_section = (
        html.Div(
            [
                html.H6("Détail des notes", className="text-muted small mb-2"),
                html.Div(
                    footnote_detail_rows,
                    style={"maxHeight": "200px", "overflowY": "auto"},
                ),
            ],
            className="mb-3",
        )
        if footnote_detail_rows
        else html.Div()
    )

    proofs_section = (
        build_proofs_section(
            item=item,
            img_t1_b64=img_t1_b64,
            img_t2_b64=img_t2_b64,
            proof_display_mode=proof_display_mode,
        )
        if show_proofs
        else html.Div()
    )

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
                html.H6(
                    t("decision_analyst", "Decision de l'Analyste"), className="mb-3"
                ),
                decision_indicator_label,
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-check-lg me-2"),
                                        t("btn_approve", "Valider"),
                                    ],
                                    id="btn-approve",
                                    color="success"
                                    if current_ind_status == REVIEW_STATUS_APPROVED
                                    else "outline-success",
                                    className="me-2 mb-2 w-100 text-start",
                                ),
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-x-lg me-2"),
                                        t("btn_reject", "Rejeter"),
                                    ],
                                    id="btn-reject",
                                    color="danger"
                                    if current_ind_status == REVIEW_STATUS_REJECTED
                                    else "outline-danger",
                                    className="w-100 text-start",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Textarea(
                                    id="review-comment",
                                    placeholder=t(
                                        "comment_optional", "Commentaire (Optionnel)"
                                    ),
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

    match_metadata_section = html.Div()
    genai_analysis_section = _build_genai_analysis_section(item)

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


# ---------------------------------------------------------------------------
# Main detail panel: branch by event_type (business rule)
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
    """Build the right-side review detail panel. Dispatches by event_type:
    - table_added / table_removed: table-only view (proof + GenAI + single Validate table). No indicator list.
    - matched_pair / footnote_only: indicator-diff view (Indicateurs (N), AJOUT/SUPPRESSION, per-indicator decisions)."""
    event_type = (item.get("event_type") or "").strip() or EVENT_TYPE_MATCHED_PAIR
    if event_type in (EVENT_TYPE_TABLE_ADDED, EVENT_TYPE_TABLE_REMOVED):
        return render_table_only_view(
            item=item,
            img_t1_b64=img_t1_b64,
            img_t2_b64=img_t2_b64,
            current_idx=current_idx,
            total_items=total_items,
            proof_display_mode=proof_display_mode,
            show_proofs=show_proofs,
        )
    return render_indicator_diff_view(
        item=item,
        img_t1_b64=img_t1_b64,
        img_t2_b64=img_t2_b64,
        current_idx=current_idx,
        total_items=total_items,
        proof_display_mode=proof_display_mode,
        indicator_idx=indicator_idx,
        show_proofs=show_proofs,
    )
