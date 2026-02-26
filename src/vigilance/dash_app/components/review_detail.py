"""Review Detail Component -- table-grouped view."""

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
    REVIEW_STATUS_REJECTED,
)


def _table_display_label(item: dict) -> str:
    """Format tableau label: 'Tableau n°X: Titre' when id exists, else 'Tableau: Titre'."""
    title = (item.get("table_name") or item.get("table_title_raw") or "").strip() or "Sans titre"
    change_type = item.get("change_type", "")
    if change_type == CHANGE_TYPE_TABLE_ADDED:
        table_id = (item.get("table_id_t2") or "").strip()
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        table_id = (item.get("table_id_t1") or "").strip()
    else:
        table_id = (item.get("table_id_t2") or item.get("table_id_t1") or "").strip()
    if table_id:
        return f"{t('table_no_prefix', 'Tableau n°')}{table_id}: {title}"
    return f"{t('table', 'Tableau')}: {title}"


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


def build_review_detail(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    current_idx: int,
    total_items: int,
) -> html.Div:
    """Build the right-side review detail panel (table-grouped)."""

    table_display = _table_display_label(item)
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
    review_status = item.get("review_status")
    indicators = item.get("indicators", [])

    header_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5(t("detail_changement", "Detail du Changement"), className="mb-1"),
                    html.Div(
                        [
                            html.Span(f"Tableau {current_idx + 1}/{total_items}", className="me-3 fw-bold"),
                        ],
                        className="small text-muted mb-2",
                    ),
                    html.Div(
                        [
                            html.Span(table_display, className="d-block text-truncate fw-semibold"),
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
    for ind in indicators:
        name = ind.get("name", "")
        change_type = ind.get("type", "")
        indicator_rows.append(
            html.Div(
                [
                    _indicator_badge(change_type),
                    html.Span(name, className="small"),
                ],
                className="d-flex align-items-center py-1 border-bottom",
            )
        )

    item_type = item.get("item_type", "indicator")
    if item_type == "footnote":
        section_title = f"Notes de bas de tableau ({len(indicators)})"
    else:
        section_title = f"{t('indicators')} ({len(indicators)})"

    indicators_section = html.Div(
        [
            html.H6(section_title, className="text-muted small mb-2"),
            html.Div(
                indicator_rows if indicator_rows else [html.P(t("no_indicators", "Aucun indicateur"), className="text-muted small")],
                className="mb-3",
                style={"maxHeight": "180px", "overflowY": "auto"},
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

    footnote_detail_section = html.Div(
        [
            html.H6("Detail des notes", className="text-muted small mb-2"),
            html.Div(footnote_detail_rows, style={"maxHeight": "200px", "overflowY": "auto"}),
        ],
        className="mb-3",
    ) if footnote_detail_rows else html.Div()

    def _img_card(b64, label, placeholder=None):
        """Placeholder: message explicite si pas d'image (ex. tableau ajoute/supprime)."""
        if not b64:
            msg = placeholder if placeholder else t("image_unavailable", "Image non disponible")
            return dbc.Card(
                dbc.CardBody(html.P(msg, className="text-muted text-center small")),
                className="h-100 bg-light border-0",
            )
        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/png;base64,{b64}",
                    top=True,
                    style={"objectFit": "contain", "maxHeight": "calc(50vh - 60px)"},
                ),
                dbc.CardFooter(html.Small(label, className="text-muted"), className="border-0 bg-white p-1"),
            ],
            className="h-100 border shadow-sm overflow-hidden",
        )

    change_type = item.get("change_type", "")
    if change_type == CHANGE_TYPE_TABLE_ADDED:
        card_t1 = _img_card(None, "T1 (Ancien)", placeholder=t("no_table_added_t2", "Aucun tableau (ajoute en T2)"))
        card_t2 = _img_card(img_t2_b64, "T2 (Nouveau)")
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        card_t1 = _img_card(img_t1_b64, "T1 (Ancien)")
        card_t2 = _img_card(None, "T2 (Nouveau)", placeholder=t("no_table_removed_t2", "Aucun tableau (supprime en T2)"))
    else:
        card_t1 = _img_card(img_t1_b64, "T1 (Ancien)")
        card_t2 = _img_card(img_t2_b64, "T2 (Nouveau)")

    proofs_row = dbc.Row(
        [
            dbc.Col(card_t1, width=6),
            dbc.Col(card_t2, width=6),
        ],
        className="mb-4 g-2",
        style={"height": "50vh", "minHeight": "400px"},
    )

    decision_section = dbc.Card(
        dbc.CardBody(
            [
                html.H6(t("decision_analyst", "Decision de l'Analyste"), className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="bi bi-check-lg me-2"), t("btn_approve", "Valider")],
                                    id="btn-approve",
                                    color="success" if review_status == REVIEW_STATUS_APPROVED else "outline-success",
                                    className="me-2 mb-2 w-100 text-start",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-x-lg me-2"), t("btn_reject", "Rejeter")],
                                    id="btn-reject",
                                    color="danger" if review_status == REVIEW_STATUS_REJECTED else "outline-danger",
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
                                        dbc.Button(t("btn_apply", "Appliquer"), id="btn-apply", color="primary", size="sm", className="me-2"),
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
            genai_analysis_section,
            footnote_detail_section,
            proofs_row,
            decision_section,
        ],
        className="h-100 d-flex flex-column",
    )
