"""Page des resultats de comparaison avec design moderne et onglets par section."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigilance.i18n import t


def build_analyst_kpi_card(
    title: str,
    value: str | int,
    delta_icon: str | None = None,
    color: str = "light",
    helper_text: str | None = None,
) -> dbc.Card:
    """Construit une carte KPI individuelle pour le panneau analyste.

    Args:
        title: Libelle affiche au-dessus de la valeur.
        value: Valeur principale du KPI (texte ou entier).
        delta_icon: Icone optionnelle de variation affichee sous la valeur.
        color: Classe de couleur Bootstrap pour le fond de la carte.
        helper_text: Texte d'aide optionnel affiche sous la valeur.

    Returns:
        Composant ``dbc.Card`` representant la carte KPI.
    """
    body_children = [
        html.P(title, className="text-muted mb-1 small"),
        html.H3(str(value), className="mb-0 fw-bold"),
    ]
    if helper_text:
        body_children.append(html.P(helper_text, className="text-muted mb-0 small"))
    if delta_icon:
        body_children.append(html.Span(delta_icon, className="text-muted small"))
    return dbc.Card(
        dbc.CardBody(body_children, className="text-center py-3"),
        className=f"shadow-sm border-0 bg-{color}",
    )


def _format_priority_badge(raw_priority: str) -> dbc.Badge | None:
    """Retourne un badge Bootstrap correspondant au niveau de priorite."""
    priority = str(raw_priority or "").strip().lower()
    if not priority:
        return None

    mapping = {
        "critique": ("Critique", "danger"),
        "prioritaire": ("Prioritaire", "warning"),
        "normal": ("Normal", "secondary"),
    }
    label, color = mapping.get(priority, (priority.capitalize(), "secondary"))
    return dbc.Badge(label, color=color, className="me-2")


def _format_confidence_badge(score: float | int | None) -> dbc.Badge | None:
    """Retourne un badge colore selon le score de confiance."""
    if score is None:
        return None
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return None

    if numeric >= 0.9:
        color = "success"
    elif numeric >= 0.8:
        color = "warning"
    else:
        color = "danger"
    return dbc.Badge(f"Confiance {numeric:.2f}", color=color, className="me-2")


def _build_change_badges(comp: dict) -> list[dbc.Badge]:
    """Construit la liste de badges resumant les changements d'une comparaison."""
    badges: list[dbc.Badge] = []
    added = len(comp.get("added_indicators", []) or [])
    removed = len(comp.get("removed_indicators", []) or [])
    renamed = len(comp.get("renamed_indicators", []) or [])
    fn_counts = comp.get("footnotes_counts", {}) or {}
    fn_total = sum(int(fn_counts.get(key, 0) or 0) for key in ("added", "removed", "modified"))

    if added:
        badges.append(dbc.Badge(f"+{added}", color="success", className="me-2"))
    if removed:
        badges.append(dbc.Badge(f"-{removed}", color="danger", className="me-2"))
    if renamed:
        badges.append(dbc.Badge(f"~{renamed}", color="warning", text_color="dark", className="me-2"))
    if fn_total:
        badges.append(dbc.Badge(f"FN {fn_total}", color="info", className="me-2"))
    return badges


def _build_indicator_detail_lines(comp: dict) -> list:
    """Construit les lignes de detail des indicateurs ajoutes, supprimes et renommes."""
    details: list = []
    added = [str(v).strip() for v in (comp.get("added_indicators", []) or []) if str(v).strip()]
    removed = [str(v).strip() for v in (comp.get("removed_indicators", []) or []) if str(v).strip()]
    renamed = comp.get("renamed_indicators", []) or []
    footnotes_diff = comp.get("footnotes_diff", {}) or {}

    if added:
        details.append(
            html.Div(
                [html.Strong("Ajouts: "), html.Span(", ".join(added[:3]))],
                className="small mb-1",
            )
        )
    if removed:
        details.append(
            html.Div(
                [html.Strong("Suppressions: "), html.Span(", ".join(removed[:3]))],
                className="small mb-1",
            )
        )
    if renamed:
        preview = []
        for item in renamed[:2]:
            if isinstance(item, dict):
                prev = str(item.get("from", "")).strip()
                curr = str(item.get("to", "")).strip()
                if prev or curr:
                    preview.append(f"{prev} -> {curr}")
        if preview:
            details.append(
                html.Div(
                    [html.Strong("Renommages: "), html.Span(" | ".join(preview))],
                    className="small mb-1",
                )
            )

    footnote_refs: list[str] = []
    for bucket in ("added", "removed", "modified"):
        for item in footnotes_diff.get(bucket, []) or []:
            ref = str(item.get("footnote_ref", "")).strip()
            if ref:
                footnote_refs.append(f"[{ref}]")
    if footnote_refs:
        details.append(
            html.Div(
                [html.Strong("Notes: "), html.Span(", ".join(footnote_refs[:4]))],
                className="small mb-1",
            )
        )

    analyst_summary = str(
        (comp.get("match_metadata", {}) or {}).get("analyst_summary")
        or (comp.get("genai_analysis", {}) or {}).get("analyst_summary")
        or ""
    ).strip()
    if analyst_summary:
        details.append(html.P(analyst_summary, className="small text-muted mb-0"))

    return details


def _build_comparison_overview_card(comp: dict) -> dbc.Card:
    """Construit une carte resume pour un tableau avec changements."""
    title = (
        comp.get("title_t2")
        or comp.get("title_t1")
        or comp.get("table_title")
        or comp.get("table_id_t2")
        or comp.get("table_id_t1")
        or "Sans titre"
    )
    page_t1 = comp.get("page_t1")
    page_t2 = comp.get("page_t2")
    priority_badge = _format_priority_badge(
        str(
            (comp.get("match_metadata", {}) or {}).get("review_priority")
            or (comp.get("genai_analysis", {}) or {}).get("review_priority")
            or ""
        )
    )
    confidence_badge = _format_confidence_badge(comp.get("match_score"))
    change_badges = _build_change_badges(comp)
    detail_lines = _build_indicator_detail_lines(comp)

    badge_row = [badge for badge in [priority_badge, confidence_badge, *change_badges] if badge is not None]

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6(title, className="mb-1"),
                        html.Small(
                            f"Pages: préc. p.{page_t1 if page_t1 is not None else '-'} / cour. p.{page_t2 if page_t2 is not None else '-'}",
                            className="text-muted",
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(badge_row, className="mb-2") if badge_row else html.Div(),
                html.Div(detail_lines)
                if detail_lines
                else html.P(
                    "Aucun détail textuel supplémentaire.",
                    className="small text-muted mb-0",
                ),
            ]
        ),
        className="shadow-sm border-0 mb-3",
    )


def _build_table_presence_card(table: dict, *, change_label: str, color: str) -> dbc.Card:
    """Construit une carte pour un tableau ajoute ou retire."""
    title = table.get("title") or table.get("table_title") or table.get("table_id") or "Sans titre"
    page = table.get("page")
    indicators = [
        str(v).strip()
        for v in (
            table.get("first_column_indicators_raw")
            or table.get("all_indicators_t1")
            or table.get("all_indicators_t2")
            or []
        )
        if str(v).strip()
    ]
    excerpt = ", ".join(indicators[:3]) if indicators else "Aucun indicateur exploitable"
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6(title, className="mb-1"),
                        dbc.Badge(change_label, color=color, className="me-2"),
                        html.Small(
                            f"Page p.{page}" if page is not None else "Page inconnue",
                            className="text-muted",
                        ),
                    ],
                    className="mb-2",
                ),
                html.P(excerpt, className="small text-muted mb-0"),
            ]
        ),
        className="shadow-sm border-0 mb-3",
    )


def build_section_accordion_item(
    section_name: str,
    tables_with_changes: list,
    tables_added: list,
    tables_removed: list,
    item_id: str,
) -> dbc.AccordionItem:
    """Construit un element d'accordeon pour l'onglet des changements par section.

    Args:
        section_name: Nom de la section affiche dans le titre de l'accordeon.
        tables_with_changes: Comparaisons de tableaux comportant des changements.
        tables_added: Tableaux presents uniquement dans le trimestre courant.
        tables_removed: Tableaux presents uniquement dans le trimestre precedent.
        item_id: Identifiant HTML de l'element d'accordeon.

    Returns:
        Composant ``dbc.AccordionItem`` representant la section.
    """
    parts = []
    if tables_with_changes:
        n = len(tables_with_changes)
        cards = [_build_comparison_overview_card(comp) for comp in tables_with_changes[:6]]
        parts.append(
            html.Div(
                [
                    html.Strong(f"{n} tableau(x) avec changements"),
                    html.Div(cards, className="mt-3"),
                    html.Small(
                        f"{n - len(cards)} autre(s) tableau(x) dans cette section." if n > len(cards) else "",
                        className="text-muted",
                    ),
                ],
                className="mb-2",
            )
        )
    if tables_added:
        cards = [
            _build_table_presence_card(
                table,
                change_label=t("table_added"),
                color="success",
            )
            for table in tables_added[:3]
        ]
        parts.append(
            html.Div(
                [
                    html.Strong(f"{t('table_added_plural')}: {len(tables_added)}"),
                    html.Div(cards, className="mt-3"),
                    html.Small(
                        f"{len(tables_added) - len(cards)} autre(s) tableau(x) ajouté(s)."
                        if len(tables_added) > len(cards)
                        else "",
                        className="text-muted",
                    ),
                ],
                className="mb-2",
            )
        )
    if tables_removed:
        cards = [
            _build_table_presence_card(
                table,
                change_label=t("table_removed"),
                color="danger",
            )
            for table in tables_removed[:3]
        ]
        parts.append(
            html.Div(
                [
                    html.Strong(f"{t('table_removed_plural')}: {len(tables_removed)}"),
                    html.Div(cards, className="mt-3"),
                    html.Small(
                        f"{len(tables_removed) - len(cards)} autre(s) tableau(x) retiré(s)."
                        if len(tables_removed) > len(cards)
                        else "",
                        className="text-muted",
                    ),
                ],
                className="mb-2",
            )
        )
    body = html.Div(parts, className="p-2") if parts else html.Div("Aucun détail.", className="text-muted p-2")
    return dbc.AccordionItem(
        body,
        title=section_name,
        id=item_id,
    )


def build_page_results() -> html.Div:
    """Construit le layout principal de la page de resultats.

    Inclut les KPI, le panneau de revue analyste et les statistiques de validation.

    Returns:
        Composant ``html.Div`` contenant l'ensemble de la page de resultats.
    """
    return html.Div(
        id="results-content",
        children=[
            # Header Section
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H2(t("analyse_comparative"), className="mb-1"),
                                    html.Div(id="results-header", className="text-muted mb-0"),
                                ],
                                width=9,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [
                                        html.I(className="bi bi-arrow-clockwise me-2"),
                                        t("nouvelle_analyse"),
                                    ],
                                    id="btn-reset",
                                    color="outline-secondary",
                                    size="sm",
                                    className="float-end mt-2",
                                ),
                                width=3,
                            ),
                        ]
                    )
                ],
                className="mb-3",
            ),
            # Tabs: Dashboard / Indicateurs / Analyse Textuelle
            html.Div(
                dbc.Tabs(
                    [
                        dbc.Tab(
                            html.Div(id="vigie-cockpit-tab-content"),
                            label="Dashboard",
                            tab_id="tab-cockpit",
                        ),
                        dbc.Tab(
                            html.Div(
                                [
                                    html.Div(id="results-executive-summary", className="mb-3 mt-3"),
                                    html.Div(id="results-kpis"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                build_analyst_kpi_card(t("file_review_total"), "0", color="white"),
                                                width=3,
                                                className="mb-3",
                                                id="kpi-queue-total",
                                            ),
                                            dbc.Col(
                                                build_analyst_kpi_card(t("validated"), "0", color="white"),
                                                width=3,
                                                className="mb-3",
                                                id="kpi-queue-approved",
                                            ),
                                            dbc.Col(
                                                build_analyst_kpi_card(t("rejected"), "0", color="white"),
                                                width=3,
                                                className="mb-3",
                                                id="kpi-queue-rejected",
                                            ),
                                            dbc.Col(
                                                build_analyst_kpi_card(t("pending"), "0", color="white"),
                                                width=3,
                                                className="mb-3",
                                                id="kpi-queue-pending",
                                            ),
                                        ],
                                        className="g-3 mb-4",
                                    ),
                                    html.Div(id="results-sections-tab", style={"display": "none"}),
                                ]
                            ),
                            label="Indicateurs",
                            tab_id="tab-indicateurs",
                        ),
                        dbc.Tab(
                            html.Div(id="text-analysis-tab-content", className="mt-3"),
                            label="Analyse Textuelle",
                            tab_id="tab-texte",
                        ),
                    ],
                    id="results-main-tabs",
                    active_tab="tab-cockpit",
                ),
                className="mb-5",
            ),
            html.Div(
                dbc.Row(
                    [
                        # Left Panel: Review Queue
                        dbc.Col(
                            html.Div(
                                id="review-queue-container",
                                className="bg-white p-3 shadow-sm rounded h-100",
                                style={"overflowY": "hidden"},
                            ),
                            md=4,
                            className="h-100",
                        ),
                        # Right Panel: Detail View + Nav (buttons in layout so callbacks fire)
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(id="review-progress-banner", className="mb-3"),
                                        html.Div(
                                            id="review-proof-container",
                                            className="mb-3",
                                        ),
                                        html.Div(id="review-meta-container"),
                                    ],
                                    id="review-detail-container",
                                    className="bg-white p-4 shadow-sm rounded",
                                    style={"overflowY": "auto", "minHeight": "400px"},
                                ),
                            ],
                            md=8,
                            className="h-100 d-flex flex-column",
                        ),
                    ],
                    className="g-4 h-100",
                    style={"minHeight": "600px", "height": "calc(100vh - 280px)"},
                ),
                className="mb-5",
            ),
            # Footer: Validation Client Statistics
            html.Div(
                [
                    html.H5(t("statistiques_validation"), className="mb-3"),
                    dbc.Progress(
                        [
                            dbc.Progress(
                                value=0,
                                color="success",
                                bar=True,
                                id="progress-approved",
                            ),
                            dbc.Progress(
                                value=0,
                                color="danger",
                                bar=True,
                                id="progress-rejected",
                            ),
                            dbc.Progress(
                                value=100,
                                color="warning",
                                bar=True,
                                id="progress-pending",
                            ),
                        ],
                        className="mb-3",
                        style={"height": "20px"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="stats-validation-time",
                                    className="text-muted small",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(
                                    id="stats-export-status",
                                    className="text-end text-muted small",
                                ),
                                width=6,
                            ),
                        ]
                    ),
                    html.Hr(className="my-4"),
                    html.Div(id="results-export-tab"),
                ],
                className="p-4 bg-light rounded shadow-sm",
            ),
            html.Div(id="results-table-tab", style={"display": "none"}),
        ],
        className="p-4",
    )
