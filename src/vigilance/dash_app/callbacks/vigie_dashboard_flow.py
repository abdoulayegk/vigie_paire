"""Dashboard de vigie read-only ajoute comme troisieme onglet resultats.

Le rendu est reparti dans le sous-package ``vigie_dashboard`` : mise en forme,
agregation des indicateurs, composants graphiques et rapport PDF. Ce module
conserve les callbacks Dash et reste la facade publique.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from vigilance.quarter_utils import get_payload_quarter_context

from .vigie_dashboard.charts import (  # noqa: F401 - re-export de compatibilite
    _bar_chart,
    _chart_card,
    _donut_chart,
    _kpi,
    _priority_rows,
    _priority_table,
    _review_status_chart,
    _top_text,
)
from .vigie_dashboard.formatting import (  # noqa: F401 - re-export de compatibilite
    _badge_class,
    _change_label,
    _change_total,
    _comparisons,
    _confidence_label,
    _footnote_counts,
    _format_number,
    _impact_label,
    _low_confidence,
    _plot_layout,
    _quarter_label,
    _safe_int,
    _table_title,
    _updated_at,
)
from .vigie_dashboard.metrics import (  # noqa: F401 - re-export de compatibilite
    _indicator_metrics,
    _review_counts,
    _text_changes,
    _text_metrics,
)
from .vigie_dashboard.pdf_report import _build_pdf_report  # noqa: F401 - re-export


@callback(
    Output("vigie-cockpit-tab-content", "children"),
    Input("store-indicator-result", "data"),
    Input("store-comparison-result", "data"),
    Input("store-text-comparison", "data"),
    Input("store-review-items", "data"),
    Input("store-review-queue", "data"),
    Input("store-show-results-page", "data"),
    Input("store-vigie-dashboard-theme", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def render_vigie_cockpit(
    indicator,
    comparison,
    text_data,
    review_items,
    review_queue,
    show_results,
    dashboard_theme,
    indicator_meta,
):
    """Rendre le dashboard sans modifier les onglets existants."""
    if not show_results:
        raise PreventUpdate
    payload = indicator or comparison or {}
    if not payload and not text_data:
        return html.Div("Aucun résultat disponible pour le dashboard.", className="text-muted p-3")

    quarter_context = get_payload_quarter_context(payload if isinstance(payload, dict) else {})
    bank = str((payload or text_data or {}).get("bank_code") or "N/A").upper()
    current_label = _quarter_label(quarter_context["current"]["label"])
    previous_label = _quarter_label(quarter_context["previous"]["label"])
    text_metrics = _text_metrics(text_data)
    indicator_metrics = _indicator_metrics(payload)
    counts = _review_counts(review_queue, review_items, payload)
    theme = "light" if dashboard_theme == "light" else "dark"
    review_total = counts["total"]
    text_total = text_metrics["total"]
    indicator_total = indicator_metrics["total_changes"]
    bars = {
        "Ajouts": text_metrics["added_changes"]
        + indicator_metrics["indicator_added"]
        + indicator_metrics["tables_added"]
        + indicator_metrics["footnote_added"],
        "Suppressions": text_metrics["removed_changes"]
        + indicator_metrics["indicator_removed"]
        + indicator_metrics["tables_removed"]
        + indicator_metrics["footnote_removed"],
        "Modifications": text_metrics["modified"] + indicator_metrics["footnote_modified"],
        "Renommages": text_metrics["renamed_changes"] + indicator_metrics["renamed"],
    }
    pertinence = {"ELEVEE": "Élevée", "MOYENNE": "Moyenne", "FAIBLE": "Faible"}.get(
        text_metrics["pertinence"], text_metrics["pertinence"]
    )

    def _pct(value: int) -> str:
        """Retourne le pourcentage formaté d'une valeur par rapport au total de revue."""
        return f"({value / review_total:.0%})" if review_total else "(0%)"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Dashboard de vigie bancaire", className="vigie-cockpit-title"),
                            html.Div(
                                f"{bank} - Comparaison {current_label} vs {previous_label}",
                                className="vigie-cockpit-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Pipeline Texte & Indicateurs", className="vigie-cockpit-pipeline"),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.I(
                                                className=f"bi {'bi-sun' if theme == 'dark' else 'bi-moon'} me-2"
                                            ),
                                            "Mode clair" if theme == "dark" else "Mode sombre",
                                        ],
                                        id="btn-vigie-dashboard-theme",
                                        color="outline-light" if theme == "dark" else "outline-secondary",
                                        size="sm",
                                        className="vigie-cockpit-theme-toggle",
                                        n_clicks=0,
                                    ),
                                ],
                                className="vigie-cockpit-theme-control",
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-file-earmark-pdf me-2"), "Télécharger rapport PDF"],
                                id="btn-download-vigie-dashboard-pdf",
                                color="primary",
                                size="sm",
                                className="fw-semibold",
                            ),
                            html.Div(
                                f"Dernière mise à jour : {_updated_at(indicator_meta, payload, text_data)}",
                                className="vigie-cockpit-updated",
                            ),
                        ],
                        className="vigie-cockpit-header-meta",
                    ),
                ],
                className="vigie-cockpit-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("RÉSULTATS TEXTUELS", className="vigie-cockpit-section-title"),
                                    html.Span(f"Pertinence : {pertinence}", className=_badge_class(pertinence)),
                                ],
                                className="vigie-cockpit-panel-head",
                            ),
                            html.Div(
                                [
                                    _kpi(
                                        "bi-chat-square-text",
                                        "Total changements",
                                        text_metrics["total"],
                                        "Lignes auditables Excel",
                                        "info",
                                    ),
                                    _kpi(
                                        "bi-exclamation-triangle",
                                        "Majeur(s)",
                                        text_metrics["major"],
                                        "Cas à lire en priorité",
                                        "danger",
                                    ),
                                    _kpi(
                                        "bi-exclamation-circle",
                                        "Modéré(s)",
                                        text_metrics["moderate"],
                                        "Cas à surveiller",
                                        "warning",
                                    ),
                                    _kpi(
                                        "bi-check2-square",
                                        "Pertinents / analysés",
                                        f"{text_metrics['relevant']} / {text_metrics['analyzed']}",
                                        "Couverture de triage",
                                        "success",
                                    ),
                                    _kpi(
                                        "bi-bullseye",
                                        "Sections affectées",
                                        text_metrics["sections"],
                                        "Zones touchées",
                                        "danger",
                                    ),
                                    _kpi(
                                        "bi-shield-exclamation",
                                        "Changements réglementaires",
                                        text_metrics["regulatory"],
                                        "Signaux conformité",
                                        "neutral",
                                    ),
                                ],
                                className="vigie-cockpit-kpi-grid",
                            ),
                            html.Div(
                                "Principaux changements textuels détectés", className="vigie-cockpit-subpanel-title"
                            ),
                            _top_text(text_metrics),
                        ],
                        className="vigie-cockpit-pipeline-panel is-text",
                    ),
                    html.Div(
                        [
                            html.Div("RÉSULTATS INDICATEURS", className="vigie-cockpit-section-title"),
                            html.Div(
                                [
                                    _kpi(
                                        "bi-window",
                                        "Paires comparées",
                                        indicator_metrics["matched"],
                                        "Tableaux appariés",
                                        "info",
                                    ),
                                    _kpi(
                                        "bi-trash3",
                                        "Tableaux supprimés",
                                        indicator_metrics["tables_removed"],
                                        "Absents maintenant",
                                        "danger",
                                    ),
                                    _kpi(
                                        "bi-graph-up-arrow",
                                        "Indicateurs ajoutés",
                                        indicator_metrics["indicator_added"],
                                        "Ajouts identifiés",
                                        "success",
                                    ),
                                    _kpi(
                                        "bi-graph-down-arrow",
                                        "Indicateurs supprimés",
                                        indicator_metrics["indicator_removed"],
                                        "Suppressions identifiées",
                                        "danger",
                                    ),
                                    _kpi(
                                        "bi-pencil-square",
                                        "Indicateurs renommés",
                                        indicator_metrics["indicator_renamed"],
                                        "Renommages identifiés",
                                        "warning",
                                    ),
                                    _kpi(
                                        "bi-journal-plus",
                                        "Notes ajoutées",
                                        indicator_metrics["footnote_added"],
                                        "Nouvelles notes",
                                        "success",
                                    ),
                                    _kpi(
                                        "bi-journal-minus",
                                        "Notes supprimées",
                                        indicator_metrics["footnote_removed"],
                                        "Retraits de notes",
                                        "danger",
                                    ),
                                    _kpi(
                                        "bi-journal-text",
                                        "Notes modifiées",
                                        indicator_metrics["footnote_modified"],
                                        "Contenu ou portée",
                                        "warning",
                                    ),
                                    _kpi(
                                        "bi-clipboard-data",
                                        "Tableaux prioritaires",
                                        indicator_metrics["priority"],
                                        "Cas à traiter",
                                        "info",
                                    ),
                                    _kpi(
                                        "bi-exclamation-triangle",
                                        "Faible confiance",
                                        indicator_metrics["low_confidence"],
                                        "Appariements à relire",
                                        "danger",
                                    ),
                                ],
                                className="vigie-cockpit-kpi-grid",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("File de revue", className="vigie-cockpit-review-label"),
                                            html.Div(str(review_total), className="vigie-cockpit-review-value"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Validés", className="vigie-cockpit-review-label"),
                                            html.Div(str(counts["approved"]), className="vigie-cockpit-review-value"),
                                            html.Div(_pct(counts["approved"]), className="vigie-cockpit-review-pct"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Rejetés", className="vigie-cockpit-review-label"),
                                            html.Div(str(counts["rejected"]), className="vigie-cockpit-review-value"),
                                            html.Div(_pct(counts["rejected"]), className="vigie-cockpit-review-pct"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("En attente", className="vigie-cockpit-review-label"),
                                            html.Div(str(counts["pending"]), className="vigie-cockpit-review-value"),
                                            html.Div(_pct(counts["pending"]), className="vigie-cockpit-review-pct"),
                                        ]
                                    ),
                                ],
                                className="vigie-cockpit-review-strip",
                            ),
                        ],
                        className="vigie-cockpit-pipeline-panel",
                    ),
                ],
                className="vigie-cockpit-pipeline-layout",
            ),
            html.Div(
                [
                    _donut_chart(text_total, indicator_total, theme=theme),
                    _bar_chart(bars, theme=theme),
                    _review_status_chart(counts, theme=theme),
                ],
                className="vigie-cockpit-chart-grid",
            ),
            _priority_table(indicator_metrics, review_queue),
        ],
        className=f"vigie-cockpit is-{theme}",
    )


@callback(
    Output("store-vigie-dashboard-theme", "data"),
    Input("btn-vigie-dashboard-theme", "n_clicks"),
    State("store-vigie-dashboard-theme", "data"),
    prevent_initial_call=True,
)
def update_vigie_dashboard_theme(n_clicks, current_theme):
    """Memoriser le mode de luminosite du dashboard."""
    if not n_clicks:
        raise PreventUpdate
    return "dark" if current_theme == "light" else "light"


@callback(
    Output("download-vigie-dashboard-pdf", "data"),
    Input("btn-download-vigie-dashboard-pdf", "n_clicks"),
    State("store-indicator-result", "data"),
    State("store-comparison-result", "data"),
    State("store-text-comparison", "data"),
    State("store-review-items", "data"),
    State("store-review-queue", "data"),
    State("store-show-results-page", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def download_vigie_dashboard_pdf(
    n_clicks,
    indicator,
    comparison,
    text_data,
    review_items,
    review_queue,
    show_results,
    indicator_meta,
):
    """Télécharger le rapport PDF du dashboard de vigie."""
    if not n_clicks or not show_results:
        raise PreventUpdate
    payload = indicator or comparison or {}
    if not payload and not text_data:
        raise PreventUpdate

    quarter_context = get_payload_quarter_context(payload if isinstance(payload, dict) else {})
    bank = str((payload or text_data or {}).get("bank_code") or "bank").upper()
    current_label = _quarter_label(quarter_context["current"]["label"])
    previous_label = _quarter_label(quarter_context["previous"]["label"])
    text_metrics = _text_metrics(text_data)
    indicator_metrics = _indicator_metrics(payload)
    counts = _review_counts(review_queue, review_items, payload)
    bars = {
        "Ajouts": text_metrics["added_changes"]
        + indicator_metrics["indicator_added"]
        + indicator_metrics["tables_added"]
        + indicator_metrics["footnote_added"],
        "Suppressions": text_metrics["removed_changes"]
        + indicator_metrics["indicator_removed"]
        + indicator_metrics["tables_removed"]
        + indicator_metrics["footnote_removed"],
        "Modifications": text_metrics["modified"] + indicator_metrics["footnote_modified"],
        "Renommages": text_metrics["renamed_changes"] + indicator_metrics["renamed"],
    }
    priority_rows = _priority_rows(indicator_metrics, review_queue, limit=8)
    updated_at = _updated_at(indicator_meta, payload, text_data)
    pdf_bytes = _build_pdf_report(
        bank=bank,
        current_label=current_label,
        previous_label=previous_label,
        updated_at=updated_at,
        text_metrics=text_metrics,
        indicator_metrics=indicator_metrics,
        review_counts=counts,
        bars=bars,
        priority_rows=priority_rows,
        text_data=text_data,
    )
    filename = f"Rapport_Vigie_{bank}_{current_label}_vs_{previous_label}.pdf".replace(" ", "_")
    return dcc.send_bytes(pdf_bytes, filename)
