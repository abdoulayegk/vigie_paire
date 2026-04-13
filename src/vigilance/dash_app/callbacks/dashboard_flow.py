"""Callbacks de rendu du tableau de bord : resultats, KPIs, onglet sections, onglet tableau, initialisation des items de revue."""

from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from vigilance.comparison_canonical import (
    get_meta_value,
    is_canonical_comparison,
)
from vigilance.dash_app.components.review_display_shared import (
    section_display_label,
)
from vigilance.dash_app.layouts.page_results import build_analyst_kpi_card
from vigilance.dash_app.services.comparison_context import (
    _comparison_path_from_meta,
    _pdf_paths_from_comparison_meta,
)
from vigilance.dash_app.services.export_helpers import (
    _comparison_has_changes,
    _footnote_change_total,
    _is_high_priority_item,
    _is_low_confidence_comparison,
)
from vigilance.dash_app.services.review_navigation import (
    _build_kpi_card,
    _format_duration,
    _remember_selection,
    _resolve_selection,
)
from vigilance.dash_app.services.review_persistence import (
    _load_review_state_for_comparison,
    _persist_review_state,
    _stored_review_items_from_state,
)
from vigilance.i18n import t
from vigilance.quarter_utils import (
    get_payload_quarter_context,
    quarter_label_from_payload,
)
from vigilance.review_adapters import build_review_items_from_indicator_result
from vigilance.review_priority import sort_review_items_by_priority
from vigilance.review_queue_normalizer import build_normalized_review_queue
from vigilance.review_storage import is_review_state_compatible
from vigilance.ui_indicators import build_indicator_change_rows

logger = logging.getLogger(__name__)


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
        meta = indicator.get("meta", {}) or {}
        genai_text = get_meta_value(meta, "executive_summary", "content") or ""
        if genai_text:
            executive_summary = dbc.Alert(
                html.P(genai_text, className="mb-0 small"),
                color="info",
                className="mb-3 shadow-sm",
            )
        else:
            executive_summary = dbc.Alert(
                html.P(
                    "Résumé en cours de génération ou non disponible.",
                    className="mb-0 small text-muted",
                ),
                color="light",
                className="mb-3",
            )

    kpis = []
    if indicator:
        kpi = indicator.get("summary", indicator.get("kpi_metier", {}))
        comparisons = indicator.get("table_comparisons", []) or []
        tables_removed = indicator.get("tables_removed", []) or []
        tables_added = indicator.get("tables_added", []) or []

        notes_total = sum(_footnote_change_total(comp) for comp in comparisons)
        high_priority_tables = sum(
            1
            for comp in comparisons
            if _comparison_has_changes(comp) and _is_high_priority_item(comp)
        ) + sum(
            1
            for table in [*tables_added, *tables_removed]
            if _is_high_priority_item(table)
        )
        low_confidence_tables = sum(
            1
            for comp in comparisons
            if _comparison_has_changes(comp) and _is_low_confidence_comparison(comp)
        )

        top_kpi_cards = [
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_compared_pairs"),
                    int(kpi.get("tables_matched", 0) or 0),
                    color="white",
                    helper_text="Tableaux apparies entre les deux trimestres",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_removed_tables"),
                    len(tables_removed),
                    color="white",
                    helper_text="Tableaux presents avant, absents maintenant",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_added"),
                    int(kpi.get("total_added_indicators", 0) or 0)
                    + int(kpi.get("total_removed_indicators", 0) or 0)
                    + int(kpi.get("total_renamed_indicators", 0) or 0),
                    color="white",
                    helper_text="Ajouts, suppressions et renommages d'indicateurs",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_notes_modified"),
                    notes_total,
                    color="white",
                    helper_text="Toutes les notes de bas de tableau qui changent",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_priority_tables"),
                    high_priority_tables,
                    color="white",
                    helper_text="Cas a traiter en premier par l'analyste",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                build_analyst_kpi_card(
                    t("kpi_low_confidence_tables"),
                    low_confidence_tables,
                    color="white",
                    helper_text="Appariements ou diffs a relire avec prudence",
                ),
                xl=2,
                md=4,
                className="mb-3",
            ),
        ]
        kpis.append(dbc.Row(top_kpi_cards, className="g-3 mb-2"))
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


@callback(
    Output("kpi-tables-matched", "children"),
    Output("kpi-added-indicators", "children"),
    Output("kpi-removed-indicators", "children"),
    Output("kpi-renamed-indicators", "children"),
    Input("store-indicator-result", "data"),
)
def render_main_kpis(indicator_result):
    """Afficher les cartes KPI principales."""
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
    """Afficher la rangee de KPIs secondaires avec le temps de validation."""
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
    """Afficher l'onglet des changements par section avec accordeon."""
    if not show_results:
        raise PreventUpdate
    from vigilance.dash_app.layouts.page_results import build_section_accordion_item

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

    for tab in tables_added:
        section = tab.get("section", "Autres")
        if section not in sections:
            sections[section] = {"changes": [], "added": [], "removed": []}
        sections[section]["added"].append(tab)

    for tab in tables_removed:
        section = tab.get("section", "Autres")
        if section not in sections:
            sections[section] = {"changes": [], "added": [], "removed": []}
        sections[section]["removed"].append(tab)

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
    Output("store-review-last-positions", "data"),
    Output("store-current-change-idx", "data"),  # V2: reset change index
    Input("store-indicator-result", "data"),
    Input("store-pdf-paths", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def init_review_items(indicator_result, paths, indicator_meta):
    """Construire les ReviewItems depuis indicator_result pour la revue.

    Construit egalement la file de revue V2 dedupliquee.
    """
    if not indicator_result:
        raise PreventUpdate

    runtime_paths = paths if isinstance(paths, dict) else {}
    meta_paths = _pdf_paths_from_comparison_meta(indicator_meta, indicator_result)
    effective_paths = dict(runtime_paths)
    for key, value in meta_paths.items():
        if value:
            effective_paths[key] = value
    path_t1 = effective_paths.get("pdf_previous", "") or effective_paths.get(
        "pdf_t1", ""
    )
    path_t2 = effective_paths.get("pdf_current", "") or effective_paths.get(
        "pdf_t2", ""
    )
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

    persisted_state = _load_review_state_for_comparison(
        indicator_meta=indicator_meta,
        indicator_result=indicator_result,
    )
    stored_items = _stored_review_items_from_state(persisted_state)
    if persisted_state:
        if not is_review_state_compatible(
            persisted_state,
            review_items=serialized,
            stored_review_items=stored_items,
        ):
            logger.info(
                "Persisted review state is incompatible with current comparison payload — ignoring saved review items."
            )
            persisted_state = None
            stored_items = None

    if persisted_state and isinstance(stored_items, list) and stored_items:
        serialized = stored_items

    # V2: Build deduplicated review queue from the active review-items only.
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
    persisted_selection = (
        persisted_state.get("review_selection")
        if isinstance(persisted_state, dict)
        else None
    )
    resolved_selection, sel_table_idx, sel_change_idx = _resolve_selection(
        serialized_v2, persisted_selection or {"review_id": None, "change_id": None}
    )
    logger.info(
        "[init_review_items] v1_count=%s v2_count=%s dedup_merged=%s persisted_selection=%s",
        total,
        total_v2,
        dedup_merged,
        bool(persisted_selection),
    )
    _persist_review_state(
        indicator_meta=indicator_meta,
        indicator_result=indicator_result,
        review_items=serialized,
        review_queue=serialized_v2,
        review_selection=resolved_selection,
        review_current_idx=sel_table_idx,
        current_change_idx=sel_change_idx,
        preferred_store="review_queue",
        source="init_review_items",
    )
    return (
        serialized,
        serialized_v2,
        resolved_selection,
        _remember_selection({}, resolved_selection),
        sel_change_idx,
    )


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
