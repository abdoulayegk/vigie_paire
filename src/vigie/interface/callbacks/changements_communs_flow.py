"""Callbacks for read-only common cross-bank changes."""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate

from vigie.comparaison.changements_communs import (
    build_changements_communs_source_stats,
    changements_communs_output_path,
    collect_changements_communs_records,
    latest_changements_communs_report_path,
    load_changements_communs_report,
)
from vigie.interface.layouts.page_changements_communs import (
    build_changements_communs_tab,
)
from vigie.support.quarter_utils import get_payload_quarter_context


@callback(
    Output("changements-communs-tab-content", "children"),
    Input("store-show-results-page", "data"),
    State("store-indicator-result", "data"),
    State("store-comparison-result", "data"),
    prevent_initial_call=True,
)
def render_changements_communs_tab(show_results, indicator_result, comparison_result):
    """Affiche le rapport interbanques associé à l'analyse terminée.

    Les stores de visibilité et de résultats déterminent le rapport à charger;
    la sortie remplace le contenu de l'onglet des changements communs.
    """
    if not show_results:
        raise PreventUpdate

    selected_payload = (
        indicator_result
        if isinstance(indicator_result, dict)
        else comparison_result
        if isinstance(comparison_result, dict)
        else None
    )
    selected_period = _period_from_payload(selected_payload)
    report = load_changements_communs_report(period=selected_period) if selected_period else None
    report_path = changements_communs_output_path(selected_period) if selected_period else None
    report_period = selected_period
    if report is None and not selected_period:
        report_path = latest_changements_communs_report_path()
        report = load_changements_communs_report(path=report_path) if report_path else None
        report_period = str((report or {}).get("period") or "").strip() or None

    records = collect_changements_communs_records(period=report_period)
    stats = build_changements_communs_source_stats(records)
    return build_changements_communs_tab(
        stats,
        report,
        selected_period=selected_period,
        report_path=str(report_path) if report_path else None,
    )


def _period_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    ctx = get_payload_quarter_context(payload)
    current = ctx.get("current") if isinstance(ctx, dict) else {}
    previous = ctx.get("previous") if isinstance(ctx, dict) else {}
    if not isinstance(current, dict) or not isinstance(previous, dict):
        return None
    current_year = current.get("year")
    previous_year = previous.get("year")
    current_code = str(current.get("code") or "").strip().lower()
    previous_code = str(previous.get("code") or "").strip().lower()
    if not all([current_year, previous_year, current_code, previous_code]):
        return None
    return f"{int(current_year)}_{current_code}_vs_{int(previous_year)}_{previous_code}"
