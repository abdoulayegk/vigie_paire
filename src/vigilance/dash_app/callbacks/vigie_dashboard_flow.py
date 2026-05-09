"""Dashboard de vigie read-only ajoute comme troisieme onglet resultats."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from datetime import datetime
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from vigilance.dash_app.services.export_helpers import _is_high_priority_item
from vigilance.dash_app.services.review_navigation import _table_decision_bucket
from vigilance.quarter_utils import get_payload_quarter_context
from vigilance.text_comparison.text_comparison_excel import _should_exclude

_PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#d7dee9", "size": 11},
    "margin": {"l": 36, "r": 18, "t": 8, "b": 28},
    "legend": {"font": {"color": "#c8d2e0", "size": 11}},
}

_PDF_NAVY = colors.HexColor("#0b1725")
_PDF_PANEL = colors.HexColor("#101f31")
_PDF_TEXT = colors.HexColor("#172033")
_PDF_MUTED = colors.HexColor("#5f6b7a")
_PDF_BLUE = colors.HexColor("#4b74f2")
_PDF_RED = colors.HexColor("#e45142")
_PDF_AMBER = colors.HexColor("#f3b23c")
_PDF_GREEN = colors.HexColor("#68b976")


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _format_number(value: int | float | str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ")
    return str(value)


def _pdf_text(value: Any, max_len: int | None = None) -> str:
    text = str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = " ".join(text.split())
    if max_len and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _quarter_label(label: Any) -> str:
    text = str(label or "").strip().upper().replace("_", "-")
    if text.startswith("T"):
        return text.replace("T", "Q", 1)
    return text or "N/D"


def _comparisons(indicator: dict | None) -> list[dict]:
    if not isinstance(indicator, dict):
        return []
    table_comparisons = indicator.get("table_comparisons")
    if isinstance(table_comparisons, list):
        return table_comparisons
    pair_comparisons = indicator.get("pair_comparisons")
    return pair_comparisons if isinstance(pair_comparisons, list) else []


def _technical_diff(comp: dict) -> dict:
    return comp.get("technical_diff", {}) or {}


def _table_title(comp: dict) -> str:
    current = comp.get("current_table", {}) or {}
    previous = comp.get("previous_table", {}) or {}
    return str(
        comp.get("title_t2")
        or comp.get("title_t1")
        or comp.get("table_title")
        or current.get("title")
        or previous.get("title")
        or comp.get("current_table_id")
        or comp.get("previous_table_id")
        or "Sans titre"
    )


def _indicator_confidence(comp: dict) -> float | None:
    raw = comp.get("match_score", comp.get("match_confidence"))
    if raw is None:
        return None
    return _safe_float(raw)


def _low_confidence(comp: dict) -> bool:
    score = _indicator_confidence(comp)
    if score is not None:
        normalized = score if score <= 1 else score / 100
        if normalized < 0.85:
            return True
    status = str(comp.get("table_status", "") or "").strip().lower()
    meta = comp.get("match_metadata", {}) or {}
    return status in {"incertain", "needs_review"} or bool(meta.get("drastic_row_drop", False))


def _count_list(comp: dict, canonical_key: str, diff_key: str) -> int:
    diff = _technical_diff(comp)
    return len(comp.get(canonical_key, []) or diff.get(diff_key, []) or [])


def _footnote_counts(comp: dict) -> dict[str, int]:
    existing = comp.get("footnotes_counts", {}) or {}
    diff = _technical_diff(comp)
    diff_renamed = len(diff.get("footnotes_renamed", []) or [])
    if existing:
        renamed = _safe_int(existing.get("renamed"), diff_renamed)
        modified = _safe_int(existing.get("modified"))
        if not existing.get("renamed") and diff_renamed and modified >= diff_renamed:
            modified -= diff_renamed
        return {
            "added": _safe_int(existing.get("added")),
            "removed": _safe_int(existing.get("removed")),
            "modified": modified,
            "renamed": renamed,
        }
    return {
        "added": len(diff.get("footnotes_added", []) or []),
        "removed": len(diff.get("footnotes_removed", []) or []),
        "modified": len(diff.get("footnotes_modified", []) or []),
        "renamed": diff_renamed,
    }


def _change_total(comp: dict) -> int:
    notes = _footnote_counts(comp)
    return (
        _count_list(comp, "added_indicators", "indicators_added")
        + _count_list(comp, "removed_indicators", "indicators_removed")
        + _count_list(comp, "renamed_indicators", "indicators_renamed")
        + notes["added"]
        + notes["removed"]
        + notes["modified"]
    )


def _change_label(comp: dict) -> str:
    notes = _footnote_counts(comp)
    if _count_list(comp, "added_indicators", "indicators_added"):
        return "Ajout d'indicateurs"
    if _count_list(comp, "removed_indicators", "indicators_removed"):
        return "Suppression d'indicateurs"
    if _count_list(comp, "renamed_indicators", "indicators_renamed"):
        return "Renommage d'indicateurs"
    if sum(notes.values()):
        return "Note(s) modifiée(s)"
    return str(_technical_diff(comp).get("table_level_change") or "Modification")


def _impact_label(comp: dict) -> str:
    triage = comp.get("genai_triage", {}) or comp.get("genai_analysis", {}) or {}
    assessment = comp.get("analyst_assessment", {}) or {}
    raw = triage.get("impact_level") or assessment.get("change_significance") or assessment.get("review_priority") or ""
    normalized = str(raw).strip().upper()
    if normalized in {"ELEVE", "ELEVEE", "CRITIQUE", "PRIORITAIRE"}:
        return "Élevé"
    if normalized in {"MODERE", "MOYEN", "MOYENNE"}:
        return "Moyen"
    if normalized in {"FAIBLE", "MINEUR", "NORMAL", "NORMALE"}:
        return "Faible"
    if _low_confidence(comp) or _is_high_priority_item(comp):
        return "Élevé"
    return "Moyen" if _change_total(comp) else "Faible"


def _confidence_label(score: float | None) -> str:
    if score is None:
        return "N/D"
    pct = score * 100 if score <= 1 else score
    if pct >= 80:
        return "Élevée"
    if pct >= 50:
        return "Moyenne"
    return "Faible"


def _badge_class(label: str) -> str:
    normalized = str(label or "").lower()
    if any(token in normalized for token in ("majeur", "élev", "eleve", "critique", "faible")):
        return "vigie-cockpit-badge is-danger"
    if any(token in normalized for token in ("mod", "moyen", "prioritaire", "attente")):
        return "vigie-cockpit-badge is-warning"
    if any(token in normalized for token in ("valid", "élevée", "elevee")):
        return "vigie-cockpit-badge is-success"
    return "vigie-cockpit-badge is-muted"


def _text_changes(text_data: dict | None, *, relevant_only: bool = True) -> list[tuple[dict, str]]:
    if not isinstance(text_data, dict):
        return []
    rows: list[tuple[dict, str]] = []
    for section in text_data.get("section_comparisons") or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("section_title") or section.get("section_key") or "Section")
        for change in section.get("all_block_comparisons") or []:
            if not isinstance(change, dict):
                continue
            triage = change.get("genai_triage") or {}
            if change.get("diff_type") == "unchanged" or triage.get("source") == "skip":
                continue
            if _should_exclude(change):
                continue
            if relevant_only:
                if triage and not bool(triage.get("is_relevant", False)):
                    continue
            rows.append((change, section_title))
    return rows


def _text_metrics(text_data: dict | None) -> dict[str, Any]:
    summary = (text_data or {}).get("global_summary") or (text_data or {}).get("all_changes_summary") or {}
    counts = summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}
    exportable_changes = _text_changes(text_data, relevant_only=False)
    relevant_changes = [
        (change, section)
        for change, section in exportable_changes
        if bool((change.get("genai_triage") or {}).get("is_relevant", False))
    ]
    sections = {section for _, section in exportable_changes}
    added_words = 0
    removed_words = 0
    added_changes = 0
    removed_changes = 0
    modified = 0
    renamed_changes = 0
    regulatory = 0
    confidences: list[float] = []
    top: list[dict[str, str]] = []
    for change, section in exportable_changes:
        triage = change.get("genai_triage") or {}
        diff_type = str(change.get("diff_type") or "")
        if diff_type == "added":
            added_changes += 1
        if diff_type == "removed":
            removed_changes += 1
        if diff_type == "modified":
            modified += 1
        if diff_type == "renamed":
            renamed_changes += 1
        themes = {str(v).upper() for v in triage.get("themes_amf") or []}
        if "EXIGENCES_REGLEMENTAIRES" in themes or str(triage.get("category", "")).upper() == "REGLEMENTAIRE":
            regulatory += 1
        added_words += len(str(change.get("source_text_t2") or change.get("semantic_text_t2") or "").split())
        removed_words += len(str(change.get("source_text_t1") or change.get("semantic_text_t1") or "").split())
        if triage.get("confidence") is not None:
            confidences.append(_safe_float(triage.get("confidence")))
        top.append(
            {
                "summary": str(
                    change.get("change_summary") or triage.get("explanation") or "Changement textuel détecté"
                ),
                "impact": str(triage.get("impact_level") if triage.get("is_relevant") else "NON_PERTINENT").upper(),
                "section": section,
            }
        )
    top.sort(key=lambda item: {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2, "NON_PERTINENT": 3}.get(item["impact"], 9))
    return {
        "major": _safe_int(by_impact.get("MAJEUR")),
        "moderate": _safe_int(by_impact.get("MODERE")),
        "total": len(exportable_changes),
        "relevant": len(relevant_changes),
        "analyzed": len(exportable_changes),
        "sections": len(sections),
        "modified": modified,
        "renamed_changes": renamed_changes,
        "added_changes": added_changes,
        "removed_changes": removed_changes,
        "added_mentions": added_words,
        "removed_mentions": removed_words,
        "words_impacted": added_words + removed_words,
        "regulatory": regulatory or _safe_int((counts.get("by_category") or {}).get("REGLEMENTAIRE")),
        "pertinence": str(summary.get("pertinence_globale") or "N/D").upper(),
        "confidence_values": confidences,
        "top": top[:5],
    }


def _indicator_metrics(indicator: dict | None) -> dict[str, Any]:
    if not isinstance(indicator, dict):
        return {"comparisons": [], "confidence_values": [], "total_changes": 0}
    summary = indicator.get("summary", indicator.get("kpi_metier", {})) or {}
    comparisons = _comparisons(indicator)
    tables_added = indicator.get("tables_added", []) or []
    tables_removed = indicator.get("tables_removed", []) or []
    added = sum(_count_list(comp, "added_indicators", "indicators_added") for comp in comparisons)
    removed = sum(_count_list(comp, "removed_indicators", "indicators_removed") for comp in comparisons)
    indicator_renamed = sum(_count_list(comp, "renamed_indicators", "indicators_renamed") for comp in comparisons)
    footnote_added = sum(_footnote_counts(comp).get("added", 0) for comp in comparisons)
    footnote_removed = sum(_footnote_counts(comp).get("removed", 0) for comp in comparisons)
    footnote_modified = sum(_footnote_counts(comp).get("modified", 0) for comp in comparisons)
    footnote_renamed = sum(_footnote_counts(comp).get("renamed", 0) for comp in comparisons)
    renamed = indicator_renamed + footnote_renamed
    notes = footnote_added + footnote_removed + footnote_modified + footnote_renamed
    confidence_values = [score for score in (_indicator_confidence(comp) for comp in comparisons) if score is not None]
    tables_added_count = len(tables_added) or _safe_int(summary.get("tables_added_total"))
    tables_removed_count = len(tables_removed) or _safe_int(summary.get("tables_removed_total"))
    indicator_added_count = added or _safe_int(summary.get("total_added_indicators"))
    indicator_removed_count = removed or _safe_int(summary.get("total_removed_indicators"))
    indicator_renamed_count = indicator_renamed or _safe_int(summary.get("total_renamed_indicators"))
    renamed_count = indicator_renamed_count + footnote_renamed
    notes_count = notes or _safe_int(summary.get("footnote_changes_total"))
    return {
        "matched": _safe_int(
            summary.get("tables_matched"), _safe_int(summary.get("matched_pairs_total"), len(comparisons))
        ),
        "tables_removed": tables_removed_count,
        "tables_added": tables_added_count,
        "indicator_added": indicator_added_count,
        "indicator_removed": indicator_removed_count,
        "indicator_renamed": indicator_renamed_count,
        "footnote_added": footnote_added,
        "footnote_removed": footnote_removed,
        "footnote_modified": footnote_modified,
        "footnote_renamed": footnote_renamed,
        "renamed": renamed_count,
        "notes": notes_count,
        "priority": _safe_int(
            summary.get("high_priority_items_total"),
            sum(1 for comp in comparisons if _change_total(comp) and _is_high_priority_item(comp)),
        ),
        "low_confidence": sum(1 for comp in comparisons if _low_confidence(comp)),
        "total_changes": (
            indicator_added_count
            + indicator_removed_count
            + renamed_count
            + footnote_added
            + footnote_removed
            + footnote_modified
            + tables_added_count
            + tables_removed_count
        ),
        "confidence_values": confidence_values,
        "comparisons": comparisons,
    }


def _review_counts(review_queue: list | None, review_items: list | None, indicator: dict | None) -> dict[str, int]:
    queue = review_queue if isinstance(review_queue, list) else []
    if queue:
        total = len(queue)
        approved = sum(1 for item in queue if _table_decision_bucket(item) == "approved")
        rejected = sum(1 for item in queue if _table_decision_bucket(item) == "rejected")
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": max(0, total - approved - rejected),
        }
    items = review_items if isinstance(review_items, list) else []
    if items:
        statuses = Counter(str(item.get("status", "pending")).lower() for item in items if isinstance(item, dict))
        total = len(items)
        approved = statuses.get("approved", 0) + statuses.get("validated", 0) + statuses.get("valide", 0)
        rejected = statuses.get("rejected", 0) + statuses.get("rejete", 0)
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": max(0, total - approved - rejected),
        }
    summary = (indicator or {}).get("review_decisions_summary", {}) if isinstance(indicator, dict) else {}
    total = _safe_int(summary.get("matched")) + _safe_int(summary.get("unmatched"))
    pending = _safe_int(summary.get("pending"), total)
    return {"total": total, "approved": max(0, total - pending), "rejected": 0, "pending": pending}


def _kpi(icon: str, label: str, value: int | str, helper: str, tone: str = "neutral") -> html.Div:
    return html.Div(
        [
            html.Div(html.I(className=f"bi {icon}"), className=f"vigie-cockpit-kpi-icon is-{tone}"),
            html.Div(
                [
                    html.Div(label, className="vigie-cockpit-kpi-label"),
                    html.Div(_format_number(value), className="vigie-cockpit-kpi-value"),
                    html.Div(helper, className="vigie-cockpit-kpi-helper"),
                ],
                className="min-w-0",
            ),
        ],
        className="vigie-cockpit-kpi-card",
    )


def _chart_card(title: str, figure: go.Figure) -> html.Div:
    figure.update_layout(**_PLOT_LAYOUT)
    return html.Div(
        [
            html.Div(title, className="vigie-cockpit-panel-title"),
            dcc.Graph(figure=figure, config={"displayModeBar": False}, className="vigie-cockpit-graph"),
        ],
        className="vigie-cockpit-panel",
    )


def _donut_chart(text_total: int, indicator_total: int) -> html.Div:
    total = text_total + indicator_total
    if total <= 0:
        return html.Div("Aucun changement disponible", className="vigie-cockpit-empty")
    fig = go.Figure(
        go.Pie(
            labels=["Changements textuels", "Changements indicateurs"],
            values=[text_total, indicator_total],
            hole=0.62,
            marker={"colors": ["#e45142", "#4b74f2"]},
            textinfo="none",
        )
    )
    fig.add_annotation(text=str(total), x=0.5, y=0.55, showarrow=False, font={"size": 28, "color": "#f4f7fb"})
    fig.add_annotation(
        text="Total des<br>changements", x=0.5, y=0.42, showarrow=False, font={"size": 11, "color": "#c8d2e0"}
    )
    return _chart_card("APERÇU COMBINÉ", fig)


def _bar_chart(values: dict[str, int]) -> html.Div:
    fig = go.Figure(
        go.Bar(
            x=list(values.values()),
            y=list(values.keys()),
            orientation="h",
            marker={"color": ["#4b74f2", "#e45142", "#f3b23c", "#68b976"]},
            text=[str(v) for v in values.values()],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(xaxis={"gridcolor": "#223248"}, yaxis={"gridcolor": "rgba(0,0,0,0)"})
    return _chart_card("RÉPARTITION PAR NATURE", fig)


def _global_evolution_chart(values: dict[str, int]) -> html.Div:
    fig = go.Figure(
        go.Bar(
            x=list(values.keys()),
            y=list(values.values()),
            marker={"color": ["#68b976", "#e45142", "#f3b23c", "#4b74f2"]},
            text=[str(v) for v in values.values()],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        yaxis={
            "title": {"text": "Nombre de changements", "font": {"color": "#c8d2e0", "size": 11}},
            "gridcolor": "#223248",
            "rangemode": "tozero",
        },
        xaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
    return _chart_card("ÉVOLUTION GLOBALE", fig)


def _top_text(text_metrics: dict[str, Any]) -> html.Div:
    rows = []
    for item in text_metrics.get("top") or []:
        label = {
            "MAJEUR": "Majeur",
            "MODERE": "Modéré",
            "MINEUR": "Faible",
            "NON_PERTINENT": "Non pertinent",
        }.get(item["impact"], item["impact"].title())
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(item["summary"], className="vigie-cockpit-change-title"),
                            html.Div(item["section"], className="vigie-cockpit-change-meta"),
                        ],
                        className="min-w-0",
                    ),
                    html.Span(label, className=_badge_class(label)),
                ],
                className="vigie-cockpit-change-row",
            )
        )
    return html.Div(rows or ["Aucun changement textuel prioritaire."], className="vigie-cockpit-change-list")


def _priority_table(indicator_metrics: dict[str, Any], review_queue: list | None) -> html.Div:
    rows = _priority_rows(indicator_metrics, review_queue, limit=6)
    body = []
    for idx, row in enumerate(rows, start=1):
        body.append(
            html.Tr(
                [
                    html.Td(str(idx)),
                    html.Td(row["title"], className="vigie-cockpit-table-name"),
                    html.Td(row["change"]),
                    html.Td("Indicateurs"),
                    html.Td(html.Span(row["impact"], className=_badge_class(row["impact"]))),
                    html.Td(html.Span(row["confidence"], className=_badge_class(row["confidence"]))),
                    html.Td(html.Span(row["status_label"], className=_badge_class(row["status_label"]))),
                ]
            )
        )
    return html.Div(
        [
            html.Div("TOP TABLEAUX À PRIORISER", className="vigie-cockpit-panel-title"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("#"),
                                html.Th("Nom du tableau"),
                                html.Th("Type"),
                                html.Th("Pipeline"),
                                html.Th("Impact"),
                                html.Th("Confiance"),
                                html.Th("Statut"),
                            ]
                        )
                    ),
                    html.Tbody(body or [html.Tr(html.Td("Aucun tableau prioritaire détecté.", colSpan=7))]),
                ],
                borderless=True,
                responsive=True,
                size="sm",
                className="vigie-cockpit-table mb-0",
            ),
        ],
        className="vigie-cockpit-panel",
    )


def _priority_rows(indicator_metrics: dict[str, Any], review_queue: list | None, *, limit: int = 6) -> list[dict[str, Any]]:
    queue_lookup = {}
    if isinstance(review_queue, list):
        for table in review_queue:
            if isinstance(table, dict):
                queue_lookup[str(table.get("table_name") or table.get("table_title") or "")] = _table_decision_bucket(
                    table
                )
    rows = []
    for comp in indicator_metrics.get("comparisons") or []:
        if _change_total(comp) <= 0:
            continue
        title = _table_title(comp)
        score = _indicator_confidence(comp)
        rows.append(
            {
                "title": title,
                "change": _change_label(comp),
                "impact": _impact_label(comp),
                "confidence": _confidence_label(score),
                "status": queue_lookup.get(title, "pending"),
                "rank": (0 if _is_high_priority_item(comp) else 1, 0 if _low_confidence(comp) else 1, -(score or 0)),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    result: list[dict[str, Any]] = []
    for row in rows[:limit]:
        status = {"approved": "Validé", "rejected": "Rejeté", "pending": "En attente"}.get(row["status"], "En attente")
        result.append({**row, "status_label": status})
    return result


def _updated_at(indicator_meta: dict | None, indicator: dict | None, text_data: dict | None) -> str:
    for source in (indicator_meta, indicator, text_data):
        if not isinstance(source, dict):
            continue
        for key in ("updated_at", "created_at", "generated_at"):
            raw = source.get(key)
            if raw:
                try:
                    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime("%d %b %Y à %Hh%M")
                except ValueError:
                    return str(raw)
    return "Non disponible"


class _DashboardCharts(Flowable):
    """Dessine les graphiques principaux du rapport PDF."""

    def __init__(self, *, text_total: int, indicator_total: int, bars: dict[str, int]) -> None:
        super().__init__()
        self.text_total = text_total
        self.indicator_total = indicator_total
        self.bars = bars
        self.width = 7.2 * inch
        self.height = 3.45 * inch

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        self._draw_donut(canvas, 0.05 * inch, 0.2 * inch)
        self._draw_bars(canvas, 3.1 * inch, 0.2 * inch)
        canvas.restoreState()

    def _draw_donut(self, canvas, x: float, y: float) -> None:
        total = max(1, self.text_total + self.indicator_total)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(x, y + 2.95 * inch, "APERÇU COMBINÉ")
        cx = x + 1.05 * inch
        cy = y + 1.55 * inch
        radius = 0.82 * inch
        start = 90
        for value, color in ((self.text_total, _PDF_RED), (self.indicator_total, _PDF_BLUE)):
            extent = -360 * (value / total)
            canvas.setFillColor(color)
            canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius, start, extent, stroke=0, fill=1)
            start += extent
        canvas.setFillColor(colors.white)
        inner = 0.43 * inch
        canvas.circle(cx, cy, inner, stroke=0, fill=1)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawCentredString(cx, cy + 0.05 * inch, str(self.text_total + self.indicator_total))
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_PDF_MUTED)
        canvas.drawCentredString(cx, cy - 0.18 * inch, "changements")
        legend_x = x + 2.15 * inch
        canvas.setFont("Helvetica", 8)
        for idx, (label, value, color) in enumerate(
            (
                ("Texte narratif", self.text_total, _PDF_RED),
                ("Indicateurs / notes", self.indicator_total, _PDF_BLUE),
            )
        ):
            yy = y + 2.15 * inch - idx * 0.25 * inch
            canvas.setFillColor(color)
            canvas.rect(legend_x, yy, 0.11 * inch, 0.11 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.drawString(legend_x + 0.18 * inch, yy, f"{label}: {value}")

    def _draw_bars(self, canvas, x: float, y: float) -> None:
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(x, y + 2.95 * inch, "RÉPARTITION PAR NATURE")
        max_value = max(1, *(self.bars.values() or [1]))
        colors_by_label = {
            "Ajouts": _PDF_GREEN,
            "Suppressions": _PDF_RED,
            "Modifications": _PDF_AMBER,
            "Renommages": _PDF_BLUE,
        }
        bar_width = 2.75 * inch
        for idx, (label, value) in enumerate(self.bars.items()):
            yy = y + 2.35 * inch - idx * 0.48 * inch
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(_PDF_TEXT)
            canvas.drawString(x, yy + 0.03 * inch, label)
            canvas.setFillColor(colors.HexColor("#edf2f7"))
            canvas.rect(x + 1.15 * inch, yy, bar_width, 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(colors_by_label.get(label, _PDF_BLUE))
            canvas.rect(x + 1.15 * inch, yy, bar_width * (value / max_value), 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(x + 4.0 * inch, yy + 0.02 * inch, str(value))


class _ReportBarChart(Flowable):
    """Dessine un graphique a barres pleine page pour le rapport PDF."""

    def __init__(
        self,
        *,
        title: str,
        values: list[tuple[str, int, colors.Color]],
        orientation: str = "vertical",
        width: float = 7.0 * inch,
        height: float = 4.8 * inch,
    ) -> None:
        super().__init__()
        self.title = title
        self.values = values
        self.orientation = orientation
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(0, self.height - 0.15 * inch, self.title)
        if self.orientation == "horizontal":
            self._draw_horizontal(canvas)
        else:
            self._draw_vertical(canvas)
        canvas.restoreState()

    def _draw_vertical(self, canvas) -> None:
        chart_x = 0.45 * inch
        chart_y = 0.55 * inch
        chart_w = self.width - 0.9 * inch
        chart_h = self.height - 1.25 * inch
        values = [value for _, value, _ in self.values] or [0]
        max_value = max(1, max(values))
        canvas.setStrokeColor(colors.HexColor("#d7dee9"))
        canvas.setLineWidth(0.5)
        canvas.line(chart_x, chart_y, chart_x + chart_w, chart_y)
        canvas.line(chart_x, chart_y, chart_x, chart_y + chart_h)
        for step in range(1, 5):
            yy = chart_y + chart_h * step / 4
            canvas.setStrokeColor(colors.HexColor("#edf2f7"))
            canvas.line(chart_x, yy, chart_x + chart_w, yy)
            canvas.setFillColor(_PDF_MUTED)
            canvas.setFont("Helvetica", 7)
            canvas.drawRightString(chart_x - 0.08 * inch, yy - 0.03 * inch, str(round(max_value * step / 4)))
        gap = 0.18 * inch
        bar_w = min(0.75 * inch, (chart_w - gap * (len(self.values) + 1)) / max(1, len(self.values)))
        for idx, (label, value, color) in enumerate(self.values):
            x = chart_x + gap + idx * (bar_w + gap)
            h = chart_h * value / max_value
            canvas.setFillColor(color)
            canvas.rect(x, chart_y, bar_w, h, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawCentredString(x + bar_w / 2, chart_y + h + 0.08 * inch, str(value))
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(x + bar_w / 2, chart_y - 0.22 * inch, _pdf_text(label, 18))

    def _draw_horizontal(self, canvas) -> None:
        chart_x = 1.65 * inch
        chart_y = 0.55 * inch
        chart_w = self.width - 2.0 * inch
        values = [value for _, value, _ in self.values] or [0]
        max_value = max(1, max(values))
        row_h = min(0.48 * inch, (self.height - 1.35 * inch) / max(1, len(self.values)))
        for idx, (label, value, color) in enumerate(self.values):
            y = chart_y + (len(self.values) - idx - 1) * row_h
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(chart_x - 0.12 * inch, y + 0.11 * inch, _pdf_text(label, 26))
            canvas.setFillColor(colors.HexColor("#edf2f7"))
            canvas.rect(chart_x, y + 0.06 * inch, chart_w, 0.18 * inch, stroke=0, fill=1)
            canvas.setFillColor(color)
            canvas.rect(chart_x, y + 0.06 * inch, chart_w * value / max_value, 0.18 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(chart_x + chart_w + 0.08 * inch, y + 0.08 * inch, str(value))


class _ReportDonutChart(Flowable):
    """Dessine un graphique anneau pleine page pour le rapport PDF."""

    def __init__(
        self,
        *,
        title: str,
        values: list[tuple[str, int, colors.Color]],
        width: float = 7.0 * inch,
        height: float = 4.8 * inch,
    ) -> None:
        super().__init__()
        self.title = title
        self.values = values
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(0, self.height - 0.15 * inch, self.title)
        total = max(1, sum(value for _, value, _ in self.values))
        cx = 1.75 * inch
        cy = 2.25 * inch
        radius = 1.3 * inch
        start = 90
        for _, value, color in self.values:
            extent = -360 * value / total
            canvas.setFillColor(color)
            canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius, start, extent, stroke=0, fill=1)
            start += extent
        canvas.setFillColor(colors.white)
        canvas.circle(cx, cy, 0.68 * inch, stroke=0, fill=1)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(cx, cy + 0.08 * inch, str(sum(value for _, value, _ in self.values)))
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(_PDF_MUTED)
        canvas.drawCentredString(cx, cy - 0.22 * inch, "changements")
        legend_x = 3.55 * inch
        for idx, (label, value, color) in enumerate(self.values):
            yy = 3.1 * inch - idx * 0.42 * inch
            pct = value / total
            canvas.setFillColor(color)
            canvas.rect(legend_x, yy, 0.16 * inch, 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(legend_x + 0.28 * inch, yy, f"{label}: {value} ({pct:.0%})")


def _pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "VigieTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=_PDF_NAVY,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "VigieH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=_PDF_NAVY,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "VigieH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=_PDF_TEXT,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "VigieBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=_PDF_TEXT,
            spaceAfter=4,
        ),
        "muted": ParagraphStyle(
            "VigieMuted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=_PDF_MUTED,
        ),
        "center": ParagraphStyle(
            "VigieCenter",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=_PDF_TEXT,
            alignment=TA_CENTER,
        ),
    }


def _pdf_table(data: list[list[Any]], widths: list[float] | None = None, *, header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7dee9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _PDF_NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _build_pdf_report(
    *,
    bank: str,
    current_label: str,
    previous_label: str,
    updated_at: str,
    text_metrics: dict[str, Any],
    indicator_metrics: dict[str, Any],
    review_counts: dict[str, int],
    bars: dict[str, int],
    priority_rows: list[dict[str, Any]],
    text_data: dict | None,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"Rapport de vigie {bank}",
    )
    styles = _pdf_styles()
    story: list[Any] = []

    def p(text: Any, style: str = "body") -> Paragraph:
        return Paragraph(_pdf_text(text), styles[style])

    story.append(p("Rapport de vigie bancaire", "title"))
    story.append(p("Document interne - confidentiel", "h1"))
    story.append(p(f"{bank} - Comparaison {current_label} vs {previous_label}", "h1"))
    story.append(p(f"Sources: pipelines texte et indicateurs | Généré le {updated_at}", "muted"))
    story.append(Spacer(1, 0.18 * inch))
    total_changes = text_metrics["total"] + indicator_metrics["total_changes"]
    executive = (
        f"{total_changes} changement(s) détecté(s), dont {text_metrics['major']} majeur(s), "
        f"{text_metrics['moderate']} modéré(s), {indicator_metrics['priority']} tableau(x) prioritaire(s) "
        f"et {indicator_metrics['low_confidence']} appariement(s) à faible confiance."
    )
    story.append(p(executive, "body"))
    story.append(Spacer(1, 0.12 * inch))
    kpis = [
        ["Dimension", "Valeur", "Lecture analyste"],
        ["Total changements", _format_number(total_changes), "Volume global à couvrir"],
        ["Texte narratif", _format_number(text_metrics["total"]), "Tous changements auditables"],
        ["Indicateurs / notes", _format_number(indicator_metrics["total_changes"]), "Tables, indicateurs et notes"],
        ["Pertinents / analysés", f"{text_metrics['relevant']} / {text_metrics['analyzed']}", "Couverture de triage"],
        ["File de revue", _format_number(review_counts["total"]), "Cas à suivre côté revue"],
        ["En attente", _format_number(review_counts["pending"]), "Actions ouvertes"],
    ]
    story.append(_pdf_table(kpis, [1.55 * inch, 1.1 * inch, 4.15 * inch]))

    story.append(PageBreak())
    story.append(p("Vue de synthèse", "h1"))
    story.append(
        _ReportDonutChart(
            title="Répartition des changements par source",
            values=[
                ("Texte narratif", text_metrics["total"], _PDF_RED),
                ("Indicateurs et notes", indicator_metrics["total_changes"], _PDF_BLUE),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    source_rows = [
        ["Source", "Nombre", "Lecture analyste"],
        ["Texte narratif", _format_number(text_metrics["total"]), "Changements détectés dans les sections narratives"],
        ["Indicateurs et notes", _format_number(indicator_metrics["total_changes"]), "Changements détectés dans les tableaux et notes"],
    ]
    story.append(_pdf_table(source_rows, [1.7 * inch, 1.0 * inch, 4.0 * inch]))

    story.append(PageBreak())
    story.append(p("Répartition par nature", "h1"))
    story.append(
        _ReportBarChart(
            title="Types de changements détectés",
            values=[
                ("Ajouts", bars.get("Ajouts", 0), _PDF_GREEN),
                ("Suppressions", bars.get("Suppressions", 0), _PDF_RED),
                ("Modifications", bars.get("Modifications", 0), _PDF_AMBER),
                ("Renommages", bars.get("Renommages", 0), _PDF_BLUE),
            ],
            orientation="horizontal",
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    nature = [["Nature", "Nombre"], *[[key, _format_number(value)] for key, value in bars.items()]]
    story.append(_pdf_table(nature, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Évolution globale", "h1"))
    story.append(
        _ReportBarChart(
            title="Volumes consolidés par type",
            values=[
                ("Ajouts", bars.get("Ajouts", 0), _PDF_GREEN),
                ("Suppressions", bars.get("Suppressions", 0), _PDF_RED),
                ("Modifications", bars.get("Modifications", 0), _PDF_AMBER),
                ("Renommages", bars.get("Renommages", 0), _PDF_BLUE),
            ],
        )
    )

    story.append(PageBreak())
    story.append(p("Sévérité des changements textuels", "h1"))
    story.append(
        _ReportBarChart(
            title="Classification des changements narratifs",
            values=[
                ("Majeur", text_metrics.get("major", 0), _PDF_RED),
                ("Modéré", text_metrics.get("moderate", 0), _PDF_AMBER),
                ("Pertinent", text_metrics.get("relevant", 0), _PDF_BLUE),
                ("Analysé", text_metrics.get("analyzed", 0), _PDF_GREEN),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    text_summary_rows = [
        ["Indicateur texte", "Valeur"],
        ["Total changements", _format_number(text_metrics["total"])],
        ["Pertinents / analysés", f"{text_metrics['relevant']} / {text_metrics['analyzed']}"],
        ["Changements majeurs", _format_number(text_metrics["major"])],
        ["Changements modérés", _format_number(text_metrics["moderate"])],
    ]
    story.append(_pdf_table(text_summary_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Indicateurs et notes", "h1"))
    story.append(
        _ReportBarChart(
            title="Changements structurés par catégorie",
            values=[
                ("Ind. ajoutés", indicator_metrics["indicator_added"], _PDF_GREEN),
                ("Ind. supprimés", indicator_metrics["indicator_removed"], _PDF_RED),
                ("Ind. renommés", indicator_metrics["indicator_renamed"], _PDF_BLUE),
                ("Notes ajoutées", indicator_metrics["footnote_added"], _PDF_GREEN),
                ("Notes supprimées", indicator_metrics["footnote_removed"], _PDF_RED),
                ("Notes modifiées", indicator_metrics["footnote_modified"], _PDF_AMBER),
                ("Notes renommées", indicator_metrics["footnote_renamed"], _PDF_BLUE),
            ],
            orientation="horizontal",
        )
    )

    story.append(PageBreak())
    story.append(p("File de revue", "h1"))
    story.append(
        _ReportBarChart(
            title="Statut de validation analyste",
            values=[
                ("Validés", review_counts.get("approved", 0), _PDF_GREEN),
                ("Rejetés", review_counts.get("rejected", 0), _PDF_RED),
                ("En attente", review_counts.get("pending", 0), _PDF_AMBER),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    review_rows = [
        ["Statut", "Nombre"],
        ["Total", _format_number(review_counts.get("total", 0))],
        ["Validés", _format_number(review_counts.get("approved", 0))],
        ["Rejetés", _format_number(review_counts.get("rejected", 0))],
        ["En attente", _format_number(review_counts.get("pending", 0))],
    ]
    story.append(_pdf_table(review_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Priorités analyste", "h1"))
    priority_data = [["#", "Tableau", "Type", "Impact", "Confiance", "Statut"]]
    for idx, row in enumerate(priority_rows, start=1):
        priority_data.append(
            [
                str(idx),
                p(row["title"], "body"),
                p(row["change"], "body"),
                row["impact"],
                row["confidence"],
                row["status_label"],
            ]
        )
    if len(priority_data) == 1:
        priority_data.append(["-", "Aucun tableau prioritaire détecté.", "", "", "", ""])
    story.append(_pdf_table(priority_data, [0.3 * inch, 2.55 * inch, 1.35 * inch, 0.75 * inch, 0.75 * inch, 0.9 * inch]))

    story.append(PageBreak())
    story.append(p("Changements textuels principaux", "h1"))
    text_rows = [["#", "Section", "Impact", "Résumé"]]
    for idx, item in enumerate(text_metrics.get("top") or [], start=1):
        text_rows.append([str(idx), p(item["section"], "body"), item["impact"], p(item["summary"], "body")])
    if len(text_rows) == 1:
        text_rows.append(["-", "", "", "Aucun changement textuel prioritaire."])
    story.append(_pdf_table(text_rows, [0.3 * inch, 1.45 * inch, 0.85 * inch, 4.2 * inch]))

    story.append(PageBreak())
    story.append(p("Synthèse indicateurs et notes", "h1"))
    indicator_rows = [
        ["Catégorie", "Nombre"],
        ["Indicateurs ajoutés", _format_number(indicator_metrics["indicator_added"])],
        ["Indicateurs supprimés", _format_number(indicator_metrics["indicator_removed"])],
        ["Indicateurs renommés", _format_number(indicator_metrics["indicator_renamed"])],
        ["Notes ajoutées", _format_number(indicator_metrics["footnote_added"])],
        ["Notes supprimées", _format_number(indicator_metrics["footnote_removed"])],
        ["Notes modifiées", _format_number(indicator_metrics["footnote_modified"])],
        ["Notes renommées", _format_number(indicator_metrics["footnote_renamed"])],
        ["Tableaux ajoutés", _format_number(indicator_metrics["tables_added"])],
        ["Tableaux supprimés", _format_number(indicator_metrics["tables_removed"])],
    ]
    story.append(_pdf_table(indicator_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Annexe - changements détectés", "h1"))
    annex_rows = [["#", "Pipeline", "Type", "Section / tableau", "Résumé"]]
    for idx, (change, section) in enumerate(_text_changes(text_data, relevant_only=False)[:30], start=1):
        annex_rows.append(
            [
                str(idx),
                "Texte",
                str(change.get("diff_type") or ""),
                p(section, "body"),
                p(change.get("change_summary") or change.get("source_text_t2") or change.get("source_text_t1"), "body"),
            ]
        )
    for idx, row in enumerate(priority_rows[:10], start=len(annex_rows)):
        annex_rows.append([str(idx), "Indicateurs", row["change"], p(row["title"], "body"), row["impact"]])
    if len(annex_rows) == 1:
        annex_rows.append(["-", "", "", "", "Aucun changement détecté."])
    story.append(_pdf_table(annex_rows, [0.3 * inch, 0.7 * inch, 0.85 * inch, 1.55 * inch, 3.4 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(p("Note: l'annexe est un extrait opérationnel; l'export Excel conserve le détail complet ligne par ligne.", "muted"))

    def _page(canvas, doc_obj) -> None:
        canvas.saveState()
        canvas.setFillColor(_PDF_NAVY)
        canvas.rect(0, letter[1] - 0.22 * inch, letter[0], 0.22 * inch, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(0.45 * inch, letter[1] - 0.14 * inch, "Document interne - confidentiel")
        canvas.setFillColor(_PDF_MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.45 * inch, 0.25 * inch, "Usage interne - vigie bancaire")
        canvas.drawRightString(letter[0] - 0.45 * inch, 0.25 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return buffer.getvalue()


@callback(
    Output("vigie-cockpit-tab-content", "children"),
    Input("store-indicator-result", "data"),
    Input("store-comparison-result", "data"),
    Input("store-text-comparison", "data"),
    Input("store-review-items", "data"),
    Input("store-review-queue", "data"),
    Input("store-show-results-page", "data"),
    State("store-indicator-meta", "data"),
    prevent_initial_call=True,
)
def render_vigie_cockpit(indicator, comparison, text_data, review_items, review_queue, show_results, indicator_meta):
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
    evolution_bars = {
        "Ajouts": bars["Ajouts"],
        "Suppressions": bars["Suppressions"],
        "Modifications": bars["Modifications"],
        "Renommages": bars["Renommages"],
    }
    pertinence = {"ELEVEE": "Élevée", "MOYENNE": "Moyenne", "FAIBLE": "Faible"}.get(
        text_metrics["pertinence"], text_metrics["pertinence"]
    )

    def _pct(value: int) -> str:
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
                                        "bi-journal-bookmark",
                                        "Notes modifiées",
                                        indicator_metrics["notes"],
                                        "Notes de bas tableau",
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
                    _donut_chart(text_total, indicator_total),
                    _bar_chart(bars),
                    _global_evolution_chart(evolution_bars),
                ],
                className="vigie-cockpit-chart-grid",
            ),
            _priority_table(indicator_metrics, review_queue),
        ],
        className="vigie-cockpit",
    )


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
