"""Mise en page de l'onglet consultatif des changements communs interbanques."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

_INDETERMINATE_VALUES = {
    "INDETERMINE",
    "INDETERMINEE",
    "INDÉTERMINÉ",
    "INDÉTERMINÉE",
    "INDETERMINATE",
}


def build_changements_communs_tab(
    stats: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    *,
    selected_period: str | None = None,
    report_path: str | None = None,
) -> html.Div:
    """Construit l'onglet des changements communs entre banques."""
    effective_period = str((report or {}).get("period") or selected_period or "").strip() or "Periode non selectionnee"
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Changements communs entre banques", className="mb-1"),
                            html.P(
                                (
                                    "Lecture de l'analyse generee apres les "
                                    "comparaisons des banques pour une meme periode."
                                ),
                                className="text-muted mb-0",
                            ),
                        ],
                        lg=8,
                    ),
                    dbc.Col(
                        dbc.Alert(
                            [
                                html.I(className="bi bi-shield-check me-2"),
                                "Consultation des resultats, sans appel LLM",
                            ],
                            color="info",
                            className="py-2 mb-0",
                        ),
                        lg=4,
                    ),
                ],
                className="align-items-center mb-4",
            ),
            dbc.Alert(
                [
                    html.Strong("Periode analysee: "),
                    html.Code(effective_period),
                ],
                color="light",
                className="border py-2",
            ),
            html.Div(
                _build_source_stats(stats or {}),
                id="changements-communs-source-stats",
                className="mb-4",
            ),
            html.Div(
                build_changements_communs_report_view(
                    report,
                    selected_period=selected_period,
                    report_path=report_path,
                ),
                id="changements-communs-results",
            ),
        ],
        className="mt-3",
    )


def build_changements_communs_report_view(
    report: dict[str, Any] | None,
    *,
    selected_period: str | None = None,
    report_path: str | None = None,
) -> html.Div:
    """Présente un rapport de changements communs déjà généré."""
    if not report:
        period_text = str(selected_period or "").strip()
        expected_path = (
            f"outputs/resultats/changements_communs_banques/{period_text}/changements_communs_banques.json"
            if period_text
            else "outputs/resultats/changements_communs_banques/<periode>/changements_communs_banques.json"
        )
        return html.Div(
            dbc.Alert(
                [
                    html.Strong("Aucune analyse sauvegardee. "),
                    (
                        "L'analyse des changements communs entre banques doit "
                        "etre generee en fin de batch pour la periode chargee, "
                        "puis sauvegardee dans "
                    ),
                    html.Code(expected_path),
                    ".",
                ],
                color="light",
                className="border",
            )
        )

    signals = report.get("signals") or []
    if not signals:
        return html.Div(
            dbc.Alert(
                "L'analyse sauvegardee ne contient aucun regroupement probant.",
                color="warning",
            )
        )

    consensus = [signal for signal in signals if isinstance(signal, dict) and bool(signal.get("min_banks_met"))]
    minor = [signal for signal in signals if isinstance(signal, dict) and not bool(signal.get("min_banks_met"))]
    return html.Div(
        [
            _build_report_summary(report, report_path=report_path),
            _build_signal_section(
                "Consensus communs",
                "3 banques ou plus",
                consensus,
            ),
            _build_signal_section(
                "Signaux mineurs a surveiller",
                "2 banques seulement - non consensus",
                minor,
            ),
        ]
    )


def _build_source_stats(stats: dict[str, Any]) -> dbc.Row:
    total = int(stats.get("total_changes", 0) or 0)
    bank_count = int(stats.get("bank_count", 0) or 0)
    period_count = int(stats.get("period_count", 0) or 0)
    banks = ", ".join(str(bank).upper() for bank in stats.get("banks", []) or [])
    return dbc.Row(
        [
            dbc.Col(_metric_card("Changements indexables", total), md=3),
            dbc.Col(_metric_card("Banques couvertes", bank_count), md=3),
            dbc.Col(_metric_card("Periodes analysees", period_count), md=3),
            dbc.Col(_metric_card("Univers", banks or "-"), md=3),
        ],
        className="g-3",
    )


def _metric_card(title: str, value: str | int) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="small text-muted mb-1"),
                html.Div(str(value), className="h5 fw-bold mb-0"),
            ]
        ),
        className="shadow-sm border-0 bg-white h-100",
    )


def _build_report_summary(report: dict[str, Any], *, report_path: str | None = None) -> dbc.Card:
    """Construit l'en-tête du rapport et ses compteurs de traçabilité."""
    topic = str(report.get("topic") or report.get("theme") or "").strip() or "Theme non precise"
    period = str(report.get("period") or "").strip() or "Periode non precisee"
    source_count = int(report.get("source_change_count", 0) or 0)
    candidate_count = int(report.get("candidate_count", 0) or 0)
    signal_counts = report.get("signal_counts") if isinstance(report.get("signal_counts"), dict) else {}
    signal_count = int(signal_counts.get("total", len(report.get("signals") or [])) or 0)
    consensus_count = int(signal_counts.get("consensus", 0) or 0)
    minor_count = int(signal_counts.get("minor", 0) or 0)
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Theme analyse", className="small text-muted"),
                                html.H5(topic, className="mb-0"),
                                html.Div(
                                    [
                                        html.Span("Periode: ", className="text-muted"),
                                        html.Code(period),
                                    ],
                                    className="small mt-1",
                                ),
                                html.Div(
                                    [
                                        html.Span("Fichier: ", className="text-muted"),
                                        html.Code(report_path),
                                    ],
                                    className="small mt-1",
                                )
                                if report_path
                                else html.Div(),
                            ],
                            lg=6,
                        ),
                        dbc.Col(_inline_stat("Sources", source_count), lg=2),
                        dbc.Col(_inline_stat("Candidats LLM", candidate_count), lg=2),
                        dbc.Col(
                            _inline_stat(
                                "Regroupements",
                                f"{signal_count} ({consensus_count}+{minor_count})",
                            ),
                            lg=2,
                        ),
                    ],
                    className="align-items-center",
                )
            ]
        ),
        className="shadow-sm border-0 mb-3",
    )


def _inline_stat(label: str, value: int | str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="small text-muted"),
            html.Div(str(value), className="fw-bold"),
        ],
        className="text-lg-end",
    )


def _build_signal_section(
    title: str,
    subtitle: str,
    signals: list[dict[str, Any]],
) -> html.Div:
    if not signals:
        return html.Div()
    return html.Div(
        [
            html.Div(
                [
                    html.H5(title, className="mb-0"),
                    html.Div(subtitle, className="small text-muted"),
                ],
                className="mt-4 mb-2",
            ),
            *[_build_signal_card(signal) for signal in signals],
        ]
    )


def _build_signal_card(signal: dict[str, Any]) -> dbc.Card:
    banks = [str(bank).upper() for bank in signal.get("banks", []) or []]
    min_banks_met = bool(signal.get("min_banks_met"))
    status_color = "success" if min_banks_met else "warning"
    status_label = "Present dans au moins 3 banques" if min_banks_met else "Signal mineur - 2 banques"

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5(str(signal.get("theme") or "Signal sans theme"), className="mb-1"),
                        dbc.Badge(status_label, color=status_color, className="me-2"),
                        dbc.Badge(", ".join(banks) or "Banques inconnues", color="secondary"),
                    ],
                    className="mb-2",
                ),
                html.P(str(signal.get("summary") or ""), className="mb-2"),
                _muted_block("Impact IT", _displayable_impact_it(signal.get("impact_it"))),
                _muted_block(
                    "Evolution de la posture de gestion",
                    signal.get("posture_summary"),
                ),
                _muted_block(
                    "Mise en oeuvre",
                    signal.get("mise_en_oeuvre_summary"),
                ),
                _muted_block(
                    "Confiance de la posture",
                    signal.get("confiance_posture"),
                ),
                _muted_block("Pourquoi ces changements sont regroupes", signal.get("rationale")),
                _muted_block("Differences entre banques", signal.get("differences")),
                _build_evidence_accordion(signal.get("evidence") or []),
            ]
        ),
        className="shadow-sm border-0 mb-3",
    )


def _muted_block(title: str, value: Any) -> html.Div:
    text = str(value or "").strip()
    if not text:
        return html.Div()
    return html.Div(
        [
            html.Div(title, className="small fw-semibold text-muted"),
            html.P(text, className="small mb-2"),
        ]
    )


def _displayable_impact_it(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.upper() in _INDETERMINATE_VALUES:
        return ""
    return text


def _build_evidence_accordion(evidence: list[dict[str, Any]]) -> html.Div:
    if not evidence:
        return html.Div(
            "Aucune preuve exploitable rattachee aux changements sources.",
            className="small text-muted",
        )
    items = []
    for idx, item in enumerate(evidence, start=1):
        title = f"{str(item.get('bank_code') or '').upper()} - {item.get('section') or 'Section inconnue'}"
        items.append(
            dbc.AccordionItem(
                _build_evidence_body(item),
                title=title,
                item_id=f"evidence-{idx}",
            )
        )
    return dbc.Accordion(items, start_collapsed=True, flush=True)


def _build_evidence_body(item: dict[str, Any]) -> html.Div:
    """Présente une preuve bancaire avec sa posture, ses textes et sa source."""
    quote = str(item.get("quote") or "").strip()
    before = str(item.get("text_before") or "").strip()
    after = str(item.get("text_after") or "").strip()
    return html.Div(
        [
            html.Div(
                [
                    dbc.Badge(str(item.get("diff_type") or "change"), color="info", className="me-2"),
                    html.Span(str(item.get("subsection") or ""), className="small text-muted"),
                ],
                className="mb-2",
            ),
            html.Div(str(item.get("change_summary") or ""), className="small mb-2"),
            _muted_block(
                "Posture source",
                item.get("changement_posture"),
            ),
            _muted_block(
                "Justification de la posture",
                item.get("justification_posture"),
            ),
            _muted_block(
                "Statut de mise en oeuvre",
                item.get("statut_mise_en_oeuvre"),
            ),
            _muted_block(
                "Confiance de la posture",
                item.get("confiance_posture"),
            ),
            html.Blockquote(quote, className="small border-start ps-3 text-muted") if quote else html.Div(),
            _text_box("Avant", before),
            _text_box("Apres", after),
            html.Div(
                [
                    html.Code(str(item.get("change_id") or "")),
                    html.Span(" | ", className="text-muted"),
                    html.Code(str(item.get("source_path") or "")),
                ],
                className="small text-muted mt-2",
            ),
        ]
    )


def _text_box(title: str, value: str) -> html.Div:
    if not value:
        return html.Div()
    return html.Div(
        [
            html.Div(title, className="small fw-semibold text-muted mb-1"),
            html.Pre(
                value,
                className="small bg-light border rounded p-2",
                style={"whiteSpace": "pre-wrap", "maxHeight": "220px", "overflowY": "auto"},
            ),
        ],
        className="mb-2",
    )
