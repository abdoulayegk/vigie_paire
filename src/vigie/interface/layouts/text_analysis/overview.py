"""Bandeau executif et indicateurs d'avancement de la revue textuelle.

Extrait de ``page_text_analysis.py`` sans modification.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from .labels import _badge, _plural_count


def _build_executive_overview_text(
    global_summary: dict[str, Any],
    auditable_changes: int | None,
) -> str:
    """Construit le résumé analytique affiché dans la bannière texte."""
    counts = global_summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}

    n_detected = auditable_changes if auditable_changes is not None else int(counts.get("total", 0) or 0)
    n_substantive = int(counts.get("total_relevant", counts.get("total", 0)) or 0)
    n_maj = int(by_impact.get("MAJEUR", 0) or 0)
    n_mod = int(by_impact.get("MODERE", 0) or 0)

    detected_label = _plural_count(
        n_detected,
        "changement textuel détecté",
        "changements textuels détectés",
    )

    if n_substantive <= 0:
        access_sentence = (
            "Tous les changements restent accessibles afin de permettre une revue complète par l'analyste."
            if n_detected
            else ""
        )
        return (
            f"{detected_label}. Aucun changement n'est classé comme substantiel "
            f"à prioriser pour revue experte. {access_sentence}"
        ).strip()

    substantive_label = _plural_count(
        n_substantive,
        "changement substantiel",
        "changements substantiels",
    )
    major_label = _plural_count(n_maj, "majeur", "majeurs")
    moderate_label = _plural_count(n_mod, "modéré", "modérés")
    access_sentence = (
        "Les autres changements restent accessibles afin de permettre une revue complète par l'analyste."
        if n_detected > n_substantive
        else "Tous les changements restent accessibles afin de permettre une revue complète par l'analyste."
    )

    return (
        f"{detected_label}. L'analyse en classe {substantive_label}, "
        f"à prioriser pour revue experte : {major_label} et {moderate_label}. "
        f"{access_sentence}"
    )


def _build_executive_banner(
    global_summary: dict[str, Any],
    bank: str,
    q_cur: str,
    q_prev: str,
    auditable_changes: int | None = None,
) -> dbc.Alert:
    """Bannière exécutive avec résumé, compteurs et bouton export."""
    overview = _build_executive_overview_text(global_summary, auditable_changes)
    pertinence = (global_summary.get("pertinence_globale") or "FAIBLE").upper()
    counts = global_summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}

    pertinence_color = {"ELEVEE": "danger", "MOYENNE": "warning", "FAIBLE": "success"}.get(pertinence, "secondary")
    pertinence_label = {"ELEVEE": "Élevée", "MOYENNE": "Moyenne", "FAIBLE": "Faible"}.get(pertinence, pertinence)

    # Compteurs
    n_maj = by_impact.get("MAJEUR", 0)
    n_mod = by_impact.get("MODERE", 0)
    n_auditable = auditable_changes if auditable_changes is not None else counts.get("total", 0)

    return dbc.Alert(
        [
            # Ligne 1 : banque + trimestres + badge pertinence
            html.Div(
                [
                    html.Strong(f"{bank} · {q_cur} vs {q_prev}  "),
                    _badge(f"Pertinence : {pertinence_label}", pertinence_color),
                ],
                className="mb-2 d-flex align-items-center flex-wrap",
            ),
            # Ligne 2 : résumé exécutif
            html.P(overview, className="mb-2 small") if overview else None,
            # Ligne 3 : compteurs + bouton Excel
            html.Div(
                [
                    _badge(f"{n_maj} Majeur(s)", "danger") if n_maj else None,
                    _badge(f"{n_mod} Modéré(s)", "warning") if n_mod else None,
                    _badge(f"{n_auditable} changement(s) textuel(s)", "primary") if n_auditable else None,
                    dbc.Button(
                        "↓ Télécharger Excel",
                        id="btn-download-text-excel",
                        color="light",
                        size="sm",
                        className="ms-auto border",
                    ),
                ],
                className="d-flex align-items-center flex-wrap gap-1 mt-1",
            ),
        ],
        color=pertinence_color,
        className="mb-3",
    )


def _count_auditable_text_changes(section_comparisons: list[dict[str, Any]]) -> int:
    """Compte tous les changements textuels affichables pour revue analyste."""
    total = 0
    for sec in section_comparisons:
        for change in sec.get("all_block_comparisons") or []:
            if change.get("diff_type") == "unchanged":
                continue
            total += 1
    return total


def _text_review_progress(section_comparisons: list[dict[str, Any]]) -> dict[str, int]:
    """Calcule les décisions de revue sur le périmètre textuel auditable."""
    counts = {"approved": 0, "rejected": 0, "skipped": 0, "pending": 0}
    for section in section_comparisons:
        for change in section.get("all_block_comparisons") or []:
            if change.get("diff_type") == "unchanged":
                continue
            review = change.get("_analyst_review") or {}
            status = str(review.get("status") or "pending").strip().lower()
            counts[status if status in counts else "pending"] += 1

    total = sum(counts.values())
    decided = counts["approved"] + counts["rejected"]
    remaining = counts["pending"] + counts["skipped"]
    percent = round(decided / total * 100) if total else 0
    return {
        **counts,
        "total": total,
        "decided": decided,
        "remaining": remaining,
        "percent": percent,
    }


def _build_text_review_progress(section_comparisons: list[dict[str, Any]]) -> html.Div:
    """Construit la bannière globale d'avancement de la revue textuelle."""
    progress = _text_review_progress(section_comparisons)
    percent = progress["percent"]
    complete = bool(progress["total"]) and progress["remaining"] == 0

    summary_items: list[Any] = [
        html.Span(
            (
                f"{progress['decided']} / {progress['total']} "
                f"{('décision rendue' if progress['decided'] == 1 else 'décisions rendues')}"
            ),
            className="fw-semibold me-2",
        ),
        dbc.Badge(
            _plural_count(progress["approved"], "validé", "validés"),
            color="success",
            className="me-1",
        ),
        dbc.Badge(
            _plural_count(progress["rejected"], "rejeté", "rejetés"),
            color="danger",
            className="me-1",
        ),
        dbc.Button(
            f"{progress['remaining']} à traiter",
            id="text-progress-remaining",
            color="primary" if progress["remaining"] else "success",
            outline=bool(progress["remaining"]),
            size="sm",
            className="text-review-remaining-button",
            disabled=not progress["remaining"],
            title="Afficher les changements qui nécessitent encore une décision",
        ),
    ]
    if progress["skipped"]:
        summary_items.append(
            html.Span(
                f"dont {_plural_count(progress['skipped'], 'passé', 'passés')} à reprendre",
                className="small text-muted ms-2",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Avancement de la revue textuelle", className="fw-semibold"),
                    dbc.Badge(
                        "Revue complète" if complete else f"{percent}%",
                        color="success" if complete else "primary",
                        className="ms-auto",
                    ),
                ],
                className="d-flex align-items-center mb-2",
            ),
            dbc.Progress(
                value=percent,
                color="success" if complete else "primary",
                className="text-review-progress-bar mb-2",
            ),
            html.Div(summary_items, className="d-flex align-items-center flex-wrap gap-1"),
        ],
        id="text-review-progress",
        className="text-review-progress-banner px-3 py-2 rounded border",
    )
