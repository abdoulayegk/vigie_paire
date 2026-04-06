"""Layout de l'onglet Analyse Textuelle — affiche les changements de paragraphes.

Lecture seule. Toute la classification a été faite en amont par le pipeline texte.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

# ---------------------------------------------------------------------------
# Constantes d'affichage
# ---------------------------------------------------------------------------

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_BADGE: dict[str, tuple[str, str]] = {
    "MAJEUR": ("Majeur", "danger"),
    "MODERE": ("Modéré", "warning"),
    "MINEUR": ("Mineur", "secondary"),
}

_DIFF_BADGE: dict[str, tuple[str, str]] = {
    "added": ("Ajouté", "success"),
    "removed": ("Supprimé", "danger"),
    "modified": ("Modifié", "warning"),
    "unchanged": ("Inchangé", "light"),
}

_CATEGORY_BADGE: dict[str, tuple[str, str]] = {
    "REGLEMENTAIRE": ("Réglementaire", "primary"),
    "RISQUE": ("Risque", "warning"),
    "CAPITAL": ("Capital", "info"),
    "STRUCTURE": ("Structure", "secondary"),
    "COSMETIQUE": ("Cosmétique", "light"),
    "INCONNU": ("Inconnu", "light"),
}

_SIGNAL_LABELS: dict[str, str] = {
    "regulatory_reference_added": "Référence réglementaire",
    "methodology_change": "Changement méthodologique",
    "tone_change": "Changement de ton",
    "forward_looking": "Prospectif",
    "quantitative_change": "Valeur quantitative",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _badge(label: str, color: str, **kwargs) -> dbc.Badge:
    """Crée un badge Bootstrap avec espacement standard."""
    return dbc.Badge(label, color=color, className="me-1", **kwargs)


def _impact_badge(impact_level: str) -> dbc.Badge:
    """Badge coloré selon le niveau d'impact (MAJEUR/MODERE/MINEUR)."""
    label, color = _IMPACT_BADGE.get(impact_level, (impact_level, "secondary"))
    return _badge(label, color)


def _diff_badge(diff_type: str) -> dbc.Badge:
    """Badge coloré selon le type de diff (added/removed/modified/unchanged)."""
    label, color = _DIFF_BADGE.get(diff_type, (diff_type, "secondary"))
    return _badge(label, color)


def _category_badge(category: str) -> dbc.Badge:
    """Badge coloré selon la catégorie réglementaire."""
    label, color = _CATEGORY_BADGE.get(category, (category, "secondary"))
    return _badge(label, color)


def _signal_badges(signals: dict[str, bool]) -> list[dbc.Badge]:
    """Retourne un badge pill pour chaque signal actif (True)."""
    return [_badge(_SIGNAL_LABELS.get(k, k), "info", pill=True) for k, v in (signals or {}).items() if v]


# ---------------------------------------------------------------------------
# Block comparison card
# ---------------------------------------------------------------------------


def _build_block_card(change: dict[str, Any]) -> dbc.Card:
    """Construit la carte d'un changement de paragraphe.

    Affiche les badges de type/impact/catégorie, le diff textuel T1/T2,
    l'explication GenAI, l'impact, la référence réglementaire et l'action
    requise. Les paragraphes ``unchanged`` sont ignorés (retourne None).

    Args:
        change: Dict issu de ``text_comparison.json`` contenant les clés
            ``diff_type``, ``block_t1``, ``block_t2``, ``genai_triage``,
            ``change_summary`` et ``signals``.

    Returns:
        ``dbc.Card`` ou ``None`` si le changement est inchangé/skip.
    """
    triage = change.get("genai_triage") or {}
    diff_type = change.get("diff_type", "")
    source = triage.get("source", "")

    # Skip unchanged
    if diff_type == "unchanged" or source == "skip":
        return None  # type: ignore[return-value]

    impact_level = triage.get("impact_level", "MINEUR")
    category = triage.get("category", "INCONNU")
    is_relevant = triage.get("is_relevant", False)
    explanation = triage.get("explanation", "")
    impact_desc = triage.get("impact_description", "")
    action = triage.get("action_requise", "aucune")
    ref_reg = triage.get("reference_reglementaire", "")
    signals = change.get("signals") or triage.get("signals") or {}
    change_summary = change.get("change_summary", "")

    block_t1 = change.get("block_t1") or {}
    block_t2 = change.get("block_t2") or {}

    # Header badges
    header_badges = [
        _diff_badge(diff_type),
        _impact_badge(impact_level),
        _category_badge(category),
    ]
    if not is_relevant:
        header_badges.append(_badge("Non pertinent", "light"))

    signal_list = _signal_badges(signals)

    # Border color by impact
    border_color = {
        "MAJEUR": "danger",
        "MODERE": "warning",
        "MINEUR": "secondary",
    }.get(impact_level, "secondary")

    card_children = [
        # Top badges row
        html.Div(header_badges + signal_list, className="mb-2"),
        # Change summary
        html.P(change_summary, className="text-muted small mb-2") if change_summary else None,
        # Text blocks
        _build_text_diff(diff_type, block_t1, block_t2),
        html.Hr(className="my-2"),
        # GenAI analysis
        html.Div(
            [
                html.Span("Explication : ", className="fw-semibold small"),
                html.Span(explanation or "—", className="small text-muted"),
            ],
            className="mb-1",
        )
        if explanation
        else None,
        html.Div(
            [
                html.Span("Impact : ", className="fw-semibold small"),
                html.Span(impact_desc or "—", className="small text-muted"),
            ],
            className="mb-1",
        )
        if impact_desc
        else None,
        html.Div(
            [
                html.Span("Référence : ", className="fw-semibold small"),
                html.Span(ref_reg, className="small font-monospace text-primary"),
            ],
            className="mb-1",
        )
        if ref_reg
        else None,
        html.Div(
            [
                html.Span("Action : ", className="fw-semibold small"),
                _badge(action.capitalize(), "dark" if action == "escalade" else "secondary"),
            ]
        )
        if action and action != "aucune"
        else None,
    ]

    # Filter None entries
    card_children = [c for c in card_children if c is not None]

    return dbc.Card(
        dbc.CardBody(card_children, className="p-3"),
        className=f"mb-3 border-start border-{border_color} border-3",
    )


def _build_text_diff(
    diff_type: str,
    block_t1: dict[str, Any],
    block_t2: dict[str, Any],
) -> html.Div:
    """Rend le diff visuel T1/T2 : ajout (vert), suppression (rouge) ou modifié (côte à côte)."""
    text_t1 = str(block_t1.get("text", "") or "")
    text_t2 = str(block_t2.get("text", "") or "")
    page_t1 = block_t1.get("page", "")
    page_t2 = block_t2.get("page", "")

    if diff_type == "added":
        return html.Div(
            [
                html.Small(f"Page {page_t2}" if page_t2 else "", className="text-muted d-block mb-1"),
                html.Div(
                    text_t2[:600] + ("…" if len(text_t2) > 600 else ""),
                    className="bg-success bg-opacity-10 p-2 rounded small border-start border-success border-2",
                ),
            ]
        )
    elif diff_type == "removed":
        return html.Div(
            [
                html.Small(f"Page {page_t1}" if page_t1 else "", className="text-muted d-block mb-1"),
                html.Div(
                    text_t1[:600] + ("…" if len(text_t1) > 600 else ""),
                    className="bg-danger bg-opacity-10 p-2 rounded small border-start border-danger border-2",
                ),
            ]
        )
    else:  # modified
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small(
                                    f"Précédent (p.{page_t1})" if page_t1 else "Précédent",
                                    className="text-muted d-block mb-1",
                                ),
                                html.Div(
                                    text_t1[:500] + ("…" if len(text_t1) > 500 else ""),
                                    className="bg-danger bg-opacity-10 p-2 rounded small border-start border-danger border-2 h-100",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Small(
                                    f"Courant (p.{page_t2})" if page_t2 else "Courant",
                                    className="text-muted d-block mb-1",
                                ),
                                html.Div(
                                    text_t2[:500] + ("…" if len(text_t2) > 500 else ""),
                                    className="bg-success bg-opacity-10 p-2 rounded small border-start border-success border-2 h-100",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="g-2",
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Section accordion item
# ---------------------------------------------------------------------------


def _build_section_accordion(section: dict[str, Any]) -> dbc.AccordionItem:
    """Construit un AccordionItem pour une section du rapport.

    Affiche les badges de comptage (ajoutés, supprimés, modifiés,
    pertinents) dans le titre et les cartes de changement dans le corps.

    Args:
        section: Dict avec ``section_key``, ``section_title``, ``summary``
            et ``block_comparisons``.

    Returns:
        ``dbc.AccordionItem`` dépliable.
    """
    section_key = section.get("section_key", "")
    section_title = section.get("section_title") or _SECTION_LABELS.get(section_key, section_key)
    summary = section.get("summary") or {}
    changes = section.get("block_comparisons") or []

    added = summary.get("added", 0)
    removed = summary.get("removed", 0)
    modified = summary.get("modified", 0)
    relevant = sum(
        1 for c in changes if (c.get("genai_triage") or {}).get("is_relevant") and c.get("diff_type") != "unchanged"
    )

    # Section summary badges
    badges = []
    if added:
        badges.append(_badge(f"+{added} ajouté(s)", "success"))
    if removed:
        badges.append(_badge(f"-{removed} supprimé(s)", "danger"))
    if modified:
        badges.append(_badge(f"~{modified} modifié(s)", "warning"))
    if relevant:
        badges.append(_badge(f"{relevant} pertinent(s)", "primary"))

    # Build cards, skip unchanged
    cards = []
    for change in changes:
        card = _build_block_card(change)
        if card is not None:
            cards.append(card)

    body = (
        html.Div(cards)
        if cards
        else html.P(
            "Aucun changement significatif détecté dans cette section.",
            className="text-muted p-3",
        )
    )

    title = html.Span(
        [
            html.Strong(section_title),
            html.Span(badges, className="ms-2"),
        ]
    )

    return dbc.AccordionItem(body, title=title, item_id=f"text-sec-{section_key}")


# ---------------------------------------------------------------------------
# Global summary banner
# ---------------------------------------------------------------------------


def _build_global_summary(global_summary: dict[str, Any]) -> html.Div:
    """Construit la bannière de résumé exécutif de l'analyse textuelle.

    Affiche l'overview, les points clés, les compteurs d'impact et la
    pertinence globale sous forme d'alerte Bootstrap colorée.

    Args:
        global_summary: Dict ``global_summary`` issu de ``text_comparison.json``.

    Returns:
        ``dbc.Alert`` coloré selon la pertinence globale.
    """
    overview = global_summary.get("executive_overview", "")
    highlights = global_summary.get("key_highlights") or []
    pertinence = global_summary.get("pertinence_globale", "FAIBLE")
    counts = global_summary.get("counts") or {}

    pertinence_color = {"ELEVEE": "danger", "MOYENNE": "warning", "FAIBLE": "success"}.get(pertinence, "secondary")

    items = []
    if overview:
        items.append(html.P(overview, className="mb-2"))
    if highlights:
        items.append(html.Ul([html.Li(h, className="small") for h in highlights], className="mb-2"))

    count_badges = []
    by_impact = counts.get("by_impact") or {}
    if by_impact.get("MAJEUR"):
        count_badges.append(_badge(f"{by_impact['MAJEUR']} Majeur(s)", "danger"))
    if by_impact.get("MODERE"):
        count_badges.append(_badge(f"{by_impact['MODERE']} Modéré(s)", "warning"))
    if counts.get("total_relevant"):
        count_badges.append(
            _badge(f"{counts['total_relevant']} pertinent(s) / {counts.get('total', 0)} total", "primary")
        )

    if count_badges:
        items.append(html.Div(count_badges, className="mt-1"))

    return dbc.Alert(
        [
            html.Div(
                [
                    html.Strong("Résumé exécutif — Analyse textuelle "),
                    _badge(pertinence.capitalize(), pertinence_color),
                ],
                className="mb-2",
            ),
            *items,
        ],
        color=pertinence_color,
        className="mb-4",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_text_analysis_tab(text_data: dict[str, Any] | None) -> html.Div:
    """Construit l'onglet complet d'analyse textuelle.

    Args:
        text_data: Contenu de text_comparison.json, ou None si non disponible.

    Returns:
        html.Div contenant l'onglet complet.
    """
    if not text_data:
        return html.Div(
            dbc.Alert(
                [
                    html.Strong("Analyse textuelle non disponible. "),
                    html.Span(
                        "Lancez le pipeline texte pour cette banque : "
                        "python run_text_pipeline.py --bank <BANK> --year <YEAR> --T2"
                    ),
                ],
                color="secondary",
                className="mt-3",
            )
        )

    global_summary = text_data.get("global_summary") or {}
    section_comparisons = text_data.get("section_comparisons") or []

    q_cur = text_data.get("quarter_current", "")
    q_prev = text_data.get("quarter_previous", "")
    bank = str(text_data.get("bank_code", "")).upper()

    header = html.Div(
        [
            html.H5(f"Analyse textuelle — {bank} : {q_cur} vs {q_prev}", className="mb-1"),
            html.P(
                "Changements sémantiques détectés dans les paragraphes des sections réglementaires.",
                className="text-muted small mb-3",
            ),
        ]
    )

    summary_banner = _build_global_summary(global_summary)

    accordion = dbc.Accordion(
        [_build_section_accordion(sec) for sec in section_comparisons],
        start_collapsed=False,
        always_open=True,
        className="mb-4",
    )

    return html.Div([header, summary_banner, accordion], className="pt-3")
