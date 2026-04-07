"""Layout de l'onglet Analyse Textuelle — vue analyste.

Affiche tous les changements textuels détectés hors ``unchanged``. Les filtres
restants sont gérés dans ``text_flow.py``.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

# ---------------------------------------------------------------------------
# Constantes d'affichage
# ---------------------------------------------------------------------------

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_ORDER: dict[str, int] = {"MAJEUR": 0, "MODERE": 1}

_IMPACT_BADGE: dict[str, tuple[str, str]] = {
    "MAJEUR": ("Majeur", "danger"),
    "MODERE": ("Modéré", "warning"),
    "MINEUR": ("Mineur", "secondary"),
}

_DIFF_LABELS: dict[str, str] = {
    "added": "Ajouté",
    "removed": "Supprimé",
    "modified": "Modifié",
}

_CATEGORY_BADGE: dict[str, tuple[str, str]] = {
    "REGLEMENTAIRE": ("Réglementaire", "primary"),
    "RISQUE": ("Risque", "warning"),
    "CAPITAL": ("Capital", "info"),
    "STRUCTURE": ("Structure", "secondary"),
    "COSMETIQUE": ("Cosmétique", "light"),
    "INCONNU": ("Inconnu", "light"),
}

_ACTION_BADGE: dict[str, tuple[str, str]] = {
    "escalade": ("Escalade", "danger"),
    "investigation": ("Investigation", "warning"),
    "confirmation": ("Confirmation", "success"),
    "information": ("Information", "info"),
    "aucune": ("Aucune", "secondary"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _badge(label: str, color: str, **kwargs) -> dbc.Badge:
    return dbc.Badge(label, color=color, className="me-1", **kwargs)


# ---------------------------------------------------------------------------
# Change card (vue analyste)
# ---------------------------------------------------------------------------


def _build_change_card(change: dict[str, Any], section_title: str) -> dbc.Card:
    """Carte analytique pour un changement pertinent.

    Args:
        change: Dict bloc issu de text_comparison.json.
        section_title: Nom affiché de la section/sous-section.

    Returns:
        dbc.Card stylisée ou None si unchanged/skip.
    """
    triage = change.get("genai_triage") or {}
    diff_type = change.get("diff_type", "")

    if diff_type == "unchanged" or triage.get("source") == "skip":
        return None  # type: ignore[return-value]

    impact_level = (triage.get("impact_level") or "MINEUR").upper()
    category = (triage.get("category") or "INCONNU").upper()
    action = (triage.get("action_requise") or "aucune").lower()
    explanation = (triage.get("explanation") or "").strip()
    nouvelle_idee = triage.get("nouvelle_idee", False)

    evidence_t1 = change.get("evidence_t1") or {}
    evidence_t2 = change.get("evidence_t2") or {}

    if diff_type == "removed":
        phrase = f"[SUPPRIMÉ] {(change.get('semantic_text_t1') or '').strip()}"
        pages = evidence_t1.get("pages") or []
    else:
        phrase = (change.get("semantic_text_t2") or "").strip()
        pages = evidence_t2.get("pages") or []
    page_label = ", ".join(str(p) for p in pages if p) if pages else ""

    # Couleur border-left
    border_color = {"MAJEUR": "danger", "MODERE": "warning"}.get(impact_level, "secondary")

    # Ligne 1 — badges
    impact_lbl, impact_color = _IMPACT_BADGE.get(impact_level, (impact_level, "secondary"))
    action_lbl, action_color = _ACTION_BADGE.get(action, (action.capitalize(), "secondary"))
    cat_lbl, cat_color = _CATEGORY_BADGE.get(category, (category, "secondary"))

    badge_row = html.Div(
        [
            _badge(impact_lbl, impact_color),
            _badge(action_lbl, action_color),
            _badge(cat_lbl, cat_color),
            dbc.Badge(
                "★ Nouvelle idée",
                color="primary",
                pill=True,
                className="me-1",
            )
            if nouvelle_idee
            else None,
        ],
        className="mb-1 d-flex flex-wrap align-items-center",
    )

    # Ligne 2 — meta
    diff_label = _DIFF_LABELS.get(diff_type, diff_type.capitalize())
    meta_text = f"{section_title} · pages {page_label} · {diff_label}" if page_label else f"{section_title} · {diff_label}"
    meta = html.Small(meta_text, className="text-muted d-block mb-2")

    text_block = html.Div(
        phrase,
        className="bg-light p-2 rounded small mb-3 border-start border-2 "
        + ("border-danger" if diff_type == "removed" else "border-primary"),
        style={"whiteSpace": "pre-wrap"},
    )

    evidence_snippet = (
        (evidence_t1.get("snippet") or "").strip()
        if diff_type == "removed"
        else (evidence_t2.get("snippet") or "").strip()
    )
    evidence_block = (
        html.Div(
            [
                html.Div("Preuve source", className="small fw-semibold text-muted mb-1"),
                html.Div(evidence_snippet, className="small text-muted", style={"whiteSpace": "pre-wrap"}),
            ],
            className="mb-3",
        )
        if evidence_snippet
        else None
    )

    # Analyse IA
    ia_children = [
        html.Div(
            [
                html.Div(
                    className="border-start border-primary border-3 ps-2 mb-2",
                    children=[
                        html.Span("Analyse IA", className="fw-semibold small text-primary"),
                    ],
                ),
                html.P(explanation, className="small mb-1", style={"whiteSpace": "pre-wrap"}),
            ]
        )
        if explanation
        else None,
    ]
    ia_block = html.Div([c for c in ia_children if c is not None])

    card_children = [c for c in [badge_row, meta, text_block, evidence_block, ia_block] if c is not None]

    return dbc.Card(
        dbc.CardBody(card_children, className="p-3"),
        className=f"mb-3 border-start border-{border_color} border-3",
    )


# ---------------------------------------------------------------------------
# Executive banner
# ---------------------------------------------------------------------------


def _build_executive_banner(
    global_summary: dict[str, Any],
    bank: str,
    q_cur: str,
    q_prev: str,
) -> dbc.Alert:
    """Bannière exécutive avec résumé, compteurs et bouton export."""
    overview = global_summary.get("executive_overview", "")
    highlights = global_summary.get("key_highlights") or []
    pertinence = (global_summary.get("pertinence_globale") or "FAIBLE").upper()
    counts = global_summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}

    pertinence_color = {"ELEVEE": "danger", "MOYENNE": "warning", "FAIBLE": "success"}.get(pertinence, "secondary")
    pertinence_label = {"ELEVEE": "Élevée", "MOYENNE": "Moyenne", "FAIBLE": "Faible"}.get(pertinence, pertinence)

    # Compteurs
    n_maj = by_impact.get("MAJEUR", 0)
    n_mod = by_impact.get("MODERE", 0)
    n_rel = counts.get("total_relevant", 0)
    n_tot = counts.get("total", 0)

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
            # Ligne 3 : points clés
            html.Ul(
                [html.Li(h, className="small") for h in highlights],
                className="mb-2 ps-3",
            )
            if highlights
            else None,
            # Ligne 4 : compteurs + bouton Excel
            html.Div(
                [
                    _badge(f"{n_maj} Majeur(s)", "danger") if n_maj else None,
                    _badge(f"{n_mod} Modéré(s)", "warning") if n_mod else None,
                    _badge(f"{n_rel} pertinents / {n_tot} analysés", "primary") if n_rel else None,
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


# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------


def _build_filter_bar(section_options: list[dict]) -> html.Div:
    """Barre de filtres : section / impact / action + compteur."""
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-section",
                        options=section_options,
                        placeholder="Toutes les sections",
                        clearable=True,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-impact",
                        options=[
                            {"label": "Majeur", "value": "MAJEUR"},
                            {"label": "Modéré", "value": "MODERE"},
                            {"label": "Mineur", "value": "MINEUR"},
                        ],
                        placeholder="Tous les impacts",
                        clearable=True,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-action",
                        options=[
                            {"label": "Escalade", "value": "escalade"},
                            {"label": "Investigation", "value": "investigation"},
                            {"label": "Confirmation", "value": "confirmation"},
                            {"label": "Information", "value": "information"},
                            {"label": "Aucune", "value": "aucune"},
                        ],
                        placeholder="Toutes les actions",
                        clearable=True,
                    ),
                    md=3,
                ),
                dbc.Col(
                    html.Span(id="text-filter-count", className="small text-muted align-self-center"),
                    md=2,
                    className="d-flex",
                ),
            ],
            className="g-2 align-items-center",
        ),
        className="mb-3 p-3 bg-white rounded border",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_text_analysis_tab(text_data: dict[str, Any] | None) -> html.Div:
    """Construit l'onglet analyse textuelle — vue analyste.

    Args:
        text_data: Contenu de text_comparison.json, ou None si non disponible.

    Returns:
        html.Div contenant banner + filtres + container de cartes (vide,
        rempli par callback text_flow.py).
    """
    if not text_data:
        return html.Div(
            dbc.Alert(
                [
                    html.Strong("Analyse textuelle non disponible. "),
                    html.Span(
                        "Lancez le pipeline texte pour cette banque : "
                        "uv run python -m vigilance.cli.run_text_compare "
                        "--bank <BANK> --year <YEAR> --T2"
                    ),
                ],
                color="secondary",
                className="mt-3",
            )
        )

    global_summary = text_data.get("all_changes_summary") or text_data.get("global_summary") or {}
    section_comparisons = text_data.get("section_comparisons") or []
    q_cur = text_data.get("quarter_current", "")
    q_prev = text_data.get("quarter_previous", "")
    bank = str(text_data.get("bank_code", "")).upper()

    # Options de filtre section dynamiques
    section_options = []
    seen: set[str] = set()
    for sec in section_comparisons:
        key = sec.get("section_key", "")
        title = sec.get("section_title") or _SECTION_LABELS.get(key, key)
        if key and key not in seen:
            section_options.append({"label": title, "value": key})
            seen.add(key)

    return html.Div(
        [
            _build_executive_banner(global_summary, bank, q_cur, q_prev),
            _build_filter_bar(section_options),
            html.Div(id="text-cards-container"),
        ],
        className="pt-3",
    )
