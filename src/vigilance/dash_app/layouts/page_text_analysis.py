"""Layout de l'onglet Analyse Textuelle — vue analyste.

Affiche tous les changements textuels détectés hors ``unchanged``. Les filtres
restants sont gérés dans ``text_flow.py``.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigilance.text_comparison.justification import build_text_triage_justification

# ---------------------------------------------------------------------------
# Constantes d'affichage
# ---------------------------------------------------------------------------

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_ORDER: dict[str, int] = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}

_IMPACT_BADGE: dict[str, tuple[str, str]] = {
    "MAJEUR": ("Majeur", "danger"),
    "MODERE": ("Modéré", "warning"),
    "MINEUR": ("Mineur", "secondary"),
}

_DIFF_LABELS: dict[str, str] = {
    "added": "Ajouté",
    "removed": "Supprimé",
    "modified": "Modifié",
    "renamed": "Renommé",
}

_THEMES_AMF_SHORT: dict[str, str] = {
    "DIVULGATION_AJOUT": "Ajout divulgation",
    "DIVULGATION_RETRAIT": "Retrait divulgation",
    "MODIFICATION_TEXTE_RISQUE": "Modif. texte risque",
    "MODIFICATION_METHODOLOGIE": "Modif. méthodologie",
    "FACTEUR_RISQUE_CHANGEMENT": "Facteur risque",
    "CAPITAL_REGLEMENTAIRE": "Capital régl.",
    "LIQUIDITE": "Liquidité",
    "FONDS_PROPRES_REGLEMENTAIRES": "Fonds propres",
    "EXIGENCES_REGLEMENTAIRES": "Exigences régl.",
    "RATIOS_REGLEMENTAIRES": "Ratios régl.",
    "STRUCTURE_RAPPORT": "Structure rapport",
    "HYPOTHESES_EXPLICATIONS_RISQUES": "Hypothèses risques",
    "ESG_CLIMATIQUE": "ESG / Climat",
    "RISQUE_EMERGENT": "Risque émergent",
    "GOUVERNANCE_RISQUES": "Gouvernance",
    "CONTROLE_CONFORMITE": "Contrôle / Conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "Nouvelle mention régl.",
    "MONTANT_REGLEMENTAIRE": "Montant régl.",
}

_ACTION_BADGE: dict[str, tuple[str, str]] = {
    "revue_prioritaire": ("Revue prioritaire", "danger"),
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


# Styles inline pour les highlights — couleurs métier banque (rouge=retiré, vert=ajouté)
_HIGHLIGHT_REMOVED_STYLE = {
    "backgroundColor": "#fde2e2",
    "color": "#9b1c1c",
    "padding": "0 2px",
    "borderRadius": "2px",
    "fontWeight": "500",
}
_HIGHLIGHT_ADDED_STYLE = {
    "backgroundColor": "#dcfce7",
    "color": "#14532d",
    "padding": "0 2px",
    "borderRadius": "2px",
    "fontWeight": "500",
}


def _highlight_text(text: str, highlights: list[str], style: dict[str, str]) -> list:
    """Découpe ``text`` en spans dont les portions matching ``highlights`` portent ``style``.

    Recherche par ``str.find()`` insensible à la casse mais avec le texte
    verbatim de GPT. Si un highlight n'est pas trouvable dans le texte source
    (hallucination GPT), il est silencieusement ignoré.

    Args:
        text: Texte source complet (T1 ou T2).
        highlights: Liste de fragments à surligner.
        style: Dict de style CSS appliqué aux spans surlignés.

    Returns:
        Liste de ``html.Span`` (alternance segments normaux / surlignés).
    """
    if not text:
        return []
    if not highlights:
        return [html.Span(text)]

    # Collecte les intervalles (start, end) des fragments trouvables
    intervals: list[tuple[int, int]] = []
    for highlight in highlights:
        if not highlight or not highlight.strip():
            continue
        start = 0
        while True:
            idx = text.find(highlight, start)
            if idx < 0:
                break
            intervals.append((idx, idx + len(highlight)))
            start = idx + len(highlight)

    if not intervals:
        return [html.Span(text)]

    # Tri + fusion des intervalles chevauchants
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    # Construit la liste de spans alternés
    spans: list = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            spans.append(html.Span(text[cursor:start]))
        spans.append(html.Span(text[start:end], style=style))
        cursor = end
    if cursor < len(text):
        spans.append(html.Span(text[cursor:]))
    return spans


def _build_side_by_side(
    *,
    text_t1: str,
    text_t2: str,
    page_t1: str,
    page_t2: str,
    change_segments: list[dict],
    diff_type: str,
) -> html.Div:
    """Affiche T1/T2 côte à côte avec highlights des segments AMF v2.

    - ``added``  : segment surligné en VERT dans la colonne T2.
    - ``removed``: segment surligné en ROUGE dans la colonne T1.
    - ``modified`` ou ``renamed``: les deux côtés sont affichés côte à côte.

    Pour ``diff_type=added`` seul T2 est affiché ; pour ``removed`` seul T1.
    Pour ``modified`` et ``renamed`` les deux colonnes sont visibles côte à côte.
    """
    highlights_t1 = [
        seg.get("text_t1", "")
        for seg in change_segments
        if seg.get("kind") in ("removed", "modified") and seg.get("text_t1")
    ]
    highlights_t2 = [
        seg.get("text_t2", "")
        for seg in change_segments
        if seg.get("kind") in ("added", "modified") and seg.get("text_t2")
    ]

    base_card_style = {
        "whiteSpace": "pre-wrap",
        "overflowWrap": "anywhere",
        "wordBreak": "break-word",
        "lineHeight": "1.55",
    }

    def _column(label: str, text: str, highlights: list[str], style: dict[str, str]) -> html.Div:
        return html.Div(
            [
                html.Div(
                    label,
                    className="fw-semibold border-bottom px-2 py-1 small text-muted",
                ),
                html.Div(
                    _highlight_text(text, highlights, style),
                    className="px-2 py-2 small",
                    style=base_card_style,
                ),
            ],
            className="border rounded bg-white overflow-hidden flex-grow-1",
            style={"minWidth": "0"},
        )

    label_t1 = f"Précédent (p.{page_t1})" if page_t1 else "Précédent"
    label_t2 = f"Courant (p.{page_t2})" if page_t2 else "Courant"

    if diff_type == "added":
        return html.Div(
            [_column(label_t2, text_t2, highlights_t2, _HIGHLIGHT_ADDED_STYLE)],
            className="mb-3",
        )
    if diff_type == "removed":
        return html.Div(
            [_column(label_t1, text_t1, highlights_t1, _HIGHLIGHT_REMOVED_STYLE)],
            className="mb-3",
        )

    return html.Div(
        [
            _column(label_t1, text_t1, highlights_t1, _HIGHLIGHT_REMOVED_STYLE),
            _column(label_t2, text_t2, highlights_t2, _HIGHLIGHT_ADDED_STYLE),
        ],
        className="mb-3 d-flex gap-2",
    )


# ---------------------------------------------------------------------------
# Change card (vue analyste)
# ---------------------------------------------------------------------------


def _build_change_card(change: dict[str, Any], section_title: str) -> dbc.Card:
    """Carte analytique pour un changement détecté.

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

    is_relevant = bool(triage.get("is_relevant", False))
    impact_level = (triage.get("impact_level") or "MINEUR").upper()
    action = (triage.get("action_requise") or "aucune").lower()
    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    nouvelle_idee_justification = build_text_triage_justification(change)
    themes_amf = list(triage.get("themes_amf") or [])

    evidence_t1 = change.get("evidence_t1") or {}
    evidence_t2 = change.get("evidence_t2") or {}

    text_t1 = (change.get("source_text_t1") or change.get("semantic_text_t1") or "").strip()
    text_t2 = (change.get("source_text_t2") or change.get("semantic_text_t2") or "").strip()
    pages_t1 = evidence_t1.get("pages") or []
    pages_t2 = evidence_t2.get("pages") or []
    page_t1_label = ", ".join(str(p) for p in pages_t1 if p) if pages_t1 else ""
    page_t2_label = ", ".join(str(p) for p in pages_t2 if p) if pages_t2 else ""

    # Pages affichées dans la ligne meta (priorité T2 si disponible)
    page_label = page_t2_label or page_t1_label
    change_segments = list(triage.get("change_segments") or [])

    # Couleur border-left dérivée du niveau d'impact
    border_color = {"MAJEUR": "danger", "MODERE": "warning"}.get(impact_level, "secondary")

    # Ligne 1 — badges (nouvelle idée + impact + action)
    impact_lbl, impact_color = _IMPACT_BADGE.get(impact_level, (impact_level, "secondary"))
    action_lbl, action_color = _ACTION_BADGE.get(action, (action.capitalize(), "secondary"))

    badge_children: list = []
    if nouvelle_idee:
        badge_children.append(
            dbc.Badge(
                "Nouvelle idée",
                color="primary",
                className="me-1",
            )
        )
    if not is_relevant:
        badge_children.append(_badge("Non pertinent", "secondary"))
    badge_children.append(_badge(impact_lbl, impact_color))
    if action and action != "aucune":
        badge_children.append(_badge(action_lbl, action_color))

    badge_row = html.Div(
        badge_children,
        className="mb-2 d-flex flex-wrap align-items-center",
    )

    # Ligne 1 bis — chips thèmes AMF (max 4 + overflow)
    themes_chips: list = []
    visible_themes = themes_amf[:4]
    overflow_themes = themes_amf[4:]
    for theme in visible_themes:
        themes_chips.append(
            dbc.Badge(
                _THEMES_AMF_SHORT.get(theme, theme),
                color="light",
                text_color="dark",
                className="me-1 mb-1 border",
            )
        )
    if overflow_themes:
        tooltip = ", ".join(_THEMES_AMF_SHORT.get(t, t) for t in overflow_themes)
        themes_chips.append(
            dbc.Badge(
                f"+{len(overflow_themes)}",
                color="secondary",
                className="me-1 mb-1",
                title=tooltip,
            )
        )
    themes_row = html.Div(themes_chips, className="mb-2 d-flex flex-wrap") if themes_chips else None

    # Ligne 2 — meta
    diff_label = _DIFF_LABELS.get(diff_type, diff_type.capitalize())
    meta_text = (
        f"{section_title} · pages {page_label} · {diff_label}" if page_label else f"{section_title} · {diff_label}"
    )
    meta = html.Small(meta_text, className="text-muted d-block mb-2")

    text_block = _build_side_by_side(
        text_t1=text_t1,
        text_t2=text_t2,
        page_t1=page_t1_label,
        page_t2=page_t2_label,
        change_segments=change_segments,
        diff_type=diff_type,
    )

    # Bloc preuve source : retiré du nouveau design — la preuve EST le texte
    # source affiché dans le side-by-side avec les highlights AMF v2.
    evidence_block = None

    # Justification de triage (champ AMF v2 — note d'analyste structurée)
    ia_block: html.Div | None = None
    if nouvelle_idee_justification:
        ia_block = html.Div(
            [
                html.Div(
                    className="border-start border-primary border-3 ps-2 mb-2",
                    children=[
                        html.Span(
                            "Justification de triage",
                            className="fw-semibold small text-primary",
                        ),
                    ],
                ),
                html.P(
                    nouvelle_idee_justification,
                    className="small mb-1",
                    style={"whiteSpace": "pre-wrap"},
                ),
            ]
        )

    card_children = [c for c in [badge_row, themes_row, meta, text_block, evidence_block, ia_block] if c is not None]

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
                            {"label": "Revue prioritaire", "value": "revue_prioritaire"},
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

    global_summary = text_data.get("global_summary") or text_data.get("all_changes_summary") or {}
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
