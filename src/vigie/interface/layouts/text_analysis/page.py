"""Layout de l'onglet Analyse Textuelle — vue analyste.

Affiche tous les changements textuels détectés hors ``unchanged``. Les filtres
restants sont gérés dans ``text_flow.py``.

Le rendu est reparti dans les modules du sous-package ``text_analysis`` ;
ce module assemble l'onglet Analyse Textuelle.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigie.comparaison.analyst_change_presentation import change_scope
from vigie.interface.components.text_change_presentation import (
    atomic_parent_context,
    atomic_parent_key,
    build_atomic_change_group,
)
from vigie.support.quarter_utils import quarter_label_from_payload

from .change_card import (
    _build_change_card,
)
from .labels import (
    _IMPACT_ORDER,
    _SECTION_LABELS,
    _UNSET,
    _plural_count,
)
from .overview import (
    _build_executive_banner,
    _build_text_review_progress,
    _count_auditable_text_changes,
)


def _section_has_auditable_text_changes(section: dict[str, Any]) -> bool:
    """Indique si une section contient au moins un changement texte affichable."""
    for change in section.get("all_block_comparisons") or []:
        if change.get("diff_type") == "unchanged":
            continue
        return True
    return False


def _default_text_section(section_comparisons: list[dict[str, Any]]) -> str | None:
    """Retourne la première section affichable à sélectionner par défaut."""
    first_key: str | None = None
    for section in section_comparisons:
        key = str(section.get("section_key") or "").strip()
        if not key:
            continue
        if first_key is None:
            first_key = key
        if _section_has_auditable_text_changes(section):
            return key
    return first_key


def _empty_text_state() -> list[html.Div]:
    """Composant Dash affiché lorsque aucun changement ne passe les filtres."""
    return [
        html.Div(
            "Aucun changement détecté correspondant aux filtres sélectionnés.",
            className="text-muted text-center py-4",
        )
    ]


def build_filtered_text_cards(
    text_data: dict[str, Any],
    filter_section: str | None,
    filter_impact: str | None,
    filter_action: str | None,
    filter_status: str | None = None,
    filter_scope: str | None = "qualitative",
) -> tuple[list[Any], str]:
    """Construit les cartes texte selon les filtres courants."""
    items: list[tuple[tuple[int, int, str, str, str], dict[str, Any], str]] = []
    current_label = quarter_label_from_payload(text_data, "current")
    previous_label = quarter_label_from_payload(text_data, "previous")
    bank_code = str(text_data.get("bank_code") or "").strip()
    for sec in text_data.get("section_comparisons") or []:
        key = sec.get("section_key", "")
        title = sec.get("section_title") or _SECTION_LABELS.get(key, key)

        if filter_section and key != filter_section:
            continue

        for change in sec.get("all_block_comparisons") or []:
            diff_type = change.get("diff_type", "")
            if diff_type == "unchanged":
                continue
            triage = change.get("genai_triage") or {}
            scope = change_scope(change)
            if scope == "hidden":
                continue
            if filter_scope == "qualitative" and scope != "qualitative":
                continue
            if filter_scope == "secondary" and scope != "secondary":
                continue

            review = change.get("_analyst_review") or {}
            review_status = str(review.get("status") or "pending").strip().lower()
            if filter_status == "remaining" and review_status not in {"pending", "skipped"}:
                continue
            if filter_status in {"approved", "rejected", "skipped"} and review_status != filter_status:
                continue

            impact = (triage.get("impact_level") or "MINEUR").upper()
            action = (triage.get("action_requise") or "aucune").lower()
            nouvelle_idee = bool(triage.get("nouvelle_idee", False))
            pages = change.get("pages_t2") or change.get("pages_t1") or []
            page_sort = ""
            if pages:
                try:
                    page_sort = f"{int(pages[0]):06d}"
                except (TypeError, ValueError):
                    page_sort = str(pages[0])

            if filter_impact and impact != filter_impact.upper():
                continue
            if filter_action and action != filter_action.lower():
                continue

            sort_key = (
                0 if triage.get("is_relevant", False) else 1,
                _IMPACT_ORDER.get(impact, 99),
                0 if nouvelle_idee else 1,
                title,
                page_sort,
                diff_type,
            )
            items.append((sort_key, change, title))

    items.sort(key=lambda x: x[0])

    grouped_items: dict[
        tuple[str, str, str, str],
        list[tuple[dict[str, Any], str]],
    ] = {}
    for index, (_, change, title) in enumerate(items):
        atomic_key = atomic_parent_key(change, section_title=title)
        if atomic_key is None:
            group_key = ("single", str(index), "", "")
        else:
            group_key = ("atomic", *atomic_key)
        grouped_items.setdefault(group_key, []).append((change, title))

    rendered: list[Any] = []
    displayed_count = 0
    for group_key, grouped_changes in grouped_items.items():
        group_cards: list[dbc.Card] = []
        for change, title in grouped_changes:
            card = _build_change_card(
                change,
                title,
                bank_code=bank_code,
                current_quarter_label=current_label,
                previous_quarter_label=previous_label,
            )
            if card is not None:
                group_cards.append(card)
        if not group_cards:
            continue
        displayed_count += len(group_cards)
        if group_key[0] == "atomic":
            rendered.append(
                build_atomic_change_group(
                    parent_context=atomic_parent_context(grouped_changes[0][0]),
                    cards=group_cards,
                )
            )
        else:
            rendered.extend(group_cards)

    if filter_scope == "secondary":
        count_text = _plural_count(
            displayed_count,
            "changement secondaire affiché",
            "changements secondaires affichés",
        )
    elif filter_scope == "all":
        count_text = _plural_count(
            displayed_count,
            "changement affiché",
            "changements affichés",
        )
    else:
        count_text = _plural_count(
            displayed_count,
            "changement qualitatif affiché",
            "changements qualitatifs affichés",
        )
    return rendered or _empty_text_state(), count_text


# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------


def _build_filter_bar(
    section_options: list[dict],
    selected_section: str | None,
    selected_scope: str,
    selected_impact: str | None,
    selected_action: str | None,
    selected_status: str,
    initial_count: str,
) -> html.Div:
    """Barre de filtres : section / impact / action / décision + compteur."""
    filters = html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-section",
                        options=section_options,
                        value=selected_section,
                        placeholder="Toutes les sections",
                        clearable=True,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-scope",
                        options=[
                            {
                                "label": "Changements qualitatifs",
                                "value": "qualitative",
                            },
                            {
                                "label": "Tous les changements",
                                "value": "all",
                            },
                            {
                                "label": "Secondaires / bruit",
                                "value": "secondary",
                            },
                        ],
                        value=selected_scope,
                        clearable=False,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-impact",
                        options=[
                            {"label": "Majeur", "value": "MAJEUR"},
                            {"label": "Modéré", "value": "MODERE"},
                            {"label": "Mineur", "value": "MINEUR"},
                        ],
                        value=selected_impact,
                        placeholder="Tous les impacts",
                        clearable=True,
                    ),
                    md=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-action",
                        options=[
                            {"label": "Revue prioritaire", "value": "revue_prioritaire"},
                            {"label": "Analyse approfondie", "value": "investigation"},
                            {"label": "Confirmation", "value": "confirmation"},
                            {"label": "Information", "value": "information"},
                            {"label": "Aucune", "value": "aucune"},
                        ],
                        value=selected_action,
                        placeholder="Toutes les actions",
                        clearable=True,
                    ),
                    md=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="text-filter-status",
                        options=[
                            {"label": "À traiter", "value": "remaining"},
                            {"label": "Validés", "value": "approved"},
                            {"label": "Rejetés", "value": "rejected"},
                            {"label": "Passés", "value": "skipped"},
                            {"label": "Toutes les décisions", "value": "all"},
                        ],
                        value=selected_status,
                        clearable=False,
                    ),
                    md=2,
                ),
            ],
            className="g-2 align-items-center",
        ),
        className="mb-2",
    )
    return html.Div(
        [
            filters,
            html.Div(
                initial_count,
                id="text-filter-count",
                className="small text-muted mt-2",
            ),
        ],
        className="mb-3 p-3 bg-white rounded border",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_text_analysis_tab(
    text_data: dict[str, Any] | None,
    *,
    filter_section: str | None | object = _UNSET,
    filter_scope: str | None = "qualitative",
    filter_impact: str | None = None,
    filter_action: str | None = None,
    filter_status: str | None = "remaining",
) -> html.Div:
    """Construit l'onglet analyse textuelle — vue analyste.

    Args:
        text_data: Contenu de text_comparison.json, ou None si non disponible.
        filter_section: Section sélectionnée, ``None`` pour toutes les sections.
        filter_scope: Périmètre qualitatif, secondaire ou complet.
        filter_impact: Niveau d'impact sélectionné.
        filter_action: Action de vigie sélectionnée.
        filter_status: Statut de décision sélectionné.

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
                        "uv run python -m vigie.cli.run_text_compare "
                        "--banque <BANQUE> --annee <ANNEE> --T2"
                    ),
                ],
                color="secondary",
                className="mt-3",
            )
        )

    global_summary = text_data.get("global_summary") or text_data.get("all_changes_summary") or {}
    section_comparisons = text_data.get("section_comparisons") or []
    q_cur = quarter_label_from_payload(text_data, "current")
    q_prev = quarter_label_from_payload(text_data, "previous")
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

    default_section = _default_text_section(section_comparisons)
    if filter_section is _UNSET:
        selected_section = default_section
    elif filter_section is None or filter_section in seen:
        selected_section = filter_section
    else:
        selected_section = default_section
    selected_status = (
        filter_status
        if filter_status in {"remaining", "approved", "rejected", "skipped", "all"}
        else "remaining"
    )
    selected_scope = (
        filter_scope
        if filter_scope in {"qualitative", "secondary", "all"}
        else "qualitative"
    )
    initial_cards, initial_count = build_filtered_text_cards(
        text_data,
        selected_section,
        filter_impact,
        filter_action,
        selected_status,
        selected_scope,
    )

    return html.Div(
        [
            _build_executive_banner(
                global_summary,
                bank,
                q_cur,
                q_prev,
                auditable_changes=_count_auditable_text_changes(section_comparisons),
            ),
            html.Div(
                [
                    _build_text_review_progress(section_comparisons),
                    _build_filter_bar(
                        section_options,
                        selected_section,
                        selected_scope,
                        filter_impact,
                        filter_action,
                        selected_status,
                        initial_count,
                    ),
                ],
                className="text-review-sticky-tools",
            ),
            html.Div(initial_cards, id="text-cards-container"),
        ],
        className="pt-3",
    )
