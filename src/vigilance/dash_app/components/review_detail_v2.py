"""Composant de detail de revue V2 -- interface de validation par changement.

Ce composant genere le panneau droit de l'interface de revue et affiche :
- Les images de preuve (T1 et T2)
- La liste des changements pour le tableau courant
- Les boutons de validation par changement
- Les controles de navigation
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigilance.dash_app.components.review_display_shared import section_display_label
from vigilance.review_models_v2 import ChangeType

_CHANGE_TYPE_LABELS = {
    ChangeType.INDICATOR_ADDED.value: "Ajouté",
    ChangeType.INDICATOR_REMOVED.value: "Supprimé",
    ChangeType.INDICATOR_RENAMED.value: "Renommé",
    ChangeType.FOOTNOTE_ADDED.value: "Note ajoutée",
    ChangeType.FOOTNOTE_REMOVED.value: "Note supprimée",
    ChangeType.FOOTNOTE_MODIFIED.value: "Note modifiée",
    ChangeType.TABLE_ADDED.value: "Tableau ajouté",
    ChangeType.TABLE_REMOVED.value: "Tableau supprimé",
    ChangeType.STRUCTURE_CHANGE.value: "Structure modifiée",
    ChangeType.UNCERTAIN.value: "Incertain",
    ChangeType.MODIFIED.value: "Modifié",
    "indicator_added": "Ajouté",
    "indicator_removed": "Supprimé",
    "indicator_renamed": "Renommé",
    "footnote_added": "Note ajoutée",
    "footnote_removed": "Note supprimée",
    "footnote_modified": "Note modifiée",
    "table_added": "Tableau ajouté",
    "table_removed": "Tableau supprimé",
    "structure_change": "Structure modifiée",
    "uncertain": "Incertain",
    "modified": "Modifié",
}

_CHANGE_TYPE_COLORS = {
    ChangeType.INDICATOR_ADDED.value: "success",
    ChangeType.INDICATOR_REMOVED.value: "danger",
    ChangeType.INDICATOR_RENAMED.value: "warning",
    ChangeType.FOOTNOTE_ADDED.value: "info",
    ChangeType.FOOTNOTE_REMOVED.value: "dark",
    ChangeType.FOOTNOTE_MODIFIED.value: "info",
    ChangeType.TABLE_ADDED.value: "success",
    ChangeType.TABLE_REMOVED.value: "danger",
    ChangeType.STRUCTURE_CHANGE.value: "primary",
    ChangeType.UNCERTAIN.value: "secondary",
    ChangeType.MODIFIED.value: "primary",
    "indicator_added": "success",
    "indicator_removed": "danger",
    "indicator_renamed": "warning",
    "footnote_added": "info",
    "footnote_removed": "dark",
    "footnote_modified": "info",
    "table_added": "success",
    "table_removed": "danger",
    "structure_change": "primary",
    "uncertain": "secondary",
    "modified": "primary",
}

_FOOTNOTE_CHANGE_TYPES = {
    ChangeType.FOOTNOTE_ADDED.value,
    ChangeType.FOOTNOTE_REMOVED.value,
    ChangeType.FOOTNOTE_MODIFIED.value,
    "footnote_added",
    "footnote_removed",
    "footnote_modified",
}


def _format_section(section: str) -> str:
    """Formate le nom de section pour l'affichage."""
    return section_display_label(section)


def _normalize_text(value: object) -> str:
    """Normalise une valeur en chaine nettoyee."""
    return str(value or "").strip()


def _is_footnote_change(change_type: str) -> bool:
    """Verifie si le type de changement concerne une note de bas de page."""
    return str(change_type or "") in _FOOTNOTE_CHANGE_TYPES


def _get_change_row_summary(change: dict) -> str:
    """Retourne un libelle compact et lisible pour la ligne de changement.

    Le resume reste volontairement court. Le texte integral des notes longues
    est affiche separement pour le changement selectionne.
    """
    change_type = str(change.get("change_type", "") or "")
    payload = change.get("payload", {}) or {}

    if change_type in (
        "indicator_added",
        "indicator_removed",
        ChangeType.INDICATOR_ADDED.value,
        ChangeType.INDICATOR_REMOVED.value,
    ):
        return _normalize_text(payload.get("indicator_name")) or "(indicateur)"

    if change_type in ("indicator_renamed", ChangeType.INDICATOR_RENAMED.value):
        from_val = _normalize_text(payload.get("from"))
        to_val = _normalize_text(payload.get("to"))
        return f"{from_val} → {to_val}"

    if _is_footnote_change(change_type):
        ref = _normalize_text(payload.get("footnote_ref"))
        ref_label = f" [{ref}]" if ref else ""
        if change_type in ("footnote_added", ChangeType.FOOTNOTE_ADDED.value):
            return f"Note ajoutée{ref_label}"
        if change_type in ("footnote_removed", ChangeType.FOOTNOTE_REMOVED.value):
            return f"Note supprimée{ref_label}"
        return f"Note modifiée{ref_label}"

    if change_type in (
        "table_added",
        "table_removed",
        ChangeType.TABLE_ADDED.value,
        ChangeType.TABLE_REMOVED.value,
    ):
        return _normalize_text(payload.get("description")) or "Tableau entier"

    return _normalize_text(payload.get("description")) or "Changement"


def _build_detail_block(label: str, text: str, muted: bool = False) -> html.Div:
    """Construit un bloc de detail avec libelle et texte."""
    content = text or "Élément absent"
    text_class = "text-muted fst-italic" if muted or not text else "text-dark"
    return html.Div(
        [
            html.Div(label, className="fw-semibold border-bottom px-3 py-2"),
            html.Div(
                content,
                className=f"px-3 py-3 {text_class}",
                style={
                    "whiteSpace": "pre-wrap",
                    "overflowWrap": "anywhere",
                    "wordBreak": "break-word",
                    "lineHeight": "1.55",
                },
            ),
        ],
        className="border rounded bg-white overflow-hidden",
    )


def _build_change_full_detail(change: dict) -> html.Div | None:
    """Construit le détail complet d'un changement (note de bas de page et/ou justification GPT)."""
    change_type = str(change.get("change_type", "") or "")
    blocks: list[Any] = []

    # --- Détail texte des notes de bas de page ---
    if _is_footnote_change(change_type):
        payload = change.get("payload", {}) or {}
        old_text = _normalize_text(payload.get("old_text"))
        new_text = _normalize_text(payload.get("new_text"))

        if change_type in ("footnote_added", ChangeType.FOOTNOTE_ADDED.value):
            blocks.append(_build_detail_block("Trimestre courant", new_text))
        elif change_type in ("footnote_removed", ChangeType.FOOTNOTE_REMOVED.value):
            blocks.append(_build_detail_block("Trimestre précédent", old_text))
            blocks.append(_build_detail_block("Trimestre courant", "", muted=True))
        else:
            blocks.append(_build_detail_block("Trimestre précédent", old_text, muted=not old_text))
            blocks.append(_build_detail_block("Trimestre courant", new_text, muted=not new_text))

    # --- Justification GPT par changement ---
    payload = change.get("payload", {}) or {}
    assessment = payload.get("analyst_assessment") or {}
    justification = str(assessment.get("justification", "") or "").strip()
    if justification:
        relevance_level = assessment.get("relevance_level")
        level_labels = {1: "Critique / Réglementaire", 2: "Élevé / Structurel", 3: "Faible / Non substantif"}
        level_colors = {1: "danger", 2: "warning", 3: "secondary"}
        level_badge = (
            dbc.Badge(
                level_labels.get(relevance_level, ""),
                color=level_colors.get(relevance_level, "secondary"),
                className="me-2",
            )
            if relevance_level in level_labels
            else None
        )
        blocks.append(
            html.Div(
                [
                    html.Small(
                        [level_badge, "Justification IA :"] if level_badge else "Justification IA :",
                        className="fw-bold text-muted",
                    ),
                    html.P(justification, className="mb-0 mt-1 small fst-italic"),
                ],
                className="mt-2 p-2 bg-light rounded",
            )
        )

    if not blocks:
        return None

    return html.Div(blocks, className="d-grid gap-3 mt-3")


_THEMES_AMF_DISPLAY: dict[str, str] = {
    "DIVULGATION_AJOUT": "Ajout de divulgation",
    "DIVULGATION_RETRAIT": "Retrait de divulgation",
    "MODIFICATION_TEXTE_RISQUE": "Modif. texte risque",
    "MODIFICATION_METHODOLOGIE": "Modif. méthodologie",
    "FACTEUR_RISQUE_CHANGEMENT": "Facteur de risque",
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

_EXCLUSION_REASON_DISPLAY: dict[str, str] = {
    "variation_numerique_propre_banque": "Variation chiffrée propre à la banque",
    "reformulation_mineure": "Reformulation sans nouveau fond",
    "deplacement_texte": "Déplacement de texte sans modification",
    "formatage_visuel": "Formatage visuel",
    "non_pertinent_autre": "Non pertinent",
}

_IMPACT_LEVEL_DISPLAY: dict[str, str] = {
    "MAJEUR": "MAJEUR",
    "MODERE": "MODÉRÉ",
    "MINEUR": "MINEUR",
}

_IMPACT_LEVEL_COLORS: dict[str, str] = {
    "MAJEUR": "danger",
    "MODERE": "warning",
    "MINEUR": "info",
}

_ACTION_REQUISE_DISPLAY: dict[str, str] = {
    "escalade": "Escalade",
    "investigation": "Investigation",
    "confirmation": "À confirmer",
    "information": "Pour information",
    "aucune": "Aucune",
}


def _build_themes_amf_chips(themes: list[str], *, max_visible: int = 4) -> html.Div:
    """Affiche les thèmes AMF en chips gris (max ``max_visible`` puis « +N »)."""
    if not themes:
        return html.Div()
    visible = themes[:max_visible]
    overflow = themes[max_visible:]
    chips: list = [
        dbc.Badge(
            _THEMES_AMF_DISPLAY.get(theme, theme),
            color="light",
            text_color="dark",
            className="me-1 mb-1 border",
        )
        for theme in visible
    ]
    if overflow:
        tooltip = ", ".join(_THEMES_AMF_DISPLAY.get(t, t) for t in overflow)
        chips.append(
            dbc.Badge(
                f"+{len(overflow)}",
                color="secondary",
                className="me-1 mb-1",
                title=tooltip,
            )
        )
    return html.Div(chips, className="mb-2")


def _build_non_relevant_card(exclusion_reason: str | None) -> html.Div:
    """Carte minimaliste pour les changements jugés non pertinents par GPT.

    Affiche la raison d'exclusion AMF (sans le « Activez GPT » fallback ancien).
    """
    reason_label = _EXCLUSION_REASON_DISPLAY.get(
        str(exclusion_reason or ""), "Non pertinent"
    )
    return html.Div(
        [
            html.H6("Explication IA générative", className="mb-2"),
            html.Div(
                dbc.Badge(
                    "Non pertinent",
                    color="secondary",
                    className="me-2",
                ),
                className="mb-2",
            ),
            html.Small(
                [html.Strong("Raison : "), reason_label],
                className="text-muted",
            ),
        ],
        className="mb-4",
    )


def _build_genai_section(table: dict) -> html.Div:
    """Génère le bloc d'explication IA aligné taxonomie AMF.

    Hiérarchie d'affichage (alignée avec la charge cognitive analyste) :
    1. Bandeau du haut : ✨ Nouvelle idée + badge impact_level (couleur)
    2. Thèmes AMF en chips gris (max 4 + overflow)
    3. Justification IA (nouvelle_idee_justification — note d'analyste)
    4. Action suggérée (discrète, en bas)

    Si ``is_relevant=False`` → carte minimaliste avec la raison d'exclusion.
    Si ``genai_analysis`` est totalement vide (legacy data sans triage) →
    message court signalant l'absence de classification.
    """
    ga = table.get("genai_analysis") or {}

    if not ga:
        return html.Div(
            [
                html.H6("Explication IA générative", className="mb-2"),
                html.P(
                    "Aucune classification IA disponible pour cet élément.",
                    className="text-muted mb-0",
                ),
            ],
            className="mb-4",
        )

    is_relevant = bool(ga.get("is_relevant", False))
    if not is_relevant:
        return _build_non_relevant_card(ga.get("exclusion_reason"))

    nouvelle_idee = bool(ga.get("nouvelle_idee", False))
    nouvelle_idee_justification = str(
        ga.get("nouvelle_idee_justification", "") or ""
    ).strip()
    themes_amf = list(ga.get("themes_amf") or [])
    impact_level = str(ga.get("impact_level", "") or "").upper()
    action_requise = str(ga.get("action_requise", "") or "").lower()

    # Bandeau principal : nouvelle idée + impact (deux signaux les plus
    # importants pour le triage analyste).
    header_badges: list = []
    if nouvelle_idee:
        header_badges.append(
            dbc.Badge(
                "✨ Nouvelle idée",
                color="primary",
                className="me-2",
            )
        )
    else:
        header_badges.append(
            dbc.Badge(
                "Pas une nouvelle idée",
                color="secondary",
                className="me-2",
            )
        )
    if impact_level:
        header_badges.append(
            dbc.Badge(
                _IMPACT_LEVEL_DISPLAY.get(impact_level, impact_level),
                color=_IMPACT_LEVEL_COLORS.get(impact_level, "secondary"),
                className="me-2",
            )
        )

    # Justification IA — schéma AMF v2 strict (plus de fallback legacy).
    # Si vide, on affiche un message explicite : ré-exécuter la pipeline.
    justification = nouvelle_idee_justification

    # Action suggérée (discrète, sous la justification).
    action_line: html.Small | None = None
    if action_requise and action_requise != "aucune":
        action_line = html.Small(
            [
                html.Strong("Action suggérée : "),
                _ACTION_REQUISE_DISPLAY.get(action_requise, action_requise.capitalize()),
            ],
            className="d-block text-muted mt-2",
        )

    body: list = [
        html.H6("Explication IA générative", className="mb-2"),
        html.Div(header_badges, className="mb-2"),
        _build_themes_amf_chips(themes_amf),
        html.P(
            justification
            or "Justification AMF non disponible — relancer la pipeline pour obtenir le triage.",
            className="mb-0 small",
            style={"whiteSpace": "pre-wrap"},
        ),
    ]
    if action_line is not None:
        body.append(action_line)

    return html.Div(body, className="mb-4")


def build_change_list_v2(
    changes: list[dict],
    current_change_idx: int,
) -> dbc.ListGroup:
    """Construit la liste des changements pour un tableau.

    Args:
        changes: Liste de dictionnaires ``ChangeItem``.
        current_change_idx: Index du changement actuellement selectionne.

    Returns:
        Un ``Div`` contenant la liste des changements.
    """
    if not changes:
        return html.Div(
            [
                html.P("Aucun changement dans ce tableau.", className="text-muted"),
            ]
        )

    change_rows = []
    for idx, change in enumerate(changes):
        is_current = idx == current_change_idx
        status = change.get("validation_status", "pending")
        change_type = change.get("change_type", "")
        is_required = change.get("is_required", True)
        change_id = str(change.get("change_id", "") or f"idx_{idx}")

        # Status icon
        if status == "approved":
            status_icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif status == "rejected":
            status_icon = html.I(className="bi bi-x-circle-fill text-danger me-2")
        elif status == "skipped":
            status_icon = html.I(className="bi bi-dash-circle text-secondary me-2")
        else:
            status_icon = html.I(className="bi bi-circle text-warning me-2")

        # Change type badge
        type_label = _CHANGE_TYPE_LABELS.get(change_type, change_type)
        type_color = _CHANGE_TYPE_COLORS.get(change_type, "secondary")

        # Description
        description = _get_change_row_summary(change)
        full_detail = _build_change_full_detail(change) if is_current else None

        # Required indicator
        required_badge = None
        if not is_required:
            required_badge = dbc.Badge(
                "Optionnel", color="light", text_color="dark", className="ms-2"
            )

        # Current item highlight
        current_class = (
            "bg-primary bg-opacity-10 border-start border-3 border-primary"
            if is_current
            else ""
        )

        # "Modifier" button for already-validated changes
        reset_button = (
            dbc.Button(
                [html.I(className="bi bi-pencil me-1"), "Modifier"],
                id={"type": "btn-reset-change-v2", "change_id": change_id},
                color="light",
                size="sm",
                className="ms-auto flex-shrink-0",
                title="Réinitialiser la décision pour re-valider",
            )
            if status in ("approved", "rejected", "skipped")
            else None
        )

        row = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                status_icon,
                                dbc.Badge(
                                    type_label,
                                    color=type_color,
                                    className="me-2",
                                ),
                                html.Span(
                                    description,
                                    className="small flex-grow-1",
                                    style={"wordBreak": "break-word"},
                                ),
                                required_badge,
                            ],
                            className="d-flex align-items-center flex-wrap flex-grow-1 gap-1",
                        ),
                        reset_button,
                    ],
                    className="d-flex align-items-start gap-2",
                ),
                # Show validation notes if present
                html.Small(
                    change.get("validation_notes", ""),
                    className="text-muted d-block mt-1 fst-italic",
                )
                if change.get("validation_notes")
                else None,
                full_detail,
                # Store the change exact text for highlight callback
                dcc.Store(
                    id={"type": "change-text-data-v2", "change_id": change_id},
                    data=change,
                ),
            ],
            id={"type": "change-row-v2", "change_id": change_id},
            className=f"p-2 {current_class}",
            style={"cursor": "pointer"},
            action=True,
        )
        change_rows.append(row)

    # Added active-highlight store inside the list container to avoid duplication
    # but still available in the DOM for callbacks
    return html.Div(
        [
            dbc.ListGroup(change_rows, flush=True, className="mb-3"),
            dcc.Store(id="active-highlight-store", data=None),
        ]
    )


def build_validation_panel_v2(
    table: dict,
    current_change_idx: int,
) -> html.Div:
    """Construit les boutons de validation et le champ de notes pour le changement courant.

    Args:
        table: Dictionnaire ``ReviewTableItem`` courant.
        current_change_idx: Index du changement courant dans le tableau.

    Returns:
        Un ``Div`` contenant les controles de validation.
    """
    changes = table.get("changes", [])
    n_changes = len(changes)

    if not changes or current_change_idx >= n_changes:
        return html.Div()

    current_change = changes[current_change_idx]
    change_type = current_change.get("change_type", "")
    status = current_change.get("validation_status", "pending")
    description = _get_change_row_summary(current_change)

    # Current change info
    change_info = html.Div(
        [
            html.H6("Changement actuel", className="mb-2"),
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Badge(
                                _CHANGE_TYPE_LABELS.get(change_type, change_type),
                                color=_CHANGE_TYPE_COLORS.get(change_type, "secondary"),
                                className="me-2",
                            ),
                            html.Span(description, className="fw-semibold"),
                        ],
                        className="d-flex align-items-center flex-wrap gap-1",
                    ),
                ],
                className="mb-2",
            ),
        ]
    )

    # Notes input
    notes_input = dbc.Textarea(
        id="validation-notes-v2",
        placeholder="Notes de validation (optionnel)...",
        value=current_change.get("validation_notes", ""),
        className="mb-3",
        rows=2,
    )

    # Validation buttons
    is_validated = status in ("approved", "rejected", "skipped")
    validation_buttons = html.Div(
        [
            dbc.Button(
                [html.I(className="bi bi-check-lg me-1"), "Approuver"],
                id="btn-approve-change-v2",
                color="success",
                className="me-2",
                disabled=is_validated,
            ),
            dbc.Button(
                [html.I(className="bi bi-x-lg me-1"), "Rejeter"],
                id="btn-reject-change-v2",
                color="danger",
                className="me-2",
                disabled=is_validated,
            ),
            dbc.Button(
                [html.I(className="bi bi-arrow-right me-1"), "Passer"],
                id="btn-skip-change-v2",
                color="secondary",
                outline=True,
                disabled=is_validated,
            ),
        ],
        className="mb-3",
    )

    # Status indicator if already validated
    change_id_current = str(current_change.get("change_id", "") or f"idx_{current_change_idx}")
    status_indicator = None
    if is_validated:
        _status_colors = {"approved": "success", "rejected": "danger", "skipped": "secondary"}
        _status_labels = {
            "approved": ("bi bi-check-circle me-2", "Approuvé"),
            "rejected": ("bi bi-x-circle me-2", "Rejeté"),
            "skipped": ("bi bi-dash-circle me-2", "Passé"),
        }
        icon_cls, label = _status_labels.get(status, ("bi bi-circle me-2", status))
        status_indicator = dbc.Alert(
            [
                html.Div(
                    [
                        html.Span([html.I(className=icon_cls), label]),
                        dbc.Button(
                            [html.I(className="bi bi-pencil me-1"), "Modifier"],
                            id={"type": "btn-reset-change-v2", "change_id": change_id_current},
                            color="light",
                            size="sm",
                            className="ms-auto",
                            title="Réinitialiser pour re-valider",
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between",
                )
            ],
            color=_status_colors.get(status, "secondary"),
            className="py-2",
        )

    # Navigation buttons
    nav_buttons = html.Div(
        [
            dbc.Button(
                [html.I(className="bi bi-chevron-left me-1"), "Précédent"],
                id="btn-prev-change-v2",
                color="light",
                className="me-2",
                disabled=current_change_idx <= 0,
            ),
            dbc.Button(
                ["Suivant", html.I(className="bi bi-chevron-right ms-1")],
                id="btn-next-change-v2",
                color="light",
                disabled=current_change_idx >= n_changes - 1,
            ),
            html.Span(
                f" {current_change_idx + 1} / {n_changes}",
                className="ms-3 text-muted small",
            ),
        ],
        className="mb-3",
    )

    return html.Div(
        [
            change_info,
            notes_input,
            validation_buttons,
            status_indicator,
            html.Hr(),
            nav_buttons,
        ]
    )


def build_review_detail_v2(
    table: dict | None,
    current_change_idx: int,
    proof_image_t1_b64: str = "",
    proof_image_t2_b64: str = "",
    show_proofs: bool = True,
) -> html.Div:
    """Construit le panneau complet de detail de revue V2.

    Args:
        table: Dictionnaire ``ReviewTableItem`` courant.
        current_change_idx: Index du changement courant.
        proof_image_t1_b64: Image de preuve T1 encodee en base64.
        proof_image_t2_b64: Image de preuve T2 encodee en base64.
        show_proofs: Si ``True``, affiche la section des preuves visuelles.

    Returns:
        Le panneau de detail de revue complet.
    """
    if not table:
        return html.Div(
            [
                html.H5("Aucun élément sélectionné"),
                html.P(
                    "Sélectionnez un tableau dans la file de revue.",
                    className="text-muted",
                ),
            ]
        )

    table_name = table.get("table_name", "Tableau")
    section = _format_section(table.get("section", ""))
    page_t1 = table.get("page_t1")
    page_t2 = table.get("page_t2")
    table_status = table.get("table_status", "pending")
    summary = table.get("summary", {})
    match_meta = table.get("match_metadata") or {}

    alert_badges = []
    if match_meta.get("drastic_row_drop"):
        alert_badges.append(
            dbc.Alert(
                [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "ALERTE CRITIQUE : Baisse drastique du nombre de lignes détectée. Vérifiez manuellement une potentielle troncature du modèle.",
                ],
                color="danger",
                className="py-2 mb-3 fw-bold",
            )
        )

    # Header with table info
    header = html.Div(
        [
            html.Div(
                [
                    html.H5(table_name, className="mb-1"),
                    html.Small(
                        [
                            f"Section : {section}",
                            html.Span(" | ", className="text-muted"),
                            f"Pages : précédent p.{page_t1 or '?'} / courant p.{page_t2 or '?'}",
                        ],
                        className="text-muted",
                    ),
                ]
            ),
            # Status badge
            dbc.Badge(
                "Complété"
                if table_status == "completed"
                else ("En cours" if table_status == "partial" else "En attente"),
                color="success"
                if table_status == "completed"
                else ("info" if table_status == "partial" else "warning"),
                className="ms-auto",
            ),
        ],
        className="d-flex justify-content-between align-items-start mb-3",
    )

    # Summary badges
    summary_badges = html.Div(
        [
            dbc.Badge(
                f"+{summary.get('indicators_added', 0)}",
                color="success",
                className="me-1",
            )
            if summary.get("indicators_added", 0)
            else None,
            dbc.Badge(
                f"-{summary.get('indicators_removed', 0)}",
                color="danger",
                className="me-1",
            )
            if summary.get("indicators_removed", 0)
            else None,
            dbc.Badge(
                f"~{summary.get('indicators_renamed', 0)}",
                color="warning",
                className="me-1",
            )
            if summary.get("indicators_renamed", 0)
            else None,
            dbc.Badge(
                f"FN {summary.get('footnotes_changed', 0)}",
                color="info",
                className="me-1",
            )
            if summary.get("footnotes_changed", 0)
            else None,
            html.Span(
                f"Validé : {summary.get('validated', 0)}/{summary.get('total_changes', 0)}",
                className="ms-2 text-muted small",
            ),
        ],
        className="mb-3",
    )

    # Proof images
    proof_section = (
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6(
                                    [
                                        html.I(className="bi bi-file-earmark-pdf me-2"),
                                        "Trimestre courant",
                                    ],
                                    className="mb-2",
                                ),
                                html.Img(
                                    src=f"data:image/png;base64,{proof_image_t2_b64}"
                                    if proof_image_t2_b64
                                    else "",
                                    className="img-fluid border rounded",
                                    style={
                                        "maxHeight": "400px",
                                        "width": "100%",
                                        "objectFit": "contain",
                                    },
                                )
                                if proof_image_t2_b64
                                else html.Div(
                                    "Image non disponible",
                                    className="text-muted p-4 bg-light rounded text-center",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.H6(
                                    [
                                        html.I(className="bi bi-file-earmark-pdf me-2"),
                                        "Trimestre précédent",
                                    ],
                                    className="mb-2",
                                ),
                                html.Img(
                                    src=f"data:image/png;base64,{proof_image_t1_b64}"
                                    if proof_image_t1_b64
                                    else "",
                                    className="img-fluid border rounded",
                                    style={
                                        "maxHeight": "400px",
                                        "width": "100%",
                                        "objectFit": "contain",
                                    },
                                )
                                if proof_image_t1_b64
                                else html.Div(
                                    "Image non disponible",
                                    className="text-muted p-4 bg-light rounded text-center",
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        if show_proofs
        else html.Div()
    )

    genai_section = _build_genai_section(table)

    # Changes list
    changes = table.get("changes", [])
    table_only_change = len(changes) == 1 and str(changes[0].get("change_type", "")) in {
        ChangeType.TABLE_ADDED.value,
        ChangeType.TABLE_REMOVED.value,
        "table_added",
        "table_removed",
    }
    changes_section = html.Div(
        [
            html.H6(
                [
                    html.I(className="bi bi-list-check me-2"),
                    (
                        "Validation au niveau tableau"
                        if table_only_change
                        else f"Changements ({len(changes)})"
                    ),
                ],
                className="mb-2",
            ),
            build_change_list_v2(changes, current_change_idx),
        ],
        className="mb-4",
    )

    # Validation panel
    validation_section = html.Div(
        [
            html.H6(
                [html.I(className="bi bi-clipboard-check me-2"), "Validation"],
                className="mb-2",
            ),
            build_validation_panel_v2(table, current_change_idx),
        ]
    )

    # Table navigation buttons
    table_nav = html.Div(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="bi bi-arrow-left me-1"),
                                    "Tableau précédent",
                                ],
                                id="btn-prev-table-v2",
                                color="outline-primary",
                                className="w-100",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    "Tableau suivant",
                                    html.I(className="bi bi-arrow-right ms-1"),
                                ],
                                id="btn-next-table-v2",
                                color="primary",
                                className="w-100",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-2",
            ),
        ],
        className="mt-4",
    )

    return html.Div(
        [
            header,
            html.Div(alert_badges) if alert_badges else None,
            summary_badges,
            html.Hr(),
            proof_section,
            genai_section,
            changes_section,
            validation_section,
            table_nav,
        ]
    )
