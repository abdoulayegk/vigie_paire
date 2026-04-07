"""Composant de detail de revue V2 -- interface de validation par changement.

Ce composant genere le panneau droit de l'interface de revue et affiche :
- Les images de preuve (T1 et T2)
- La liste des changements pour le tableau courant
- Les boutons de validation par changement
- Les controles de navigation
"""

from __future__ import annotations

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
    """Construit le detail complet d'un changement de note de bas de page."""
    change_type = str(change.get("change_type", "") or "")
    if not _is_footnote_change(change_type):
        return None

    payload = change.get("payload", {}) or {}
    old_text = _normalize_text(payload.get("old_text"))
    new_text = _normalize_text(payload.get("new_text"))

    if change_type in ("footnote_added", ChangeType.FOOTNOTE_ADDED.value):
        blocks = [_build_detail_block("Trimestre courant", new_text)]
    elif change_type in ("footnote_removed", ChangeType.FOOTNOTE_REMOVED.value):
        blocks = [
            _build_detail_block("Trimestre précédent", old_text),
            _build_detail_block("Trimestre courant", "", muted=True),
        ]
    else:
        blocks = [
            _build_detail_block("Trimestre précédent", old_text, muted=not old_text),
            _build_detail_block("Trimestre courant", new_text, muted=not new_text),
        ]

    return html.Div(blocks, className="d-grid gap-3 mt-3")


def _build_fallback_genai_message(table: dict) -> str:
    """Construit un message de repli lorsque la classification GenAI est indisponible."""
    change_types = {
        str(change.get("change_type", ""))
        for change in (table.get("changes", []) or [])
    }
    match_meta = table.get("match_metadata") or {}
    semantic_judge = (
        match_meta.get("semantic_judge") if isinstance(match_meta, dict) else None
    )
    semantic_hint = ""
    if isinstance(semantic_judge, dict):
        final_decision = str(semantic_judge.get("final_decision", "") or "").strip()
        guard_action = str(semantic_judge.get("guard_action", "") or "").strip()
        if final_decision and guard_action:
            semantic_hint = f" Contrôle sémantique : {final_decision} ({guard_action})."

    if change_types.intersection({ChangeType.TABLE_REMOVED.value, "table_removed"}):
        return (
            "Explication automatique : ce tableau est présent au trimestre précédent "
            "et absent au trimestre courant. Vérifier si le contenu a été renommé, "
            "fusionné ou déplacé dans une autre section."
            f"{semantic_hint}"
        )

    if change_types.intersection({ChangeType.TABLE_ADDED.value, "table_added"}):
        return (
            "Explication automatique : ce tableau est absent au trimestre précédent "
            "et présent au trimestre courant. Vérifier s'il s'agit d'une nouvelle "
            "divulgation ou d'une table décomposée depuis un ancien tableau."
            f"{semantic_hint}"
        )

    return (
        "Classification IA générative non disponible pour cet élément. Activez "
        "'Classer les changements avec l'IA générative (GPT-4o)' puis relancez l'analyse pour "
        "obtenir une justification détaillée."
    )


def _build_genai_section(table: dict) -> html.Div:
    """Genere le bloc d'explication GenAI concis pour la revue par l'analyste."""
    ga = table.get("genai_analysis") or {}
    relevance = str(ga.get("relevance", "") or "")
    risk = str(ga.get("risk_level", "") or "")
    confidence = ga.get("confidence", None)
    justification = str(ga.get("justification", "") or "").strip()

    if not any([relevance, risk, justification]):
        fallback_msg = _build_fallback_genai_message(table)
        return html.Div(
            [
                html.H6("Explication IA générative", className="mb-2"),
                html.P(fallback_msg, className="text-muted mb-0"),
            ],
            className="mb-4",
        )

    chips = []
    if relevance:
        chips.append(dbc.Badge(relevance, color="primary", className="me-2"))
    if risk:
        chips.append(dbc.Badge(f"Risque {risk}", color="warning", className="me-2"))
    if confidence is not None:
        try:
            conf_pct = max(0.0, min(1.0, float(confidence))) * 100
            chips.append(
                dbc.Badge(
                    f"Confiance {conf_pct:.0f}%",
                    color="light",
                    text_color="dark",
                    className="me-2",
                )
            )
        except (TypeError, ValueError):
            pass

    # New analyst fields
    impact_type = str(ga.get("impact_type", "") or "")
    project_phase = str(ga.get("project_phase", "") or "")
    action_requise = str(ga.get("action_requise", "") or "")
    impact_desc = str(ga.get("impact_description", "") or "").strip()

    _IMPACT_COLORS = {
        "structurel": "danger",
        "contenu": "primary",
        "methodologique": "warning",
        "cosmetique": "secondary",
    }
    _ACTION_COLORS = {
        "escalade": "danger",
        "investigation": "warning",
        "confirmation": "info",
        "information": "secondary",
        "aucune": "light",
    }
    _PHASE_DISPLAY = {
        "rapport_gestion": "Rapport de gestion",
        "pilier_3": "Pilier 3",
        "ifc": "IFC",
        "autre": "Autre",
    }

    secondary_chips = []
    if impact_type:
        secondary_chips.append(
            dbc.Badge(
                f"Impact : {impact_type}",
                color=_IMPACT_COLORS.get(impact_type, "secondary"),
                className="me-2",
            )
        )
    if project_phase:
        secondary_chips.append(
            dbc.Badge(
                _PHASE_DISPLAY.get(project_phase, project_phase),
                color="info",
                className="me-2",
            )
        )
    if action_requise and action_requise != "aucune":
        secondary_chips.append(
            dbc.Badge(
                f"Action : {action_requise}",
                color=_ACTION_COLORS.get(action_requise, "secondary"),
                className="me-2",
            )
        )

    detail_parts = []
    if impact_desc:
        detail_parts.append(
            html.Small(
                [html.Strong("Impact : "), impact_desc],
                className="d-block text-muted mb-1",
            )
        )

    return html.Div(
        [
            html.H6("Explication IA générative", className="mb-2"),
            html.Div(chips, className="mb-2"),
            html.Div(secondary_chips, className="mb-2")
            if secondary_chips
            else html.Div(),
            *detail_parts,
            html.P(justification or "Explication non fournie.", className="mb-0"),
        ],
        className="mb-4",
    )


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
