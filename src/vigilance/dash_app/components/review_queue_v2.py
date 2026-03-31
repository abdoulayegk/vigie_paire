"""Review Queue Component V2 - Deduplicated grouped tables.

This component renders the left panel of the review UI showing
one item per table (not per change), with progress indicators.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from vigilance.i18n import t
from vigilance.review_models_v2 import ChangeType

_CHANGE_TYPE_LABELS = {
    ChangeType.INDICATOR_ADDED.value: "Ajoute",
    ChangeType.INDICATOR_REMOVED.value: "Supprime",
    ChangeType.INDICATOR_RENAMED.value: "Renomme",
    ChangeType.FOOTNOTE_ADDED.value: "Note ajoutee",
    ChangeType.FOOTNOTE_REMOVED.value: "Note supprimee",
    ChangeType.FOOTNOTE_MODIFIED.value: "Note modifiee",
    ChangeType.TABLE_ADDED.value: "Tableau ajoute",
    ChangeType.TABLE_REMOVED.value: "Tableau supprime",
    ChangeType.STRUCTURE_CHANGE.value: "Structure modifiee",
    ChangeType.UNCERTAIN.value: "Incertain",
    ChangeType.MODIFIED.value: "Modifie",
    "indicator_added": "Ajoute",
    "indicator_removed": "Supprime",
    "indicator_renamed": "Renomme",
    "footnote_added": "Note ajoutee",
    "footnote_removed": "Note supprimee",
    "footnote_modified": "Note modifiee",
    "table_added": "Tableau ajoute",
    "table_removed": "Tableau supprime",
    "structure_change": "Structure modifiee",
    "uncertain": "Incertain",
    "modified": "Modifie",
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

_PRIORITY_DISPLAY = {
    "critique": ("Critique", "danger"),
    "prioritaire": ("Prioritaire", "warning"),
    "normal": ("Normal", "secondary"),
}

_ACTION_DISPLAY = {
    "escalade": ("Escalade", "danger"),
    "investigation": ("Investigation", "warning"),
    "confirmation": ("Confirmation", "info"),
    "information": ("Information", "secondary"),
    "aucune": None,
}

_CATEGORY_DISPLAY = {
    "REGLEMENTAIRE": ("Réglementaire", "danger"),
    "RISQUE": ("Risque", "warning"),
    "CAPITAL": ("Capital", "primary"),
    "STRUCTURE": ("Structure", "info"),
    "COSMETIQUE": ("Cosmétique", "secondary"),
    "INCONNU": None,
}

_RELEVANCE_DISPLAY = {
    "REGLEMENTAIRE": "Reglementaire",
    "NOUVELLE_DIVULGATION": "Nouvelle divulgation",
    "STRUCTUREL": "Structurel",
    "NON_SIGNIFICATIF": "Non significatif",
    "NON_CLASSIFIE": "Non classifie",
}

_RELEVANCE_COLORS = {
    "REGLEMENTAIRE": "danger",
    "NOUVELLE_DIVULGATION": "info",
    "STRUCTUREL": "primary",
    "NON_SIGNIFICATIF": "secondary",
    "NON_CLASSIFIE": "light",
}

_RISK_DISPLAY = {
    "ELEVE": "Eleve",
    "MODERE": "Modere",
    "FAIBLE": "Faible",
}

_RISK_COLORS = {
    "ELEVE": "danger",
    "MODERE": "warning",
    "FAIBLE": "success",
}

_CHANGE_FAMILY_LABELS = {
    "structure": "Structure et tableau",
    "indicators": "Indicateurs",
    "footnotes": "Notes",
    "other": "Autres",
}

_CHANGE_FAMILY_ORDER = ("structure", "indicators", "footnotes", "other")


def _tone_class(color: str) -> str:
    tone = str(color or "secondary").strip().lower()
    if tone in {"primary", "secondary", "success", "danger", "warning", "info"}:
        return tone
    return "secondary"


def _format_section(section: str) -> str:
    """Format section name for display."""
    if not section:
        return "Autre"
    # Capitalize first letter of each word
    return " ".join(w.capitalize() for w in section.replace("_", " ").split())


def _queue_page_summary_v2(table: dict) -> str:
    """Return concise page context string for queue items."""
    page_t1 = table.get("page_t1")
    page_t2 = table.get("page_t2")

    # Check if this is an added or removed table
    changes = table.get("changes", [])
    change_types = {c.get("change_type", "") for c in changes}

    if ChangeType.TABLE_ADDED.value in change_types or "table_added" in change_types:
        return f"p.{page_t2}" if page_t2 is not None else ""
    if (
        ChangeType.TABLE_REMOVED.value in change_types
        or "table_removed" in change_types
    ):
        return f"p.{page_t1}" if page_t1 is not None else ""
    if page_t1 is not None and page_t2 is not None:
        return f"Préc. p.{page_t1} / Cour. p.{page_t2}"
    if page_t2 is not None:
        return f"p.{page_t2}"
    if page_t1 is not None:
        return f"p.{page_t1}"
    return ""


def _get_change_payload(change: dict) -> dict:
    payload = change.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _collapse_spaces(value: object) -> str:
    return " ".join(_normalize_text(value).split())


def _truncate_text(value: object, max_length: int = 110) -> str:
    text = _collapse_spaces(value)
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)].rstrip() + "..."


def _get_review_priority(table: dict) -> str:
    match_meta = table.get("match_metadata") or {}
    genai = table.get("genai_analysis") or {}
    priority = (
        str((match_meta.get("review_priority") or genai.get("review_priority") or ""))
        .strip()
        .lower()
    )
    return priority


def _build_signal_chip(label: str, color: str) -> html.Span:
    return html.Span(
        label,
        className=f"review-queue-signal review-queue-signal--{_tone_class(color)}",
    )


def _build_metric_chip(label: str, value: int, tone: str) -> html.Span:
    return html.Span(
        [
            html.Span(label, className="review-queue-metric-label"),
            html.Span(str(value), className="review-queue-metric-value"),
        ],
        className=f"review-queue-metric review-queue-metric--{_tone_class(tone)}",
    )


def _build_detail_panel(label: str, text: str, empty_message: str) -> html.Div:
    content = text or empty_message
    content_class = (
        "review-queue-change-text"
        if text
        else "review-queue-change-text text-muted fst-italic"
    )
    return html.Div(
        [
            html.Div(label, className="review-queue-change-panel-label"),
            html.Div(content, className=content_class),
        ],
        className="review-queue-change-panel",
    )


def _get_change_family(change: dict) -> str:
    change_type = str(change.get("change_type", "") or "")
    if change_type in {
        ChangeType.TABLE_ADDED.value,
        ChangeType.TABLE_REMOVED.value,
        ChangeType.STRUCTURE_CHANGE.value,
        ChangeType.MODIFIED.value,
        ChangeType.UNCERTAIN.value,
        "table_added",
        "table_removed",
        "structure_change",
        "modified",
        "uncertain",
    }:
        return "structure"
    if change_type in {
        ChangeType.INDICATOR_ADDED.value,
        ChangeType.INDICATOR_REMOVED.value,
        ChangeType.INDICATOR_RENAMED.value,
        "indicator_added",
        "indicator_removed",
        "indicator_renamed",
    }:
        return "indicators"
    if change_type in {
        ChangeType.FOOTNOTE_ADDED.value,
        ChangeType.FOOTNOTE_REMOVED.value,
        ChangeType.FOOTNOTE_MODIFIED.value,
        "footnote_added",
        "footnote_removed",
        "footnote_modified",
    }:
        return "footnotes"
    return "other"


def _get_change_status_rank(change: dict) -> int:
    status = str(change.get("validation_status", "pending") or "pending")
    if status == "pending":
        return 0
    if status == "skipped":
        return 1
    if status == "rejected":
        return 2
    if status == "approved":
        return 3
    return 4


def _ordered_changes(
    changes: list[dict], current_change_id: str | None = None
) -> list[dict]:
    current_id = str(current_change_id or "")
    return sorted(
        changes,
        key=lambda change: (
            0 if str(change.get("change_id") or "") == current_id else 1,
            _get_change_status_rank(change),
            0 if bool(change.get("is_required", True)) else 1,
        ),
    )


def _get_change_preview(change: dict, max_length: int = 96) -> str:
    change_type = str(change.get("change_type", "") or "")
    payload = _get_change_payload(change)

    if change_type in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
        return _truncate_text(
            payload.get("indicator_name") or payload.get("indicator_name_clean"),
            max_length,
        )

    if change_type in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
        return _truncate_text(
            payload.get("indicator_name") or payload.get("indicator_name_clean"),
            max_length,
        )

    if change_type in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
        from_val = _collapse_spaces(payload.get("from") or payload.get("from_clean"))
        to_val = _collapse_spaces(payload.get("to") or payload.get("to_clean"))
        return _truncate_text(f"{from_val} -> {to_val}", max_length)

    if change_type in (ChangeType.FOOTNOTE_ADDED.value, "footnote_added"):
        return _truncate_text(payload.get("new_text"), max_length)

    if change_type in (ChangeType.FOOTNOTE_REMOVED.value, "footnote_removed"):
        return _truncate_text(payload.get("old_text"), max_length)

    if change_type in (ChangeType.FOOTNOTE_MODIFIED.value, "footnote_modified"):
        new_text = _collapse_spaces(payload.get("new_text"))
        old_text = _collapse_spaces(payload.get("old_text"))
        return _truncate_text(new_text or old_text, max_length)

    return _truncate_text(payload.get("description"), max_length)


def _build_change_preview_item(change: dict, is_current: bool) -> html.Div:
    change_type = str(change.get("change_type", "") or "")
    family = _get_change_family(change)

    class_name = "review-queue-preview-item"
    if is_current:
        class_name += " is-current"

    return html.Div(
        [
            html.Span(
                _CHANGE_FAMILY_LABELS.get(family, "Autres"),
                className="review-queue-preview-item-family",
            ),
            html.Span(
                _CHANGE_TYPE_LABELS.get(change_type, "Changement"),
                className="review-queue-preview-item-type",
            ),
            html.Span(
                _get_change_preview(change),
                className="review-queue-preview-item-text",
            ),
        ],
        className=class_name,
    )


def _build_preview_band(
    changes: list[dict], current_change_id: str | None
) -> html.Div | None:
    if not changes:
        return None

    ordered = _ordered_changes(changes, current_change_id)
    previews = ordered[:3]
    remaining = max(0, len(changes) - len(previews))
    unresolved = sum(
        1
        for change in changes
        if str(change.get("validation_status", "pending") or "pending")
        not in {"approved", "rejected"}
    )
    band_label = "A revoir" if unresolved else "Apercu du dossier"

    return html.Div(
        [
            html.Div(
                [
                    html.Span(band_label, className="review-queue-preview-label"),
                    html.Span(
                        f"{unresolved} restant(s)" if unresolved else "Tous parcourus",
                        className="review-queue-preview-count",
                    ),
                ],
                className="review-queue-preview-head",
            ),
            html.Div(
                [
                    *[
                        _build_change_preview_item(
                            change,
                            str(change.get("change_id") or "")
                            == str(current_change_id or ""),
                        )
                        for change in previews
                    ],
                    html.Div(
                        f"+ {remaining} autre(s) changement(s)",
                        className="review-queue-preview-more",
                    )
                    if remaining
                    else None,
                ],
                className="review-queue-preview-items",
            ),
        ],
        className="review-queue-preview-band",
    )


def _build_change_panels(change: dict) -> html.Div:
    change_type = str(change.get("change_type", "") or "")
    payload = _get_change_payload(change)
    indicator_name = _normalize_text(
        payload.get("indicator_name") or payload.get("indicator_name_clean")
    )
    from_val = _normalize_text(payload.get("from") or payload.get("from_clean"))
    to_val = _normalize_text(payload.get("to") or payload.get("to_clean"))
    old_text = _normalize_text(payload.get("old_text"))
    new_text = _normalize_text(payload.get("new_text"))
    description = _normalize_text(payload.get("description"))

    if change_type in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
        panels = [
            _build_detail_panel(
                "Indicateur ajoute",
                indicator_name,
                "Aucun libelle complet disponible.",
            )
        ]
    elif change_type in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
        panels = [
            _build_detail_panel(
                "Indicateur supprime",
                indicator_name,
                "Aucun libelle complet disponible.",
            )
        ]
    elif change_type in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
        panels = [
            _build_detail_panel(
                "Version precedente",
                from_val,
                "Ancien libelle indisponible.",
            ),
            _build_detail_panel(
                "Version courante",
                to_val,
                "Nouveau libelle indisponible.",
            ),
        ]
    elif change_type in (ChangeType.FOOTNOTE_ADDED.value, "footnote_added"):
        panels = [
            _build_detail_panel(
                "Texte ajoute",
                new_text,
                "Texte de la note indisponible.",
            )
        ]
    elif change_type in (ChangeType.FOOTNOTE_REMOVED.value, "footnote_removed"):
        panels = [
            _build_detail_panel(
                "Texte supprime",
                old_text,
                "Texte de la note indisponible.",
            )
        ]
    elif change_type in (ChangeType.FOOTNOTE_MODIFIED.value, "footnote_modified"):
        panels = [
            _build_detail_panel(
                "Version precedente",
                old_text,
                "Ancienne version indisponible.",
            ),
            _build_detail_panel(
                "Version courante",
                new_text,
                "Nouvelle version indisponible.",
            ),
        ]
    else:
        panels = [
            _build_detail_panel(
                "Description complete",
                description,
                "Aucun detail textuel disponible.",
            )
        ]

    panel_class = "review-queue-change-panels"
    if len(panels) > 1:
        panel_class += " review-queue-change-panels--split"
    return html.Div(panels, className=panel_class)


def _get_change_description(change: dict) -> str:
    change_type = str(change.get("change_type", "") or "")
    payload = _get_change_payload(change)

    if change_type in (
        ChangeType.INDICATOR_ADDED.value,
        ChangeType.INDICATOR_REMOVED.value,
        "indicator_added",
        "indicator_removed",
    ):
        return _normalize_text(payload.get("indicator_name")) or "(indicateur)"

    if change_type in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
        from_val = _normalize_text(payload.get("from"))
        to_val = _normalize_text(payload.get("to"))
        if from_val or to_val:
            return f"{from_val} -> {to_val}"

    if "footnote" in change_type:
        footnote_ref = _normalize_text(payload.get("footnote_ref"))
        if footnote_ref:
            return f"Note [{footnote_ref}]"
        return "Note"

    return _normalize_text(payload.get("description")) or "Changement"


def _get_change_state_label(change: dict) -> str:
    status = str(change.get("validation_status", "pending") or "pending")
    if status == "approved":
        return "Valide"
    if status == "rejected":
        return "Rejete"
    if status == "skipped":
        return "Passe"
    return "A revoir"


def _build_change_card(change: dict, is_current: bool) -> html.Button:
    change_id = str(change.get("change_id", "") or "")
    change_type = str(change.get("change_type", "") or "")
    status = str(change.get("validation_status", "pending") or "pending")
    is_required = bool(change.get("is_required", True))
    footnote_ref = _normalize_text(_get_change_payload(change).get("footnote_ref"))

    if status == "approved":
        status_icon = html.I(className="bi bi-check-circle-fill text-success")
    elif status == "rejected":
        status_icon = html.I(className="bi bi-x-circle-fill text-danger")
    elif status == "skipped":
        status_icon = html.I(className="bi bi-dash-circle text-secondary")
    else:
        status_icon = html.I(className="bi bi-circle text-warning")

    class_name = "review-queue-change-card"
    if is_current:
        class_name += " is-current"

    meta_badges: list = [
        dbc.Badge(
            _CHANGE_TYPE_LABELS.get(change_type, change_type or "Changement"),
            color=_CHANGE_TYPE_COLORS.get(change_type, "secondary"),
            className="me-2",
        )
    ]
    if footnote_ref:
        meta_badges.append(
            dbc.Badge(
                f"Ref. {footnote_ref}",
                color="light",
                text_color="dark",
                className="me-2",
            )
        )
    if not is_required:
        meta_badges.append(dbc.Badge("Optionnel", color="light", text_color="dark"))

    notes = _normalize_text(change.get("validation_notes"))
    preview_text = _get_change_preview(change, max_length=140)
    state_class = f"review-queue-change-state review-queue-change-state--{status}"

    return html.Button(
        [
            html.Div(
                [
                    html.Div(status_icon, className="review-queue-change-status"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        meta_badges,
                                        className="review-queue-change-meta",
                                    ),
                                    html.Span(
                                        _get_change_state_label(change),
                                        className=state_class,
                                    ),
                                ],
                                className="review-queue-change-topline",
                            ),
                            html.Div(
                                _get_change_description(change),
                                className="review-queue-change-title",
                            ),
                            html.Div(
                                preview_text,
                                className="review-queue-change-preview",
                            )
                            if preview_text
                            else None,
                        ],
                        className="review-queue-change-copy flex-grow-1",
                    ),
                ],
                className="d-flex align-items-start gap-2",
            ),
            html.Div(
                [
                    html.Div(_build_change_panels(change), className="mt-3"),
                    html.Div(notes, className="review-queue-change-note mt-2")
                    if notes
                    else None,
                ],
                className="review-queue-change-details",
            )
            if is_current
            else None,
        ],
        id={"type": "change-row-v2", "change_id": change_id},
        className=class_name,
        n_clicks=0,
    )


def _build_change_group(
    family: str,
    changes: list[dict],
    current_change_id: str | None,
    is_active: bool,
) -> html.Div:
    if not changes:
        return html.Div()

    ordered = _ordered_changes(changes, current_change_id if is_active else None)
    current_id = str(current_change_id or "") if is_active else ""

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        _CHANGE_FAMILY_LABELS.get(family, "Autres"),
                        className="review-queue-group-title",
                    ),
                    html.Div(
                        f"{len(changes)} changement(s)",
                        className="review-queue-group-count",
                    ),
                ],
                className="review-queue-group-header",
            ),
            html.Div(
                [
                    _build_change_card(
                        change,
                        str(change.get("change_id") or "") == current_id,
                    )
                    for change in ordered
                ],
                className="review-queue-group-items",
            ),
        ],
        className=f"review-queue-group review-queue-group--{family}",
    )


def _build_genai_summary_row(table: dict) -> html.Div | None:
    """Build the GenAI action badge + category badge + narrative line for a queue card."""
    ga = table.get("genai_analysis")
    if not isinstance(ga, dict) or not ga:
        return None

    action = str(ga.get("action_requise", "") or "").strip().lower()
    category = str(ga.get("category", "") or "").strip().upper()
    narrative = str(ga.get("impact_description", "") or ga.get("justification", "") or "").strip()

    chips = []

    action_info = _ACTION_DISPLAY.get(action)
    if action_info:
        label, color = action_info
        chips.append(
            dbc.Badge(label, color=color, className="me-1", style={"fontSize": "0.65rem"})
        )

    category_info = _CATEGORY_DISPLAY.get(category)
    if category_info:
        label, color = category_info
        chips.append(
            dbc.Badge(label, color=color, className="me-1", style={"fontSize": "0.65rem"})
        )

    if not chips and not narrative:
        return None

    parts: list = []
    if chips:
        parts.append(html.Div(chips, className="d-flex flex-wrap gap-1 mb-1"))
    if narrative:
        # Truncate to keep card compact
        display = narrative if len(narrative) <= 90 else narrative[:87] + "…"
        parts.append(
            html.Div(
                display,
                className="review-queue-narrative",
                title=narrative,
            )
        )

    return html.Div(parts, className="review-queue-genai-summary mt-1")


def _build_table_metric_badges(summary: dict) -> list[html.Span]:
    badges: list[html.Span] = []
    n_added = int(summary.get("indicators_added", 0) or 0)
    n_removed = int(summary.get("indicators_removed", 0) or 0)
    n_renamed = int(summary.get("indicators_renamed", 0) or 0)
    n_footnotes = int(summary.get("footnotes_changed", 0) or 0)

    if n_added:
        badges.append(
            html.Span(f"+{n_added}", className="review-queue-stat-chip is-added")
        )
    if n_removed:
        badges.append(
            html.Span(f"-{n_removed}", className="review-queue-stat-chip is-removed")
        )
    if n_renamed:
        badges.append(
            html.Span(f"RN {n_renamed}", className="review-queue-stat-chip is-renamed")
        )
    if n_footnotes:
        badges.append(
            html.Span(
                f"FN {n_footnotes}",
                className="review-queue-stat-chip is-footnotes",
            )
        )

    return badges


def _build_progress_pill(
    validated: int, total: int, table_status: str, is_active: bool
) -> html.Span | None:
    if total <= 0:
        return None

    class_name = "review-queue-progress-pill"
    if table_status == "completed":
        class_name += " is-completed"
    elif table_status == "partial":
        class_name += " is-partial"
    else:
        class_name += " is-pending"
    if is_active:
        class_name += " is-active"

    return html.Span(f"{validated}/{total}", className=class_name)


def build_review_queue_v2(
    tables: list[dict],
    current_review_id: str | None,
    current_change_id: str | None = None,
    active_filters: dict | None = None,
) -> html.Div:
    """Build the left-side review queue panel V2 with grouped tables.

    Args:
        tables: List of ReviewTableItem dicts (from store-review-queue)
        current_review_id: Currently selected review_id
        active_filters: Optional filters (section, status)

    Returns:
        Div containing the queue panel
    """
    if not tables:
        return html.Div(
            [
                html.H5(t("file_review")),
                html.P("Aucun élément à réviser.", className="text-muted"),
            ]
        )

    # Extract all sections for filter
    all_sections = sorted(set(t.get("section", "Autre") for t in tables))

    active_section = (active_filters or {}).get("section", "all")
    active_status = (active_filters or {}).get("status", "all")

    def _matches_filters(table: dict) -> bool:
        if active_section and active_section != "all":
            if table.get("section") != active_section:
                return False
        if active_status and active_status != "all":
            if table.get("table_status") != active_status:
                return False
        return True

    # Filter tables
    filtered_with_idx = [
        (idx, table) for idx, table in enumerate(tables) if _matches_filters(table)
    ]
    filtered_tables = [t for _, t in filtered_with_idx]

    # Compute stats
    total = len(filtered_tables)
    completed = sum(1 for t in filtered_tables if t.get("table_status") == "completed")
    partial = sum(1 for t in filtered_tables if t.get("table_status") == "partial")
    pending = sum(1 for t in filtered_tables if t.get("table_status") == "pending")

    # Build filter buttons
    filter_buttons: list = [
        dbc.Button(
            [
                html.I(className="bi bi-funnel me-2"),
                f"{t('all_sections')} ({len(tables)})",
            ],
            id={"type": "filter-section-v2", "value": "all"},
            color="light",
            size="sm",
            className=(
                "review-queue-filter-button w-100 text-start mb-1"
                + (" is-active" if active_section == "all" else "")
            ),
        )
    ]
    for section in all_sections:
        section_count = sum(1 for tb in tables if tb.get("section") == section)
        section_label = _format_section(section)
        filter_buttons.append(
            dbc.Button(
                [
                    html.I(className="bi bi-folder me-2"),
                    f"{section_label} ({section_count})",
                ],
                id={"type": "filter-section-v2", "value": section},
                color="light",
                size="sm",
                className=(
                    "review-queue-filter-button w-100 text-start mb-1"
                    + (" is-active" if active_section == section else "")
                ),
            )
        )

    filter_bar = html.Div(
        filter_buttons, className="review-queue-filter-bar mb-3 p-2 rounded border"
    )

    queue_items = []
    for _, table in filtered_with_idx:
        summary = table.get("summary", {})
        n_total = summary.get("total_changes", 0)
        n_validated = summary.get("validated", 0)
        table_status = table.get("table_status", "pending")
        if table_status == "completed":
            icon = html.I(className="bi bi-check-circle-fill text-success me-2")
        elif table_status == "partial":
            icon = html.I(className="bi bi-pie-chart-fill text-info me-2")
        else:
            icon = html.I(className="bi bi-circle text-warning me-2")

        review_id = str(table.get("review_id") or table.get("table_key") or "")
        is_active = bool(review_id) and review_id == str(current_review_id or "")

        # Table display info
        section = _format_section(table.get("section", ""))
        table_name = (
            table.get("table_name")
            or table.get("table_title_raw")
            or table.get("table_id_t2")
            or table.get("table_id_t1")
            or "Tableau"
        )
        page_summary = _queue_page_summary_v2(table)
        context_text = f"{section} - {page_summary}" if page_summary else section
        stat_badges = _build_table_metric_badges(summary)
        progress_badge = _build_progress_pill(
            int(n_validated or 0),
            int(n_total or 0),
            str(table_status or "pending"),
            is_active,
        )
        genai_summary = _build_genai_summary_row(table)

        row_class = "review-queue-table-row"
        if is_active:
            row_class += " is-active border-primary"
        if table_status == "completed":
            row_class += " is-completed"
        elif table_status == "partial":
            row_class += " is-partial"
        else:
            row_class += " is-pending"

        queue_items.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(icon, className="review-queue-table-icon"),
                                    html.Div(
                                        table_name,
                                        className="review-queue-table-title",
                                        title=table_name,
                                    ),
                                ],
                                className="review-queue-table-head-main",
                            ),
                            progress_badge,
                        ],
                        className="review-queue-table-head",
                    ),
                    html.Div(
                        context_text,
                        className="review-queue-table-context",
                    ),
                    genai_summary,
                    html.Div(
                        [
                            html.Span(
                                f"{n_total} changement(s)",
                                className="review-queue-table-count",
                            ),
                            html.Div(
                                stat_badges,
                                className="review-queue-table-stats",
                            )
                            if stat_badges
                            else None,
                        ],
                        className="review-queue-table-foot",
                    ),
                ],
                id={"type": "queue-table-item-v2", "review_id": review_id},
                className=row_class,
                n_clicks=0,
            )
        )

    return html.Div(
        [
            html.H5(t("file_review"), className="mb-3"),
            html.Div(
                [
                    html.Span(f"Total: {total}", className="me-3 fw-bold"),
                    html.Span(
                        [
                            html.I(
                                className="bi bi-check-circle-fill text-success me-1"
                            ),
                            f"{completed}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-pie-chart-fill text-info me-1"),
                            f"{partial}",
                        ],
                        className="me-3 small",
                    ),
                    html.Span(
                        [
                            html.I(className="bi bi-circle text-warning me-1"),
                            f"{pending}",
                        ],
                        className="small",
                    ),
                ],
                className="mb-3 small text-muted",
            ),
            filter_bar,
            html.Div(
                queue_items,
                className="review-queue-v2-list overflow-auto",
                style={"maxHeight": "calc(100vh - 340px)"},
            ),
        ],
        className="h-100",
    )
