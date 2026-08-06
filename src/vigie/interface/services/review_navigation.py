"""Navigation, selection et filtrage de la revue pour les callbacks Dash."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from vigie.interface.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
)
from vigie.interface.review_models_v2 import ChangeType


def _build_kpi_card(title: str, value: str | int, delta_icon: str | None = None, color: str = "light") -> dbc.Card:
    """Construit une carte KPI individuelle pour le panneau analyste.

    Args:
        title: Libelle affiche au-dessus de la valeur.
        value: Valeur principale du KPI (texte ou entier).
        delta_icon: Icone optionnelle de variation.
        color: Classe de couleur Bootstrap pour le fond de la carte.

    Returns:
        Composant ``dbc.Card`` representant la carte KPI.
    """
    body_children = [
        html.P(title, className="text-muted mb-1 small"),
        html.H3(str(value), className="mb-0 fw-bold"),
    ]
    if delta_icon:
        body_children.append(html.Span(delta_icon, className="text-muted small"))
    return dbc.Card(
        dbc.CardBody(body_children, className="text-center py-3"),
        className=f"shadow-sm border-0 bg-{color}",
    )


def _format_duration(seconds: int | None) -> str:
    """Formate une duree en secondes sous la forme ``MM:SS``."""
    if seconds is None:
        return "--:--"
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"


def _review_id(table: dict) -> str:
    """Extrait l'identifiant de revue d'un tableau."""
    return str(table.get("review_id") or table.get("table_key") or "")


def _table_matches_filters(table: dict, filters: dict | None) -> bool:
    """Verifie si un tableau correspond aux filtres de section et de statut."""
    active_section = (filters or {}).get("section", "all")
    active_status = (filters or {}).get("status", "all")
    if active_section and active_section != "all":
        if table.get("section") != active_section:
            return False
    if active_status and active_status != "all":
        if table.get("table_status") != active_status:
            return False
    return True


def _visible_review_ids(queue: list[dict], filters: dict | None) -> list[str]:
    """Retourne les identifiants de revue visibles apres application des filtres."""
    return [_review_id(table) for table in queue if _review_id(table) and _table_matches_filters(table, filters)]


def _get_table_by_review_id(queue: list[dict], review_id: str | None) -> tuple[int, dict | None]:
    """Recherche un tableau par son identifiant de revue dans la file."""
    if not queue:
        return -1, None
    rid = str(review_id or "")
    if rid:
        for idx, table in enumerate(queue):
            if _review_id(table) == rid:
                return idx, table
    return -1, None


def _normalize_last_positions(data: dict | None) -> dict[str, str]:
    """Normalise le dictionnaire des dernieres positions de selection."""
    if not isinstance(data, dict):
        return {}

    normalized: dict[str, str] = {}
    for review_id, change_id in data.items():
        rid = str(review_id or "").strip()
        cid = str(change_id or "").strip()
        if rid and cid:
            normalized[rid] = cid
    return normalized


def _remember_selection(
    last_positions: dict | None,
    selection: dict | None,
) -> dict[str, str]:
    """Met a jour les dernieres positions avec la selection courante."""
    remembered = dict(_normalize_last_positions(last_positions))
    if not isinstance(selection, dict):
        return remembered

    review_id = str(selection.get("review_id") or "").strip()
    change_id = str(selection.get("change_id") or "").strip()
    if review_id and change_id:
        remembered[review_id] = change_id
    return remembered


def _resolve_change_id(
    table: dict,
    preferred_change_id: str | None = None,
    last_positions: dict | None = None,
) -> str | None:
    """Resout l'identifiant du changement actif pour un tableau donne."""
    changes = table.get("changes", []) or []
    preferred = str(preferred_change_id or "")
    if preferred and any(str(c.get("change_id", "")) == preferred for c in changes):
        return preferred

    remembered_change_id = _normalize_last_positions(last_positions).get(_review_id(table), "")
    if remembered_change_id and any(str(c.get("change_id", "")) == remembered_change_id for c in changes):
        return remembered_change_id

    for change in changes:
        status = str(change.get("validation_status", "pending"))
        if change.get("is_required", True) and status == "pending":
            return str(change.get("change_id", "")) or None
    for change in changes:
        status = str(change.get("validation_status", "pending"))
        if change.get("is_required", True) and status == "skipped":
            return str(change.get("change_id", "")) or None
    if changes:
        return str(changes[0].get("change_id", "")) or None
    return None


def _resolve_selection(
    queue: list[dict],
    selection: dict | None,
    filters: dict | None = None,
    last_positions: dict | None = None,
) -> tuple[dict, int, int]:
    """Resout la selection courante dans la file de revue.

    Prend en compte les filtres actifs et les dernieres positions
    memorisees pour determiner le tableau et le changement selectionnes.

    Args:
        queue: File de revue des tableaux.
        selection: Selection courante (``review_id``, ``change_id``).
        filters: Filtres actifs de section et de statut.
        last_positions: Dernieres positions memorisees par tableau.

    Returns:
        Tuple ``(selection_resolue, index_tableau, index_changement)``.
    """
    if not queue:
        return {"review_id": None, "change_id": None}, 0, 0

    visible_ids = _visible_review_ids(queue, filters) or [_review_id(t) for t in queue if _review_id(t)]
    preferred_review_id = str((selection or {}).get("review_id") or "")
    if preferred_review_id not in visible_ids:
        preferred_review_id = visible_ids[0] if visible_ids else _review_id(queue[0])

    table_idx, table = _get_table_by_review_id(queue, preferred_review_id)
    if table is None:
        table_idx = 0
        table = queue[0]
        preferred_review_id = _review_id(table)

    resolved_change_id = _resolve_change_id(
        table,
        preferred_change_id=(selection or {}).get("change_id"),
        last_positions=last_positions,
    )
    changes = table.get("changes", []) or []
    change_idx = 0
    if resolved_change_id:
        for idx, change in enumerate(changes):
            if str(change.get("change_id", "")) == resolved_change_id:
                change_idx = idx
                break
    return (
        {"review_id": preferred_review_id, "change_id": resolved_change_id},
        table_idx,
        change_idx,
    )


def _display_label_from_v2(change_type: str) -> str:
    """Convertit un type de changement V2 en libelle d'affichage legacy."""
    ctype = str(change_type or "")
    if ctype in (ChangeType.TABLE_ADDED.value, "table_added"):
        return CHANGE_TYPE_TABLE_ADDED
    if ctype in (ChangeType.TABLE_REMOVED.value, "table_removed"):
        return CHANGE_TYPE_TABLE_REMOVED
    if ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added"):
        return CHANGE_TYPE_ADDED
    if ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed"):
        return CHANGE_TYPE_REMOVED
    if ctype in (ChangeType.INDICATOR_RENAMED.value, "indicator_renamed"):
        return CHANGE_TYPE_RENAMED
    if "footnote" in ctype:
        return CHANGE_TYPE_MODIFIED
    return CHANGE_TYPE_MODIFIED


def _selected_change_from_table(table: dict, selection: dict | None) -> dict | None:
    """Retourne le changement selectionne dans un tableau ou le premier par defaut."""
    changes = table.get("changes", []) or []
    selected_change_id = str((selection or {}).get("change_id") or "")
    for change in changes:
        if str(change.get("change_id", "")) == selected_change_id:
            return change
    return changes[0] if changes else None


def _effective_proof_display_mode(
    table: dict,
    selection: dict | None,
    requested_mode: str | None,
) -> str:
    """Determine le mode d'affichage effectif de la preuve selon le contexte."""
    mode = (requested_mode or "crop").strip().lower()
    if mode not in {"crop", "full", "footnote"}:
        mode = "crop"

    selected_change = _selected_change_from_table(table, selection)
    change_type = str((selected_change or {}).get("change_type") or "").strip().lower()

    if "footnote" in change_type and mode == "crop":
        return "footnote"
    if "footnote" not in change_type and mode == "footnote":
        return "crop"
    if change_type == ChangeType.TABLE_ADDED.value and mode == "crop":
        if table.get("page_t2") is not None and table.get("bbox_t2") is None:
            return "full"
    if change_type == ChangeType.TABLE_REMOVED.value and mode == "crop":
        if table.get("page_t1") is not None and table.get("bbox_t1") is None:
            return "full"
    return mode


def _table_to_proof_item(table: dict, selection: dict | None) -> dict:
    """Convertit un tableau de la file de revue en dictionnaire compatible avec le rendu de preuve."""
    changes = table.get("changes", []) or []
    selected_change = _selected_change_from_table(table, selection)
    selected_change_type = selected_change.get("change_type", "") if isinstance(selected_change, dict) else ""
    primary_type = _display_label_from_v2(selected_change_type)
    added_indicators: list[str] = []
    removed_indicators: list[str] = []
    for change in changes:
        ctype = str(change.get("change_type", ""))
        payload = change.get("payload", {}) or {}
        indicator_name = str(payload.get("indicator_name", "")).strip()
        if ctype in (ChangeType.INDICATOR_ADDED.value, "indicator_added") and indicator_name:
            added_indicators.append(indicator_name)
        if ctype in (ChangeType.INDICATOR_REMOVED.value, "indicator_removed") and indicator_name:
            removed_indicators.append(indicator_name)

    return {
        "change_type": primary_type,
        "selected_change_type": selected_change_type,
        "selected_change_payload": (selected_change.get("payload", {}) if isinstance(selected_change, dict) else {}),
        "table_name": table.get("table_name", ""),
        "table_title_raw": table.get("table_title", "") or table.get("table_name", ""),
        "table_number": table.get("table_number", ""),
        "section": table.get("section", ""),
        "page_t1": table.get("page_t1"),
        "page_t2": table.get("page_t2"),
        "bbox_t1": table.get("bbox_t1"),
        "bbox_t2": table.get("bbox_t2"),
        "table_status": table.get("table_status", ""),
        "source_ref_t1": table.get("source_pdf_t1", ""),
        "source_ref_t2": table.get("source_pdf_t2", ""),
        "confidence": float(table.get("confidence", 0.0) or 0.0),
        "genai_analysis": table.get("genai_analysis") or {},
        "added_indicators": added_indicators,
        "removed_indicators": removed_indicators,
    }


def _table_decision_bucket(table: dict) -> str:
    """Determine le statut global de decision d'un tableau (approved, rejected ou pending)."""
    changes = table.get("changes", []) or []
    if not changes:
        return "pending"
    required = [c for c in changes if c.get("is_required", True)]
    if not required:
        required = changes
    statuses = [str(c.get("validation_status", "pending")) for c in required]
    if all(s == "approved" for s in statuses):
        return "approved"
    if all(s == "rejected" for s in statuses):
        return "rejected"
    return "pending"
