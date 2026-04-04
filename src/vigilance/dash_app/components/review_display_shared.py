"""Fonctions d'affichage partagees pour l'interface de revue Dash V2."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from vigilance.i18n import t
from vigilance.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
)
from vigilance.utils.indicator_cleaner import strip_dates_from_table_title

_CHANGE_TYPES_WITH_VISUAL_FLAG = frozenset(
    {
        CHANGE_TYPE_TABLE_ADDED,
        CHANGE_TYPE_TABLE_REMOVED,
        CHANGE_TYPE_ADDED,
        CHANGE_TYPE_REMOVED,
        CHANGE_TYPE_RENAMED,
        CHANGE_TYPE_MODIFIED,
        "uncertain",
        "structure_change",
    }
)

_SECTION_LABELS = {
    "gestion_capital": "Gestion du capital",
    "capital_management": "Gestion du capital",
    "capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "risk_management": "Gestion des risques",
    "risk": "Gestion des risques",
    "gestion_reglementation": "Reglementation",
    "regulatory_updates": "Reglementation",
    "reglementation": "Reglementation",
    "regulatory": "Reglementation",
    "unknown_section": "Section non determinee",
    "credit_risk": "Risque de credit",
    "market_risk": "Risque de marche",
    "liquidity_risk": "Risque de liquidite",
    "operational_risk": "Risque operationnel",
}


def section_display_label(section: str | None) -> str:
    """Convertit un identifiant de section en libelle lisible pour l'affichage.

    Args:
        section: Identifiant brut de la section (peut etre ``None``).

    Returns:
        Libelle francise de la section, ou ``"Autre section"`` par defaut.
    """
    value = (section or "").strip()
    if not value:
        return "Autre section"
    lowered = value.lower()
    if lowered in _SECTION_LABELS:
        return _SECTION_LABELS[lowered]
    if "capital" in lowered or "fonds propres" in lowered:
        return "Gestion du capital"
    if "credit" in lowered and "risk" in lowered:
        return "Risque de credit"
    if "market" in lowered and "risk" in lowered:
        return "Risque de marche"
    if "liquidity" in lowered and "risk" in lowered:
        return "Risque de liquidite"
    if "operational" in lowered and "risk" in lowered:
        return "Risque operationnel"
    if "risque" in lowered or "risk" in lowered:
        return "Gestion des risques"
    if "reglement" in lowered or "regulatory" in lowered:
        return "Reglementation"
    if "unknown" in lowered:
        return "Section non determinee"
    return "Autre section"


def _clean_title_for_display(raw_title: str) -> str:
    """Nettoie un titre brut de tableau pour l'affichage."""
    title = (raw_title or "").strip()
    if not title:
        return ""
    cleaned = strip_dates_from_table_title(title).strip(" -:;,")
    lowered = cleaned.lower()
    if lowered in {"au", "aux", "as at"}:
        return ""
    return cleaned or ""


def table_display_label(item: dict) -> str:
    """Formate un libelle de tableau de type rapport pour l'affichage.

    Args:
        item: Dictionnaire contenant les metadonnees du tableau
            (``table_name``, ``table_number``, ``page_t1``/``page_t2``,
            ``section``, ``change_type``).

    Returns:
        Libelle formate du tableau.
    """
    raw_title = (item.get("table_name") or item.get("table_title_raw") or "").strip()
    title = _clean_title_for_display(raw_title)
    has_title = bool(title)
    table_num = (item.get("table_number") or "").strip()
    change_type = item.get("change_type", "")
    if change_type == CHANGE_TYPE_TABLE_ADDED:
        page = item.get("page_t2")
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        page = item.get("page_t1")
    else:
        page = item.get("page_t2") or item.get("page_t1")
    section = section_display_label(item.get("section"))
    prefix = t("table", "Tableau")

    if table_num and has_title:
        return f"{prefix} {table_num}: {title}"
    if table_num:
        page_part = f" (p.{page})" if page is not None else ""
        section_part = f" - {section}" if section else ""
        return f"{prefix} {table_num}{page_part}{section_part}" or f"{prefix} {table_num}"
    if has_title:
        return f"{prefix}: {title}"
    if page is not None and section:
        return f"{prefix} (p.{page}) - {section}"
    if page is not None:
        return f"{prefix} (p.{page})"
    if section:
        return f"{prefix} - {section}"
    return f"{prefix} (sans titre)"


def compute_flag_state(item: dict) -> dict:
    """Calcule les indicateurs visuels des cartes de preuve a partir du payload de changement.

    Args:
        item: Dictionnaire du changement contenant ``change_type``,
            ``added_indicators``, ``removed_indicators`` et ``indicators``.

    Returns:
        Dictionnaire avec les classes CSS et libelles de badges pour T1/T2.
    """
    change_type = (item.get("change_type") or "").strip()
    added = item.get("added_indicators") or []
    removed = item.get("removed_indicators") or []
    indicators = item.get("indicators") or []
    added_count = len(added) if isinstance(added, list) else 0
    removed_count = len(removed) if isinstance(removed, list) else 0
    renamed_count = sum(
        1
        for ind in indicators
        if isinstance(ind, dict) and ind.get("type") == CHANGE_TYPE_RENAMED
    )

    has_change = (
        change_type in _CHANGE_TYPES_WITH_VISUAL_FLAG
        or added_count + removed_count + renamed_count > 0
    )
    if not has_change:
        return {
            "has_change": False,
            "t1_class": "proof-card",
            "t2_class": "proof-card",
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "neutral",
            "badge_class_t2": "neutral",
        }
    if change_type == CHANGE_TYPE_TABLE_ADDED:
        return {
            "has_change": True,
            "t1_class": "proof-card",
            "t2_class": "proof-card proof-flag-t2",
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "neutral",
            "badge_class_t2": "t2",
        }
    if change_type == CHANGE_TYPE_TABLE_REMOVED:
        return {
            "has_change": True,
            "t1_class": "proof-card proof-flag-t1",
            "t2_class": "proof-card",
            "badge_t1": "Trimestre précédent",
            "badge_t2": "Trimestre courant",
            "badge_class_t1": "t1",
            "badge_class_t2": "neutral",
        }
    return {
        "has_change": True,
        "t1_class": "proof-card proof-flag-t1",
        "t2_class": "proof-card proof-flag-t2",
        "badge_t1": "Trimestre précédent",
        "badge_t2": "Trimestre courant",
        "badge_class_t1": "t1",
        "badge_class_t2": "t2",
    }


def _bbox_normalized_for_overlay(bbox: list | None) -> list[float] | None:
    """Retourne ``[left, top, right, bottom]`` normalise en 0..1 si valide pour les overlays CSS."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
    except (TypeError, ValueError):
        return None
    if not (
        0 <= left <= 1 and 0 <= top <= 1 and 0 <= right <= 1 and 0 <= bottom <= 1
    ):
        return None
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def build_proofs_section(
    item: dict,
    img_t1_b64: str | None,
    img_t2_b64: str | None,
    proof_display_mode: str = "crop",
    proof_result_t1: dict | None = None,
    proof_result_t2: dict | None = None,
) -> html.Div:
    """Construit le panneau de preuves visuelles partage par la vue de detail V2.

    Args:
        item: Dictionnaire du changement contenant les metadonnees du tableau.
        img_t1_b64: Image base64 de la preuve T1 (trimestre precedent).
        img_t2_b64: Image base64 de la preuve T2 (trimestre courant).
        proof_display_mode: Mode d'affichage (``"crop"``, ``"full"``,
            ``"footnote"`` ou ``"full_without_bbox"``).
        proof_result_t1: Resultat de rendu pour la preuve T1.
        proof_result_t2: Resultat de rendu pour la preuve T2.

    Returns:
        Un ``Div`` contenant les cartes de preuve T1/T2 et le selecteur de mode.
    """
    flag_state = compute_flag_state(item)
    normalized_mode = (proof_display_mode or "crop").strip().lower()
    mode_label_map = {
        "crop": "Mode focus tableau",
        "full": "Mode page complète + bbox",
        "footnote": "Mode note de bas de tableau",
        "full_without_bbox": "Mode page complète sans bbox",
    }

    def _mode_label(mode_value: str | None) -> str:
        """Retourne le libelle francais du mode d'affichage."""
        return mode_label_map.get(
            (mode_value or normalized_mode).strip().lower(),
            "Mode focus tableau",
        )

    def _proof_caption(base_label: str, page: int | None, mode_value: str | None) -> str:
        """Genere la legende d'une carte de preuve."""
        page_label = f"Page {page}" if page is not None else "Page indisponible"
        return f"{base_label} · {page_label} · {_mode_label(mode_value)}"

    def _proof_placeholder(
        placeholder: str | None, render_result: dict | None, mode_value: str
    ) -> str:
        """Retourne le message de substitution quand l'image n'est pas disponible."""
        if placeholder:
            return placeholder
        status = str((render_result or {}).get("status") or "").strip().lower()
        if status == "bbox_missing":
            if mode_value == "footnote":
                return "Zone footnote indisponible: bbox absente pour ce tableau."
            return "Crop indisponible: bbox absente pour ce tableau."
        if status == "page_missing":
            return "Page indisponible pour cette preuve."
        if status == "render_failed":
            return "Rendu impossible pour cette preuve."
        return t("image_unavailable", "Image non disponible")

    def _img_card(
        b64,
        label,
        *,
        mode_value: str,
        placeholder=None,
        bbox_norm=None,
        render_result=None,
        side: str = "t1",
    ):
        """Construit une carte image de preuve avec overlay bbox optionnel."""
        if not b64:
            msg = _proof_placeholder(placeholder, render_result, mode_value)
            return dbc.Card(
                dbc.CardBody(html.P(msg, className="text-muted text-center small")),
                className="h-100 bg-light border-0",
            )
        if mode_value == "full" and bbox_norm:
            left, top, right, bottom = bbox_norm
            overlay_style = {
                "position": "absolute",
                "left": f"{left * 100}%",
                "top": f"{top * 100}%",
                "width": f"{(right - left) * 100}%",
                "height": f"{(bottom - top) * 100}%",
                "pointerEvents": "none",
            }
            wrapper = html.Div(
                [
                    html.Img(
                        src=f"data:image/png;base64,{b64}",
                        style={
                            "objectFit": "contain",
                            "maxHeight": "calc(50vh - 60px)",
                            "width": "100%",
                            "height": "auto",
                        },
                    ),
                    html.Div(className="bbox-rect", style=overlay_style),
                ],
                style={
                    "position": "relative",
                    "display": "inline-block",
                    "width": "100%",
                },
                className=f"proof-img-with-bbox proof-bbox-{side}",
            )
            return dbc.Card(
                [
                    dbc.CardBody(wrapper),
                    dbc.CardFooter(
                        html.Small(label, className="text-muted"),
                        className="border-0 bg-white p-1",
                    ),
                ],
                className="h-100 border shadow-sm overflow-hidden",
            )
        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/png;base64,{b64}",
                    top=True,
                    style={"objectFit": "contain", "maxHeight": "calc(50vh - 60px)"},
                ),
                dbc.CardFooter(
                    html.Small(label, className="text-muted"),
                    className="border-0 bg-white p-1",
                ),
            ],
            className="h-100 border shadow-sm overflow-hidden",
        )

    def _proof_wrapper(card, card_class: str, badge_text: str, badge_class: str):
        """Enveloppe une carte de preuve avec un badge de trimestre."""
        badge = html.Span(badge_text, className=f"proof-badge {badge_class}")
        return html.Div([badge, card], className=card_class)

    change_type = item.get("change_type", "")
    mode_t1 = str((proof_result_t1 or {}).get("mode_effective") or normalized_mode)
    mode_t2 = str((proof_result_t2 or {}).get("mode_effective") or normalized_mode)
    bbox_t1_norm = (
        _bbox_normalized_for_overlay(item.get("bbox_t1")) if mode_t1 == "full" else None
    )
    bbox_t2_norm = (
        _bbox_normalized_for_overlay(item.get("bbox_t2")) if mode_t2 == "full" else None
    )

    if change_type == CHANGE_TYPE_TABLE_ADDED:
        card_t1 = _img_card(
            None,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            placeholder=t(
                "no_table_added_t2",
                "Aucun tableau dans le trimestre précédent",
            ),
            render_result=proof_result_t1,
        )
        card_t2 = _img_card(
            img_t2_b64,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            bbox_norm=bbox_t2_norm,
            render_result=proof_result_t2,
            side="t2",
        )
    elif change_type == CHANGE_TYPE_TABLE_REMOVED:
        card_t1 = _img_card(
            img_t1_b64,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            bbox_norm=bbox_t1_norm,
            render_result=proof_result_t1,
            side="t1",
        )
        card_t2 = _img_card(
            None,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            placeholder=t(
                "no_table_removed_t2",
                "Aucun tableau au trimestre courant (supprimé depuis le trimestre précédent)",
            ),
            render_result=proof_result_t2,
        )
    else:
        card_t1 = _img_card(
            img_t1_b64,
            _proof_caption("Trimestre précédent", item.get("page_t1"), mode_t1),
            mode_value=mode_t1,
            bbox_norm=bbox_t1_norm,
            render_result=proof_result_t1,
            side="t1",
        )
        card_t2 = _img_card(
            img_t2_b64,
            _proof_caption("Trimestre courant", item.get("page_t2"), mode_t2),
            mode_value=mode_t2,
            bbox_norm=bbox_t2_norm,
            render_result=proof_result_t2,
            side="t2",
        )

    proof_mode_toggle = dbc.Row(
        dbc.Col(
            [
                html.Label("Affichage preuve", className="small text-muted me-2"),
                dbc.RadioItems(
                    id="proof-display-mode",
                    options=[
                        {"label": "Crop (focus)", "value": "crop"},
                        {"label": "Page complète + bbox", "value": "full"},
                        {"label": "Note de bas de table", "value": "footnote"},
                    ],
                    value=normalized_mode,
                    inline=True,
                    className="small",
                ),
            ],
            width=12,
        ),
        className="mb-2",
    )

    proofs_row = dbc.Row(
        [
            dbc.Col(
                _proof_wrapper(
                    card_t2,
                    flag_state["t2_class"],
                    flag_state["badge_t2"],
                    flag_state["badge_class_t2"],
                ),
                width=6,
            ),
            dbc.Col(
                _proof_wrapper(
                    card_t1,
                    flag_state["t1_class"],
                    flag_state["badge_t1"],
                    flag_state["badge_class_t1"],
                ),
                width=6,
            ),
        ],
        className="mb-4 g-2",
        style={"height": "50vh", "minHeight": "400px"},
    )

    return html.Div(
        [
            html.H6("Preuves visuelles T1/T2", className="mb-1"),
            proof_mode_toggle,
            proofs_row,
        ]
    )


__all__ = [
    "build_proofs_section",
    "compute_flag_state",
    "section_display_label",
    "table_display_label",
]
