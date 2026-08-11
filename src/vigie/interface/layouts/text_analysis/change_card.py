"""Rendu d'une carte de changement : bloc observe, details IA et vue cote a cote.

Extrait de ``page_text_analysis.py`` sans modification.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigie.analyse_texte.text_comparison.justification import build_text_triage_justification
from vigie.comparaison.analyst_change_presentation import (
    build_analyst_narrative,
    build_change_presentation,
    canonicalize_analyst_narrative,
)
from vigie.comparaison.triage.amf_taxonomy import (
    IMPACT_IT_DETAIL_LABELS,
    POSTURE_DETAIL_LABELS,
    _compact_complete_sentence_parts,
    extract_labeled_analysis,
)
from vigie.interface.components.text_change_presentation import build_source_evidence_details
from vigie.support.i18n.fr import sanitize_analyst_french

from .highlight import (
    _HIGHLIGHT_ADDED_STYLE,
    _HIGHLIGHT_REMOVED_STYLE,
    _change_segments_are_usable,
    _diff_highlight_intervals,
    _find_highlight_intervals,
    _highlight_text_by_intervals,
)
from .labels import (
    _ACTION_BADGE,
    _DIFF_LABELS,
    _IMPACT_BADGE,
    _IMPACT_DOMAIN_BY_THEME,
    _IMPACT_DOMAIN_PRIORITY,
    _IMPLEMENTATION_DETAIL_LABEL,
    _POSTURE_BADGE,
    _POSTURE_CONFIDENCE_DETAIL_LABEL,
    _TEXT_REVIEW_STATUS_BADGES,
    _THEMES_AMF_SHORT,
    _TRIAGE_DETAIL_LABELS,
    _badge,
)


def _ai_detail_item(label: str, value: str) -> html.Div:
    """Affiche une rubrique courte dans le volet de détails IA."""
    if not value:
        return html.Div()
    return html.Div(
        [
            html.Div(label, className="small fw-semibold text-muted mb-1"),
            html.P(value, className="small mb-2"),
        ]
    )


def _impact_domain(themes_amf: list[str], section_title: str) -> str:
    """Retourne un domaine métier concis à partir des thèmes détectés."""
    theme_set = set(themes_amf)
    for theme in _IMPACT_DOMAIN_PRIORITY:
        if theme in theme_set:
            return _IMPACT_DOMAIN_BY_THEME[theme]
    return section_title.lower()


def _first_complete_sentence(text: str) -> str:
    """Retourne la première phrase complète ponctuée, ou le texte nettoyé."""
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return ""
    parts = _compact_complete_sentence_parts(normalized)
    if parts:
        return parts[0]
    return normalized


def _build_observed_block(
    *,
    impact_level: str,
    impact_domain: str,
    justification_sections: dict[str, str],
    change_summary: str,
    relevance_reason: str = "",
    observed_text: str = "",
    business_relevance: str = "",
) -> html.Div:
    """Affiche le résumé métier canonique et l'impact contextualisé."""
    impact_label = _IMPACT_BADGE.get(
        impact_level,
        (impact_level.capitalize(), "secondary"),
    )[0]
    observed = (
        observed_text
        or _first_complete_sentence(relevance_reason)
        or justification_sections.get("Ce qui change")
        or change_summary
        or "Le changement est visible dans les passages comparés ci-dessus."
    )
    observed = sanitize_analyst_french(observed)
    return html.Div(
        [
            html.Div(
                "Changement constaté",
                className="small fw-semibold text-primary mb-1",
            ),
            html.P(observed, className="small mb-2"),
            (
                html.Div(
                    [
                        html.Div(
                            "Pertinence métier",
                            className="small fw-semibold text-primary mb-1",
                        ),
                        html.P(
                            business_relevance,
                            className="small mb-2",
                        ),
                    ]
                )
                if business_relevance
                else None
            ),
            html.Div(
                f"Impact {impact_domain} — {impact_label}",
                className="small fw-semibold text-muted",
            ),
        ],
        className="border-start border-primary border-3 ps-2 mb-3",
    )


def _build_ai_details(
    *,
    impact_it_justification: str,
    impact_level: str,
    impact_domain: str,
    justification_sections: dict[str, str],
    changement_posture: str,
    justification_posture: str,
    statut_mise_en_oeuvre: str,
    confiance_posture: str,
) -> tuple[html.Div | None, html.Details | None]:
    """Construit la preuve de posture visible et les explications repliées."""
    impact_sections = extract_labeled_analysis(
        impact_it_justification,
        IMPACT_IT_DETAIL_LABELS,
    )
    posture_sections = extract_labeled_analysis(
        justification_posture,
        POSTURE_DETAIL_LABELS,
    )

    posture_proof = posture_sections.get("Preuve", "")
    if not posture_proof and justification_posture:
        posture_proof = justification_posture

    proof_block: html.Div | None = None
    if posture_proof:
        proof_block = html.Div(
            [
                html.Div(
                    "Preuve de posture",
                    className="small fw-semibold text-primary mb-1",
                ),
                html.P(posture_proof, className="small mb-0"),
            ],
            className="border-start border-primary border-3 ps-2 mt-3",
        )

    detail_sections: list = []
    surveillance = sanitize_analyst_french(justification_sections.get("Point de surveillance", ""))
    subject = sanitize_analyst_french(justification_sections.get("Sujet détecté", ""))
    if surveillance or subject or impact_sections:
        impact_label = _IMPACT_BADGE.get(
            impact_level,
            (impact_level.capitalize(), "secondary"),
        )[0]
        detail_sections.append(
            html.Div(
                [
                    html.H6(
                        f"Impact {impact_domain} — {impact_label}",
                        className="fw-semibold mb-2",
                    ),
                    _ai_detail_item("Domaine détecté", subject or impact_domain),
                    _ai_detail_item("Point de surveillance", surveillance),
                    _ai_detail_item(
                        "Conséquence probable",
                        sanitize_analyst_french(impact_sections.get("Conséquence probable", "")),
                    ),
                    _ai_detail_item(
                        "Limite de l’analyse",
                        sanitize_analyst_french(impact_sections.get("Limite de l'analyse", "")),
                    ),
                ],
                className="mb-3",
            )
        )

    if changement_posture in _POSTURE_BADGE and justification_posture:
        posture_label = _POSTURE_BADGE[changement_posture][0]
        detail_sections.append(
            html.Div(
                [
                    html.H6(posture_label, className="fw-semibold mb-2"),
                    _ai_detail_item(
                        "Effet sur la gestion du risque",
                        posture_sections.get(
                            "Effet sur la gestion du risque",
                            "",
                        ),
                    ),
                    _ai_detail_item(
                        "Mise en œuvre",
                        (
                            f"{_IMPLEMENTATION_DETAIL_LABEL.get(statut_mise_en_oeuvre, statut_mise_en_oeuvre.capitalize())} — "
                            f"{posture_sections.get('Justification du statut', '')}"
                        ).rstrip(" —"),
                    ),
                    _ai_detail_item(
                        "Confiance",
                        (
                            f"{_POSTURE_CONFIDENCE_DETAIL_LABEL.get(confiance_posture, confiance_posture.capitalize())} — "
                            f"{posture_sections.get('Justification de la confiance', '')}"
                        ).rstrip(" —"),
                    ),
                ]
            )
        )

    if not detail_sections:
        return proof_block, None

    details = html.Details(
        [
            html.Summary(
                "Voir les détails de l’évaluation IA",
                className="fw-semibold small text-primary",
                style={"cursor": "pointer"},
            ),
            html.Div(
                detail_sections,
                className="pt-3 px-2",
            ),
        ],
        open=False,
        className="mt-3 border rounded bg-light p-2",
    )
    return proof_block, details


def _build_side_by_side(
    *,
    text_t1: str,
    text_t2: str,
    page_t1: str,
    page_t2: str,
    change_segments: list[dict],
    diff_type: str,
    current_quarter_label: str = "Trimestre courant",
    previous_quarter_label: str = "Trimestre précédent",
) -> html.Div:
    """Affiche T2/T1 côte à côte avec highlights des segments AMF v2.

    - ``added``  : segment surligné en VERT dans la colonne T2.
    - ``removed``: segment surligné en AMBRE dans la colonne T1.
    - ``modified`` ou ``renamed``: les deux côtés sont affichés côte à côte.

    Le rapport courant est toujours affiché en premier, puis le rapport
    précédent. Pour un ajout ou une suppression, le côté absent présente
    explicitement la nature du changement.
    """
    usable_change_segments = change_segments if _change_segments_are_usable(change_segments) else []
    highlights_t1 = [
        seg.get("text_t1", "")
        for seg in usable_change_segments
        if seg.get("kind") in ("removed", "modified") and seg.get("text_t1")
    ]
    highlights_t2 = [
        seg.get("text_t2", "")
        for seg in usable_change_segments
        if seg.get("kind") in ("added", "modified") and seg.get("text_t2")
    ]
    intervals_t1 = _find_highlight_intervals(text_t1, highlights_t1)
    intervals_t2 = _find_highlight_intervals(text_t2, highlights_t2)
    diff_intervals_t1, diff_intervals_t2 = _diff_highlight_intervals(text_t1, text_t2)

    if diff_type == "added" and not intervals_t2:
        intervals_t2 = diff_intervals_t2 or ([(0, len(text_t2))] if text_t2 else [])
    elif diff_type == "removed" and not intervals_t1:
        intervals_t1 = diff_intervals_t1 or ([(0, len(text_t1))] if text_t1 else [])
    elif diff_type in {"modified", "renamed"}:
        if not intervals_t1:
            intervals_t1 = diff_intervals_t1
        if not intervals_t2:
            intervals_t2 = diff_intervals_t2

    base_card_style = {
        "whiteSpace": "pre-wrap",
        "overflowWrap": "anywhere",
        "wordBreak": "break-word",
        "lineHeight": "1.55",
    }

    def _column(
        label: str,
        text: str,
        intervals: list[tuple[int, int]],
        style: dict[str, str],
        *,
        empty_message: str = "",
    ) -> html.Div:
        """Construit une colonne (T1 ou T2) avec son libellé et son texte mis en surbrillance."""
        content = (
            _highlight_text_by_intervals(text, intervals, style)
            if text
            else html.Span(empty_message, className="fst-italic text-muted")
        )
        return html.Div(
            [
                html.Div(
                    label,
                    className="fw-semibold border-bottom px-2 py-1 small text-muted",
                ),
                html.Div(
                    content,
                    className="px-2 py-2 small",
                    style=base_card_style,
                ),
            ],
            className="border rounded bg-white overflow-hidden flex-grow-1",
            style={"minWidth": "0"},
        )

    current_label = str(current_quarter_label or "").strip()
    previous_label = str(previous_quarter_label or "").strip()
    has_period_labels = (
        current_label
        and previous_label
        and current_label != "Trimestre courant"
        and previous_label != "Trimestre précédent"
    )
    if has_period_labels:
        label_t1 = f"Précédent - {previous_label} (p.{page_t1})" if page_t1 else f"Précédent - {previous_label}"
        label_t2 = f"Courant - {current_label} (p.{page_t2})" if page_t2 else f"Courant - {current_label}"
    else:
        label_t1 = f"Précédent (p.{page_t1})" if page_t1 else "Précédent"
        label_t2 = f"Courant (p.{page_t2})" if page_t2 else "Courant"

    current_empty_message = (
        "Aucun texte dans le rapport courant — contenu retiré."
        if diff_type == "removed"
        else "Aucun texte dans le rapport courant."
    )
    previous_empty_message = (
        "Aucun texte dans le rapport précédent — contenu ajouté."
        if diff_type == "added"
        else "Aucun texte dans le rapport précédent."
    )

    return html.Div(
        [
            _column(
                label_t2,
                text_t2,
                intervals_t2,
                _HIGHLIGHT_ADDED_STYLE,
                empty_message=current_empty_message,
            ),
            _column(
                label_t1,
                text_t1,
                intervals_t1,
                _HIGHLIGHT_REMOVED_STYLE,
                empty_message=previous_empty_message,
            ),
        ],
        className="mb-3 d-flex gap-2",
    )


# ---------------------------------------------------------------------------
# Change card (vue analyste)
# ---------------------------------------------------------------------------


def _build_change_card(
    change: dict[str, Any],
    section_title: str,
    *,
    bank_code: str = "",
    current_quarter_label: str = "Trimestre courant",
    previous_quarter_label: str = "Trimestre précédent",
) -> dbc.Card:
    """Carte analytique pour un changement détecté.

    Args:
        change: Dict bloc issu de text_comparison.json.
        section_title: Nom affiché de la section/sous-section.
        bank_code: Code court de la banque utilisé comme sujet du résumé.
        current_quarter_label: Libelle du trimestre courant, si disponible.
        previous_quarter_label: Libelle du trimestre precedent, si disponible.

    Returns:
        dbc.Card stylisée ou None si unchanged/skip.
    """
    triage = change.get("genai_triage") or {}
    diff_type = change.get("diff_type", "")
    change_id = str(change.get("change_id") or "").strip()

    if diff_type == "unchanged" or triage.get("source") == "skip":
        return None  # type: ignore[return-value]

    analyst_narrative = build_analyst_narrative(
        change,
        bank_code=bank_code,
    )
    presentation = build_change_presentation(
        change,
        bank_code=bank_code,
        candidate_summary=analyst_narrative.changement_constate,
    )
    is_relevant = bool(triage.get("is_relevant", False))
    impact_level = (triage.get("impact_level") or "MINEUR").upper()
    impact_it_justification = str(triage.get("impact_it_justification") or "").strip()
    changement_posture = (triage.get("changement_posture") or "INDETERMINE").upper()
    justification_posture = str(triage.get("justification_posture") or "").strip()
    statut_mise_en_oeuvre = (triage.get("statut_mise_en_oeuvre") or "INDETERMINE").upper()
    confiance_posture = (triage.get("confiance_posture") or "INDETERMINE").upper()
    action = (triage.get("action_requise") or "aucune").lower()
    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    nouvelle_idee_justification = canonicalize_analyst_narrative(
        build_text_triage_justification(change),
        bank_code=bank_code,
    )
    impact_it_justification = canonicalize_analyst_narrative(
        impact_it_justification,
        bank_code=bank_code,
    )
    justification_posture = canonicalize_analyst_narrative(
        justification_posture,
        bank_code=bank_code,
    )
    themes_amf = list(triage.get("themes_amf") or [])
    justification_sections = extract_labeled_analysis(
        nouvelle_idee_justification,
        _TRIAGE_DETAIL_LABELS,
    )
    impact_domain = _impact_domain(themes_amf, section_title)
    business_relevance = analyst_narrative.pertinence_metier if presentation.scope == "qualitative" else ""

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
    badge_children.append(_badge(presentation.nature_label, "primary"))
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
    if presentation.scope == "secondary":
        badge_children.append(_badge("Secondaire / bruit", "light", text_color="dark"))
    if presentation.quality_status == "review":
        badge_children.append(
            _badge(
                "Résumé à valider",
                "warning",
                title=", ".join(presentation.quality_issues),
            )
        )
    badge_children.append(_badge(impact_lbl, impact_color))
    posture_badge = _POSTURE_BADGE.get(changement_posture)
    if posture_badge:
        badge_children.append(_badge(*posture_badge))
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
        current_quarter_label=current_quarter_label,
        previous_quarter_label=previous_quarter_label,
    )
    evidence_block = build_source_evidence_details(text_block)

    observed_block = _build_observed_block(
        impact_level=impact_level,
        impact_domain=impact_domain,
        justification_sections=justification_sections,
        change_summary=presentation.summary,
        relevance_reason="",
        observed_text=presentation.summary,
        business_relevance=business_relevance,
    )

    posture_proof_block, ai_details = _build_ai_details(
        impact_it_justification=impact_it_justification,
        impact_level=impact_level,
        impact_domain=impact_domain,
        justification_sections=justification_sections,
        changement_posture=changement_posture,
        justification_posture=justification_posture,
        statut_mise_en_oeuvre=statut_mise_en_oeuvre,
        confiance_posture=confiance_posture,
    )

    review = change.get("_analyst_review") or {}
    review_status = str(review.get("status") or "").strip().lower()
    review_comment = str(review.get("comment") or "").strip()
    review_badge = None
    if review_status in _TEXT_REVIEW_STATUS_BADGES:
        review_label, review_color = _TEXT_REVIEW_STATUS_BADGES[review_status]
        review_badge = _badge(f"Décision : {review_label}", review_color)

    review_controls = html.Div(
        [
            html.Div(
                [
                    html.Span("Revue analyste", className="fw-semibold small text-muted me-2"),
                    review_badge,
                ],
                className="mb-2 d-flex align-items-center flex-wrap",
            ),
            dcc.Textarea(
                id={"type": "text-review-comment", "change_id": change_id},
                value=review_comment,
                placeholder="Commentaire analyste (optionnel)...",
                className="form-control form-control-sm mb-2",
                style={"minHeight": "64px", "resize": "vertical"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Valider",
                        id={"type": "text-review-action", "change_id": change_id, "action": "approved"},
                        color="success",
                        size="sm",
                        outline=review_status != "approved",
                        className="me-2",
                        disabled=not change_id,
                    ),
                    dbc.Button(
                        "Rejeter",
                        id={"type": "text-review-action", "change_id": change_id, "action": "rejected"},
                        color="danger",
                        size="sm",
                        outline=review_status != "rejected",
                        className="me-2",
                        disabled=not change_id,
                    ),
                    dbc.Button(
                        "Passer",
                        id={"type": "text-review-action", "change_id": change_id, "action": "skipped"},
                        color="secondary",
                        size="sm",
                        outline=review_status != "skipped",
                        disabled=not change_id,
                    ),
                ],
                className="d-flex flex-wrap",
            ),
        ],
        className="mt-3 pt-3 border-top",
    )

    card_children = [
        c
        for c in [
            badge_row,
            themes_row,
            meta,
            observed_block,
            evidence_block,
            posture_proof_block,
            ai_details,
            review_controls,
        ]
        if c is not None
    ]

    return dbc.Card(
        dbc.CardBody(card_children, className="p-3"),
        className=f"mb-3 border-start border-{border_color} border-3",
    )


# ---------------------------------------------------------------------------
# Executive banner
# ---------------------------------------------------------------------------
