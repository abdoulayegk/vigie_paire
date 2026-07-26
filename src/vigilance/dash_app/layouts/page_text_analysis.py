"""Layout de l'onglet Analyse Textuelle — vue analyste.

Affiche tous les changements textuels détectés hors ``unchanged``. Les filtres
restants sont gérés dans ``text_flow.py``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from vigilance.analyst_change_presentation import (
    build_analyst_narrative,
    build_change_presentation,
    canonicalize_analyst_narrative,
    change_scope,
)
from vigilance.amf_taxonomy import (
    IMPACT_IT_DETAIL_LABELS,
    POSTURE_DETAIL_LABELS,
    _compact_complete_sentence_parts,
    extract_labeled_analysis,
)
from vigilance.dash_app.components.text_change_presentation import (
    atomic_parent_context,
    atomic_parent_key,
    build_atomic_change_group,
    build_source_evidence_details,
)
from vigilance.i18n.fr import sanitize_analyst_french
from vigilance.quarter_utils import quarter_label_from_payload
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

_POSTURE_BADGE: dict[str, tuple[str, str]] = {
    "RENFORCEMENT": ("Posture renforcée", "success"),
    "ALLEGEMENT": ("Posture allégée", "warning"),
    "NOUVEAU_DISPOSITIF": ("Nouveau dispositif", "primary"),
    "RETRAIT_DISPOSITIF": ("Dispositif retiré", "danger"),
    "AUCUN": ("Posture inchangée", "secondary"),
}

_IMPLEMENTATION_DETAIL_LABEL: dict[str, str] = {
    "ANNONCE": "Annoncée",
    "PLANIFIE": "Planifiée",
    "EN_COURS": "En cours",
    "MIS_EN_OEUVRE": "Mise en œuvre",
    "INDETERMINE": "Indéterminée",
}

_POSTURE_CONFIDENCE_DETAIL_LABEL: dict[str, str] = {
    "ELEVEE": "Élevée",
    "MOYENNE": "Moyenne",
    "FAIBLE": "Faible",
    "INDETERMINE": "Indéterminée",
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
    "RISQUE_DONNEES": "Risque données",
    "RISQUE_TIERS_CLOUD": "Tiers / Cloud",
    "RISQUE_MACRO_GEOPOLITIQUE": "Commercial / géopolitique",
    "GOUVERNANCE_RISQUES": "Gouvernance",
    "CONTROLE_CONFORMITE": "Contrôle / Conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "Nouvelle mention régl.",
    "MONTANT_REGLEMENTAIRE": "Montant régl.",
}

_IMPACT_DOMAIN_BY_THEME: dict[str, str] = {
    "CAPITAL_REGLEMENTAIRE": "capital réglementaire",
    "LIQUIDITE": "liquidité",
    "FONDS_PROPRES_REGLEMENTAIRES": "fonds propres réglementaires",
    "EXIGENCES_REGLEMENTAIRES": "exigences réglementaires",
    "RATIOS_REGLEMENTAIRES": "ratios prudentiels",
    "MONTANT_REGLEMENTAIRE": "seuils réglementaires",
    "ESG_CLIMATIQUE": "ESG / climat",
    "RISQUE_EMERGENT": "risques émergents",
    "RISQUE_DONNEES": "données",
    "RISQUE_TIERS_CLOUD": "tiers / cloud",
    "RISQUE_MACRO_GEOPOLITIQUE": "risques macroéconomiques / géopolitiques",
    "GOUVERNANCE_RISQUES": "gouvernance des risques",
    "CONTROLE_CONFORMITE": "contrôle / conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "réglementation",
    "MODIFICATION_METHODOLOGIE": "méthodologie de risque",
    "MODIFICATION_TEXTE_RISQUE": "gestion des risques",
    "FACTEUR_RISQUE_CHANGEMENT": "facteurs de risque",
    "HYPOTHESES_EXPLICATIONS_RISQUES": "hypothèses de risque",
}

_IMPACT_DOMAIN_PRIORITY = (
    "RISQUE_DONNEES",
    "RISQUE_TIERS_CLOUD",
    "RISQUE_EMERGENT",
    "ESG_CLIMATIQUE",
    "RISQUE_MACRO_GEOPOLITIQUE",
    "CAPITAL_REGLEMENTAIRE",
    "LIQUIDITE",
    "FONDS_PROPRES_REGLEMENTAIRES",
    "RATIOS_REGLEMENTAIRES",
    "MONTANT_REGLEMENTAIRE",
    "EXIGENCES_REGLEMENTAIRES",
    "CONTROLE_CONFORMITE",
    "GOUVERNANCE_RISQUES",
    "MODIFICATION_METHODOLOGIE",
    "FACTEUR_RISQUE_CHANGEMENT",
    "MODIFICATION_TEXTE_RISQUE",
    "HYPOTHESES_EXPLICATIONS_RISQUES",
    "NOUVELLE_MENTION_REGLEMENTAIRE",
)

_TRIAGE_DETAIL_LABELS = (
    "Nouvel élément à surveiller",
    "Sujet détecté",
    "Ce qui change",
    "Pertinence métier",
    "Point de surveillance",
)

_ACTION_BADGE: dict[str, tuple[str, str]] = {
    "revue_prioritaire": ("Revue prioritaire", "danger"),
    "investigation": ("Analyse approfondie", "warning"),
    "confirmation": ("Confirmation", "success"),
    "information": ("Information", "info"),
    "aucune": ("Aucune", "secondary"),
}

_TEXT_REVIEW_STATUS_BADGES: dict[str, tuple[str, str]] = {
    "approved": ("Validé", "success"),
    "rejected": ("Rejeté", "danger"),
    "skipped": ("Passé", "secondary"),
}

_UNSET = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _badge(label: str, color: str, **kwargs) -> dbc.Badge:
    """Construit un ``dbc.Badge`` standardisé pour la page d'analyse texte."""
    return dbc.Badge(label, color=color, className="me-1", **kwargs)


def _plural_count(count: int, singular: str, plural: str) -> str:
    """Retourne un libellé compté avec accord simple."""
    return f"{count} {singular if count == 1 else plural}"


def _build_executive_overview_text(
    global_summary: dict[str, Any],
    auditable_changes: int | None,
) -> str:
    """Construit le résumé analytique affiché dans la bannière texte."""
    counts = global_summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}

    n_detected = auditable_changes if auditable_changes is not None else int(counts.get("total", 0) or 0)
    n_substantive = int(counts.get("total_relevant", counts.get("total", 0)) or 0)
    n_maj = int(by_impact.get("MAJEUR", 0) or 0)
    n_mod = int(by_impact.get("MODERE", 0) or 0)

    detected_label = _plural_count(
        n_detected,
        "changement textuel détecté",
        "changements textuels détectés",
    )

    if n_substantive <= 0:
        access_sentence = (
            "Tous les changements restent accessibles afin de permettre une revue complète par l'analyste."
            if n_detected
            else ""
        )
        return (
            f"{detected_label}. Aucun changement n'est classé comme substantiel "
            f"à prioriser pour revue experte. {access_sentence}"
        ).strip()

    substantive_label = _plural_count(
        n_substantive,
        "changement substantiel",
        "changements substantiels",
    )
    major_label = _plural_count(n_maj, "majeur", "majeurs")
    moderate_label = _plural_count(n_mod, "modéré", "modérés")
    access_sentence = (
        "Les autres changements restent accessibles afin de permettre une revue complète par l'analyste."
        if n_detected > n_substantive
        else "Tous les changements restent accessibles afin de permettre une revue complète par l'analyste."
    )

    return (
        f"{detected_label}. L'analyse en classe {substantive_label}, "
        f"à prioriser pour revue experte : {major_label} et {moderate_label}. "
        f"{access_sentence}"
    )


# Styles inline pour les highlights — couleurs métier banque
# (ambre=retiré, vert=ajouté).
_HIGHLIGHT_REMOVED_STYLE = {
    "backgroundColor": "#fef3c7",
    "color": "#92400e",
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
def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fusionne des intervalles de caractères chevauchants."""
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_highlight_intervals(text: str, highlights: list[str]) -> list[tuple[int, int]]:
    """Retourne les positions des fragments GPT retrouvables dans le texte."""
    intervals: list[tuple[int, int]] = []
    lower_text = text.lower()
    for highlight in highlights:
        if not highlight or not highlight.strip():
            continue
        needle = highlight.lower()
        start = 0
        while True:
            idx = lower_text.find(needle, start)
            if idx < 0:
                break
            intervals.append((idx, idx + len(highlight)))
            start = idx + len(highlight)
    return _merge_intervals(intervals)


def _highlight_text_by_intervals(
    text: str,
    intervals: list[tuple[int, int]],
    style: dict[str, str],
) -> list:
    """Découpe ``text`` en spans selon des intervalles déjà calculés."""
    if not text:
        return []
    merged = _merge_intervals(intervals)
    if not merged:
        return [html.Span(text)]

    spans: list = []
    cursor = 0
    for start, end in merged:
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if cursor < start:
            spans.append(html.Span(text[cursor:start]))
        if start < end:
            spans.append(html.Span(text[start:end], style=style))
        cursor = end
    if cursor < len(text):
        spans.append(html.Span(text[cursor:]))
    return spans


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

    return _highlight_text_by_intervals(text, _find_highlight_intervals(text, highlights), style)


def _change_segments_are_usable(change_segments: list[dict]) -> bool:
    """Ecarte les segments trop fragmentés qui produisent du faux surlignage."""
    lengths: list[int] = []
    for seg in change_segments:
        if not isinstance(seg, dict):
            continue
        parts = [
            str(seg.get("text_t1") or "").strip(),
            str(seg.get("text_t2") or "").strip(),
        ]
        substantive = [part for part in parts if re.search(r"\w", part, flags=re.UNICODE)]
        if not substantive:
            continue
        longest = max(len(part) for part in substantive)
        if longest < 3:
            return False
        lengths.append(longest)

    if not lengths:
        return False
    if len(lengths) >= 8:
        tiny_count = sum(1 for length in lengths if length < 12)
        if tiny_count / len(lengths) >= 0.35:
            return False
    return True


def _token_intervals(text: str) -> list[tuple[str, int, int]]:
    """Tokenise un texte en mots avec positions pour calculer un diff lisible."""
    return [(match.group(0).lower(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _diff_highlight_intervals(text_t1: str, text_t2: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Calcule les intervalles modifiés directement depuis le diff T1/T2."""
    tokens_t1 = _token_intervals(text_t1)
    tokens_t2 = _token_intervals(text_t2)
    if not tokens_t1 and not tokens_t2:
        return [], []
    if tokens_t1 and not tokens_t2:
        return [(0, len(text_t1))], []
    if tokens_t2 and not tokens_t1:
        return [], [(0, len(text_t2))]

    words_t1 = [token for token, _, _ in tokens_t1]
    words_t2 = [token for token, _, _ in tokens_t2]
    intervals_t1: list[tuple[int, int]] = []
    intervals_t2: list[tuple[int, int]] = []
    matcher = SequenceMatcher(None, words_t1, words_t2, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"delete", "replace"} and i1 < i2:
            intervals_t1.append((tokens_t1[i1][1], tokens_t1[i2 - 1][2]))
        if tag in {"insert", "replace"} and j1 < j2:
            intervals_t2.append((tokens_t2[j1][1], tokens_t2[j2 - 1][2]))
    return _merge_intervals(intervals_t1), _merge_intervals(intervals_t2)


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
    surveillance = sanitize_analyst_french(
        justification_sections.get("Point de surveillance", "")
    )
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
                        sanitize_analyst_french(
                            impact_sections.get("Conséquence probable", "")
                        ),
                    ),
                    _ai_detail_item(
                        "Limite de l’analyse",
                        sanitize_analyst_french(
                            impact_sections.get("Limite de l'analyse", "")
                        ),
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
    usable_change_segments = (
        change_segments if _change_segments_are_usable(change_segments) else []
    )
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
        label_t1 = (
            f"Précédent - {previous_label} (p.{page_t1})"
            if page_t1
            else f"Précédent - {previous_label}"
        )
        label_t2 = (
            f"Courant - {current_label} (p.{page_t2})"
            if page_t2
            else f"Courant - {current_label}"
        )
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
    impact_it_justification = str(
        triage.get("impact_it_justification") or ""
    ).strip()
    changement_posture = (
        triage.get("changement_posture") or "INDETERMINE"
    ).upper()
    justification_posture = str(
        triage.get("justification_posture") or ""
    ).strip()
    statut_mise_en_oeuvre = (
        triage.get("statut_mise_en_oeuvre") or "INDETERMINE"
    ).upper()
    confiance_posture = (
        triage.get("confiance_posture") or "INDETERMINE"
    ).upper()
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
    business_relevance = (
        analyst_narrative.pertinence_metier
        if presentation.scope == "qualitative"
        else ""
    )

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


def _build_executive_banner(
    global_summary: dict[str, Any],
    bank: str,
    q_cur: str,
    q_prev: str,
    auditable_changes: int | None = None,
) -> dbc.Alert:
    """Bannière exécutive avec résumé, compteurs et bouton export."""
    overview = _build_executive_overview_text(global_summary, auditable_changes)
    pertinence = (global_summary.get("pertinence_globale") or "FAIBLE").upper()
    counts = global_summary.get("counts") or {}
    by_impact = counts.get("by_impact") or {}

    pertinence_color = {"ELEVEE": "danger", "MOYENNE": "warning", "FAIBLE": "success"}.get(pertinence, "secondary")
    pertinence_label = {"ELEVEE": "Élevée", "MOYENNE": "Moyenne", "FAIBLE": "Faible"}.get(pertinence, pertinence)

    # Compteurs
    n_maj = by_impact.get("MAJEUR", 0)
    n_mod = by_impact.get("MODERE", 0)
    n_auditable = auditable_changes if auditable_changes is not None else counts.get("total", 0)

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
            # Ligne 3 : compteurs + bouton Excel
            html.Div(
                [
                    _badge(f"{n_maj} Majeur(s)", "danger") if n_maj else None,
                    _badge(f"{n_mod} Modéré(s)", "warning") if n_mod else None,
                    _badge(f"{n_auditable} changement(s) textuel(s)", "primary") if n_auditable else None,
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


def _count_auditable_text_changes(section_comparisons: list[dict[str, Any]]) -> int:
    """Compte tous les changements textuels affichables pour revue analyste."""
    total = 0
    for sec in section_comparisons:
        for change in sec.get("all_block_comparisons") or []:
            if change.get("diff_type") == "unchanged":
                continue
            total += 1
    return total


def _text_review_progress(section_comparisons: list[dict[str, Any]]) -> dict[str, int]:
    """Calcule les décisions de revue sur le périmètre textuel auditable."""
    counts = {"approved": 0, "rejected": 0, "skipped": 0, "pending": 0}
    for section in section_comparisons:
        for change in section.get("all_block_comparisons") or []:
            if change.get("diff_type") == "unchanged":
                continue
            review = change.get("_analyst_review") or {}
            status = str(review.get("status") or "pending").strip().lower()
            counts[status if status in counts else "pending"] += 1

    total = sum(counts.values())
    decided = counts["approved"] + counts["rejected"]
    remaining = counts["pending"] + counts["skipped"]
    percent = round(decided / total * 100) if total else 0
    return {
        **counts,
        "total": total,
        "decided": decided,
        "remaining": remaining,
        "percent": percent,
    }


def _build_text_review_progress(section_comparisons: list[dict[str, Any]]) -> html.Div:
    """Construit la bannière globale d'avancement de la revue textuelle."""
    progress = _text_review_progress(section_comparisons)
    percent = progress["percent"]
    complete = bool(progress["total"]) and progress["remaining"] == 0

    summary_items: list[Any] = [
        html.Span(
            (
                f"{progress['decided']} / {progress['total']} "
                f"{('décision rendue' if progress['decided'] == 1 else 'décisions rendues')}"
            ),
            className="fw-semibold me-2",
        ),
        dbc.Badge(
            _plural_count(progress["approved"], "validé", "validés"),
            color="success",
            className="me-1",
        ),
        dbc.Badge(
            _plural_count(progress["rejected"], "rejeté", "rejetés"),
            color="danger",
            className="me-1",
        ),
        dbc.Button(
            f"{progress['remaining']} à traiter",
            id="text-progress-remaining",
            color="primary" if progress["remaining"] else "success",
            outline=bool(progress["remaining"]),
            size="sm",
            className="text-review-remaining-button",
            disabled=not progress["remaining"],
            title="Afficher les changements qui nécessitent encore une décision",
        ),
    ]
    if progress["skipped"]:
        summary_items.append(
            html.Span(
                f"dont {_plural_count(progress['skipped'], 'passé', 'passés')} à reprendre",
                className="small text-muted ms-2",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Avancement de la revue textuelle", className="fw-semibold"),
                    dbc.Badge(
                        "Revue complète" if complete else f"{percent}%",
                        color="success" if complete else "primary",
                        className="ms-auto",
                    ),
                ],
                className="d-flex align-items-center mb-2",
            ),
            dbc.Progress(
                value=percent,
                color="success" if complete else "primary",
                className="text-review-progress-bar mb-2",
            ),
            html.Div(summary_items, className="d-flex align-items-center flex-wrap gap-1"),
        ],
        id="text-review-progress",
        className="text-review-progress-banner px-3 py-2 rounded border",
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
