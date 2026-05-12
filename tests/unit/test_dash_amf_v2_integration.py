"""Tests d'intégration Dash pour le schéma AMF v2 unifié.

Vérifie que :
- ``review_queue_v2._build_genai_summary_row`` filtre les items non pertinents
- ``review_detail_v2._build_themes_amf_chips`` rend correctement l'overflow ``+N``
- ``page_text_analysis._build_change_card`` affiche les nouveaux champs AMF
  et garde les changements ``is_relevant=False`` visibles pour revue humaine
"""

from __future__ import annotations

import pytest
from dash import html
from dash.development.base_component import Component

from vigilance.dash_app.components.review_detail_v2 import _build_themes_amf_chips
from vigilance.dash_app.components.review_queue_v2 import _build_genai_summary_row
from vigilance.dash_app.layouts.page_text_analysis import _build_change_card


def _flatten_text(node: object) -> str:
    """Aplatit récursivement les enfants Dash en une seule chaîne texte."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (int, float)):
        return str(node)
    if isinstance(node, list):
        return " ".join(_flatten_text(child) for child in node)
    if isinstance(node, Component):
        children = getattr(node, "children", None)
        return _flatten_text(children)
    return ""


# --- Filtrage is_relevant=False dans la queue (review_queue_v2) ---


def test_queue_summary_row_filters_non_pertinent() -> None:
    """Une table avec is_relevant=False ne doit produire aucune ligne synthèse."""
    table = {
        "genai_analysis": {
            "is_relevant": False,
            "themes_amf": [],
            "nouvelle_idee": False,
            "exclusion_reason": "reformulation_mineure",
        }
    }
    assert _build_genai_summary_row(table) is None


def test_queue_summary_row_renders_relevant_with_amf_fields() -> None:
    """Une table pertinente doit produire la ligne synthèse avec badges AMF."""
    table = {
        "genai_analysis": {
            "is_relevant": True,
            "nouvelle_idee": True,
            "nouvelle_idee_justification": ("OUI nouveau ratio TLAC ajoute. Cela aligne sur les exigences BSIF."),
            "themes_amf": ["DIVULGATION_AJOUT", "RATIOS_REGLEMENTAIRES"],
            "impact_level": "MAJEUR",
            "action_requise": "revue_prioritaire",
        }
    }
    row = _build_genai_summary_row(table)
    assert row is not None
    text = _flatten_text(row)
    assert "Nouvelle idée" in text
    assert "MAJEUR" in text
    assert "Revue prioritaire" in text


# --- Overflow des chips themes_amf (review_detail_v2) ---


def test_themes_amf_chips_show_overflow_count_when_more_than_max() -> None:
    """Au-delà de max_visible, on affiche un badge `+N` avec tooltip."""
    themes = [
        "DIVULGATION_AJOUT",
        "RATIOS_REGLEMENTAIRES",
        "MONTANT_REGLEMENTAIRE",
        "EXIGENCES_REGLEMENTAIRES",
        "NOUVELLE_MENTION_REGLEMENTAIRE",
        "GOUVERNANCE_RISQUES",
    ]
    chips_div = _build_themes_amf_chips(themes, max_visible=4)
    text = _flatten_text(chips_div)
    # Les 4 premiers thèmes affichés
    assert "Ajout de divulgation" in text
    assert "Ratios régl." in text
    # Badge overflow visible
    assert "+2" in text


def test_themes_amf_chips_returns_empty_div_when_no_themes() -> None:
    """Sans thèmes, on retourne un Div vide (pas de None)."""
    chips_div = _build_themes_amf_chips([])
    assert isinstance(chips_div, html.Div)


# --- page_text_analysis affiche les nouveaux champs AMF ---


def test_text_analysis_change_card_renders_amf_fields() -> None:
    """La carte texte narratif rend les badges AMF v2 et la justification."""
    change = {
        "diff_type": "modified",
        "source_text_t2": "Texte courant T2",
        "source_text_t1": "Texte précédent T1",
        "evidence_t2": {"pages": [10], "snippet": "preuve"},
        "genai_triage": {
            "is_relevant": True,
            "themes_amf": ["MODIFICATION_METHODOLOGIE", "EXIGENCES_REGLEMENTAIRES"],
            "impact_level": "MAJEUR",
            "nouvelle_idee": True,
            "nouvelle_idee_justification": (
                "OUI - methodologie modifiee au T2. Cela touche les exigences BSIF (MODIFICATION_METHODOLOGIE)."
            ),
            "action_requise": "revue_prioritaire",
        },
    }
    card = _build_change_card(change, "Gestion des risques")
    assert card is not None
    text = _flatten_text(card)
    assert "Nouvelle idée" in text
    assert "Majeur" in text  # libellé impact (capitalized) pour la page texte
    assert "Modif. méthodologie" in text
    assert "OUI" in text
    assert "Revue prioritaire" in text
    assert "Justification" in text
    assert "Justification de triage" not in text


def test_text_analysis_change_card_keeps_non_pertinent() -> None:
    """Une carte avec is_relevant=False reste visible pour revue humaine."""
    change = {
        "diff_type": "modified",
        "source_text_t2": "Variation chiffrée",
        "evidence_t2": {"pages": [5], "snippet": "preuve"},
        "genai_triage": {
            "is_relevant": False,
            "themes_amf": [],
            "impact_level": "MINEUR",
            "nouvelle_idee": False,
            "exclusion_reason": "variation_numerique_propre_banque",
        },
    }
    card = _build_change_card(change, "Gestion des risques")
    assert card is not None
    text = _flatten_text(card)
    assert "Non pertinent" in text
    assert "Variation chiffrée" in text


# --- Side-by-side avec highlights AMF v2 ---


from vigilance.dash_app.layouts.page_text_analysis import (
    _HIGHLIGHT_ADDED_STYLE,
    _HIGHLIGHT_REMOVED_STYLE,
    _build_side_by_side,
    _highlight_text,
)


def test_highlight_text_marks_matching_segments() -> None:
    """Les fragments présents dans le texte source sont surlignés."""
    text = "Le seuil prudentiel CET1 minimal applicable est de 4,5 %."
    spans = _highlight_text(text, ["4,5 %"], _HIGHLIGHT_REMOVED_STYLE)
    assert len(spans) == 3
    # Le fragment surligné porte le style demandé
    styled = [s for s in spans if getattr(s, "style", None)]
    assert len(styled) == 1
    assert styled[0].children == "4,5 %"


def test_highlight_text_skips_hallucinated_segments() -> None:
    """Un fragment GPT introuvable dans le texte source est ignoré silencieusement."""
    text = "Le seuil est de 4,5 %."
    spans = _highlight_text(text, ["FRAGMENT_INEXISTANT"], _HIGHLIGHT_ADDED_STYLE)
    # Aucun span n'est surligné
    assert all(getattr(s, "style", None) is None for s in spans)
    # Le texte complet est rendu
    assert _flatten_text(spans) == text


def test_highlight_text_handles_multiple_occurrences() -> None:
    """Toutes les occurrences d'un fragment sont surlignées."""
    text = "TLAC est important. TLAC permet la résolution. TLAC s'applique aux BISI."
    spans = _highlight_text(text, ["TLAC"], _HIGHLIGHT_ADDED_STYLE)
    styled = [s for s in spans if getattr(s, "style", None)]
    assert len(styled) == 3


def test_highlight_text_merges_overlapping_segments() -> None:
    """Deux highlights chevauchants sont fusionnés en un seul span surligné."""
    text = "approche par modèles internes avancés"
    # Deux fragments overlap : « modèles internes » et « internes avancés »
    spans = _highlight_text(
        text,
        ["modèles internes", "internes avancés"],
        _HIGHLIGHT_ADDED_STYLE,
    )
    styled = [s for s in spans if getattr(s, "style", None)]
    # Les deux highlights chevauchants → un seul span fusionné
    assert len(styled) == 1
    assert styled[0].children == "modèles internes avancés"


def test_side_by_side_modified_renders_two_columns() -> None:
    """Pour diff_type=modified, on affiche les 2 colonnes T1 et T2."""
    sbs = _build_side_by_side(
        text_t1="Le seuil est de 4,5 %.",
        text_t2="Le seuil est de 5,0 %.",
        page_t1="22",
        page_t2="25",
        change_segments=[{"kind": "modified", "text_t1": "4,5 %", "text_t2": "5,0 %"}],
        diff_type="modified",
    )
    text = _flatten_text(sbs)
    assert "Précédent (p.22)" in text
    assert "Courant (p.25)" in text
    assert "4,5 %" in text
    assert "5,0 %" in text


def test_side_by_side_added_renders_only_t2() -> None:
    """Pour diff_type=added, seule la colonne T2 est affichée."""
    sbs = _build_side_by_side(
        text_t1="",
        text_t2="Nouveau cadre IA générative.",
        page_t1="",
        page_t2="30",
        change_segments=[{"kind": "added", "text_t1": "", "text_t2": "Nouveau cadre IA générative."}],
        diff_type="added",
    )
    text = _flatten_text(sbs)
    assert "Précédent" not in text
    assert "Courant (p.30)" in text
    assert "Nouveau cadre IA générative." in text


def test_side_by_side_removed_renders_only_t1() -> None:
    """Pour diff_type=removed, seule la colonne T1 est affichée."""
    sbs = _build_side_by_side(
        text_t1="Mention cybermenaces (DDoS, ransomwares).",
        text_t2="",
        page_t1="18",
        page_t2="",
        change_segments=[
            {
                "kind": "removed",
                "text_t1": "Mention cybermenaces (DDoS, ransomwares).",
                "text_t2": "",
            }
        ],
        diff_type="removed",
    )
    text = _flatten_text(sbs)
    assert "Précédent (p.18)" in text
    assert "Courant" not in text
    assert "cybermenaces" in text


def test_change_segment_pydantic_invariants() -> None:
    """Pydantic rejette les ChangeSegment incohérents."""
    from pydantic import ValidationError as _PydErr

    from vigilance.amf_taxonomy import ChangeSegment

    # Valid added
    s = ChangeSegment(kind="added", text_t2="x")
    assert s.kind == "added"

    # Invalid: added with text_t1 set
    with pytest.raises(_PydErr, match="added"):
        ChangeSegment(kind="added", text_t1="non vide", text_t2="x")

    # Invalid: removed without text_t1
    with pytest.raises(_PydErr, match="removed"):
        ChangeSegment(kind="removed", text_t1="", text_t2="")

    # Invalid: modified with one side empty
    with pytest.raises(_PydErr, match="modified"):
        ChangeSegment(kind="modified", text_t1="ok", text_t2="")
