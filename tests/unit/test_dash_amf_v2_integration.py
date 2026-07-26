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

from vigilance.dash_app.components.review_detail_v2 import (
    _build_genai_section,
    _build_themes_amf_chips,
)
from vigilance.dash_app.components.review_queue_v2 import _build_genai_summary_row
from vigilance.dash_app.layouts.page_changements_communs import _build_signal_card
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


def _find_component_by_type(node: object, type_name: str) -> Component:
    """Retourne le premier composant Dash portant le nom de type demandé."""
    if isinstance(node, Component):
        if type(node).__name__ == type_name:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for child in children:
                try:
                    return _find_component_by_type(child, type_name)
                except LookupError:
                    pass
        elif children is not None:
            return _find_component_by_type(children, type_name)
    raise LookupError(type_name)


def _find_component_by_type_and_text(
    node: object,
    type_name: str,
    expected_text: str,
) -> Component:
    """Retourne le composant du type demandé contenant un libellé précis."""
    if isinstance(node, Component):
        if type(node).__name__ == type_name and expected_text in _flatten_text(node):
            return node
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for child in children:
                try:
                    return _find_component_by_type_and_text(
                        child,
                        type_name,
                        expected_text,
                    )
                except LookupError:
                    pass
        elif children is not None:
            return _find_component_by_type_and_text(
                children,
                type_name,
                expected_text,
            )
    raise LookupError(f"{type_name}: {expected_text}")


def _styled_texts(node: object) -> list[tuple[str, dict]]:
    """Retourne les textes portés par des composants stylés."""
    results: list[tuple[str, dict]] = []
    if isinstance(node, Component):
        style = getattr(node, "style", None)
        if style:
            results.append((_flatten_text(node), style))
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for child in children:
                results.extend(_styled_texts(child))
        elif children is not None:
            results.extend(_styled_texts(children))
    return results


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


def test_themes_amf_chips_render_data_and_third_party_cloud_labels() -> None:
    chips_div = _build_themes_amf_chips(
        ["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"]
    )
    text = _flatten_text(chips_div)
    assert "Risque données" in text
    assert "Tiers / Cloud" in text


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
            "impact_it": "ELEVE",
            "impact_it_justification": (
                "Éléments observés : Le changement exige une migration des "
                "données et de nouveaux contrôles technologiques.\n\n"
                "Conséquence probable : Les processus et l'architecture IT "
                "devront être adaptés à la migration.\n\n"
                "Limite de l'analyse : Le rapport ne précise pas le calendrier "
                "technique complet."
            ),
            "changement_posture": "RENFORCEMENT",
            "justification_posture": (
                "Preuve : La banque renforce les contrôles et la surveillance "
                "associés à la migration des données.\n\n"
                "Effet sur la gestion du risque : Le niveau d'encadrement des "
                "données et du fournisseur augmente.\n\n"
                "Justification du statut : Le rapport décrit un déploiement en "
                "cours, mais pas encore achevé.\n\n"
                "Justification de la confiance : Le renforcement et son état "
                "d'avancement sont formulés explicitement."
            ),
            "statut_mise_en_oeuvre": "EN_COURS",
            "confiance_posture": "ELEVEE",
            "nouvelle_idee": True,
            "nouvelle_idee_justification": (
                "OUI — Nouvel élément à surveiller : Oui.\n\n"
                "Sujet détecté : Exigences réglementaires et méthodologie.\n\n"
                "Ce qui change : Le T2 modifie la méthode applicable aux "
                "exigences réglementaires.\n\n"
                "Pertinence métier : La modification peut changer la manière "
                "dont la banque applique le cadre réglementaire.\n\n"
                "Point de surveillance : Exigences réglementaires — Vérifier "
                "les adaptations du dispositif de conformité."
            ),
            "action_requise": "revue_prioritaire",
        },
    }
    card = _build_change_card(change, "Gestion des risques", bank_code="BNC")
    assert card is not None
    text = _flatten_text(card)
    assert "Nouvelle idée" in text
    assert "Majeur" in text  # libellé impact (capitalized) pour la page texte
    assert "Impact exigences réglementaires — Majeur" in text
    assert "Impact IT" not in text
    assert "Posture renforcée" in text
    assert "Preuve de posture" in text
    assert "Voir les détails de l’évaluation IA" in text
    assert "Changement constaté" in text
    assert "Pertinence métier" in text
    assert (
        "La modification peut changer la manière dont la banque applique le cadre "
        "réglementaire."
    ) in text
    assert "Voir la preuve source" in text
    assert "Conséquence probable" in text
    assert "Limite de l’analyse" in text
    assert "Mise en œuvre En cours" in text
    assert "Confiance Élevée" in text
    assert "Modif. méthodologie" in text
    assert "Revue prioritaire" in text

    details = _find_component_by_type_and_text(
        card,
        "Details",
        "Voir les détails de l’évaluation IA",
    )
    assert getattr(details, "open", None) is False
    details_text = _flatten_text(details)
    assert "Voir les détails de l’évaluation IA" in details_text
    assert "Changement constaté" not in details_text
    assert "Pertinence métier" not in details_text

    card_body = getattr(card, "children")
    card_sections = [_flatten_text(child) for child in getattr(card_body, "children")]
    proof_index = next(
        index for index, value in enumerate(card_sections) if "Preuve de posture" in value
    )
    observed_index = next(
        index for index, value in enumerate(card_sections) if "Changement constaté" in value
    )
    evidence_index = next(
        index for index, value in enumerate(card_sections) if "Voir la preuve source" in value
    )
    details_index = next(
        index
        for index, value in enumerate(card_sections)
        if "Voir les détails de l’évaluation IA" in value
    )
    assert observed_index < evidence_index < proof_index < details_index

    badge_row = getattr(card_body, "children")[0]
    badge_text = _flatten_text(badge_row)
    assert "Mise en œuvre" not in badge_text
    assert "Confiance" not in badge_text


def test_text_analysis_change_card_shows_unchanged_posture() -> None:
    change = {
        "diff_type": "modified",
        "source_text_t2": "Le risque est décrit plus précisément.",
        "source_text_t1": "Le risque était déjà décrit.",
        "genai_triage": {
            "is_relevant": True,
            "themes_amf": ["RISQUE_DONNEES"],
            "impact_level": "MINEUR",
            "impact_it": "INDETERMINE",
            "changement_posture": "AUCUN",
            "nouvelle_idee": False,
            "nouvelle_idee_justification": "NON - aucune nouvelle idée.",
            "action_requise": "information",
        },
    }

    card = _build_change_card(change, "Gestion des risques")
    text = _flatten_text(card)

    assert "Posture inchangée" in text
    details = _find_component_by_type_and_text(
        card,
        "Details",
        "Voir les détails de l’évaluation IA",
    )
    details_text = _flatten_text(details)
    assert "Voir les détails de l’évaluation IA" in details_text
    assert "Impact données — Mineur" in details_text
    assert "Impact IT" not in details_text
    assert "Posture indéterminée" not in details_text


def test_text_analysis_change_card_always_exposes_ai_details_fold() -> None:
    change = {
        "diff_type": "modified",
        "source_text_t2": "Le cadre est présenté plus brièvement.",
        "source_text_t1": "Le cadre était présenté avec davantage de détails.",
        "genai_triage": {
            "is_relevant": True,
            "themes_amf": ["MODIFICATION_METHODOLOGIE"],
            "impact_level": "MODERE",
            "impact_it": "INDETERMINE",
            "changement_posture": "INDETERMINE",
            "nouvelle_idee": True,
            "nouvelle_idee_justification": "OUI - présentation modifiée.",
            "action_requise": "investigation",
        },
    }

    card = _build_change_card(change, "Gestion du capital")
    details = _find_component_by_type_and_text(
        card,
        "Details",
        "Voir les détails de l’évaluation IA",
    )
    details_text = _flatten_text(details)

    assert getattr(details, "open", None) is False
    assert "Voir les détails de l’évaluation IA" in details_text
    assert "Impact méthodologie de risque — Modéré" in details_text
    assert "Impact IT" not in details_text
    assert "Posture indéterminée" not in details_text


def test_review_detail_renders_posture_evidence_and_implementation_status() -> None:
    table = {
        "genai_analysis": {
            "is_relevant": True,
            "themes_amf": ["RISQUE_TIERS_CLOUD"],
            "impact_level": "MODERE",
            "impact_it": "MOYEN",
            "impact_it_justification": (
                "Éléments observés : Le rapport décrit de nouveaux contrôles "
                "contractuels appliqués aux fournisseurs.\n\n"
                "Conséquence probable : Les processus de suivi IT et les "
                "rapports de contrôle devront être adaptés.\n\n"
                "Limite de l'analyse : Aucune migration ni modification "
                "d'architecture n'est décrite."
            ),
            "changement_posture": "RENFORCEMENT",
            "justification_posture": (
                "Preuve : La banque indique que la surveillance des fournisseurs "
                "critiques a été renforcée.\n\n"
                "Effet sur la gestion du risque : Le niveau d'encadrement des "
                "tiers critiques augmente.\n\n"
                "Justification du statut : Le texte décrit les contrôles comme "
                "déjà mis en œuvre.\n\n"
                "Justification de la confiance : La formulation du rapport est "
                "explicite et ne repose pas sur une inférence."
            ),
            "statut_mise_en_oeuvre": "MIS_EN_OEUVRE",
            "confiance_posture": "ELEVEE",
            "nouvelle_idee": False,
            "nouvelle_idee_justification": "NON - dispositif déjà connu.",
            "action_requise": "information",
        }
    }

    text = _flatten_text(_build_genai_section(table))

    assert "Posture renforcée" in text
    assert "Mise en œuvre réalisée" in text
    assert "Confiance posture élevée" in text
    assert "Posture de gestion" in text
    assert "surveillance des fournisseurs" in text


def test_review_detail_hides_impact_it_without_justification() -> None:
    table = {
        "genai_analysis": {
            "is_relevant": True,
            "themes_amf": ["RISQUE_DONNEES"],
            "impact_level": "MODERE",
            "impact_it": "FAIBLE",
            "impact_it_justification": "",
            "changement_posture": "AUCUN",
            "nouvelle_idee": False,
            "nouvelle_idee_justification": "NON - aucun signal IT explicite.",
            "action_requise": "information",
        }
    }

    text = _flatten_text(_build_genai_section(table))

    assert "Impact IT" not in text


def test_changements_communs_hides_indeterminate_impact_it() -> None:
    signal = {
        "theme": "Signal commun",
        "summary": "Synthèse du signal.",
        "impact_it": "INDETERMINE",
        "banks": ["bmo", "td"],
        "min_banks_met": False,
    }

    text = _flatten_text(_build_signal_card(signal))

    assert "Impact IT" not in text


def test_text_analysis_shows_observed_change_before_fold() -> None:
    """Le changement observé reste visible et les explications sont repliées."""
    change = {
        "diff_type": "removed",
        "source_text_t1": "Contexte géopolitique volatile.",
        "evidence_t1": {"pages": [12], "snippet": "preuve"},
        "genai_triage": {
            "is_relevant": True,
            "themes_amf": ["FACTEUR_RISQUE_CHANGEMENT"],
            "impact_level": "MAJEUR",
            "nouvelle_idee": True,
            "nouvelle_idee_justification": (
                "OUI — Nouvel élément à surveiller : Oui.\n\n"
                "Sujet détecté : Risques géopolitiques.\n\n"
                "Ce qui change : Le T2 retire la description du contexte géopolitique "
                "et économique volatile, incluant les mesures commerciales, la guerre "
                "russo-ukrainienne, et les affrontements entre Israël et le Hamas, "
                "qui étaient mentionnés au T1.\n\n"
                "Pertinence métier : Le retrait modifie le niveau de détail fourni.\n\n"
                "Point de surveillance : Risques géopolitiques — Suivre la transparence."
            ),
            "action_requise": "revue_prioritaire",
        },
    }

    card = _build_change_card(change, "Gestion des risques", bank_code="BNC")
    card_body = getattr(card, "children")
    card_sections = [_flatten_text(child) for child in getattr(card_body, "children")]
    observed_index = next(
        index for index, value in enumerate(card_sections) if "Changement constaté" in value
    )
    details_index = next(
        index
        for index, value in enumerate(card_sections)
        if "Voir les détails de l’évaluation IA" in value
    )

    assert observed_index < details_index
    assert "BNC retire la description du contexte géopolitique" in card_sections[
        observed_index
    ]
    assert "Changement constaté" in card_sections[observed_index]
    assert "…" not in card_sections[observed_index].split("Changement constaté", 1)[-1][:200]
    assert "Impact facteurs de risque — Majeur" in card_sections[observed_index]
    assert "Pertinence métier" in card_sections[observed_index]
    assert "Le retrait modifie le niveau de détail fourni." in card_sections[
        observed_index
    ]
    assert "Pertinence métier" not in card_sections[details_index]


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


def test_text_analysis_hides_structured_non_relevance_reason_from_main_card() -> None:
    change = {
        "diff_type": "modified",
        "source_text_t1": "BMO Harris Bank N.A. était mentionnée.",
        "source_text_t2": "BMO Bank N.A. est désormais mentionnée.",
        "genai_triage": {
            "is_relevant": False,
            "themes_amf": [],
            "impact_level": "MINEUR",
            "nouvelle_idee": False,
            "exclusion_reason": "reformulation_mineure",
            "changement_constate": (
                "BMO remplace BMO Harris Bank N.A. par BMO Bank N.A."
            ),
            "signification_metier": "",
            "comparaison_interbanques": "",
            "limite_interpretation": "",
            "motif_non_pertinence": (
                "Cette reformulation ne révèle aucune nouvelle pratique comparable."
            ),
            "relevance_reason": (
                "RAISON LEGACY qui ne doit pas être utilisée par la carte."
            ),
        },
    }

    card = _build_change_card(change, "Gestion des risques", bank_code="BMO")
    text = _flatten_text(card)

    assert "BMO remplace BMO Harris Bank N.A. par BMO Bank N.A." in text
    assert "Pertinence métier" not in text
    assert "Cette reformulation ne révèle aucune nouvelle pratique comparable" not in text
    assert "RAISON LEGACY" not in text


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
    """Pour diff_type=modified, on affiche Courant a gauche puis Precedent."""
    sbs = _build_side_by_side(
        text_t1="Le seuil est de 4,5 %.",
        text_t2="Le seuil est de 5,0 %.",
        page_t1="22",
        page_t2="25",
        change_segments=[{"kind": "modified", "text_t1": "4,5 %", "text_t2": "5,0 %"}],
        diff_type="modified",
        current_quarter_label="T3 2025",
        previous_quarter_label="T2 2025",
    )
    text = _flatten_text(sbs)
    assert "Précédent - T2 2025 (p.22)" in text
    assert "Courant - T3 2025 (p.25)" in text
    assert "4,5 %" in text
    assert "5,0 %" in text
    assert _flatten_text(sbs.children[0]).startswith("Courant - T3 2025 (p.25)")
    assert _flatten_text(sbs.children[1]).startswith("Précédent - T2 2025 (p.22)")


def test_side_by_side_added_renders_current_then_empty_previous() -> None:
    """Pour un ajout, Courant précède le panneau vide Précédent."""
    sbs = _build_side_by_side(
        text_t1="",
        text_t2="Nouveau cadre IA générative.",
        page_t1="",
        page_t2="30",
        change_segments=[{"kind": "added", "text_t1": "", "text_t2": "Nouveau cadre IA générative."}],
        diff_type="added",
    )
    text = _flatten_text(sbs)
    assert "Courant (p.30)" in text
    assert "Précédent" in text
    assert "Nouveau cadre IA générative." in text
    assert "Aucun texte dans le rapport précédent — contenu ajouté." in text
    assert _flatten_text(sbs.children[0]).startswith("Courant (p.30)")
    assert _flatten_text(sbs.children[1]).startswith("Précédent")


def test_side_by_side_removed_renders_empty_current_then_previous() -> None:
    """Pour une suppression, le panneau vide Courant précède Précédent."""
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
    assert "Courant" in text
    assert "cybermenaces" in text
    assert "Aucun texte dans le rapport courant — contenu retiré." in text
    assert _flatten_text(sbs.children[0]).startswith("Courant")
    assert _flatten_text(sbs.children[1]).startswith("Précédent (p.18)")


def test_side_by_side_modified_highlights_diff_when_change_segments_do_not_match() -> None:
    """Sans segment IA exploitable, le diff T1/T2 surligne la partie source retirée."""
    removed_sentence = (
        "Depuis quelques années, la Banque fait face à un contexte volatile. "
        "Le contexte géopolitique, notamment les mesures commerciales, la guerre "
        "russo-ukrainienne et les affrontements entre Israël et le Hamas, crée des incertitudes."
    )
    sbs = _build_side_by_side(
        text_t1=(
            "Le risque de marché est le risque de pertes financières liées à la variation "
            f"des prix de marché. {removed_sentence}"
        ),
        text_t2=(
            "Le risque de marché est le risque de pertes financières liées à la variation "
            "des prix de marché."
        ),
        page_t1="30",
        page_t2="33",
        change_segments=[],
        diff_type="modified",
    )

    styled = _styled_texts(sbs)
    removed_highlights = [
        text
        for text, style in styled
        if style.get("backgroundColor") == "#fef3c7" and "contexte géopolitique" in text
    ]
    assert removed_highlights
    assert any("affrontements entre Israël et le Hamas" in text for text in removed_highlights)


def test_side_by_side_ignores_over_fragmented_change_segments() -> None:
    """Les segments caractère-par-caractère sont ignorés au profit du diff mot-à-mot."""
    text_t1 = (
        "Le 28 août 2023, la Banque a annoncé que la Bourse de Toronto (TSX) "
        "et le BSIF ont approuvé une offre publique de rachat dans le cours normal "
        "des activités visant à racheter, pour annulation, jusqu'à 90 millions de "
        "ses actions ordinaires. L'offre publique de rachat dans le cours normal "
        "des activités..."
    )
    text_t2 = (
        "La Banque prévoit lancer un nouveau programme d'offre publique de rachat "
        "dans le cours normal des activités au terme de l'OPRCNA de 2025, sous "
        "réserve de l'approbation des autorités de réglementation. Dans le cadre "
        "du nouveau programme, la Banque entend racheter de 6 à 7 milliards de "
        "dollars d'actions ordinaires au cours de l'exercice 2026, selon les "
        "conditions du marché."
    )
    noisy_segments = [
        {"kind": "modified", "text_t1": "a ann", "text_t2": "prév"},
        {"kind": "modified", "text_t1": "ncé que", "text_t2": "it"},
        {"kind": "modified", "text_t1": "B", "text_t2": "ncer un n"},
        {"kind": "added", "text_t1": "", "text_t2": "veau p"},
        {"kind": "modified", "text_t1": "s", "text_t2": "ogramm"},
        {"kind": "added", "text_t1": "", "text_t2": "on de"},
        {"kind": "modified", "text_t1": "90", "text_t2": "7"},
        {"kind": "modified", "text_t1": "se", "text_t2": "dollar"},
    ]

    sbs = _build_side_by_side(
        text_t1=text_t1,
        text_t2=text_t2,
        page_t1="80",
        page_t2="79",
        change_segments=noisy_segments,
        diff_type="modified",
    )

    added_highlights = [
        text
        for text, style in _styled_texts(sbs)
        if style.get("backgroundColor") == "#dcfce7"
    ]
    assert "prév" not in added_highlights
    assert "it" not in added_highlights
    assert "veau p" not in added_highlights
    assert any("prévoit lancer un nouveau programme d'offre" in text for text in added_highlights)


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
