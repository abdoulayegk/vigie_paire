from __future__ import annotations

from typing import Any

from dash.development.base_component import Component

from vigilance.dash_app.callbacks.text_flow import (
    download_text_excel,
    filter_text_cards,
    render_text_analysis,
    show_remaining_text_changes,
)
from vigilance.dash_app.layouts.page_text_analysis import build_text_analysis_tab


def _flat_text(node: object) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list | tuple):
        return " ".join(_flat_text(child) for child in node)
    if isinstance(node, Component):
        return _flat_text(getattr(node, "children", None))
    return ""


def _find_by_id(node: object, target_id: str) -> Component:
    if isinstance(node, Component):
        if getattr(node, "id", None) == target_id:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, list | tuple):
            for child in children:
                try:
                    return _find_by_id(child, target_id)
                except LookupError:
                    pass
        elif children is not None:
            return _find_by_id(children, target_id)
    raise LookupError(target_id)


def test_download_text_excel_reload_latest_payload_before_export(monkeypatch) -> None:
    stale_payload = {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
        "section_comparisons": [{"block_comparisons": [{"change_id": "strict"}], "all_block_comparisons": []}],
    }
    latest_payload = {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
        "section_comparisons": [
            {"block_comparisons": [{"change_id": "strict"}], "all_block_comparisons": [{"change_id": "all"}]}
        ],
    }
    captured: dict[str, Any] = {}

    def _fake_load_text_comparison_for_dash(**kwargs):
        captured["load_kwargs"] = kwargs
        return latest_payload

    def _fake_generate_text_comparison_excel(payload, output_path=None):
        captured["export_payload"] = payload
        assert output_path is None
        return b"excel-bytes"

    monkeypatch.setattr(
        "vigilance.dash_app.services.text_comparison_store.load_text_comparison_for_dash",
        _fake_load_text_comparison_for_dash,
    )
    monkeypatch.setattr(
        "vigilance.text_comparison.generate_text_comparison_excel",
        _fake_generate_text_comparison_excel,
    )

    response = download_text_excel(1, stale_payload)

    assert captured["load_kwargs"] == {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
    }
    assert captured["export_payload"] is latest_payload
    assert response["filename"] == "veille_textuelle_TD_T1_2026.xlsx"


def test_filter_text_cards_prioritizes_qualitative_and_keeps_secondary_accessible() -> None:
    """La vue principale masque le bruit, qui reste accessible par filtre."""
    text_data = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "major_existing",
                        "diff_type": "modified",
                        "semantic_text_t2": "Majeur existant",
                        "pages_t2": [5],
                        "evidence_t2": {"pages": [5], "snippet": "preuve 1"},
                        "genai_triage": {
                            "is_relevant": True,
                            "themes_amf": ["MODIFICATION_TEXTE_RISQUE"],
                            "impact_level": "MAJEUR",
                            "action_requise": "revue_prioritaire",
                            "nouvelle_idee": False,
                            "nouvelle_idee_justification": (
                                "NON le concept existait deja au T1. "
                                "Seule la formulation a evolue de maniere substantive."
                            ),
                        },
                    },
                    {
                        "change_id": "new_moderate",
                        "diff_type": "added",
                        "semantic_text_t2": "Nouvelle idée",
                        "pages_t2": [9],
                        "evidence_t2": {"pages": [9], "snippet": "preuve 2"},
                        "genai_triage": {
                            "is_relevant": True,
                            "themes_amf": ["DIVULGATION_AJOUT", "STRUCTURE_RAPPORT"],
                            "impact_level": "MODERE",
                            "action_requise": "information",
                            "nouvelle_idee": True,
                            "nouvelle_idee_justification": (
                                "OUI nouvelle divulgation absente au T1. "
                                "Cela introduit un sujet nouveau dans le rapport."
                            ),
                        },
                    },
                    {
                        "change_id": "non_pertinent",
                        "diff_type": "modified",
                        "semantic_text_t2": "Variation chiffree",
                        "pages_t2": [2],
                        "evidence_t2": {"pages": [2], "snippet": "preuve 3"},
                        "genai_triage": {
                            "is_relevant": False,
                            "themes_amf": [],
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                            "exclusion_reason": "variation_numerique_propre_banque",
                        },
                    },
                ],
            }
        ]
    }

    cards, count_text = filter_text_cards(text_data, None, None, None)
    all_cards, all_count_text = filter_text_cards(
        text_data,
        None,
        None,
        None,
        None,
        "all",
    )

    assert count_text == "2 changements qualitatifs affichés"
    assert len(cards) == 2
    assert all_count_text == "3 changements affichés"
    assert len(all_cards) == 3
    # La structure de la carte est : badge_row, themes_row?, meta, side_by_side, ...
    # Le side-by-side rend les textes T1/T2 dans des spans imbriqués.
    # On aplatit tous les enfants pour vérifier la présence des phrases.
    from dash.development.base_component import Component as _DashComponent

    def _flat_text(node) -> str:
        if node is None:
            return ""
        if isinstance(node, str):
            return node
        if isinstance(node, list):
            return " ".join(_flat_text(c) for c in node)
        if isinstance(node, _DashComponent):
            return _flat_text(getattr(node, "children", None))
        return ""

    first_text = _flat_text(cards[0])
    second_text = _flat_text(cards[1])
    # Tri : impact d'abord, puis nouvelle idée
    assert "Majeur existant" in first_text
    assert "Nouvelle idée" in second_text
    assert "Variation chiffree" not in _flat_text(cards)
    assert "Variation chiffree" in _flat_text(all_cards)
    assert "Secondaire / bruit" in _flat_text(all_cards)


def test_filter_text_cards_places_dates_and_reformulations_in_secondary_scope() -> None:
    text_data = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "date_update",
                        "diff_type": "modified",
                        "change_summary": "La date de référence est passée de janvier à avril.",
                        "source_text_t1": "Données au 31 janvier.",
                        "source_text_t2": "Données au 30 avril.",
                        "pages_t2": [5],
                        "evidence_t1": {"pages": [4]},
                        "evidence_t2": {"pages": [5]},
                        "genai_triage": {
                            "is_relevant": False,
                            "themes_amf": [],
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                            "exclusion_reason": "variation_numerique_propre_banque",
                        },
                    },
                    {
                        "change_id": "reformulation",
                        "diff_type": "modified",
                        "change_summary": "Légère reformulation sans changement de fond.",
                        "source_text_t1": "La banque surveille ce risque.",
                        "source_text_t2": "Ce risque est surveillé par la banque.",
                        "pages_t2": [7],
                        "evidence_t1": {"pages": [6]},
                        "evidence_t2": {"pages": [7]},
                        "genai_triage": {
                            "is_relevant": False,
                            "themes_amf": [],
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                            "exclusion_reason": "reformulation_mineure",
                        },
                    },
                ],
            }
        ]
    }

    cards, count_text = filter_text_cards(
        text_data,
        None,
        None,
        None,
        None,
        "secondary",
    )
    rendered = _flat_text(cards)
    compact = " ".join(rendered.split())

    assert count_text == "2 changements secondaires affichés"
    assert "Données au 31 janvier." in compact
    assert "Données au 30 avril." in compact
    assert "La banque surveille ce risque." in compact
    assert "Ce risque est surveillé par la banque." in compact
    assert rendered.count("Non pertinent") == 2


def test_filter_text_cards_groups_atomic_ideas_under_their_parent_context() -> None:
    parent_context = (
        "Notre cadre d’appétit pour le risque s’articule autour de cinq objectifs."
    )
    text_data = {
        "bank_code": "bmo",
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Cadre d’appétit pour le risque",
                "all_block_comparisons": [
                    {
                        "change_id": "ai",
                        "diff_type": "modified",
                        "unit_role_t1": "item",
                        "unit_role_t2": "item",
                        "parent_chunk_id_t1": "parent-1",
                        "parent_chunk_id_t2": "parent-1",
                        "parent_context_t2": parent_context,
                        "change_summary": (
                            "Le T2 ajoute la surveillance des risques liés à "
                            "l’intelligence artificielle."
                        ),
                        "source_text_t1": "Surveiller les risques technologiques.",
                        "source_text_t2": (
                            "Surveiller les risques technologiques et ceux liés à l’IA."
                        ),
                        "genai_triage": {
                            "is_relevant": True,
                            "nouvelle_idee": True,
                            "impact_level": "MODERE",
                        },
                    },
                    {
                        "change_id": "crisis",
                        "diff_type": "modified",
                        "unit_role_t1": "item",
                        "unit_role_t2": "item",
                        "parent_chunk_id_t1": "parent-1",
                        "parent_chunk_id_t2": "parent-1",
                        "parent_context_t2": parent_context,
                        "change_summary": (
                            "Le T2 ajoute le renforcement de sa capacité à absorber "
                            "les périodes de crise."
                        ),
                        "source_text_t1": "Maintenir une solide situation de capital.",
                        "source_text_t2": (
                            "Maintenir une situation permettant d’absorber les crises."
                        ),
                        "genai_triage": {
                            "is_relevant": True,
                            "nouvelle_idee": True,
                            "impact_level": "MODERE",
                        },
                    },
                ],
            }
        ],
    }

    groups, count_text = filter_text_cards(text_data, None, None, None)
    rendered = _flat_text(groups)

    assert count_text == "2 changements qualitatifs affichés"
    assert len(groups) == 1
    assert "Bloc de liste analysé" in rendered
    assert "2 idées modifiées" in rendered
    assert parent_context in rendered
    assert "BMO ajoute la surveillance des risques liés à l’intelligence artificielle." in rendered
    assert "BMO ajoute le renforcement de sa capacité à absorber les périodes de crise." in rendered
    assert rendered.count("Changement constaté") == 2


def test_text_analysis_banner_uses_auditable_text_total_not_retained_total() -> None:
    """Le badge principal texte suit le même périmètre que l'Excel."""
    changes = [
        {
            "change_id": f"c{i}",
            "diff_type": "modified",
            "source_text_t1": "Ancien",
            "source_text_t2": "Nouveau",
            "genai_triage": {"impact_level": "MINEUR", "is_relevant": i < 17},
        }
        for i in range(27)
    ]
    text_data = {
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "global_summary": {
            "executive_overview": "17 changement(s) textuel(s) substantiel(s) retenu(s) pour revue experte.",
            "key_highlights": ["Exemple brut à ne pas afficher"],
            "counts": {
                "total": 32,
                "total_relevant": 17,
                "by_impact": {"MAJEUR": 4, "MODERE": 13},
            }
        },
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "block_comparisons": changes[:17],
                "all_block_comparisons": changes,
            }
        ],
    }

    view = build_text_analysis_tab(text_data)
    text = _flat_text(view)

    assert "BNC · T2 2025 vs T1 2025" in text
    assert "Courant - T2 2025" in text
    assert "Précédent - T1 2025" in text
    assert "27 changement(s) textuel(s)" in text
    assert "27 changements textuels détectés" in text
    assert "17 changements substantiels" in text
    assert "4 majeurs et 13 modérés" in text
    assert "Les autres changements restent accessibles" in text
    assert "Exemple brut à ne pas afficher" not in text
    assert "retenu(s) pour revue experte" not in text
    assert "17 pertinents / 32 analysés" not in text


def test_text_review_progress_counts_decisions_and_keeps_skipped_remaining() -> None:
    changes = [
        {
            "change_id": "approved_1",
            "diff_type": "modified",
            "source_text_t2": "Validé un",
            "_analyst_review": {"status": "approved"},
        },
        {
            "change_id": "approved_2",
            "diff_type": "modified",
            "source_text_t2": "Validé deux",
            "_analyst_review": {"status": "approved"},
        },
        {
            "change_id": "rejected_1",
            "diff_type": "modified",
            "source_text_t2": "Rejeté un",
            "_analyst_review": {"status": "rejected"},
        },
        {
            "change_id": "skipped_1",
            "diff_type": "modified",
            "source_text_t2": "Passé un",
            "_analyst_review": {"status": "skipped"},
        },
        {
            "change_id": "pending_1",
            "diff_type": "modified",
            "source_text_t2": "En attente un",
        },
        {
            "change_id": "unchanged_1",
            "diff_type": "unchanged",
            "source_text_t2": "Stable",
        },
    ]
    text_data = {
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "global_summary": {"counts": {"by_impact": {}}},
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": changes,
            }
        ],
    }

    view = build_text_analysis_tab(text_data)
    rendered = _flat_text(view)
    progress = _find_by_id(view, "text-review-progress")
    progress_bar = next(
        child
        for child in progress.children
        if isinstance(child, Component) and child.__class__.__name__ == "Progress"
    )

    assert "3 / 5 décisions rendues" in rendered
    assert "2 validés" in rendered
    assert "1 rejeté" in rendered
    assert "2 à traiter" in rendered
    assert "dont 1 passé à reprendre" in rendered
    assert progress_bar.value == 60
    assert "Passé un" in rendered
    assert "En attente un" in rendered
    assert "Validé un" not in rendered


def test_text_review_status_filter_and_remaining_shortcut() -> None:
    text_data = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "approved",
                        "diff_type": "modified",
                        "source_text_t2": "Décision validée",
                        "_analyst_review": {"status": "approved"},
                    },
                    {
                        "change_id": "pending",
                        "diff_type": "modified",
                        "source_text_t2": "Décision attendue",
                    },
                ],
            }
        ]
    }

    approved_cards, approved_count = filter_text_cards(
        text_data,
        None,
        None,
        None,
        "approved",
    )
    remaining_cards, remaining_count = filter_text_cards(
        text_data,
        None,
        None,
        None,
        "remaining",
    )

    assert approved_count == "1 changement qualitatif affiché"
    assert "Décision validée" in _flat_text(approved_cards)
    assert "Décision attendue" not in _flat_text(approved_cards)
    assert remaining_count == "1 changement qualitatif affiché"
    assert "Décision attendue" in _flat_text(remaining_cards)
    assert show_remaining_text_changes(1) == "remaining"


def test_text_analysis_preserves_filters_when_review_store_changes() -> None:
    text_data = {
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "global_summary": {"counts": {"by_impact": {}}},
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "risk_1",
                        "diff_type": "modified",
                        "source_text_t2": "Risque validé",
                        "genai_triage": {
                            "impact_level": "MAJEUR",
                            "action_requise": "revue_prioritaire",
                        },
                        "_analyst_review": {"status": "approved"},
                    }
                ],
            }
        ],
    }
    filters = {
        "section": None,
        "scope": "all",
        "impact": "MAJEUR",
        "action": "revue_prioritaire",
        "status": "approved",
    }

    view = render_text_analysis(text_data, True, filters)

    assert _find_by_id(view, "text-filter-section").value is None
    assert _find_by_id(view, "text-filter-scope").value == "all"
    assert _find_by_id(view, "text-filter-impact").value == "MAJEUR"
    assert _find_by_id(view, "text-filter-action").value == "revue_prioritaire"
    assert _find_by_id(view, "text-filter-status").value == "approved"


def test_text_analysis_tab_selects_first_auditable_section_by_default() -> None:
    text_data = {
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "global_summary": {"counts": {"total": 2, "total_relevant": 2, "by_impact": {}}},
        "section_comparisons": [
            {
                "section_key": "gestion_capital",
                "section_title": "Gestion du capital",
                "all_block_comparisons": [
                    {
                        "change_id": "capital_1",
                        "diff_type": "modified",
                        "source_text_t1": "Ancien capital",
                        "source_text_t2": "Nouveau capital",
                        "genai_triage": {"impact_level": "MAJEUR"},
                    }
                ],
            },
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "risk_1",
                        "diff_type": "modified",
                        "source_text_t1": "Ancien risque",
                        "source_text_t2": "Nouveau risque",
                        "genai_triage": {"impact_level": "MODERE"},
                    }
                ],
            },
        ],
    }

    view = build_text_analysis_tab(text_data)
    section_dropdown = _find_by_id(view, "text-filter-section")
    text = _flat_text(view)

    assert section_dropdown.value == "gestion_capital"
    assert "1 changement qualitatif affiché" in text
    assert "Nouveau" in text
    assert "capital" in text
    assert "Nouveau risque" not in text


def test_text_analysis_keeps_quarters_in_evidence_but_uses_bank_in_narrative() -> None:
    text_data = {
        "bank_code": "bnc",
        "quarter_current": "2026_t4",
        "quarter_previous": "2025_t4",
        "global_summary": {"counts": {"total": 1, "total_relevant": 1, "by_impact": {"MAJEUR": 1}}},
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "change_id": "risk_t4",
                        "diff_type": "modified",
                        "source_text_t1": "Ancien libellé.",
                        "source_text_t2": "Nouveau libellé.",
                        "evidence_t1": {"pages": [18]},
                        "evidence_t2": {"pages": [24]},
                        "change_summary": "Le T2 précise une information qui était implicite au T1.",
                        "genai_triage": {
                            "is_relevant": True,
                            "themes_amf": ["MODIFICATION_TEXTE_RISQUE"],
                            "impact_level": "MAJEUR",
                            "action_requise": "revue_prioritaire",
                            "nouvelle_idee": True,
                            "nouvelle_idee_justification": (
                                "OUI — Nouvel élément à surveiller : Oui.\n\n"
                                "Sujet détecté : Gestion des risques.\n\n"
                                "Ce qui change : Le T2 ajoute une précision absente du T1.\n\n"
                                "Pertinence métier : La mention au T2 modifie la lecture du risque.\n\n"
                                "Point de surveillance : Comparer la posture entre T1 et T2."
                            ),
                        },
                    }
                ],
            }
        ],
    }

    view = build_text_analysis_tab(text_data)
    rendered = _flat_text(view)

    assert "BNC · T4 2026 vs T4 2025" in rendered
    assert "Courant - T4 2026" in rendered
    assert "Précédent - T4 2025" in rendered
    assert "BNC ajoute une précision absente du rapport précédent" in rendered
    assert "Comparer la posture entre rapport précédent et rapport courant" in rendered
    assert "Le T2 ajoute une précision absente du T1" not in rendered
    assert "Le T4 2026 ajoute" not in rendered
