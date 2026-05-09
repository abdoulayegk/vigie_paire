from __future__ import annotations

from typing import Any

from vigilance.dash_app.callbacks.text_flow import download_text_excel, filter_text_cards


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
    assert response["filename"] == "veille_textuelle_TD_2026t1.xlsx"


def test_filter_text_cards_sorts_new_idea_first_and_keeps_non_pertinent() -> None:
    """Le filtrage Dash garde les changements non pertinents pour revue humaine.

    Tri : nouvelle idée d'abord, puis impact décroissant. Les is_relevant=False
    restent visibles afin que l'analyste puisse contester le triage.
    """
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

    assert count_text == "3 changement(s) affiché(s)"
    assert len(cards) == 3
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
    third_text = _flat_text(cards[2])
    # Tri : nouvelle idée d'abord, puis impact décroissant
    assert "Nouvelle idée" in first_text  # phrase added present in T2 column
    assert "Majeur existant" in second_text  # phrase modified present in T2 column
    assert "Variation chiffree" in third_text
    assert "Non pertinent" in third_text
