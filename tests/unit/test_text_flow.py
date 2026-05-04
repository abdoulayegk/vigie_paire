from __future__ import annotations

from typing import Any

from vigilance.dash_app.callbacks.text_flow import download_text_excel, filter_text_cards


def test_download_text_excel_reload_latest_payload_before_export(monkeypatch) -> None:
    stale_payload = {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
        "section_comparisons": [
            {"block_comparisons": [{"change_id": "strict"}], "all_block_comparisons": []}
        ],
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


def test_filter_text_cards_sorts_new_idea_first_and_keeps_minor_cosmetic() -> None:
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
                            "category": "RISQUE",
                            "impact_level": "MAJEUR",
                            "action_requise": "escalade",
                            "nouvelle_idee": False,
                        },
                    },
                    {
                        "change_id": "new_moderate",
                        "diff_type": "added",
                        "semantic_text_t2": "Nouvelle idée",
                        "pages_t2": [9],
                        "evidence_t2": {"pages": [9], "snippet": "preuve 2"},
                        "genai_triage": {
                            "category": "STRUCTURE",
                            "impact_level": "MODERE",
                            "action_requise": "information",
                            "nouvelle_idee": True,
                        },
                    },
                    {
                        "change_id": "cosmetic",
                        "diff_type": "modified",
                        "semantic_text_t2": "Cosmétique",
                        "pages_t2": [2],
                        "evidence_t2": {"pages": [2], "snippet": "preuve 3"},
                        "genai_triage": {
                            "category": "COSMETIQUE",
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                        },
                    },
                ],
            }
        ]
    }

    cards, count_text = filter_text_cards(text_data, None, None, None)

    assert count_text == "3 changement(s) affiché(s)"
    assert len(cards) == 3
    first_phrase = cards[0].children.children[2].children
    second_phrase = cards[1].children.children[2].children
    third_phrase = cards[2].children.children[2].children
    assert first_phrase == "Nouvelle idée"
    assert second_phrase == "Majeur existant"
    assert third_phrase == "Cosmétique"
