from __future__ import annotations

import io

from openpyxl import load_workbook

from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel


def test_generate_text_comparison_excel_creates_synthese_and_expert_sheets() -> None:
    payload = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "block_comparisons": [
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Ancien texte",
                        "semantic_text_t2": "Nouveau texte majeur",
                        "evidence_t2": {"pages": [12], "snippet": "preuve"},
                        "genai_triage": {
                            "is_relevant": True,
                            "impact_level": "MAJEUR",
                            "action_requise": "escalade",
                            "explanation": "Explication majeure",
                            "impact_description": "",
                            "nouvelle_idee": True,
                        },
                    }
                ],
                "expert_block_comparisons": [
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Ancien texte",
                        "semantic_text_t2": "Nouveau texte majeur",
                        "evidence_t2": {"pages": [12], "snippet": "preuve"},
                        "genai_triage": {
                            "is_relevant": True,
                            "impact_level": "MAJEUR",
                            "action_requise": "escalade",
                            "explanation": "Explication majeure",
                            "impact_description": "",
                            "nouvelle_idee": True,
                        },
                    },
                    {
                        "diff_type": "added",
                        "semantic_text_t1": "",
                        "semantic_text_t2": "Texte modere expert",
                        "evidence_t2": {"pages": [13], "snippet": "preuve moderee"},
                        "genai_triage": {
                            "is_relevant": True,
                            "impact_level": "MODERE",
                            "action_requise": "investigation",
                            "explanation": "Explication moderee",
                            "impact_description": "",
                            "nouvelle_idee": False,
                        },
                    },
                ],
            }
        ]
    }

    raw = generate_text_comparison_excel(payload, output_path=None)
    workbook = load_workbook(io.BytesIO(raw))

    assert workbook.sheetnames == ["Synthese", "Expert"]
    assert workbook["Synthese"].max_row == 2
    assert workbook["Expert"].max_row == 3
    assert workbook["Expert"]["C2"].value == "MAJEUR"
    assert workbook["Expert"]["C3"].value == "MODERE"
