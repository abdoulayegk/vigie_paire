from __future__ import annotations

import io

from openpyxl import load_workbook

from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel


def test_generate_text_comparison_excel_creates_analysis_sheet() -> None:
    payload = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "block_comparisons": [],
                "all_block_comparisons": [
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Ancien texte",
                        "semantic_text_t2": "Nouveau texte majeur",
                        "source_text_t1": "Paragraphe exact T1",
                        "source_text_t2": "Paragraphe exact T2",
                        "evidence_t1": {"pages": [10], "snippet": "preuve t1"},
                        "evidence_t2": {"pages": [12], "snippet": "preuve"},
                        "genai_triage": {
                            "is_relevant": True,
                            "category": "RISQUE",
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
                        "source_text_t1": "",
                        "source_text_t2": "Paragraphe exact ajouté",
                        "evidence_t2": {"pages": [13], "snippet": "preuve moderee"},
                        "genai_triage": {
                            "is_relevant": False,
                            "category": "STRUCTURE",
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "explanation": "Explication mineure",
                            "impact_description": "",
                            "nouvelle_idee": False,
                        },
                    },
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Texte modéré",
                        "semantic_text_t2": "Texte modéré non cosmétique",
                        "source_text_t1": "Paragraphe modéré T1",
                        "source_text_t2": "Paragraphe modéré T2",
                        "evidence_t1": {"pages": [14], "snippet": "preuve moderee t1"},
                        "evidence_t2": {"pages": [15], "snippet": "preuve moderee t2"},
                        "genai_triage": {
                            "is_relevant": False,
                            "category": "STRUCTURE",
                            "impact_level": "MODERE",
                            "action_requise": "information",
                            "explanation": "Explication modérée",
                            "impact_description": "",
                            "nouvelle_idee": False,
                        },
                    },
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Texte cosmétique T1",
                        "semantic_text_t2": "Texte cosmétique T2",
                        "source_text_t1": "Paragraphe cosmétique T1",
                        "source_text_t2": "Paragraphe cosmétique T2",
                        "evidence_t1": {"pages": [16], "snippet": "preuve cosmetique t1"},
                        "evidence_t2": {"pages": [17], "snippet": "preuve cosmetique t2"},
                        "genai_triage": {
                            "is_relevant": False,
                            "category": "COSMETIQUE",
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "explanation": "Explication cosmétique",
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

    assert workbook.sheetnames == ["Analyse complète"]
    ws = workbook["Analyse complète"]
    assert ws.max_row == 5
    assert ws["A2"].value == "Gestion des risques"
    assert ws["C2"].value == "10"
    assert ws["D2"].value == "12"
    assert ws["E2"].value == "Modification"
    assert ws["F2"].value == "Paragraphe exact T1"
    assert ws["G2"].value == "Paragraphe exact T2"
    assert ws["H2"].value == "Oui"
    assert ws["I2"].value == "Explication majeure"
    assert ws["F3"].value == "Paragraphe modéré T1"
    assert ws["G3"].value == "Paragraphe modéré T2"
    assert ws["C4"].value is None
    assert ws["D4"].value == "13"
    assert ws["F4"].value is None
    assert ws["G4"].value == "Paragraphe exact ajouté"
    assert ws["F5"].value == "Paragraphe cosmétique T1"
    assert ws["G5"].value == "Paragraphe cosmétique T2"
