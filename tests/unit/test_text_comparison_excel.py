from __future__ import annotations

import io

from openpyxl import load_workbook

from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel


def test_generate_text_comparison_excel_creates_synthese_and_all_changes_sheets() -> None:
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

    assert workbook.sheetnames == ["Synthese", "Tous_les_changements"]
    assert workbook["Synthese"].max_row == 4
    assert workbook["Tous_les_changements"].max_row == 5
    assert workbook["Synthese"]["D2"].value == "Paragraphe exact T1"
    assert workbook["Synthese"]["E2"].value == "Paragraphe exact T2"
    assert workbook["Synthese"]["D3"].value == "Paragraphe modéré T1"
    assert workbook["Synthese"]["E3"].value == "Paragraphe modéré T2"
    assert workbook["Synthese"]["D4"].value is None
    assert workbook["Synthese"]["E4"].value == "Paragraphe exact ajouté"
    assert workbook["Tous_les_changements"]["B2"].value == "10"
    assert workbook["Tous_les_changements"]["C2"].value == "12"
    assert workbook["Tous_les_changements"]["D2"].value == "Paragraphe exact T1"
    assert workbook["Tous_les_changements"]["E2"].value == "Paragraphe exact T2"
    assert workbook["Tous_les_changements"]["F2"].value == "modified"
    assert workbook["Tous_les_changements"]["H2"].value.startswith("Priorité: MAJEUR")
    assert workbook["Tous_les_changements"]["D3"].value == "Paragraphe modéré T1"
    assert workbook["Tous_les_changements"]["E3"].value == "Paragraphe modéré T2"
    assert workbook["Tous_les_changements"]["D4"].value is None
    assert workbook["Tous_les_changements"]["E4"].value == "Paragraphe exact ajouté"
    assert workbook["Tous_les_changements"]["D5"].value == "Paragraphe cosmétique T1"
    assert workbook["Tous_les_changements"]["E5"].value == "Paragraphe cosmétique T2"
