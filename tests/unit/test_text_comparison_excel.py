from __future__ import annotations

import io

from openpyxl import load_workbook

from vigilance.text_comparison.justification import build_text_triage_justification
from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel


def test_generate_text_comparison_excel_creates_analysis_sheet() -> None:
    justification_oui = (
        "OUI - le nouveau modele AIRB est introduit au T2 et n'apparaissait "
        "pas au T1. Cette methode change la facon dont la banque decrit "
        "l'evaluation du risque de credit.\n\n"
        "Cette nouveaute est pertinente pour la vigie parce qu'elle touche "
        "une methodologie prudentielle et les exigences BSIF, pas seulement "
        "une reformulation.\n\n"
        "L'analyste doit comparer cette nouvelle base methodologique avec "
        "celle du trimestre precedent et verifier l'impact sur la comparabilite "
        "inter-pairs."
    )
    justification_non = (
        "NON - la valeur du ratio change mais l'indicateur existait deja au "
        "T1. Le T2 ne cree pas de nouveau concept ni de nouvelle methode.\n\n"
        "Cette variation chiffree est propre a la banque et ne touche pas un "
        "seuil reglementaire AMF ou BSIF. Elle ne constitue donc pas une "
        "nouvelle idee de vigie.\n\n"
        "L'analyste peut l'ecarter comme mise a jour quantitative attendue, "
        "sauf si un seuil interne distinct declenche une revue separee."
    )

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
                        "evidence_t1": {"pages": [10], "snippet": "preuve T1"},
                        "evidence_t2": {"pages": [12], "snippet": "preuve"},
                        "genai_triage": {
                            "is_relevant": True,
                            "category": "RISQUE",
                            "impact_level": "MAJEUR",
                            "action_requise": "revue_prioritaire",
                            "nouvelle_idee": True,
                            "nouvelle_idee_justification": justification_oui,
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
                            "category": "NON_PERTINENT",
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                            "nouvelle_idee_justification": "",
                        },
                    },
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Texte modéré",
                        "semantic_text_t2": "Texte modéré substantif",
                        "source_text_t1": "Paragraphe modéré T1",
                        "source_text_t2": "Paragraphe modéré T2",
                        "evidence_t1": {"pages": [14], "snippet": "preuve moderee T1"},
                        "evidence_t2": {"pages": [15], "snippet": "preuve moderee T2"},
                        "genai_triage": {
                            "is_relevant": False,
                            "category": "NON_PERTINENT",
                            "impact_level": "MODERE",
                            "action_requise": "information",
                            "nouvelle_idee": False,
                            "nouvelle_idee_justification": "",
                        },
                    },
                    {
                        "diff_type": "modified",
                        "semantic_text_t1": "Texte non substantif T1",
                        "semantic_text_t2": "Texte non substantif T2",
                        "source_text_t1": "Paragraphe non substantif T1",
                        "source_text_t2": "Paragraphe non substantif T2",
                        "evidence_t1": {"pages": [16], "snippet": "preuve non substantif T1"},
                        "evidence_t2": {"pages": [17], "snippet": "preuve non substantif T2"},
                        "genai_triage": {
                            "is_relevant": False,
                            "category": "NON_PERTINENT",
                            "impact_level": "MINEUR",
                            "action_requise": "aucune",
                            "nouvelle_idee": False,
                            "nouvelle_idee_justification": "",
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
    # La colonne 9 contient maintenant nouvelle_idee_justification
    assert ws["I2"].value == justification_oui
    assert "\n\nCette nouveaute est pertinente" in ws["I2"].value
    assert ws["F3"].value == "Paragraphe modéré T1"
    assert ws["G3"].value == "Paragraphe modéré T2"
    assert ws["C4"].value is None
    assert ws["D4"].value == "13"
    assert ws["F4"].value is None
    assert ws["G4"].value == "Paragraphe exact ajouté"
    assert ws["F5"].value == "Paragraphe non substantif T1"
    assert ws["G5"].value == "Paragraphe non substantif T2"


def test_generate_text_comparison_excel_strips_control_characters() -> None:
    payload = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "diff_type": "modified",
                        "source_text_t1": "Texte\x00 T1",
                        "source_text_t2": "Texte T2",
                        "evidence_t1": {"pages": [1]},
                        "evidence_t2": {"pages": [2]},
                        "genai_triage": {
                            "is_relevant": True,
                            "category": "RISQUE",
                            "impact_level": "MAJEUR",
                            "action_requise": "revue_prioritaire",
                            "nouvelle_idee": True,
                            "nouvelle_idee_justification": "OUI - note\x00 analyste utile.",
                        },
                    }
                ],
            }
        ]
    }

    raw = generate_text_comparison_excel(payload, output_path=None)
    workbook = load_workbook(io.BytesIO(raw))
    ws = workbook["Analyse complète"]

    assert ws["F2"].value == "Texte T1"
    assert ws["I2"].value == "OUI - note analyste utile."


def test_text_justification_falls_back_for_legacy_b15_triage() -> None:
    change = {
        "diff_type": "added",
        "change_summary": (
            "Sous-section ajoutee: Faits nouveaux sur la reglementation en "
            "matiere de durabilite"
        ),
        "source_text_t1": "",
        "source_text_t2": (
            "En mars 2025, le BSIF a publie une mise a jour de la ligne "
            "directrice B-15 afin de l'harmoniser avec les exigences des "
            "normes du Conseil canadien des normes d'information sur la "
            "durabilite. Les principales modifications comprennent le report "
            "de la date de mise en oeuvre pour la communication d'informations "
            "sur les emissions de GES du champ d'application 3."
        ),
        "evidence_t2": {"pages": [43]},
        "genai_triage": {
            "is_relevant": True,
            "category": "REGLEMENTAIRE",
            "impact_level": "MODERE",
            "risk_type": "conformite",
            "action_requise": "confirmation",
            "nouvelle_idee": True,
            "explanation": (
                "Une nouvelle section a ete ajoutee concernant la mise a jour "
                "des lignes directrices pour s'aligner sur les normes de "
                "durabilite."
            ),
            "impact_description": (
                "Le report des exigences de communication sur les emissions "
                "de GES permet a la banque de mieux se preparer"
            ),
        },
    }

    justification = build_text_triage_justification(change)

    assert justification.startswith("OUI — Nouvel élément à surveiller : Oui.")
    assert "Sujet détecté : Risque climatique, ESG" in justification
    assert "nouvelle mention réglementaire" in justification
    assert "information ajoutée" in justification
    assert "Ce qui change : Le T2 ajoute" in justification
    assert "Pertinence métier :" in justification
    assert (
        "Ce changement met l'accent sur l'évolution du cadre réglementaire "
        "applicable aux divulgations climatiques des institutions financières."
    ) in justification
    assert (
        "Ce point est important à suivre, car il permet d'évaluer l'évolution "
        "des attentes prudentielles, la comparabilité des pratiques de divulgation "
        "entre pairs et le niveau de préparation des banques face aux exigences "
        "climatiques."
    ) in justification
    assert "Point de surveillance :" in justification

    raw = generate_text_comparison_excel(
        {
            "section_comparisons": [
                {
                    "section_key": "gestion_reglementation",
                    "section_title": "Faits nouveaux en matiere de reglementation",
                    "all_block_comparisons": [change],
                }
            ]
        },
        output_path=None,
    )
    workbook = load_workbook(io.BytesIO(raw))
    ws = workbook["Analyse complète"]
    assert ws["H2"].value == "Oui"
    assert ws["I2"].value == justification


def test_text_justification_rewrites_b15_pertinence_when_explicit_exists() -> None:
    change = {
        "diff_type": "added",
        "source_text_t2": "Le BSIF met à jour la ligne directrice B-15 sur les GES.",
        "genai_triage": {
            "nouvelle_idee": True,
            "nouvelle_idee_justification": (
                "OUI — Nouvel élément à surveiller : Oui.\n\n"
                "Sujet détecté : Risque climatique, ESG.\n\n"
                "Ce qui change : Le T2 ajoute une mention B-15.\n\n"
                "Pertinence métier : Ancienne formulation à remplacer.\n\n"
                "Point de surveillance : Le point à retenir demeure le suivi des exigences."
            ),
        },
    }

    justification = build_text_triage_justification(change)

    assert "Ancienne formulation à remplacer" not in justification
    assert (
        "Pertinence métier : Ce changement met l'accent sur l'évolution du cadre "
        "réglementaire applicable aux divulgations climatiques des institutions "
        "financières."
    ) in justification
    assert (
        "Point de surveillance : Risque climatique / ESG — Le changement indique "
        "que la banque tient compte de la mise à jour de la ligne directrice B-15"
    ) in justification


def test_generate_text_comparison_excel_labels_text_renames() -> None:
    payload = {
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "section_title": "Gestion des risques",
                "all_block_comparisons": [
                    {
                        "diff_type": "renamed",
                        "subsection_heading": "Ancien titre → Nouveau titre",
                        "source_text_t1": "Ancien titre",
                        "source_text_t2": "Nouveau titre",
                        "evidence_t1": {"pages": [4]},
                        "evidence_t2": {"pages": [5]},
                        "genai_triage": {
                            "is_relevant": True,
                            "category": "STRUCTURE",
                            "impact_level": "MODERE",
                            "nouvelle_idee": False,
                            "nouvelle_idee_justification": "Renommage de sous-section.",
                        },
                    }
                ],
            }
        ]
    }

    raw = generate_text_comparison_excel(payload, output_path=None)
    workbook = load_workbook(io.BytesIO(raw))
    ws = workbook["Analyse complète"]

    assert ws["B2"].value == "Ancien titre → Nouveau titre"
    assert ws["E2"].value == "Renommage"
    assert ws["F2"].value == "Ancien titre"
    assert ws["G2"].value == "Nouveau titre"
