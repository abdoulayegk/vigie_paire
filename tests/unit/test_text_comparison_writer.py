from __future__ import annotations

import json
from pathlib import Path

from vigie.analyse_texte.text_comparison.text_comparison_writer import write_text_comparison


def test_write_text_comparison_normalizes_legacy_justification_and_segments(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 3,
        "artifact_type": "text_comparison",
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "section_comparisons": [
            {
                "section_key": "gestion_capital",
                "section_title": "Gestion du capital",
                "block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "diff_type": "removed",
                        "source_text_t1": "Ancien passage réglementaire détaillé.",
                        "source_text_t2": "",
                        "genai_triage": {
                            "is_relevant": True,
                            "themes_amf": ["DIVULGATION_RETRAIT"],
                            "impact_level": "MAJEUR",
                            "nouvelle_idee": True,
                            "action_requise": "revue_prioritaire",
                            "nouvelle_idee_justification": ("OUI - retrait important à suivre par l'analyste."),
                        },
                    }
                ],
                "all_block_comparisons": [],
            }
        ],
    }

    out_path = tmp_path / "text_comparison.json"

    write_text_comparison(payload, out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    triage = written["section_comparisons"][0]["block_comparisons"][0]["genai_triage"]
    justification = triage["nouvelle_idee_justification"]

    assert justification.startswith("OUI — Nouvel élément à surveiller : Oui.")
    assert "Sujet détecté :" in justification
    assert "Ce qui change :" in justification
    assert "Pertinence métier : retrait important" in justification
    assert "Point de surveillance :" in justification
    assert triage["change_segments"] == [
        {
            "kind": "removed",
            "text_t1": "Ancien passage réglementaire détaillé.",
            "text_t2": "",
        }
    ]


def test_write_text_comparison_clears_non_relevant_segments(tmp_path: Path) -> None:
    payload = {
        "schema_version": 3,
        "artifact_type": "text_comparison",
        "bank_code": "bnc",
        "quarter_current": "2025_t2",
        "quarter_previous": "2025_t1",
        "section_comparisons": [
            {
                "section_key": "gestion_capital",
                "section_title": "Gestion du capital",
                "block_comparisons": [
                    {
                        "change_id": "chg-2",
                        "diff_type": "modified",
                        "source_text_t1": "100 %",
                        "source_text_t2": "101 %",
                        "genai_triage": {
                            "is_relevant": False,
                            "themes_amf": [],
                            "impact_level": "MINEUR",
                            "nouvelle_idee": False,
                            "action_requise": "aucune",
                            "nouvelle_idee_justification": ("NON - mise à jour quantitative seulement."),
                            "change_segments": [{"kind": "modified", "text_t1": "100 %", "text_t2": "101 %"}],
                        },
                    }
                ],
                "all_block_comparisons": [],
            }
        ],
    }

    out_path = tmp_path / "text_comparison.json"

    write_text_comparison(payload, out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    triage = written["section_comparisons"][0]["block_comparisons"][0]["genai_triage"]

    assert triage["change_segments"] == []
    assert "Ce qui change :" in triage["nouvelle_idee_justification"]
