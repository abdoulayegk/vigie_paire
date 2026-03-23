from __future__ import annotations

from copy import deepcopy

from app import comparison_canonical as cc


def _raw_report_comparison() -> dict:
    return {
        "artifact_type": "report_comparison",
        "run_id": "20260323_143015",
        "bank_code": "bnc",
        "year_previous": 2025,
        "quarter_previous": "t3",
        "year_current": 2026,
        "quarter_current": "t1",
        "source_pdf_previous": "/tmp/prev.pdf",
        "source_pdf_current": "/tmp/curr.pdf",
        "archived_pdf_previous": "/archive/run/previous_report.pdf",
        "archived_pdf_current": "/archive/run/current_report.pdf",
        "reference_resolution": {
            "mode": "automatique",
            "year_previous": 2025,
            "quarter_previous": "t3",
            "rule": "t1->t3 annee precedente",
        },
        "matching": {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 0.96,
                    "reason": "Meme concept",
                }
            ],
            "tables_added": [
                {
                    "table_id": "curr_2",
                    "title": "Liquidite",
                    "page": 14,
                    "section": "liquidite",
                    "bbox": [0.1, 0.1, 0.9, 0.6],
                    "indicators_raw": ["LCR"],
                    "indicators_normalized": ["lcr"],
                    "footnotes": [],
                    "reason": "Nouveau tableau",
                    "analyst_assessment": {
                        "theme": "liquidite",
                        "change_significance": "eleve",
                        "review_priority": "critique",
                        "analyst_summary": "Nouveau tableau de liquidite.",
                    },
                }
            ],
            "tables_removed": [],
        },
        "pair_comparisons": [
            {
                "previous_table_id": "prev_1",
                "current_table_id": "curr_1",
                "match_confidence": 0.96,
                "match_reason": "Meme concept",
                "previous_table": {
                    "table_id": "prev_1",
                    "title": "Capital reglementaire",
                    "page": 8,
                    "section": "capital",
                    "bbox": [0.1, 0.2, 0.8, 0.7],
                    "indicators_raw": ["Ratio CET1"],
                    "indicators_normalized": ["ratio cet1"],
                    "footnotes": [{"id": "1", "text": "Note A"}],
                },
                "current_table": {
                    "table_id": "curr_1",
                    "title": "Capital reglementaire",
                    "page": 10,
                    "section": "capital",
                    "bbox": [0.2, 0.2, 0.85, 0.72],
                    "indicators_raw": ["Ratio CET1", "Ratio de levier"],
                    "indicators_normalized": ["ratio cet1", "ratio de levier"],
                    "footnotes": [{"id": "1", "text": "Note A maj"}],
                },
                "technical_diff": {
                    "indicators_added": [{"value": "ratio de levier", "reason": "Ajout"}],
                    "indicators_removed": [],
                    "indicators_renamed": [],
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [
                        {
                            "previous_id": "1",
                            "current_id": "1",
                            "previous_text": "Note A",
                            "current_text": "Note A maj",
                            "reason": "Meme note",
                        }
                    ],
                    "table_level_change": "modifie",
                },
                "analyst_assessment": {
                    "theme": "capital",
                    "change_significance": "eleve",
                    "review_priority": "prioritaire",
                    "analyst_summary": "Ajout d'un indicateur capital.",
                },
                "reason": "Difference semantique.",
            }
        ],
        "summary": {
            "matched_pairs_total": 1,
            "tables_added_total": 1,
            "tables_removed_total": 0,
            "indicator_changes_total": 1,
            "footnote_changes_total": 1,
            "high_priority_items_total": 2,
        },
    }


def test_to_canonical_payload_supports_report_comparison_without_extraction_lookup() -> None:
    raw = _raw_report_comparison()

    canonical = cc.to_canonical_payload(raw)

    assert canonical["schema_version"] == cc.UI_COMPARISON_PAYLOAD_SCHEMA_VERSION
    assert canonical["quarter_from"] == "Q3-2025"
    assert canonical["quarter_to"] == "Q1-2026"
    assert canonical["summary"]["tables_t1"] == 1
    assert canonical["summary"]["tables_t2"] == 2
    assert canonical["summary"]["tables_matched"] == 1
    assert canonical["summary"]["tables_added"] == 1
    assert canonical["summary"]["total_added_indicators"] == 1
    assert canonical["summary"]["footnote_change_pairs"] == 1
    assert canonical["meta"]["source_format"] == "report_comparison"
    assert canonical["meta"]["reference_resolution"]["quarter_previous"] == "t3"
    assert canonical["meta"]["run_id"] == "20260323_143015"
    assert canonical["meta"]["pdf_paths"]["pdf_previous"] == "/archive/run/previous_report.pdf"
    assert canonical["meta"]["pdf_paths"]["pdf_current"] == "/archive/run/current_report.pdf"

    comp = canonical["table_comparisons"][0]
    assert comp["table_id_t1"] == "prev_1"
    assert comp["table_id_t2"] == "curr_1"
    assert comp["title_t1"] == "Capital reglementaire"
    assert comp["page_t2"] == 10
    assert comp["bbox_t1"] == [0.1, 0.2, 0.8, 0.7]
    assert comp["source_pdf_t1"] == "/archive/run/previous_report.pdf"
    assert comp["source_pdf_t2"] == "/archive/run/current_report.pdf"
    assert comp["added_indicators"] == ["ratio de levier"]
    assert comp["all_indicators_t1"] == ["Ratio CET1"]
    assert comp["footnotes_counts"]["modified"] == 1
    assert comp["genai_analysis"]["theme"] == "capital"

    added_table = canonical["tables_added"][0]
    assert added_table["table_id"] == "curr_2"
    assert added_table["title"] == "Liquidite"
    assert added_table["first_column_indicators_raw"] == ["LCR"]
    assert added_table["bbox_t2"] == [0.1, 0.1, 0.9, 0.6]
    assert added_table["source_pdf_t2"] == "/archive/run/current_report.pdf"


def test_report_comparison_conversion_is_stable_without_extraction_files() -> None:
    raw = _raw_report_comparison()

    first = cc.to_canonical_payload(raw)
    mutated = deepcopy(raw)
    mutated["matching"]["tables_added"][0]["title"] = "Liquidite MAJ"
    second = cc.to_canonical_payload(raw)

    assert first == second
    assert second["tables_added"][0]["title"] == "Liquidite"
    assert mutated["matching"]["tables_added"][0]["title"] == "Liquidite MAJ"
