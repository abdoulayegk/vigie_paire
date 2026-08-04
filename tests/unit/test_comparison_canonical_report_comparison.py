from __future__ import annotations

from copy import deepcopy
import logging

from vigie.comparaison import canonical as cc
from vigie.interface.review_adapters import build_review_items_from_indicator_result
from vigie.interface.review_queue_normalizer import build_normalized_review_queue


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
        "model_version": "gpt-5.4",
        "prompt_version_match": "table_match_v8",
        "prompt_version_diff": "table_diff_v4",
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
                    "indicators": ["LCR"],
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
                    "indicators": ["Ratio CET1"],
                    "footnotes": [{"id": "1", "text": "Note A"}],
                },
                "current_table": {
                    "table_id": "curr_1",
                    "title": "Capital reglementaire",
                    "page": 10,
                    "section": "capital",
                    "bbox": [0.2, 0.2, 0.85, 0.72],
                    "indicators": ["Ratio CET1", "Ratio de levier"],
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
        "run_metrics": {
            "runtime_extraction_sec": 12.5,
            "runtime_comparison_sec": 1.4,
            "vision_calls_total": 18,
            "vision_rescue_total": 2,
            "comparison_calls_total": 2,
            "prompt_tokens_total": 4200,
            "completion_tokens_total": 900,
            "total_tokens_total": 5100,
            "estimated_cost_usd": 0.123,
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
    assert canonical["meta"]["model_version"] == "gpt-5.4"
    assert canonical["meta"]["prompt_version_match"] == "table_match_v8"
    assert canonical["meta"]["prompt_version_diff"] == "table_diff_v4"
    assert canonical["meta"]["pdf_paths"]["pdf_previous"] == "/archive/run/previous_report.pdf"
    assert canonical["meta"]["pdf_paths"]["pdf_current"] == "/archive/run/current_report.pdf"
    assert canonical["meta"]["run_metrics"]["vision_calls_total"] == 18

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
    assert added_table["first_column_indicators"] == ["LCR"]
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


def test_to_canonical_payload_warns_when_visual_context_missing(caplog) -> None:
    raw = _raw_report_comparison()
    raw["pair_comparisons"][0]["previous_table"]["bbox"] = None
    raw["pair_comparisons"][0]["current_table"]["page"] = None

    with caplog.at_level(logging.WARNING):
        canonical = cc.to_canonical_payload(raw)

    assert canonical["table_comparisons"][0]["bbox_t1"] is None
    assert canonical["table_comparisons"][0]["page_t2"] is None
    assert "missing_visual_context" in caplog.text
    assert "prev_1" in caplog.text
    assert "curr_1" in caplog.text


def test_report_comparison_bbox_survives_review_queue_normalization() -> None:
    raw = _raw_report_comparison()
    canonical = cc.to_canonical_payload(raw)

    items = build_review_items_from_indicator_result(
        canonical,
        bank_code="bnc",
        quarter_from=canonical["quarter_from"],
        quarter_to=canonical["quarter_to"],
        pdf_path_t1="/tmp/prev.pdf",
        pdf_path_t2="/tmp/curr.pdf",
    )
    queue = build_normalized_review_queue(
        canonical,
        [item.to_dict() for item in items],
        "/tmp/prev.pdf",
        "/tmp/curr.pdf",
    )

    matched_table = next(
        table for table in queue if table.table_id_t1 == "prev_1" and table.table_id_t2 == "curr_1"
    )
    assert matched_table.bbox_t1 == [0.1, 0.2, 0.8, 0.7]
    assert matched_table.bbox_t2 == [0.2, 0.2, 0.85, 0.72]


def test_to_canonical_payload_separates_artifacts_and_extraction_suspects() -> None:
    raw = _raw_report_comparison()
    raw["matching"]["artifacts_confirmed_previous"] = [
        {
            "table_id": "prev_artifact",
            "title": "Rapport de gestion",
            "page": 40,
            "section": "risk_management",
            "bbox": [0.1, 0.1, 0.9, 0.8],
            "indicators": [],
            "reason": "Artefact confirme.",
        }
    ]
    raw["matching"]["artifacts_confirmed_current"] = []
    raw["matching"]["extraction_suspects_previous"] = []
    raw["matching"]["extraction_suspects_current"] = [
        {
            "table_id": "curr_suspect",
            "title": "Rapport de gestion",
            "page": 41,
            "section": "risk_management",
            "bbox": [0.1, 0.1, 0.9, 0.8],
            "indicators": [],
            "reason": "Extraction suspecte.",
        }
    ]
    raw["summary"]["artifacts_confirmed_previous_total"] = 1
    raw["summary"]["artifacts_confirmed_current_total"] = 0
    raw["summary"]["extraction_suspects_previous_total"] = 0
    raw["summary"]["extraction_suspects_current_total"] = 1

    canonical = cc.to_canonical_payload(raw)

    assert canonical["summary"]["artifacts_confirmed_previous"] == 1
    assert canonical["summary"]["extraction_suspects_current"] == 1
    assert canonical["summary"]["tables_added"] == 1
    assert canonical["summary"]["tables_added_pending_review"] == 1
    assert canonical["summary"]["review_candidates"] == 1
    assert canonical["artifacts_confirmed_previous"][0]["table_id"] == "prev_artifact"
    assert canonical["extraction_suspects_current"][0]["table_id"] == "curr_suspect"
    assert canonical["tables_added_pending_review"][0]["table_id"] == "curr_suspect"


def test_review_queue_excludes_extraction_suspects_from_visible_review() -> None:
    raw = _raw_report_comparison()
    raw["matching"]["tables_added"] = []
    raw["matching"]["artifacts_confirmed_previous"] = []
    raw["matching"]["artifacts_confirmed_current"] = []
    raw["matching"]["extraction_suspects_previous"] = []
    raw["matching"]["extraction_suspects_current"] = [
        {
            "table_id": "curr_suspect",
            "title": "Rapport de gestion",
            "page": 41,
            "section": "risk_management",
            "bbox": [0.1, 0.1, 0.9, 0.8],
            "indicators": [],
            "extraction_status": "suspect_unresolved",
            "reason": "Extraction suspecte.",
        }
    ]
    raw["summary"]["tables_added_total"] = 0
    raw["summary"]["artifacts_confirmed_previous_total"] = 0
    raw["summary"]["artifacts_confirmed_current_total"] = 0
    raw["summary"]["extraction_suspects_previous_total"] = 0
    raw["summary"]["extraction_suspects_current_total"] = 1

    canonical = cc.to_canonical_payload(raw)
    items = build_review_items_from_indicator_result(
        canonical,
        bank_code="bnc",
        quarter_from=canonical["quarter_from"],
        quarter_to=canonical["quarter_to"],
        pdf_path_t1="/tmp/prev.pdf",
        pdf_path_t2="/tmp/curr.pdf",
    )
    queue = build_normalized_review_queue(
        canonical,
        [item.to_dict() for item in items],
        "/tmp/prev.pdf",
        "/tmp/curr.pdf",
    )

    assert all(table.table_id_t2 != "curr_suspect" for table in queue)
