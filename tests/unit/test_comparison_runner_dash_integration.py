from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import comparison_runner as cr


def _write_tables_json(path: Path, *, bank: str, year: int, quarter: str, table_id: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 7,
                "bank_code": bank,
                "year": year,
                "quarter": quarter,
                "created_at": "2026-03-23T10:00:00",
                "tables": [
                    {
                        "table_id": table_id,
                        "page": 5,
                        "section": "capital",
                        "title": title,
                        "table_summary": title,
                        "bbox": [0.1, 0.2, 0.8, 0.7],
                        "row_count": 1,
                        "headers": ["Indicateur"],
                        "indicators": ["Ratio CET1"],
                        "footnotes": [{"id": "1", "text": "Note A"}],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_comparison_with_sections_returns_dash_canonical(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extraction_root = tmp_path / "outputs" / "extractions"
    comparison_root = tmp_path / "outputs" / "comparisons"
    monkeypatch.setattr(cr, "EXTRACTION_ROOT", extraction_root)
    monkeypatch.setattr(cr, "COMPARISON_ROOT", comparison_root)

    def fake_extract_tables(**kwargs):
        quarter = kwargs["quarter"]
        year = kwargs["year"]
        bank = kwargs["bank_code"]
        out_dir = extraction_root / bank / str(year) / quarter
        table_id = "prev_1" if quarter == "t3" else "curr_1"
        title = "Capital precedent" if quarter == "t3" else "Capital courant"
        _write_tables_json(
            out_dir / "tables.json",
            bank=bank,
            year=year,
            quarter=quarter,
            table_id=table_id,
            title=title,
        )
        (out_dir / "indicators.json").write_text("{}", encoding="utf-8")
        (out_dir / "footnotes.json").write_text("{}", encoding="utf-8")
        provenance = {
            "mode": "fresh",
            "artifact_dir": str(out_dir),
            "tables_path": str(out_dir / "tables.json"),
            "indicators_path": str(out_dir / "indicators.json"),
            "footnotes_path": str(out_dir / "footnotes.json"),
            "artifacts_present": {
                "tables": True,
                "indicators": True,
                "footnotes": True,
            },
            "quarter": quarter,
            "year": year,
        }
        return [], provenance

    def fake_compare_reports_gpt4o(*, previous_dir, current_dir, out_root, **kwargs):
        assert Path(previous_dir) == extraction_root / "bnc" / "2025" / "t3"
        assert Path(current_dir) == extraction_root / "bnc" / "2026" / "t1"
        assert kwargs["source_pdf_previous"].endswith("/prev.pdf")
        assert kwargs["source_pdf_current"].endswith("/curr.pdf")
        assert kwargs["runtime_extraction_sec"] >= 0.0
        assert set(kwargs["extraction_run_metrics"]) == {"previous", "current"}
        out_path = (
            Path(out_root)
            / "bnc"
            / "2026_t1_vs_2025_t3"
            / "20260323_143015"
            / "comparison.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "artifact_type": "report_comparison",
                    "run_id": "20260323_143015",
                    "bank_code": "bnc",
                    "year_previous": 2025,
                    "quarter_previous": "t3",
                    "year_current": 2026,
                    "quarter_current": "t1",
                    "source_pdf_previous": kwargs["source_pdf_previous"],
                    "source_pdf_current": kwargs["source_pdf_current"],
                    "archived_pdf_previous": "/archive/prev.pdf",
                    "archived_pdf_current": "/archive/curr.pdf",
                    "reference_resolution": kwargs["reference_resolution"],
                    "matching": {
                        "matched_pairs": [
                            {
                                "previous_table_id": "prev_1",
                                "current_table_id": "curr_1",
                                "match_confidence": 0.98,
                                "reason": "Meme concept",
                            }
                        ],
                        "tables_added": [],
                        "tables_removed": [],
                    },
                    "pair_comparisons": [
                        {
                            "previous_table_id": "prev_1",
                            "current_table_id": "curr_1",
                            "match_confidence": 0.98,
                            "match_reason": "Meme concept",
                            "previous_table": {
                                "table_id": "prev_1",
                                "title": "Capital precedent",
                                "page": 5,
                                "section": "capital",
                                "bbox": [0.1, 0.2, 0.8, 0.7],
                                "indicators": ["Ratio CET1"],
                                "footnotes": [{"id": "1", "text": "Note A"}],
                            },
                            "current_table": {
                                "table_id": "curr_1",
                                "title": "Capital courant",
                                "page": 5,
                                "section": "capital",
                                "bbox": [0.1, 0.2, 0.8, 0.7],
                                "indicators": ["Ratio CET1"],
                                "footnotes": [{"id": "1", "text": "Note A"}],
                            },
                            "technical_diff": {
                                "indicators_added": [],
                                "indicators_removed": [],
                                "indicators_renamed": [],
                                "footnotes_added": [],
                                "footnotes_removed": [],
                                "footnotes_renamed": [],
                                "table_level_change": "inchange",
                            },
                            "analyst_assessment": {
                                "theme": "capital",
                                "change_significance": "faible",
                                "review_priority": "normale",
                                "analyst_summary": "Aucun changement.",
                            },
                            "reason": "Aucun changement.",
                        }
                    ],
                    "summary": {
                        "matched_pairs_total": 1,
                        "tables_added_total": 0,
                        "tables_removed_total": 0,
                        "indicator_changes_total": 0,
                        "footnote_changes_total": 0,
                        "high_priority_items_total": 0,
                    },
                    "run_metrics": {
                        "runtime_extraction_sec": kwargs["runtime_extraction_sec"],
                        "runtime_comparison_sec": 0.5,
                        "vision_calls_total": 2,
                        "vision_rescue_total": 0,
                        "comparison_calls_total": 1,
                        "comparison_local_diff_skips": 0,
                        "prompt_tokens_total": 220,
                        "completion_tokens_total": 40,
                        "total_tokens_total": 260,
                        "estimated_cost_usd": 0.001,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_path

    monkeypatch.setattr(cr, "_extract_tables", fake_extract_tables)
    monkeypatch.setattr(cr, "compare_reports_gpt4o", fake_compare_reports_gpt4o)

    previous_pdf = tmp_path / "prev.pdf"
    current_pdf = tmp_path / "curr.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    result = cr.run_comparison_with_sections(
        pdf_path_previous=str(previous_pdf),
        pdf_path_current=str(current_pdf),
        bank_code="bnc",
        sections_previous=[{"section": "capital", "start_page": 1, "end_page": 2}],
        sections_current=[{"section": "capital", "start_page": 1, "end_page": 2}],
        current_quarter="Q1-2026",
        current_year=2026,
        api_key="test-key",
    )

    assert result["bank_code"] == "bnc"
    assert result["quarter_from"] == "Q3-2025"
    assert result["quarter_to"] == "Q1-2026"
    assert result["summary"]["tables_matched"] == 1
    assert result["meta"]["source_format"] == "report_comparison"
    assert result["meta"]["compare_path"].endswith(
        "2026_t1_vs_2025_t3/20260323_143015/comparison.json"
    )
    assert result["meta"]["reference_resolution"]["quarter_previous"] == "t3"
    assert result["meta"]["run_id"] == "20260323_143015"
    assert result["meta"]["pdf_paths"]["pdf_previous"] == "/archive/prev.pdf"
    assert result["meta"]["pdf_paths"]["pdf_current"] == "/archive/curr.pdf"
    assert result["meta"]["run_metrics"]["comparison_calls_total"] == 1
    assert result["meta"]["extraction_sources"]["previous"]["tables_path"].endswith(
        "/2025/t3/tables.json"
    )
    assert result["table_comparisons"][0]["table_id_t1"] == "prev_1"
    assert result["table_comparisons"][0]["bbox_t1"] == [0.1, 0.2, 0.8, 0.7]
    assert result["table_comparisons"][0]["bbox_t2"] == [0.1, 0.2, 0.8, 0.7]
    assert result["table_comparisons"][0]["source_pdf_t1"] == "/archive/prev.pdf"
    assert result["table_comparisons"][0]["all_indicators_t1"] == ["Ratio CET1"]
    assert result["table_comparisons"][0]["all_indicators_t2"] == ["Ratio CET1"]


def test_extract_tables_reuses_schema_v7_storage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cr, "EXTRACTION_ROOT", tmp_path / "outputs" / "extractions")

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    out_dir = cr.EXTRACTION_ROOT / "bnc" / "2025" / "t1"
    _write_tables_json(
        out_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        table_id="stored_1",
        title="Capital stocke",
    )
    (out_dir / "indicators.json").write_text("{}", encoding="utf-8")
    (out_dir / "footnotes.json").write_text("{}", encoding="utf-8")

    def should_not_extract(**kwargs):
        raise AssertionError("fresh extraction should not run when schema_version=7 is already stored")

    monkeypatch.setattr(
        "vigilance.extraction.docling_processor.extract_tables_docling_by_sections",
        should_not_extract,
    )

    tables, provenance = cr._extract_tables(
        pdf_path=str(pdf_path),
        bank_code="bnc",
        quarter="t1",
        year=2025,
        sections=[{"section": "capital", "start_page": 1, "end_page": 2}],
        use_vision_extraction=True,
        use_stored_extraction_if_available=True,
        return_provenance=True,
    )

    assert provenance["mode"] == "stored"
    assert provenance["run_metrics"]["cache_hit"] is True
    assert provenance["run_metrics"]["vision_calls_total"] == 0


def test_run_comparison_with_sections_rejects_missing_pdf_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Rapport précédent introuvable"):
        cr.run_comparison_with_sections(
            pdf_path_previous=None,
            pdf_path_current=str(tmp_path / "curr.pdf"),
            bank_code="bnc",
            sections_previous=[{"section": "capital", "start_page": 1, "end_page": 2}],
            sections_current=[{"section": "capital", "start_page": 1, "end_page": 2}],
            current_quarter="Q1-2026",
            current_year=2026,
            use_genai=False,
        )


def test_run_comparison_with_sections_rejects_nonexistent_pdf_files(
    tmp_path: Path,
) -> None:
    previous_pdf = tmp_path / "missing_prev.pdf"
    current_pdf = tmp_path / "missing_curr.pdf"
    with pytest.raises(ValueError, match="Rapport précédent introuvable"):
        cr.run_comparison_with_sections(
            pdf_path_previous=str(previous_pdf),
            pdf_path_current=str(current_pdf),
            bank_code="bnc",
            sections_previous=[{"section": "capital", "start_page": 1, "end_page": 2}],
            sections_current=[{"section": "capital", "start_page": 1, "end_page": 2}],
            current_quarter="Q1-2026",
            current_year=2026,
            use_genai=False,
        )
