"""Tests for the current comparison_runner metadata and contract propagation."""

from __future__ import annotations

import json
from pathlib import Path

from vigilance import comparison_runner as cr


def _write_report_comparison(
    path: Path,
    *,
    prompt_version_match: str = "table_match_v8",
    prompt_version_diff: str = "table_diff_v4",
    model_version: str = "gpt-5.4",
    run_metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "run_id": "20260325_143000",
                "bank_code": "bnc",
                "year_previous": 2025,
                "quarter_previous": "t3",
                "year_current": 2026,
                "quarter_current": "t1",
                "source_pdf_previous": "/tmp/prev.pdf",
                "source_pdf_current": "/tmp/curr.pdf",
                "archived_pdf_previous": "/archive/prev.pdf",
                "archived_pdf_current": "/archive/curr.pdf",
                "model_version": model_version,
                "prompt_version_match": prompt_version_match,
                "prompt_version_diff": prompt_version_diff,
                "reference_resolution": {
                    "mode": "automatique",
                    "year_previous": 2025,
                    "quarter_previous": "t3",
                    "rule": "t1->t3 annee precedente",
                },
                "matching": {
                    "matched_pairs": [],
                    "tables_added": [],
                    "tables_removed": [],
                },
                "pair_comparisons": [],
                "summary": {
                    "matched_pairs_total": 0,
                    "tables_added_total": 0,
                    "tables_removed_total": 0,
                    "indicator_changes_total": 0,
                    "footnote_changes_total": 0,
                    "high_priority_items_total": 0,
                },
                "run_metrics": run_metrics
                or {
                    "runtime_extraction_sec": 1.2,
                    "runtime_comparison_sec": 0.4,
                    "comparison_calls_total": 2,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _fake_extract_with_provenance(**kwargs):
    provenance = {
        "mode": "fresh",
        "artifact_dir": f"/tmp/{kwargs['quarter']}",
        "tables_path": f"/tmp/{kwargs['quarter']}/tables.json",
        "indicators_path": f"/tmp/{kwargs['quarter']}/indicators.json",
        "footnotes_path": f"/tmp/{kwargs['quarter']}/footnotes.json",
        "artifacts_present": {
            "tables": True,
            "indicators": True,
            "footnotes": True,
        },
        "quarter": kwargs["quarter"],
        "year": kwargs["year"],
        "run_metrics": {"mode": "fresh", "vision_calls_total": 0, "cache_hit": False},
        "source_metrics": {},
    }
    if kwargs.get("return_provenance"):
        return [], provenance
    return []


def test_run_comparison_with_sections_propagates_prompt_versions_to_dash_meta(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cr, "COMPARISON_ROOT", tmp_path / "outputs" / "comparisons")

    def _fake_compare_reports_gpt4o(*, out_root, **_kwargs):
        out_path = (
            Path(out_root)
            / "bnc"
            / "2026_t1_vs_2025_t3"
            / "20260325_143000"
            / "comparison.json"
        )
        _write_report_comparison(out_path)
        return out_path

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_with_provenance)
    monkeypatch.setattr(cr, "compare_reports_gpt4o", _fake_compare_reports_gpt4o)

    previous_pdf = tmp_path / "prev.pdf"
    current_pdf = tmp_path / "curr.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    result = cr.run_comparison_with_sections(
        pdf_path_previous=str(previous_pdf),
        pdf_path_current=str(current_pdf),
        bank_code="bnc",
        sections_previous=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        sections_current=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        current_quarter="Q1-2026",
        current_year=2026,
        api_key="test-key",
    )

    assert result["meta"]["source_format"] == "report_comparison"
    assert result["meta"]["model_version"] == "gpt-5.4"
    assert result["meta"]["prompt_version_match"] == "table_match_v8"
    assert result["meta"]["prompt_version_diff"] == "table_diff_v4"


def test_run_comparison_with_sections_propagates_compare_path_reference_and_run_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cr, "COMPARISON_ROOT", tmp_path / "outputs" / "comparisons")

    def _fake_compare_reports_gpt4o(*, out_root, **_kwargs):
        out_path = (
            Path(out_root)
            / "bnc"
            / "2026_t1_vs_2025_t3"
            / "20260325_143000"
            / "comparison.json"
        )
        _write_report_comparison(
            out_path,
            run_metrics={
                "runtime_extraction_sec": 2.5,
                "runtime_comparison_sec": 0.8,
                "comparison_calls_total": 3,
            },
        )
        return out_path

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_with_provenance)
    monkeypatch.setattr(cr, "compare_reports_gpt4o", _fake_compare_reports_gpt4o)

    previous_pdf = tmp_path / "prev.pdf"
    current_pdf = tmp_path / "curr.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    result = cr.run_comparison_with_sections(
        pdf_path_previous=str(previous_pdf),
        pdf_path_current=str(current_pdf),
        bank_code="bnc",
        sections_previous=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        sections_current=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        current_quarter="Q1-2026",
        current_year=2026,
        api_key="test-key",
    )

    assert result["meta"]["compare_path"].endswith(
        "2026_t1_vs_2025_t3/20260325_143000/comparison.json"
    )
    assert result["meta"]["reference_resolution"]["quarter_previous"] == "t3"
    assert result["meta"]["run_metrics"]["comparison_calls_total"] == 3
    assert result["meta"]["run_metrics"]["runtime_extraction_sec"] == 2.5


def test_empty_result_uses_current_ui_payload_contract() -> None:
    result = cr._empty_result(
        "bnc",
        2026,
        "Aucune section valide fournie.",
        quarter_context={
            "previous": {"code": "t3", "label": "Q3-2025", "year": 2025},
            "current": {"code": "t1", "label": "Q1-2026", "year": 2026},
            "comparison_direction": "current_vs_previous",
            "comparison_label": "Q1-2026 vs Q3-2025",
        },
    )

    assert result["schema_version"] == "comparison_canonical_v1"
    assert result["meta"]["source_format"] == "dash_runner_empty"
    assert result["meta"]["executive_summary"]["content"] == "Aucune section valide fournie."
    assert result["meta"]["extraction_sources"]["previous"]["quarter"] == "t3"
    assert result["meta"]["extraction_sources"]["current"]["quarter"] == "t1"
