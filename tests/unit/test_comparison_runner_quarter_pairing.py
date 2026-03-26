from __future__ import annotations

import json
from pathlib import Path

from app import comparison_runner as cr


def _write_report_comparison(
    path: Path,
    *,
    reference_resolution: dict | None = None,
    run_metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "run_id": "20260325_120000",
                "bank_code": "bnc",
                "year_previous": 2025,
                "quarter_previous": "t3",
                "year_current": 2026,
                "quarter_current": "t1",
                "source_pdf_previous": "/tmp/prev.pdf",
                "source_pdf_current": "/tmp/curr.pdf",
                "archived_pdf_previous": "/archive/prev.pdf",
                "archived_pdf_current": "/archive/curr.pdf",
                "reference_resolution": reference_resolution
                or {
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
                "run_metrics": run_metrics or {"comparison_calls_total": 0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_comparison_uses_current_vs_previous_quarter_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extraction_calls: list[dict] = []
    monkeypatch.setattr(cr, "COMPARISON_ROOT", tmp_path / "outputs" / "comparisons")

    def _fake_extract_tables(**kwargs):
        extraction_calls.append(dict(kwargs))
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
            "run_metrics": {"mode": "fresh", "cache_hit": False, "vision_calls_total": 0},
            "source_metrics": {},
        }
        if kwargs.get("return_provenance"):
            return [], provenance
        return []

    def _fake_compare_reports_gpt4o(*, out_root, reference_resolution, **_kwargs):
        out_path = (
            Path(out_root)
            / "bnc"
            / "2026_t1_vs_2025_t3"
            / "20260325_120000"
            / "comparison.json"
        )
        _write_report_comparison(out_path, reference_resolution=reference_resolution)
        return out_path

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "compare_reports_gpt4o", _fake_compare_reports_gpt4o)

    previous_pdf = tmp_path / "previous.pdf"
    current_pdf = tmp_path / "current.pdf"
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

    assert len(extraction_calls) == 2
    assert extraction_calls[0]["quarter"] == "t3"
    assert extraction_calls[0]["year"] == 2025
    assert extraction_calls[1]["quarter"] == "t1"
    assert extraction_calls[1]["year"] == 2026
    assert result["quarter_from"] == "Q3-2025"
    assert result["quarter_to"] == "Q1-2026"
    assert result["current_quarter"] == "Q1-2026"
    assert result["previous_quarter"] == "Q3-2025"
    assert result["meta"]["quarter_context"]["comparison_label"] == "Q1-2026 vs Q3-2025"


def test_run_comparison_includes_extraction_source_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cr, "COMPARISON_ROOT", tmp_path / "outputs" / "comparisons")

    def _fake_extract_tables(**kwargs):
        quarter = kwargs["quarter"]
        provenance = {
            "mode": "stored" if quarter == "t3" else "fresh",
            "artifact_dir": f"/tmp/{quarter}",
            "tables_path": f"/tmp/{quarter}/tables.json",
            "indicators_path": f"/tmp/{quarter}/indicators.json",
            "footnotes_path": f"/tmp/{quarter}/footnotes.json",
            "artifacts_present": {
                "tables": True,
                "indicators": True,
                "footnotes": True,
            },
            "quarter": quarter,
            "year": kwargs["year"],
            "run_metrics": {"mode": "stored" if quarter == "t3" else "fresh"},
            "source_metrics": {},
        }
        if kwargs.get("return_provenance"):
            return [], provenance
        return []

    def _fake_compare_reports_gpt4o(*, out_root, reference_resolution, **_kwargs):
        out_path = (
            Path(out_root)
            / "bnc"
            / "2026_t1_vs_2025_t3"
            / "20260325_120000"
            / "comparison.json"
        )
        _write_report_comparison(
            out_path,
            reference_resolution=reference_resolution,
            run_metrics={"comparison_calls_total": 2},
        )
        return out_path

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "compare_reports_gpt4o", _fake_compare_reports_gpt4o)

    previous_pdf = tmp_path / "previous.pdf"
    current_pdf = tmp_path / "current.pdf"
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

    prev = result["meta"]["extraction_sources"]["previous"]
    curr = result["meta"]["extraction_sources"]["current"]
    assert prev["mode"] == "stored"
    assert curr["mode"] == "fresh"
    assert prev["tables_path"].endswith("/t3/tables.json")
    assert curr["indicators_path"].endswith("/t1/indicators.json")
    assert result["meta"]["compare_path"].endswith(
        "2026_t1_vs_2025_t3/20260325_120000/comparison.json"
    )
    assert result["meta"]["run_metrics"]["comparison_calls_total"] == 2


def test_empty_result_includes_structured_extraction_sources() -> None:
    result = cr._empty_result(
        "bnc",
        2025,
        "Aucune section valide fournie.",
        quarter_context={
            "previous": {"code": "t1", "label": "Q1-2025", "year": 2025},
            "current": {"code": "t2", "label": "Q2-2025", "year": 2025},
            "comparison_direction": "current_vs_previous",
            "comparison_label": "Q2-2025 vs Q1-2025",
        },
    )

    assert result["schema_version"] == "comparison_canonical_v1"
    assert result["quarter_from"] == "Q1-2025"
    assert result["quarter_to"] == "Q2-2025"
    assert result["meta"]["source_format"] == "dash_runner_empty"
    assert result["meta"]["extraction_sources"]["previous"]["quarter"] == "t1"
    assert result["meta"]["extraction_sources"]["current"]["quarter"] == "t2"
