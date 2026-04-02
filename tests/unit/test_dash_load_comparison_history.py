from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc

from vigilance.dash_app.callbacks import load_flow as load_mod


def _write_report_comparison(
    path: Path,
    *,
    archived_previous: str,
    archived_current: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "run_id": "20260323_143015",
                "bank_code": "bnc",
                "year_previous": 2025,
                "quarter_previous": "t3",
                "year_current": 2026,
                "quarter_current": "t1",
                "source_pdf_previous": "/tmp/previous_upload.pdf",
                "source_pdf_current": "/tmp/current_upload.pdf",
                "archived_pdf_previous": archived_previous,
                "archived_pdf_current": archived_current,
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_on_load_comparison_restores_archived_pdf_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    previous_pdf = tmp_path / "archive" / "previous_report.pdf"
    current_pdf = tmp_path / "archive" / "current_report.pdf"
    previous_pdf.parent.mkdir(parents=True, exist_ok=True)
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    relative = "bnc/2026_t1_vs_2025_t3/20260323_143015/comparison.json"
    comparison_path = tmp_path / relative
    _write_report_comparison(
        comparison_path,
        archived_previous=str(previous_pdf),
        archived_current=str(current_pdf),
    )

    monkeypatch.setattr(load_mod, "INDICATOR_COMPARISON_DIR", tmp_path)
    monkeypatch.setattr(load_mod, "build_page_results", lambda: "results-page")

    result = load_mod.on_load_comparison(1, relative)

    (
        _,
        indicator_result,
        indicator_meta,
        pdf_paths,
        sections_validated,
        page,
        notification,
        show_results,
    ) = result
    assert indicator_result["meta"]["pdf_paths"]["pdf_previous"] == str(previous_pdf)
    assert indicator_meta["pdf_paths"]["pdf_current"] == str(current_pdf)
    assert indicator_meta["model_version"] == "gpt-5.4"
    assert indicator_meta["prompt_version_match"] == "table_match_v8"
    assert indicator_meta["prompt_version_diff"] == "table_diff_v4"
    assert pdf_paths["pdf_previous"] == str(previous_pdf)
    assert pdf_paths["pdf_current"] == str(current_pdf)
    assert sections_validated is True
    assert page == "results-page"
    assert isinstance(notification, dbc.Alert)
    assert notification.color == "success"
    assert show_results is True


def test_on_load_comparison_warns_when_archived_pdf_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    relative = "bnc/2026_t1_vs_2025_t3/20260323_143015/comparison.json"
    comparison_path = tmp_path / relative
    missing_previous = tmp_path / "archive" / "missing_previous.pdf"
    missing_current = tmp_path / "archive" / "missing_current.pdf"
    _write_report_comparison(
        comparison_path,
        archived_previous=str(missing_previous),
        archived_current=str(missing_current),
    )

    monkeypatch.setattr(load_mod, "INDICATOR_COMPARISON_DIR", tmp_path)
    monkeypatch.setattr(load_mod, "build_page_results", lambda: "results-page")

    result = load_mod.on_load_comparison(1, relative)

    _, _, indicator_meta, pdf_paths, _, _, notification, _ = result
    assert indicator_meta["pdf_paths"]["pdf_previous"] == str(missing_previous)
    assert pdf_paths["pdf_current"] == str(missing_current)
    assert isinstance(notification, dbc.Alert)
    assert notification.color == "warning"
