from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc

from vigilance.dash_app.callbacks import load_flow as load_mod


def _write_report_comparison(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "run_id": "20260328_194247",
                "bank_code": "td",
                "year_previous": 2025,
                "quarter_previous": "t3",
                "year_current": 2026,
                "quarter_current": "t1",
                "source_pdf_previous": "",
                "source_pdf_current": "",
                "archived_pdf_previous": "",
                "archived_pdf_current": "",
                "pair_comparisons": [],
                "matching": {
                    "matched_pairs": [],
                    "tables_added": [],
                    "tables_removed": [],
                },
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


def test_on_load_comparison_falls_back_to_run_archived_pdfs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    relative = "td/2026_t1_vs_2025_t3/comparison.json"
    comparison_path = tmp_path / relative
    _write_report_comparison(comparison_path)

    previous_pdf = comparison_path.parent / "previous_report.pdf"
    current_pdf = comparison_path.parent / "current_report.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    monkeypatch.setattr(load_mod, "INDICATOR_COMPARISON_DIR", tmp_path)
    monkeypatch.setattr(load_mod, "build_page_results", lambda: "results-page")

    result = load_mod.on_load_comparison(1, relative)

    (
        _,
        _,
        indicator_meta,
        pdf_paths,
        sections_validated,
        page,
        notification,
        show_results,
    ) = result

    assert indicator_meta["pdf_paths"]["pdf_previous"] == str(previous_pdf)
    assert indicator_meta["pdf_paths"]["pdf_current"] == str(current_pdf)
    assert pdf_paths["pdf_previous"] == str(previous_pdf)
    assert pdf_paths["pdf_current"] == str(current_pdf)
    assert sections_validated is True
    assert page == "results-page"
    assert isinstance(notification, dbc.Alert)
    assert notification.color == "success"
    assert show_results is True
