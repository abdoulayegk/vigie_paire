from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc

from vigie.interface.callbacks import load_flow as load_mod
from vigie.interface.services.comparison_store import FileComparisonStore


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
    monkeypatch.setattr(
        "vigie.interface.services.text_comparison_store.resolve_text_comparison_from_payload",
        lambda _payload: None,
    )

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
        text_comparison,
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
    assert text_comparison is None


def test_file_store_falls_back_to_repo_inputs_for_cross_platform_paths(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    result_root = repo_root / "outputs" / "resultats"
    comparison_path = (
        result_root / "rbc" / "2025_t2_vs_2025_t1" / "comparison.json"
    )
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "bank_code": "rbc",
                "year_previous": 2025,
                "quarter_previous": "t1",
                "year_current": 2025,
                "quarter_current": "t2",
                "source_pdf_previous": (
                    "/Users/producer/vigie_paire/Inputs/RBC/2025/RBC_2025_T1.pdf"
                ),
                "source_pdf_current": (
                    "/Users/producer/vigie_paire/Inputs/RBC/2025/RBC_2025_T2.pdf"
                ),
                "archived_pdf_previous": (
                    "/Users/producer/vigie_paire/outputs/previous_report.pdf"
                ),
                "archived_pdf_current": (
                    "/Users/producer/vigie_paire/outputs/current_report.pdf"
                ),
                "pair_comparisons": [],
                "matching": {},
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    previous_pdf = repo_root / "Inputs" / "RBC" / "2025" / "RBC_2025_T1.pdf"
    current_pdf = repo_root / "Inputs" / "RBC" / "2025" / "RBC_2025_T2.pdf"
    previous_pdf.parent.mkdir(parents=True)
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    payload = FileComparisonStore(root_dir=result_root).load_dash_payload(
        comparison_path,
        source="analyse_enregistree",
        source_label="Analyse enregistrée",
    )

    assert payload is not None
    assert payload["pdf_paths"]["pdf_previous"] == str(previous_pdf)
    assert payload["pdf_paths"]["pdf_current"] == str(current_pdf)
    assert payload["warning"] == ""


def test_file_store_inputs_fallback_supports_nonstandard_pdf_name(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    result_root = repo_root / "outputs" / "resultats"
    comparison_path = (
        result_root / "bmo" / "2025_t2_vs_2025_t1" / "comparison.json"
    )
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "bank_code": "bmo",
                "year_previous": 2025,
                "quarter_previous": "t1",
                "year_current": 2025,
                "quarter_current": "t2",
                "source_pdf_previous": "C:\\missing\\BMO_2025_T1.pdf",
                "source_pdf_current": "C:\\missing\\BMO_2025_T2.pdf",
                "archived_pdf_previous": "",
                "archived_pdf_current": "",
                "pair_comparisons": [],
                "matching": {},
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    previous_pdf = repo_root / "Inputs" / "BMO" / "2025" / "BMO_2025_T1.pdf"
    current_pdf = repo_root / "Inputs" / "BMO" / "2025" / "BNO_2025_T2.pdf"
    previous_pdf.parent.mkdir(parents=True)
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    payload = FileComparisonStore(root_dir=result_root).load_dash_payload(
        comparison_path,
        source="analyse_enregistree",
        source_label="Analyse enregistrée",
    )

    assert payload is not None
    assert payload["pdf_paths"]["pdf_previous"] == str(previous_pdf)
    assert payload["pdf_paths"]["pdf_current"] == str(current_pdf)
    assert payload["warning"] == ""
