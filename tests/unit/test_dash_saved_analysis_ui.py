from __future__ import annotations

import json
from pathlib import Path

from dash.development.base_component import Component

from vigilance.dash_app.layouts.sidebar import build_sidebar
from vigilance.dash_app.services.comparison_store import build_file_comparison_store


def _walk_components(node: object) -> list[Component]:
    found: list[Component] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, Component):
            found.append(current)
            children = getattr(current, "children", None)
            if isinstance(children, list):
                stack.extend(children)
            elif children is not None:
                stack.append(children)
    return found


def test_sidebar_defaults_to_saved_analyses() -> None:
    sidebar = build_sidebar()
    components = _walk_components(sidebar)
    radio = next(
        component
        for component in components
        if getattr(component, "id", None) == "data-source-type"
    )

    assert radio.value == "saved"
    labels = [option["label"] for option in radio.options]
    assert labels == ["Analyses enregistrées"]

    saved_container = next(
        component
        for component in components
        if getattr(component, "id", None) == "db-source-container"
    )
    upload_container = next(
        component
        for component in components
        if getattr(component, "id", None) == "upload-source-container"
    )
    assert saved_container.style == {"display": "block"}
    assert upload_container.style == {"display": "none"}

    source_wrapper = next(
        component
        for component in components
        if getattr(component, "id", None) == "data-source-wrapper"
    )
    assert source_wrapper.style == {"display": "none"}


def test_file_comparison_store_loads_saved_analysis_with_local_pdf_context(
    tmp_path: Path,
) -> None:
    relative = "bnc/2026_t1_vs_2025_t3/comparison.json"
    comparison_path = tmp_path / relative
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(
            {
                "artifact_type": "report_comparison",
                "run_id": "20260402_101500",
                "bank_code": "bnc",
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
    previous_pdf = comparison_path.parent / "previous_report.pdf"
    current_pdf = comparison_path.parent / "current_report.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    store = build_file_comparison_store(root_dir=tmp_path)
    loaded = store.load_dash_payload(
        relative,
        source="analyse_enregistree",
        source_label="Analyse enregistrée",
    )

    assert loaded is not None
    assert loaded["indicator_meta"]["source"] == "analyse_enregistree"
    assert loaded["indicator_meta"]["source_label"] == "Analyse enregistrée"
    assert loaded["pdf_paths"]["pdf_previous"] == str(previous_pdf)
    assert loaded["pdf_paths"]["pdf_current"] == str(current_pdf)
    assert loaded["warning"] == ""
