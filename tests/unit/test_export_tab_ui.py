from __future__ import annotations

from dash.development.base_component import Component

from vigilance.dash_app.app import render_export_tab


def _collect_component_ids(node: object, acc: list[object]) -> None:
    if isinstance(node, Component):
        comp_id = getattr(node, "id", None)
        if comp_id is not None:
            acc.append(comp_id)
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for child in children:
                _collect_component_ids(child, acc)
        elif children is not None:
            _collect_component_ids(children, acc)


def test_render_export_tab_shows_only_excel_and_txt_actions() -> None:
    tree = render_export_tab(
        review_items_data=[{"change_id": "x"}],
        indicator_result={"bank_code": "td", "quarter_from": "Q3-2025", "quarter_to": "Q1-2026"},
        show_results=True,
    )
    ids: list[object] = []
    _collect_component_ids(tree, ids)

    assert "btn-download-review-excel" in ids
    assert "btn-download-review-txt" in ids
    assert "btn-download-review-csv" not in ids
    assert "btn-download-review-json" not in ids
    assert "btn-download-indicator-json-brut" not in ids
