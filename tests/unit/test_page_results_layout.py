from __future__ import annotations

from dash.development.base_component import Component

from vigilance.dash_app.layouts.page_results import build_page_results


def _find_by_id(node: object, target_id: str) -> Component:
    if isinstance(node, Component):
        if getattr(node, "id", None) == target_id:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, list | tuple):
            for child in children:
                try:
                    return _find_by_id(child, target_id)
                except LookupError:
                    pass
        elif children is not None:
            return _find_by_id(children, target_id)
    raise LookupError(target_id)


def _collect_ids(node: object) -> set[object]:
    ids: set[object] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, Component):
            continue
        comp_id = getattr(current, "id", None)
        if comp_id is not None:
            ids.add(comp_id)
        children = getattr(current, "children", None)
        if isinstance(children, list | tuple):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return ids


def test_text_tab_does_not_include_indicator_review_panel() -> None:
    page = build_page_results()
    tabs = _find_by_id(page, "results-main-tabs")
    tab_children = getattr(tabs, "children", []) or []

    indicator_tab = next(tab for tab in tab_children if getattr(tab, "tab_id", None) == "tab-indicateurs")
    text_tab = next(tab for tab in tab_children if getattr(tab, "tab_id", None) == "tab-texte")

    indicator_ids = _collect_ids(indicator_tab)
    text_ids = _collect_ids(text_tab)

    assert "review-queue-container" in indicator_ids
    assert "review-proof-container" in indicator_ids
    assert "results-export-tab" in indicator_ids
    assert "text-analysis-tab-content" in text_ids
    assert "review-queue-container" not in text_ids
    assert "review-proof-container" not in text_ids
    assert "results-export-tab" not in text_ids
