from __future__ import annotations

from dash.development.base_component import Component

from vigilance.dash_app.components.review_queue_v2 import build_review_queue_v2


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


def _sample_table(review_id: str, table_name: str) -> dict:
    return {
        "review_id": review_id,
        "table_key": review_id,
        "section": "risk_management",
        "table_name": table_name,
        "table_status": "pending",
        "summary": {
            "total_changes": 2,
            "validated": 0,
            "pending": 2,
            "indicators_added": 1,
            "indicators_removed": 1,
            "indicators_renamed": 0,
            "footnotes_changed": 0,
        },
        "changes": [
            {"change_type": "indicator_added"},
            {"change_type": "indicator_removed"},
        ],
        "page_t1": 10,
        "page_t2": 12,
    }


def test_queue_items_use_review_id_in_pattern_id() -> None:
    tables = [
        _sample_table("rid-1", "Table 1"),
        _sample_table("rid-2", "Table 2"),
    ]
    tree = build_review_queue_v2(tables, current_review_id="rid-1")
    ids: list[object] = []
    _collect_component_ids(tree, ids)
    assert {"type": "queue-table-item-v2", "review_id": "rid-1"} in ids
    assert {"type": "queue-table-item-v2", "review_id": "rid-2"} in ids


def test_current_review_id_marks_active_row() -> None:
    tables = [
        _sample_table("rid-1", "Table 1"),
        _sample_table("rid-2", "Table 2"),
    ]
    tree = build_review_queue_v2(tables, current_review_id="rid-2")
    found_active = False
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Component):
            comp_id = getattr(node, "id", None)
            class_name = str(getattr(node, "className", "") or "")
            if comp_id == {"type": "queue-table-item-v2", "review_id": "rid-2"}:
                found_active = "border-primary" in class_name
                break
            children = getattr(node, "children", None)
            if isinstance(children, list):
                nodes.extend(children)
            elif children is not None:
                nodes.append(children)
    assert found_active
