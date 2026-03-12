from __future__ import annotations

from dash.development.base_component import Component

from vigilance.dash_app.components.review_detail_v2 import build_review_detail_v2


def _flatten_text(node: object) -> str:
    parts: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            parts.append(current)
            continue
        if isinstance(current, Component):
            children = getattr(current, "children", None)
            if isinstance(children, list):
                stack.extend(children)
            elif children is not None:
                stack.append(children)
    return " ".join(parts)


def test_table_only_view_hides_indicator_change_list_title() -> None:
    table = {
        "table_name": "Table entière",
        "section": "risk_management",
        "page_t1": None,
        "page_t2": 42,
        "table_status": "pending",
        "summary": {"total_changes": 1, "validated": 0, "pending": 1},
        "changes": [
            {
                "change_id": "chg_1",
                "change_type": "table_added",
                "payload": {"description": "Tableau entier"},
                "validation_status": "pending",
                "is_required": True,
            }
        ],
        "genai_analysis": {"justification": "Nouveau tableau détecté."},
    }
    view = build_review_detail_v2(table=table, current_change_idx=0, show_proofs=False)
    text = _flatten_text(view)
    assert "Validation au niveau tableau" in text
    assert "Changements (" not in text


def test_change_list_view_shows_changes_heading() -> None:
    table = {
        "table_name": "Table diff",
        "section": "risk_management",
        "page_t1": 10,
        "page_t2": 12,
        "table_status": "pending",
        "summary": {"total_changes": 1, "validated": 0, "pending": 1},
        "changes": [
            {
                "change_id": "chg_1",
                "change_type": "indicator_added",
                "payload": {"indicator_name": "Indicateur A"},
                "validation_status": "pending",
                "is_required": True,
            }
        ],
    }
    view = build_review_detail_v2(table=table, current_change_idx=0, show_proofs=False)
    text = _flatten_text(view)
    assert "Changements (1)" in text


def test_table_removed_without_genai_shows_fallback_explanation() -> None:
    table = {
        "table_name": "Table supprimée",
        "section": "capital_management",
        "page_t1": 33,
        "page_t2": None,
        "table_status": "pending",
        "summary": {"total_changes": 1, "validated": 0, "pending": 1},
        "changes": [
            {
                "change_id": "chg_1",
                "change_type": "table_removed",
                "payload": {"description": "Tableau entier"},
                "validation_status": "pending",
                "is_required": True,
            }
        ],
        "genai_analysis": {},
    }
    view = build_review_detail_v2(table=table, current_change_idx=0, show_proofs=False)
    text = _flatten_text(view)
    assert "Aucune explication GenAI disponible." not in text
    assert "Explication automatique" in text
    assert "absent au trimestre courant" in text
