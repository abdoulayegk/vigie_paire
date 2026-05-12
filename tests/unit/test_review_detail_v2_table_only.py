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


def test_table_removed_without_genai_shows_no_classification_message() -> None:
    """Sans triage IA disponible (genai_analysis vide), on affiche un message
    court et neutre — pas de fallback heuristique (alignement principe 100% GPT)."""
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
    assert "Aucune classification IA disponible" in text


def test_table_with_non_relevant_triage_shows_exclusion_reason() -> None:
    """Quand GPT a explicitement classé en non pertinent, on affiche la raison."""
    table = {
        "table_name": "Table reformulée",
        "section": "capital_management",
        "page_t1": 33,
        "page_t2": 35,
        "table_status": "pending",
        "summary": {"total_changes": 1, "validated": 0, "pending": 1},
        "changes": [
            {
                "change_id": "chg_1",
                "change_type": "indicator_renamed",
                "payload": {},
                "validation_status": "pending",
                "is_required": True,
            }
        ],
        "genai_analysis": {
            "is_relevant": False,
            "exclusion_reason": "reformulation_mineure",
            "themes_amf": [],
            "nouvelle_idee": False,
        },
    }
    view = build_review_detail_v2(table=table, current_change_idx=0, show_proofs=False)
    text = _flatten_text(view)
    assert "Non pertinent" in text
    assert "Reformulation sans nouveau fond" in text


def test_table_with_relevant_triage_shows_nouvelle_idee_and_themes() -> None:
    """Cas pertinent : badge nouvelle idée + impact + chips thèmes AMF + justification."""
    justification = (
        "OUI - le ratio TLAC est ajoute au TABLEAU 11 absent du T1. "
        "Cela aligne la divulgation BMO sur les attentes BSIF (DIVULGATION_AJOUT, RATIOS_REGLEMENTAIRES)."
    )
    table = {
        "table_name": "Tableau 11",
        "section": "capital_management",
        "page_t1": 23,
        "page_t2": 25,
        "table_status": "pending",
        "summary": {"total_changes": 1, "validated": 0, "pending": 1},
        "changes": [
            {
                "change_id": "chg_1",
                "change_type": "indicator_added",
                "payload": {"indicator_name": "Ratio TLAC"},
                "validation_status": "pending",
                "is_required": True,
            }
        ],
        "genai_analysis": {
            "is_relevant": True,
            "nouvelle_idee": True,
            "nouvelle_idee_justification": justification,
            "themes_amf": ["DIVULGATION_AJOUT", "RATIOS_REGLEMENTAIRES"],
            "impact_level": "MAJEUR",
            "action_requise": "revue_prioritaire",
            "exclusion_reason": None,
        },
    }
    view = build_review_detail_v2(table=table, current_change_idx=0, show_proofs=False)
    text = _flatten_text(view)
    assert "Nouvelle idée" in text
    assert "MAJEUR" in text
    assert "Ajout de divulgation" in text
    assert "Ratios régl." in text
    assert "OUI" in text
    assert "Revue prioritaire" in text


def test_detail_view_translates_english_section_label() -> None:
    table = {
        "table_name": "Credit table",
        "section": "Credit Risk",
        "page_t1": 12,
        "page_t2": 14,
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
    assert "Risque de credit" in text
    assert "Credit Risk" not in text
