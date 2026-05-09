from __future__ import annotations

from vigilance.review_priority import sort_review_items_by_priority


def test_sort_review_items_by_priority_orders_by_category_then_impact() -> None:
    """Tri AMF v2 : action > category > impact_level (sans confidence legacy)."""
    items = [
        {
            "change_id": "non_pertinent",
            "genai_analysis": {
                "action_requise": "aucune",
                "category": "NON_PERTINENT",
                "impact_level": "MAJEUR",
            },
        },
        {
            "change_id": "structure",
            "genai_analysis": {
                "action_requise": "investigation",
                "category": "STRUCTURE",
                "impact_level": "MINEUR",
            },
        },
        {
            "change_id": "reg_mod",
            "genai_analysis": {
                "action_requise": "investigation",
                "category": "REGLEMENTAIRE",
                "impact_level": "MODERE",
            },
        },
        {
            "change_id": "reg_high",
            "genai_analysis": {
                "action_requise": "escalade",
                "category": "REGLEMENTAIRE",
                "impact_level": "MAJEUR",
            },
        },
    ]

    ordered = sort_review_items_by_priority(items)
    ordered_ids = [str(item.get("change_id")) for item in ordered]
    # escalade > investigation > aucune ; puis REGLEMENTAIRE > STRUCTURE > NON_PERTINENT
    assert ordered_ids == ["reg_high", "reg_mod", "structure", "non_pertinent"]


def test_sort_review_items_by_priority_falls_back_when_no_genai() -> None:
    """Items sans genai_analysis utilisent les rangs par défaut (rangés en dernier)."""
    items = [
        {"change_id": "no_genai_a"},
        {
            "change_id": "regle_majeur",
            "genai_analysis": {
                "action_requise": "escalade",
                "category": "REGLEMENTAIRE",
                "impact_level": "MAJEUR",
            },
        },
        {"change_id": "no_genai_b"},
    ]

    ordered = sort_review_items_by_priority(items)
    ordered_ids = [str(item.get("change_id")) for item in ordered]
    # L'item avec triage AMF v2 passe en premier ; les sans-triage suivent en
    # ordre stable d'apparition.
    assert ordered_ids[0] == "regle_majeur"
    assert set(ordered_ids[1:]) == {"no_genai_a", "no_genai_b"}
