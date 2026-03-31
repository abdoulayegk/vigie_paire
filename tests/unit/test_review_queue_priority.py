from __future__ import annotations

from vigilance.review_priority import sort_review_items_by_priority


def test_sort_review_items_by_priority_orders_by_relevance_then_risk() -> None:
    items = [
        {
            "change_id": "non_signif",
            "genai_analysis": {
                "relevance": "NON_SIGNIFICATIF",
                "risk_level": "ELEVE",
                "confidence": 0.9,
            },
        },
        {
            "change_id": "structural",
            "genai_analysis": {
                "relevance": "STRUCTUREL",
                "risk_level": "FAIBLE",
                "confidence": 0.2,
            },
        },
        {
            "change_id": "reg_mod",
            "genai_analysis": {
                "relevance": "REGLEMENTAIRE",
                "risk_level": "MODERE",
                "confidence": 0.7,
            },
        },
        {
            "change_id": "reg_high",
            "genai_analysis": {
                "relevance": "REGLEMENTAIRE",
                "risk_level": "ELEVE",
                "confidence": 0.1,
            },
        },
    ]

    ordered = sort_review_items_by_priority(items)
    ordered_ids = [str(item.get("change_id")) for item in ordered]
    assert ordered_ids == ["reg_high", "reg_mod", "structural", "non_signif"]


def test_sort_review_items_by_priority_handles_english_labels() -> None:
    items = [
        {
            "change_id": "unknown",
            "genai_analysis": {
                "relevance": "UNKNOWN",
                "risk_level": "LOW",
                "confidence": 0.9,
            },
        },
        {
            "change_id": "regulatory",
            "genai_analysis": {
                "relevance": "REGULATORY",
                "risk_level": "HIGH",
                "confidence": 0.3,
            },
        },
    ]

    ordered = sort_review_items_by_priority(items)
    ordered_ids = [str(item.get("change_id")) for item in ordered]
    assert ordered_ids == ["regulatory", "unknown"]


def test_sort_review_items_by_priority_fallbacks_to_structure_change() -> None:
    items = [
        {"change_id": "fallback_structure", "change_type": "structure_change"},
        {
            "change_id": "non_signif",
            "genai_analysis": {
                "relevance": "NON_SIGNIFICATIF",
                "risk_level": "FAIBLE",
                "confidence": 0.5,
            },
        },
        {"change_id": "no_genai_other"},
    ]

    ordered = sort_review_items_by_priority(items)
    ordered_ids = [str(item.get("change_id")) for item in ordered]
    assert ordered_ids == ["fallback_structure", "non_signif", "no_genai_other"]
