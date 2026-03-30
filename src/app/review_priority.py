"""Priority helpers for review queue ordering and GenAI signal normalization."""

from __future__ import annotations

# action_requise is the primary sort key — it reflects the analyst action needed,
# which is the most direct signal of urgency.
_ACTION_PRIORITY = {
    "escalade": 0,
    "investigation": 1,
    "confirmation": 2,
    "information": 3,
    "aucune": 4,
}

_RELEVANCE_PRIORITY = {
    "REGLEMENTAIRE": 0,
    "NOUVELLE_DIVULGATION": 1,
    "STRUCTUREL": 2,
    "NON_CLASSIFIE": 3,
    "NON_SIGNIFICATIF": 4,
}

_RISK_PRIORITY = {
    "ELEVE": 0,
    "MODERE": 1,
    "FAIBLE": 2,
}


def normalize_relevance(value: str) -> str:
    raw = str(value or "").strip().upper()
    if raw == "REGULATORY":
        return "REGLEMENTAIRE"
    if raw == "NON_MATERIAL":
        return "NON_SIGNIFICATIF"
    if raw == "STRUCTURAL":
        return "STRUCTUREL"
    if raw == "NEW_DISCLOSURE":
        return "NOUVELLE_DIVULGATION"
    if raw == "UNKNOWN":
        return "NON_CLASSIFIE"
    return raw


def normalize_risk(value: str) -> str:
    raw = str(value or "").strip().upper()
    if raw == "HIGH":
        return "ELEVE"
    if raw == "MODERATE":
        return "MODERE"
    if raw == "LOW":
        return "FAIBLE"
    return raw


def get_priority_signals(item: dict) -> tuple[str, str, str, float]:
    """Return (action_requise, relevance, risk, confidence) for a queue item."""
    ga = item.get("genai_analysis")
    if not isinstance(ga, dict):
        ga = {}

    action = str(ga.get("action_requise", "") or "").strip().lower()

    relevance = normalize_relevance(str(ga.get("relevance", "") or ga.get("category", "")))
    if not relevance:
        if str(item.get("table_status", "")).strip().lower() == "structure_change":
            relevance = "STRUCTUREL"
        elif str(item.get("change_type", "")).strip().lower() == "structure_change":
            relevance = "STRUCTUREL"

    risk = normalize_risk(str(ga.get("risk_level", "")))

    confidence = ga.get("confidence", None)
    try:
        conf_f = float(confidence)
        conf_f = max(0.0, min(1.0, conf_f))
    except (TypeError, ValueError):
        conf_f = -1.0

    return action, relevance, risk, conf_f


def sort_review_items_by_priority(items: list[dict]) -> list[dict]:
    """Sort queue items by analyst urgency.

    Primary key  : action_requise (escalade → investigation → confirmation → information → aucune)
    Secondary    : relevance category (REGLEMENTAIRE → … → NON_SIGNIFICATIF)
    Tertiary     : risk level (ELEVE → MODERE → FAIBLE)
    Quaternary   : confidence (descending)
    Stable       : original index
    """
    indexed: list[tuple[int, dict]] = list(enumerate(items))

    def _priority_key(entry: tuple[int, dict]) -> tuple[int, int, int, float, int]:
        idx, item = entry
        action, relevance, risk, confidence = get_priority_signals(item)
        action_rank = _ACTION_PRIORITY.get(action, 5)
        relevance_rank = _RELEVANCE_PRIORITY.get(relevance, 5)
        risk_rank = _RISK_PRIORITY.get(risk, 3)
        conf_rank = -confidence if confidence >= 0 else 1.0
        return (action_rank, relevance_rank, risk_rank, conf_rank, idx)

    ordered = sorted(indexed, key=_priority_key)
    return [item for _, item in ordered]
