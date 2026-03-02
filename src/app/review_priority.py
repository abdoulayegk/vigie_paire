"""Priority helpers for review queue ordering and GenAI signal normalization."""

from __future__ import annotations

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


def get_priority_signals(item: dict) -> tuple[str, str, float]:
    ga = item.get("genai_analysis")
    if not isinstance(ga, dict):
        ga = {}

    relevance = normalize_relevance(str(ga.get("relevance", "")))
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

    return relevance, risk, conf_f


def sort_review_items_by_priority(items: list[dict]) -> list[dict]:
    """Sort queue items by criticity (regulatory > structural > non-significant)."""
    indexed: list[tuple[int, dict]] = list(enumerate(items))

    def _priority_key(entry: tuple[int, dict]) -> tuple[int, int, float, int]:
        idx, item = entry
        relevance, risk, confidence = get_priority_signals(item)
        relevance_rank = _RELEVANCE_PRIORITY.get(relevance, 5)
        risk_rank = _RISK_PRIORITY.get(risk, 3)
        conf_rank = -confidence if confidence >= 0 else 1.0
        return (relevance_rank, risk_rank, conf_rank, idx)

    ordered = sorted(indexed, key=_priority_key)
    return [item for _, item in ordered]
