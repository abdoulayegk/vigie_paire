"""Utilitaires de priorite pour le tri de la file de revue et la normalisation des signaux GenAI."""

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
    """Normalise un code de pertinence anglais vers la terminologie francaise.

    Args:
        value: Code de pertinence brut (ex. ``"REGULATORY"``, ``"NEW_DISCLOSURE"``).

    Returns:
        Code normalise en francais (ex. ``"REGLEMENTAIRE"``).
    """
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
    """Normalise un code de niveau de risque anglais vers le francais.

    Args:
        value: Code de risque brut (ex. ``"HIGH"``, ``"MODERATE"``).

    Returns:
        Code normalise en francais (ex. ``"ELEVE"``).
    """
    raw = str(value or "").strip().upper()
    if raw == "HIGH":
        return "ELEVE"
    if raw == "MODERATE":
        return "MODERE"
    if raw == "LOW":
        return "FAIBLE"
    return raw


def get_priority_signals(item: dict) -> tuple[str, str, str, float]:
    """Extrait les signaux de priorite d'un element de la file de revue.

    Args:
        item: Dictionnaire representant un element de revue.

    Returns:
        Tuple ``(action_requise, relevance, risk, confidence)``.
    """
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
    """Trie les elements de la file de revue par urgence analyste.

    Cle primaire : action_requise (escalade > investigation > confirmation > information > aucune).
    Secondaire   : pertinence (REGLEMENTAIRE > ... > NON_SIGNIFICATIF).
    Tertiaire    : niveau de risque (ELEVE > MODERE > FAIBLE).
    Quaternaire  : confiance (descendant).
    Stable       : index d'origine.

    Args:
        items: Elements de revue sous forme de dictionnaires.

    Returns:
        Liste triee par priorite decroissante.
    """
    indexed: list[tuple[int, dict]] = list(enumerate(items))

    def _priority_key(entry: tuple[int, dict]) -> tuple[int, int, int, float, int]:
        """Calcule la cle de tri composite pour un element indexe."""
        idx, item = entry
        action, relevance, risk, confidence = get_priority_signals(item)
        action_rank = _ACTION_PRIORITY.get(action, 5)
        relevance_rank = _RELEVANCE_PRIORITY.get(relevance, 5)
        risk_rank = _RISK_PRIORITY.get(risk, 3)
        conf_rank = -confidence if confidence >= 0 else 1.0
        return (action_rank, relevance_rank, risk_rank, conf_rank, idx)

    ordered = sorted(indexed, key=_priority_key)
    return [item for _, item in ordered]
