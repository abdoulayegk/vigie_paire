"""Utilitaires de priorite pour le tri de la file de revue.

Aligne sur la taxonomie AMF v2 unifiée :
- ``action_requise`` (revue_prioritaire > investigation > confirmation > information > aucune)
- ``category`` (REGLEMENTAIRE > RISQUE > CAPITAL > STRUCTURE > NON_PERTINENT > INCONNU)
- ``impact_level`` (MAJEUR > MODERE > MINEUR)

Plus de traduction anglais→français (les sorties GPT sont natives en français
via la taxonomie AMF). Plus de fallback sur les anciens champs ``relevance`` /
``risk_level`` translated.
"""

from __future__ import annotations

# action_requise is the primary sort key — it reflects the analyst action needed,
# which is the most direct signal of urgency.
_ACTION_PRIORITY = {
    "revue_prioritaire": 0,
    "investigation": 1,
    "confirmation": 2,
    "information": 3,
    "aucune": 4,
}

# Categorie AMF (lecture directe depuis ``genai_analysis.category``).
_CATEGORY_PRIORITY = {
    "REGLEMENTAIRE": 0,
    "RISQUE": 1,
    "CAPITAL": 2,
    "STRUCTURE": 3,
    "NON_PERTINENT": 4,
    "INCONNU": 5,
}

# Niveau d'impact AMF v2 (lecture directe depuis ``genai_analysis.impact_level``).
_IMPACT_PRIORITY = {
    "MAJEUR": 0,
    "MODERE": 1,
    "MINEUR": 2,
}


def get_priority_signals(item: dict) -> tuple[str, str, str]:
    """Extrait les signaux de priorite d'un element de la file de revue.

    Args:
        item: Dictionnaire representant un element de revue.

    Returns:
        Tuple ``(action_requise, category, impact_level)`` aux valeurs AMF v2.
    """
    ga = item.get("genai_analysis")
    if not isinstance(ga, dict):
        ga = {}

    action = str(ga.get("action_requise", "") or "").strip().lower()
    category = str(ga.get("category", "") or "").strip().upper()
    impact = str(ga.get("impact_level", "") or "").strip().upper()
    return action, category, impact


def sort_review_items_by_priority(items: list[dict]) -> list[dict]:
    """Trie les elements de la file de revue par urgence analyste.

    Cle primaire : action_requise (revue_prioritaire > investigation > confirmation > information > aucune).
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

    def _priority_key(entry: tuple[int, dict]) -> tuple[int, int, int, int]:
        """Calcule la cle de tri composite pour un element indexe."""
        idx, item = entry
        action, category, impact = get_priority_signals(item)
        action_rank = _ACTION_PRIORITY.get(action, 5)
        category_rank = _CATEGORY_PRIORITY.get(category, 6)
        impact_rank = _IMPACT_PRIORITY.get(impact, 3)
        return (action_rank, category_rank, impact_rank, idx)

    ordered = sorted(indexed, key=_priority_key)
    return [item for _, item in ordered]
