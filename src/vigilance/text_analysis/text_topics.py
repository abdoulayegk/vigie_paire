"""Detection des sujets canoniques pour les sous-sections texte."""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

from .text_normalization import _normalize_match_text

def _canonical_topic_for_text(heading: str, text: str = "") -> str:
    """Mappe un titre/contenu vers un thème canonique stable entre années."""
    value = _normalize_match_text(f"{heading} {(text or '')[:900]}")

    def has(*words: str) -> bool:
        return all(word in value for word in words)

    if has("juridique", "reglementaire") or has("conformite", "juridique", "reglementaire"):
        return "conformite_juridique_reglementaire"
    if "geopolitique" in value or ("differends" in value and "commerciaux" in value):
        return "risques_geopolitiques_commerciaux"
    if "politique" in value and ("monetaire" in value or "budgetaire" in value):
        return "politiques_monetaires_budgetaires_economie"
    if "cybersecurite" in value or ("securite" in value and "information" in value):
        return "cybersecurite"
    if "technologie" in value and ("innovation" in value or "resilience" in value):
        return "technologie_innovation"
    if "tiers" in value or "fournisseur" in value or "fournisseurs" in value or "impartition" in value:
        return "risque_tiers"
    if "actifs" in value and "ponderes" in value:
        return "actifs_ponderes_risque"
    if (
        "fonds propres" in value
        or "tlac" in value
        or "cet1" in value
        or "bale" in value
        or ("capital" in value and ("reglementaire" in value or "prudentiel" in value or "plancher" in value))
    ):
        return "capital_reglementaire"
    if ("environnemental" in value and "social" in value) or "climatique" in value or "durabilite" in value:
        return "risque_env_social_climat"
    if "fraude" in value:
        return "risque_fraude"
    if "donnees" in value or "renseignements personnels" in value or "vie privee" in value:
        return "donnees_vie_privee"
    if "intelligence artificielle" in value or re.search(r"\bia\b", value):
        return "intelligence_artificielle"
    if "liquidite" in value or "financement" in value:
        return "liquidite_financement"
    if "modele" in value or "modeles" in value:
        return "risque_modeles"
    return _normalize_match_text(heading)


def _is_specific_canonical_topic(topic: str) -> bool:
    """Indique si un thème canonique est assez métier pour matcher des titres différents."""
    return bool(topic and topic not in {"__intro__", "intro", "description"} and len(topic) > 3)
