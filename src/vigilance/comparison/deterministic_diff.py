import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)


class DeterministicRowComparator:
    """
    Comparateur structurel déterministe pour les listes d'indicateurs.

    Compare deux listes (T1 et T2) de manière indépendante de l'ordre
    pour identifier les ajouts, suppressions et éléments identiques.

    Logique:
    - Identique: Présent dans T1 et T2 (exact match après normalisation)
    - Supprimé: Présent dans T1 mais pas dans T2
    - Ajouté: Présent dans T2 mais pas dans T1
    """

    def __init__(self, normalize_whitespace: bool = True, case_insensitive: bool = False):
        """
        Initialise le comparateur.

        Args:
            normalize_whitespace: Si True, supprime les espaces superflus (début/fin)
            case_insensitive: Si True, compare sans tenir compte de la casse
        """
        self.normalize_whitespace = normalize_whitespace
        self.case_insensitive = case_insensitive

    def _normalize(self, text: str) -> str:
        """Normalise une chaîne de caractères selon la configuration."""
        if not text:
            return ""

        normalized = text
        if self.normalize_whitespace:
            normalized = normalized.strip()
        if self.case_insensitive:
            normalized = normalized.lower()

        return normalized

    def compare(self, t1_rows: List[str], t2_rows: List[str]) -> Dict[str, Any]:
        """
        Compare deux listes d'indicateurs structurellement.

        Args:
            t1_rows: Liste des indicateurs de la période précédente (T1)
            t2_rows: Liste des indicateurs de la période actuelle (T2)

        Returns:
            Dictionnaire structuré avec:
            - summary: Comptes des changements (added, removed, identical)
            - details: Liste détaillée des statuts pour chaque élément unique
            - diff: Dictionnaire avec les listes séparées (added, removed, identical)
        """
        # Normalisation pour la comparaison (set) mais on garde les originaux pour l'affichage
        # On utilise un dictionnaire {normalized: original} pour retrouver le texte d'origine
        # En cas de conflit (ex: "Total" et "total" normalisés en "total"), on prend le dernier vu de T2, puis T1

        t1_map = {self._normalize(item): item for item in t1_rows if item}
        t2_map = {self._normalize(item): item for item in t2_rows if item}

        set_t1 = set(t1_map.keys())
        set_t2 = set(t2_map.keys())

        # Calcul des ensembles
        identical_keys = set_t1.intersection(set_t2)
        removed_keys = set_t1 - set_t2
        added_keys = set_t2 - set_t1

        # Construction des listes de résultats
        identical_items = [
            t2_map[k] for k in identical_keys
        ]  # On préfère la version T2 pour l'affichage
        removed_items = [t1_map[k] for k in removed_keys]
        added_items = [t2_map[k] for k in added_keys]

        # Création de la liste détaillée (tous les éléments uniques)
        all_details = []

        for k in added_keys:
            all_details.append({"element": t2_map[k], "status": "Ajouté", "origin": "T2"})

        for k in removed_keys:
            all_details.append({"element": t1_map[k], "status": "Supprimé", "origin": "T1"})

        for k in identical_keys:
            all_details.append({"element": t2_map[k], "status": "Identique", "origin": "T1+T2"})

        # Trier par nom pour une lecture plus facile
        all_details.sort(key=lambda x: x["element"])

        return {
            "summary": {
                "total_unique": len(set_t1.union(set_t2)),
                "identical": len(identical_keys),
                "added": len(added_keys),
                "removed": len(removed_keys),
            },
            "diff": {
                "identical": sorted(identical_items),
                "added": sorted(added_items),
                "removed": sorted(removed_items),
            },
            "details": all_details,
        }
