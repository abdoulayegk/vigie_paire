"""
Module de matching intelligent des tableaux entre trimestres.
Implémente les 3 niveaux de matching: exact, sémantique, et par indicateurs.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Résultat d'un matching entre deux tableaux."""

    is_match: bool
    confidence: float
    match_method: str  # "exact", "semantic", "indicators", "structural"
    common_indicators: List[str] = field(default_factory=list)
    added_indicators: List[str] = field(default_factory=list)
    removed_indicators: List[str] = field(default_factory=list)
    different_indicators: List[str] = field(default_factory=list)  # Pour compatibilité descendante
    title_similarity: float = 0.0
    indicator_similarity: float = 0.0
    is_renamed: bool = False
    is_repositioned: bool = False

    def to_dict(self) -> dict:
        return {
            "est_correspondance": self.is_match,
            "confiance": round(self.confidence, 2),
            "methode_matching": self.match_method,
            "indicateurs_communs": self.common_indicators,
            "indicateurs_ajoutes": self.added_indicators,
            "indicateurs_supprimes": self.removed_indicators,
            "similarite_titre": round(self.title_similarity, 2),
            "similarite_indicateurs": round(self.indicator_similarity, 2),
            "est_renomme": self.is_renamed,
            "est_repositionne": self.is_repositioned,
        }


@dataclass
class LearnedPattern:
    """Pattern de matching appris."""

    bank_code: str
    old_title: str
    new_title: str
    old_page: int
    new_page: int
    quarter_from: str
    quarter_to: str
    pattern_type: str  # "rename", "reposition", "both"
    confidence: float
    occurrences: int = 1

    def to_dict(self) -> dict:
        return {
            "bank_code": self.bank_code,
            "old_title": self.old_title,
            "new_title": self.new_title,
            "old_page": self.old_page,
            "new_page": self.new_page,
            "quarter_from": self.quarter_from,
            "quarter_to": self.quarter_to,
            "pattern_type": self.pattern_type,
            "confidence": round(self.confidence, 2),
            "occurrences": self.occurrences,
        }


class IndicatorMatcher:
    """
    Algorithme de matching de tableaux basé sur les indicateurs de première colonne.
    Implémente le matching à 3 niveaux selon la spécification.
    """

    # Seuils par défaut
    EXACT_MATCH_THRESHOLD = 1.0
    SEMANTIC_TITLE_THRESHOLD = 0.90
    SEMANTIC_INDICATOR_THRESHOLD = 0.90
    STRUCTURAL_INDICATOR_THRESHOLD = 0.80
    MINIMUM_MATCH_THRESHOLD = 0.70

    def __init__(self, config_path: str = None, custom_thresholds: dict = None):
        """
        Initialise le matcher.

        Args:
            config_path: Chemin vers bank_profiles.json
            custom_thresholds: Seuils personnalisés
        """
        self.learned_patterns: Dict[str, List[LearnedPattern]] = {}
        self.patterns_file = Path("config/learned_patterns.json")

        # Charger les seuils personnalisés
        if custom_thresholds:
            self.EXACT_MATCH_THRESHOLD = custom_thresholds.get(
                "exact_match", self.EXACT_MATCH_THRESHOLD
            )
            self.SEMANTIC_TITLE_THRESHOLD = custom_thresholds.get(
                "semantic_title", self.SEMANTIC_TITLE_THRESHOLD
            )
            self.SEMANTIC_INDICATOR_THRESHOLD = custom_thresholds.get(
                "semantic_indicator", self.SEMANTIC_INDICATOR_THRESHOLD
            )
            self.STRUCTURAL_INDICATOR_THRESHOLD = custom_thresholds.get(
                "structural_indicator", self.STRUCTURAL_INDICATOR_THRESHOLD
            )

        # Charger patterns existants
        self._load_patterns()

    def _load_patterns(self):
        """Charge les patterns appris depuis le fichier."""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for bank, patterns in data.items():
                        self.learned_patterns[bank] = [LearnedPattern(**p) for p in patterns]
                logger.info(
                    f"Chargé {sum(len(p) for p in self.learned_patterns.values())} patterns"
                )
            except Exception as e:
                logger.warning(f"Erreur chargement patterns: {e}")

    def _save_patterns(self):
        """Sauvegarde les patterns appris."""
        try:
            self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                bank: [p.to_dict() for p in patterns]
                for bank, patterns in self.learned_patterns.items()
            }
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde patterns: {e}")

    def match_tables(self, table1: dict, table2: dict, bank_code: str = None) -> MatchResult:
        """
        Compare deux tableaux et détermine s'ils correspondent.

        Niveaux de matching (par priorité):
        - Niveau 0: Numéro de tableau identique (TABLEAU 28 = TABLEAU 28) → confiance 100%
        - Niveau 1: Titre exact normalisé → confiance 100%
        - Niveau 2: Similarité sémantique (titre ≥90% OU indicateurs ≥90%)
        - Niveau 3: Similarité structurelle (indicateurs ≥80%)

        Args:
            table1: Premier tableau (ancien) avec 'title', 'rows', 'page_number', 'table_number'
            table2: Deuxième tableau (nouveau)
            bank_code: Code de la banque (pour utiliser les patterns appris)

        Returns:
            MatchResult avec détails de la correspondance
        """
        title1 = table1.get("title") or table1.get("id") or ""
        title2 = table2.get("title") or table2.get("id") or ""
        title1 = str(title1).strip() if title1 else ""
        title2 = str(title2).strip() if title2 else ""

        page1 = table1.get("page_number", 0) or 0
        page2 = table2.get("page_number", 0) or 0

        # Extraire les numéros de tableaux
        table_num1 = table1.get("table_number")
        table_num2 = table2.get("table_number")

        # Extraire les indicateurs de première colonne
        indicators1 = self._extract_first_column(table1)
        indicators2 = self._extract_first_column(table2)

        # Niveau 0: Matching par numéro de tableau (le plus fiable)
        if table_num1 and table_num2 and str(table_num1) == str(table_num2):
            indicator_sim = self._calculate_indicator_similarity(indicators1, indicators2)
            common, added, removed = self._get_indicator_diff(indicators1, indicators2)
            title_sim = self._calculate_title_similarity(title1, title2)

            return MatchResult(
                is_match=True,
                confidence=1.0,
                match_method="table_number",
                common_indicators=common,
                added_indicators=added,
                removed_indicators=removed,
                different_indicators=added + removed,
                title_similarity=title_sim,
                indicator_similarity=indicator_sim,
                is_renamed=(title_sim < 1.0),
                is_repositioned=(page1 != page2),
            )

        # Niveau 1: Matching exact par titre
        if self._normalize(title1) == self._normalize(title2):
            indicator_sim = self._calculate_indicator_similarity(indicators1, indicators2)
            common, added, removed = self._get_indicator_diff(indicators1, indicators2)

            return MatchResult(
                is_match=True,
                confidence=1.0,
                match_method="exact",
                common_indicators=common,
                added_indicators=added,
                removed_indicators=removed,
                different_indicators=added + removed,
                title_similarity=1.0,
                indicator_similarity=indicator_sim,
                is_repositioned=(page1 != page2),
            )

        # Niveau 2: Matching sémantique (titre OU indicateurs ≥0.90)
        title_similarity = self._calculate_title_similarity(title1, title2)
        indicator_similarity = self._calculate_indicator_similarity(indicators1, indicators2)

        if (
            title_similarity >= self.SEMANTIC_TITLE_THRESHOLD
            or indicator_similarity >= self.SEMANTIC_INDICATOR_THRESHOLD
        ):
            common, added, removed = self._get_indicator_diff(indicators1, indicators2)
            is_renamed = (
                title_similarity < 1.0 and title_similarity >= self.SEMANTIC_TITLE_THRESHOLD
            )

            return MatchResult(
                is_match=True,
                confidence=max(title_similarity, indicator_similarity),
                match_method="semantic",
                common_indicators=common,
                added_indicators=added,
                removed_indicators=removed,
                different_indicators=added + removed,
                title_similarity=title_similarity,
                indicator_similarity=indicator_similarity,
                is_renamed=is_renamed,
                is_repositioned=(page1 != page2),
            )

        # Niveau 3: Matching structurel (indicateurs communs ≥80%)
        if indicator_similarity >= self.STRUCTURAL_INDICATOR_THRESHOLD:
            common, added, removed = self._get_indicator_diff(indicators1, indicators2)

            return MatchResult(
                is_match=True,
                confidence=indicator_similarity,
                match_method="structural",
                common_indicators=common,
                added_indicators=added,
                removed_indicators=removed,
                different_indicators=added + removed,
                title_similarity=title_similarity,
                indicator_similarity=indicator_similarity,
                is_renamed=True,
                is_repositioned=(page1 != page2),
            )

        # Vérifier les patterns appris
        if bank_code and bank_code in self.learned_patterns:
            pattern_result = self._check_learned_patterns(
                bank_code, title1, title2, page1, page2, indicators1, indicators2
            )
            if pattern_result:
                return pattern_result

        # Pas de correspondance
        return MatchResult(
            is_match=False,
            confidence=max(title_similarity, indicator_similarity),
            match_method="none",
            common_indicators=[],
            different_indicators=[],
            title_similarity=title_similarity,
            indicator_similarity=indicator_similarity,
        )

    def _normalize(self, text: str) -> str:
        """Normalise le texte pour la comparaison."""
        if not text:
            return ""
        text = text.lower().strip()
        # Supprimer ponctuation et espaces multiples
        import re

        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_first_column(self, table: dict) -> List[str]:
        """Extrait les valeurs de la première colonne."""
        # Utiliser les indicateurs pré-extraits si disponibles
        if "first_column_indicators" in table and table["first_column_indicators"]:
            return table["first_column_indicators"]

        indicators = []
        rows = table.get("rows", [])

        for row in rows:
            if row and len(row) > 0:
                cell = str(row[0]).strip()
                # Ignorer les valeurs numériques et vides
                if cell and not self._is_numeric(cell):
                    indicators.append(cell)

        return indicators

    def _is_numeric(self, text: str) -> bool:
        """Vérifie si le texte est une valeur numérique."""
        import re

        # Pattern pour nombres, pourcentages, montants
        return bool(re.match(r"^[\d\s,.\-$%()]+$", text.strip()))

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calcule la similarité entre deux titres."""
        norm1 = self._normalize(title1)
        norm2 = self._normalize(title2)

        if not norm1 or not norm2:
            return 0.0

        return SequenceMatcher(None, norm1, norm2).ratio()

    def _calculate_indicator_similarity(
        self, indicators1: List[str], indicators2: List[str]
    ) -> float:
        """Calcule la similarité entre deux listes d'indicateurs."""
        norm1 = set(self._normalize(i) for i in indicators1 if i)
        norm2 = set(self._normalize(i) for i in indicators2 if i)

        if not norm1 and not norm2:
            return 1.0
        if not norm1 or not norm2:
            return 0.0

        # Jaccard similarity
        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)

        return intersection / union if union > 0 else 0.0

    def _get_indicator_diff(
        self, indicators1: List[str], indicators2: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Retourne les indicateurs communs, ajoutés et supprimés."""
        norm1 = {self._normalize(i): i for i in indicators1 if i}
        norm2 = {self._normalize(i): i for i in indicators2 if i}

        common_keys = set(norm1.keys()) & set(norm2.keys())
        added_keys = set(norm2.keys()) - set(norm1.keys())
        removed_keys = set(norm1.keys()) - set(norm2.keys())

        common = [norm1.get(k) or norm2.get(k) for k in sorted(list(common_keys))]
        added = [norm2.get(k) for k in sorted(list(added_keys))]
        removed = [norm1.get(k) for k in sorted(list(removed_keys))]

        return common, added, removed

    def _check_learned_patterns(
        self,
        bank_code: str,
        title1: str,
        title2: str,
        page1: int,
        page2: int,
        indicators1: List[str],
        indicators2: List[str],
    ) -> Optional[MatchResult]:
        """Vérifie si un pattern appris s'applique."""
        patterns = self.learned_patterns.get(bank_code, [])

        for pattern in patterns:
            if self._normalize(pattern.old_title) == self._normalize(title1) and self._normalize(
                pattern.new_title
            ) == self._normalize(title2):
                common, added, removed = self._get_indicator_diff(indicators1, indicators2)
                indicator_sim = self._calculate_indicator_similarity(indicators1, indicators2)

                return MatchResult(
                    is_match=True,
                    confidence=pattern.confidence,
                    match_method="learned_pattern",
                    common_indicators=common,
                    added_indicators=added,
                    removed_indicators=removed,
                    different_indicators=added + removed,
                    title_similarity=self._calculate_title_similarity(title1, title2),
                    indicator_similarity=indicator_sim,
                    is_renamed=(pattern.pattern_type in ["rename", "both"]),
                    is_repositioned=(pattern.pattern_type in ["reposition", "both"]),
                )

        return None

    def learn_pattern(
        self,
        bank_code: str,
        old_title: str,
        new_title: str,
        old_page: int,
        new_page: int,
        quarter_from: str,
        quarter_to: str,
        confidence: float,
    ):
        """
        Apprend un nouveau pattern de correspondance.

        Args:
            bank_code: Code de la banque
            old_title: Ancien titre du tableau
            new_title: Nouveau titre
            old_page: Ancienne position
            new_page: Nouvelle position
            quarter_from: Trimestre source
            quarter_to: Trimestre cible
            confidence: Score de confiance
        """
        pattern_type = "none"
        if old_title != new_title and old_page != new_page:
            pattern_type = "both"
        elif old_title != new_title:
            pattern_type = "rename"
        elif old_page != new_page:
            pattern_type = "reposition"
        else:
            return  # Rien à apprendre

        pattern = LearnedPattern(
            bank_code=bank_code,
            old_title=old_title,
            new_title=new_title,
            old_page=old_page,
            new_page=new_page,
            quarter_from=quarter_from,
            quarter_to=quarter_to,
            pattern_type=pattern_type,
            confidence=confidence,
        )

        if bank_code not in self.learned_patterns:
            self.learned_patterns[bank_code] = []

        # Vérifier si le pattern existe déjà
        for existing in self.learned_patterns[bank_code]:
            if existing.old_title == old_title and existing.new_title == new_title:
                existing.occurrences += 1
                existing.confidence = (existing.confidence + confidence) / 2
                self._save_patterns()
                return

        self.learned_patterns[bank_code].append(pattern)
        self._save_patterns()
        logger.info(f"Pattern appris: {bank_code} - {old_title} → {new_title}")

    def predict_position(self, bank_code: str, title: str, current_page: int) -> Optional[int]:
        """
        Prédit la position attendue d'un tableau dans le prochain trimestre.

        Args:
            bank_code: Code de la banque
            title: Titre du tableau
            current_page: Position actuelle

        Returns:
            Position prédite ou None
        """
        patterns = self.learned_patterns.get(bank_code, [])

        # Calculer le déplacement moyen pour ce type de tableau
        position_deltas = []

        for pattern in patterns:
            if pattern.pattern_type in ["reposition", "both"]:
                delta = pattern.new_page - pattern.old_page
                position_deltas.append(delta)

        if position_deltas:
            avg_delta = sum(position_deltas) / len(position_deltas)
            predicted = current_page + int(avg_delta)
            return max(1, predicted)  # Minimum page 1

        return None

    def get_statistics(self) -> dict:
        """Retourne les statistiques des patterns appris."""
        total_patterns = sum(len(p) for p in self.learned_patterns.values())

        stats = {
            "total_patterns": total_patterns,
            "patterns_par_banque": {
                bank: len(patterns) for bank, patterns in self.learned_patterns.items()
            },
            "types_patterns": {
                "rename": 0,
                "reposition": 0,
                "both": 0,
            },
        }

        for patterns in self.learned_patterns.values():
            for p in patterns:
                if p.pattern_type in stats["types_patterns"]:
                    stats["types_patterns"][p.pattern_type] += 1

        return stats


def find_best_match(
    target_table: dict,
    candidate_tables: List[dict],
    matcher: IndicatorMatcher = None,
    bank_code: str = None,
) -> Tuple[Optional[dict], Optional[MatchResult]]:
    """
    Trouve la meilleure correspondance pour un tableau parmi les candidats.

    Args:
        target_table: Tableau à matcher
        candidate_tables: Liste des tableaux candidats
        matcher: Instance de IndicatorMatcher
        bank_code: Code de la banque

    Returns:
        Tuple (meilleur_candidat, résultat_match) ou (None, None)
    """
    if matcher is None:
        matcher = IndicatorMatcher()

    best_match = None
    best_result = None
    best_confidence = 0.0

    for candidate in candidate_tables:
        result = matcher.match_tables(target_table, candidate, bank_code)

        if result.is_match and result.confidence > best_confidence:
            best_match = candidate
            best_result = result
            best_confidence = result.confidence

    return best_match, best_result


def detect_renamed_tables(
    tables_t1: List[dict],
    tables_t2: List[dict],
    matcher: IndicatorMatcher = None,
    bank_code: str = None,
) -> List[Tuple[dict, dict, MatchResult]]:
    """
    Détecte les tableaux qui ont été renommés entre deux trimestres.

    Returns:
        Liste de tuples (table_t1, table_t2, match_result)
    """
    if matcher is None:
        matcher = IndicatorMatcher()

    renamed = []
    matched_t2 = set()

    for t1 in tables_t1:
        for i, t2 in enumerate(tables_t2):
            if i in matched_t2:
                continue

            result = matcher.match_tables(t1, t2, bank_code)

            if result.is_match and result.is_renamed:
                renamed.append((t1, t2, result))
                matched_t2.add(i)

                # Apprendre le pattern
                matcher.learn_pattern(
                    bank_code=bank_code or "unknown",
                    old_title=t1.get("title", ""),
                    new_title=t2.get("title", ""),
                    old_page=t1.get("page_number", 0),
                    new_page=t2.get("page_number", 0),
                    quarter_from="T1",
                    quarter_to="T2",
                    confidence=result.confidence,
                )
                break

    return renamed
