"""
Multi-Signal Table Matcher pour comparaison haute precision (95-98%).

Ce module implemente un systeme de matching multi-criteres pour identifier
les correspondances entre tableaux de rapports trimestriels differents.

Signaux utilises:
- table_number (35%): Numero de tableau (TABLEAU 23, T24, etc.) - signal parfait
- fuzzy_labels (30%): Similarite des libelles de la 1ere colonne
- title_match (20%): Similarite des titres de tableaux
- structure_match (10%): Correspondance structure (colonnes, lignes)
- section_context (5%): Meme section parente (Capital vs Risques)
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None

# Essayer d'importer rapidfuzz, sinon fallback sur difflib
try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher

    RAPIDFUZZ_AVAILABLE = False
    logger.info("rapidfuzz non disponible, utilisation de difflib")


@dataclass
class TableSignature:
    """Signature d'un tableau pour le matching."""

    table_id: str
    page_number: int
    title: str = ""
    table_number: str = ""  # Numero de tableau extrait (T23, TABLEAU 25, etc.)
    first_column_labels: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    num_rows: int = 0
    num_columns: int = 0
    section_type: str = ""  # gestion_capital, gestion_risques, etc.
    raw_data: Optional[dict] = None


@dataclass
class MatchResult:
    """Resultat d'un matching entre deux tableaux."""

    table_t1: TableSignature
    table_t2: TableSignature
    total_score: float
    table_number_score: float
    fuzzy_label_score: float
    structure_score: float
    title_score: float
    section_score: float
    confidence_level: str  # "confirmed", "probable", "no_match"
    needs_review: bool = False
    match_details: dict = field(default_factory=dict)


class MultiSignalMatcher:
    """
    Matcher multi-signal pour comparaison de tableaux.

    Combine 5 signaux ponderes pour calculer un score de matching:
    - Numero de tableau (TABLEAU XX): 35% (signal parfait si present)
    - Labels 1ere colonne (fuzzy): 30%
    - Titre: 20%
    - Structure (colonnes/lignes): 10%
    - Section parente: 5%
    """

    # Poids des signaux (doivent sommer a 1.0)
    WEIGHT_TABLE_NUMBER = 0.35
    WEIGHT_FUZZY_LABELS = 0.30
    WEIGHT_TITLE = 0.20
    WEIGHT_STRUCTURE = 0.10
    WEIGHT_SECTION = 0.05

    # Patterns pour detecter numeros de tableau
    TABLE_NUMBER_PATTERNS = [
        r"TABLEAU\s*(\d+)",  # TABLEAU 23
        r"TABLE\s*(\d+)",  # TABLE 23
        r"\bT(\d+)\b",  # T23, T24
        r"Tableau\s*(\d+)",  # Tableau 23 (minuscule)
    ]

    # Seuils de decision
    THRESHOLD_CONFIRMED = 0.85
    THRESHOLD_PROBABLE = 0.65
    UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

    def __init__(
        self,
        weight_table_number: float = 0.35,
        weight_fuzzy: float = 0.30,
        weight_title: float = 0.20,
        weight_structure: float = 0.10,
        weight_section: float = 0.05,
        threshold_confirmed: float = 0.85,
        threshold_probable: float = 0.65,
    ):
        """Initialiser le matcher avec poids personnalisables."""
        total = (
            weight_table_number + weight_fuzzy + weight_structure + weight_title + weight_section
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Les poids doivent sommer a 1.0, obtenu: {total}")

        self.weight_table_number = weight_table_number
        self.weight_fuzzy = weight_fuzzy
        self.weight_structure = weight_structure
        self.weight_title = weight_title
        self.weight_section = weight_section
        self.threshold_confirmed = threshold_confirmed
        self.threshold_probable = threshold_probable

    def extract_table_number(self, text: str) -> str:
        """
        Extraire le numero de tableau depuis un titre ou texte.

        Detecte: TABLEAU 23, T24, Table 25, etc.
        Retourne le numero normalise ou chaine vide.
        """
        if not text:
            return ""

        for pattern in self.TABLE_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)  # Retourne juste le numero
        return ""

    def compute_table_number_score(
        self,
        sig_t1: "TableSignature",
        sig_t2: "TableSignature",
    ) -> float:
        """
        Calculer le score de correspondance par numero de tableau.

        Si les deux tableaux ont un numero et ils correspondent: 1.0
        Si un seul a un numero: 0.5 (neutre)
        Si aucun n'a de numero: 0.5 (neutre)
        Si les deux ont un numero different: 0.0
        """
        # Extraire depuis table_number ou titre
        num_t1 = sig_t1.table_number or self.extract_table_number(sig_t1.title)
        num_t2 = sig_t2.table_number or self.extract_table_number(sig_t2.title)

        if not num_t1 and not num_t2:
            return 0.5  # Neutre si aucun numero

        if not num_t1 or not num_t2:
            return 0.5  # Neutre si un seul a un numero

        # Les deux ont un numero
        return 1.0 if num_t1 == num_t2 else 0.0

    def compute_fuzzy_label_score(self, labels_t1: list[str], labels_t2: list[str]) -> float:
        """
        Calculer le score de similarite entre les labels de la 1ere colonne.

        Utilise rapidfuzz si disponible, sinon difflib.
        """
        if not labels_t1 or not labels_t2:
            return 0.0

        # Normaliser les labels
        labels_t1_norm = [self._normalize_label(l) for l in labels_t1]
        labels_t2_norm = [self._normalize_label(l) for l in labels_t2]

        # Calculer le ratio de labels correspondants
        matches = 0
        total_comparisons = max(len(labels_t1_norm), len(labels_t2_norm))

        for label1 in labels_t1_norm:
            best_match = 0.0
            for label2 in labels_t2_norm:
                score = self._fuzzy_ratio(label1, label2)
                best_match = max(best_match, score)
            if best_match >= 0.80:  # Seuil de correspondance
                matches += 1

        return matches / total_comparisons if total_comparisons > 0 else 0.0

    def compute_structure_score(
        self,
        sig_t1: TableSignature,
        sig_t2: TableSignature,
    ) -> float:
        """
        Calculer le score de correspondance structurelle.

        Compare:
        - Nombre de colonnes (40% du score structure)
        - Nombre de lignes (30%)
        - Headers similaires (30%)
        """
        score = 0.0

        # Similarite colonnes (tolerance 1)
        col_diff = abs(sig_t1.num_columns - sig_t2.num_columns)
        if col_diff == 0:
            score += 0.40
        elif col_diff == 1:
            score += 0.30
        elif col_diff == 2:
            score += 0.15

        # Similarite lignes (tolerance proportionnelle)
        if sig_t1.num_rows > 0 and sig_t2.num_rows > 0:
            row_ratio = min(sig_t1.num_rows, sig_t2.num_rows) / max(
                sig_t1.num_rows, sig_t2.num_rows
            )
            score += 0.30 * row_ratio

        # Similarite headers
        if sig_t1.headers and sig_t2.headers:
            header_score = self._fuzzy_ratio(" ".join(sig_t1.headers), " ".join(sig_t2.headers))
            score += 0.30 * header_score

        return score

    def compute_title_score(
        self,
        title_t1: str,
        title_t2: str,
    ) -> float:
        """Calculer le score de similarite des titres."""
        if not title_t1 or not title_t2:
            return 0.5  # Score neutre si pas de titre

        return self._fuzzy_ratio(self._normalize_label(title_t1), self._normalize_label(title_t2))

    def compute_section_score(
        self,
        section_t1: str,
        section_t2: str,
    ) -> float:
        """Calculer le score de contexte de section."""
        if not section_t1 or not section_t2:
            return 0.5  # Score neutre si pas de section

        # Meme section = 1.0, sinon 0.0
        return 1.0 if section_t1.lower() == section_t2.lower() else 0.0

    def compute_match_score(
        self,
        sig_t1: TableSignature,
        sig_t2: TableSignature,
    ) -> MatchResult:
        """
        Calculer le score total de matching entre deux signatures de tableaux.

        Retourne un MatchResult avec scores detailles.
        """
        # Calculer chaque signal (5 signaux)
        table_number_score = self.compute_table_number_score(sig_t1, sig_t2)
        fuzzy_score = self.compute_fuzzy_label_score(
            sig_t1.first_column_labels, sig_t2.first_column_labels
        )
        title_score = self.compute_title_score(sig_t1.title, sig_t2.title)
        structure_score = self.compute_structure_score(sig_t1, sig_t2)
        section_score = self.compute_section_score(sig_t1.section_type, sig_t2.section_type)

        # Score total pondere (5 signaux)
        total_score = (
            self.weight_table_number * table_number_score
            + self.weight_fuzzy * fuzzy_score
            + self.weight_title * title_score
            + self.weight_structure * structure_score
            + self.weight_section * section_score
        )

        # Determiner le niveau de confiance
        if total_score >= self.threshold_confirmed:
            confidence_level = "confirmed"
            needs_review = False
        elif total_score >= self.threshold_probable:
            confidence_level = "probable"
            needs_review = True
        else:
            confidence_level = "no_match"
            needs_review = False

        return MatchResult(
            table_t1=sig_t1,
            table_t2=sig_t2,
            total_score=total_score,
            table_number_score=table_number_score,
            fuzzy_label_score=fuzzy_score,
            structure_score=structure_score,
            title_score=title_score,
            section_score=section_score,
            confidence_level=confidence_level,
            needs_review=needs_review,
            match_details={
                "weights": {
                    "table_number": self.weight_table_number,
                    "fuzzy": self.weight_fuzzy,
                    "title": self.weight_title,
                    "structure": self.weight_structure,
                    "section": self.weight_section,
                },
                "thresholds": {
                    "confirmed": self.threshold_confirmed,
                    "probable": self.threshold_probable,
                },
            },
        )

    def find_best_matches(
        self,
        tables_t1: list[TableSignature],
        tables_t2: list[TableSignature],
    ) -> tuple[list[MatchResult], list[TableSignature], list[TableSignature]]:
        """
        Trouver les meilleurs matchs entre deux ensembles de tableaux.

        Returns:
            Tuple de:
            - Liste de MatchResult (matchs trouves)
            - Liste de tableaux T1 non matches
            - Liste de tableaux T2 non matches
        """
        matches: list[MatchResult] = []
        matched_t1_ids: set[str] = set()
        matched_t2_ids: set[str] = set()

        # Calculer tous les scores possibles
        all_scores: list[tuple[float, TableSignature, TableSignature, MatchResult]] = []

        for sig_t1 in tables_t1:
            for sig_t2 in tables_t2:
                if not self._sections_strict_match(sig_t1.section_type, sig_t2.section_type):
                    continue
                result = self.compute_match_score(sig_t1, sig_t2)
                if result.total_score >= self.threshold_probable:
                    all_scores.append((result.total_score, sig_t1, sig_t2, result))

        # Trier par score decroissant
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Greedy matching: prendre les meilleurs scores en premier
        for score, sig_t1, sig_t2, result in all_scores:
            if sig_t1.table_id not in matched_t1_ids and sig_t2.table_id not in matched_t2_ids:
                matches.append(result)
                matched_t1_ids.add(sig_t1.table_id)
                matched_t2_ids.add(sig_t2.table_id)

        # Identifier les tableaux non matches
        unmatched_t1 = [t for t in tables_t1 if t.table_id not in matched_t1_ids]
        unmatched_t2 = [t for t in tables_t2 if t.table_id not in matched_t2_ids]

        logger.info(
            f"Matching termine: {len(matches)} matches, "
            f"{len(unmatched_t1)} non-matches T1, {len(unmatched_t2)} non-matches T2"
        )

        return matches, unmatched_t1, unmatched_t2

    def _canonical_section(self, value: str) -> str:
        section = (value or "").strip()
        if not section:
            return ""
        if canonicalize_section is None:
            return section.lower()
        try:
            return canonicalize_section(section)
        except Exception:
            return section.lower()

    def _sections_strict_match(self, section_t1: str, section_t2: str) -> bool:
        left = self._canonical_section(section_t1)
        right = self._canonical_section(section_t2)
        left_known = left not in self.UNKNOWN_SECTIONS
        right_known = right not in self.UNKNOWN_SECTIONS
        if left_known and right_known:
            return left == right
        return True

    def _normalize_label(self, label: str) -> str:
        """Normaliser un label indicateur pour comparaison; delegue a normalize_indicator_for_comparison."""
        from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
        return normalize_indicator_for_comparison(label or "")

    def _fuzzy_ratio(self, s1: str, s2: str) -> float:
        """Calculer le ratio de similarite entre deux chaines."""
        if not s1 or not s2:
            return 0.0

        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(s1, s2) / 100.0
        else:
            return SequenceMatcher(None, s1, s2).ratio()


def create_signature_from_extracted_table(
    table_data: dict,
    section_type: str = "",
) -> TableSignature:
    """
    Creer une signature de tableau depuis les donnees extraites.

    Args:
        table_data: Dictionnaire avec les donnees du tableau
        section_type: Type de section parente

    Returns:
        TableSignature pour le matching
    """
    return TableSignature(
        table_id=table_data.get("id", table_data.get("table_id", str(id(table_data)))),
        page_number=table_data.get("page", table_data.get("page_number", 0)),
        title=table_data.get("title", ""),
        first_column_labels=table_data.get("first_column_indicators", []),
        headers=table_data.get("headers", []),
        num_rows=len(table_data.get("rows", [])),
        num_columns=len(table_data.get("headers", [])),
        section_type=section_type,
        raw_data=table_data,
    )
