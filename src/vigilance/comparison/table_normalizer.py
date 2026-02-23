"""
Normalisateur de tableaux pour gérer les structures hétérogènes entre banques et trimestres.
Permet la comparaison sémantique de tableaux avec des structures différentes.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple
from difflib import SequenceMatcher
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class NormalizedTable:
    """Tableau normalisé pour comparaison."""

    table_id: str
    original_headers: list[str]
    normalized_headers: list[str]
    header_mapping: dict[str, str]  # original -> normalized
    rows: list[dict[str, str]]  # Liste de {header_normalisé: valeur}
    row_keys: list[str]  # Identifiants des lignes
    is_transposed: bool = False
    original_structure: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "original_headers": self.original_headers,
            "normalized_headers": self.normalized_headers,
            "header_mapping": self.header_mapping,
            "rows": self.rows,
            "row_keys": self.row_keys,
            "is_transposed": self.is_transposed,
        }

    def get_row_by_key(self, key: str, fuzzy: bool = True) -> Optional[dict]:
        """Récupérer une ligne par sa clé, avec matching flou optionnel."""
        # Recherche exacte
        for i, row_key in enumerate(self.row_keys):
            if row_key == key:
                return self.rows[i]

        # Recherche floue
        if fuzzy:
            for i, row_key in enumerate(self.row_keys):
                if SequenceMatcher(None, key.lower(), row_key.lower()).ratio() > 0.85:
                    return self.rows[i]

        return None


@dataclass
class TableMatchResult:
    """Résultat du matching entre deux tableaux."""

    is_match: bool
    confidence: float
    matched_headers: list[Tuple[str, str]]  # (header1, header2)
    unmatched_headers_1: list[str]
    unmatched_headers_2: list[str]
    matched_rows: list[Tuple[str, str]]  # (key1, key2)
    unmatched_rows_1: list[str]
    unmatched_rows_2: list[str]
    transformation_needed: str  # "none", "transpose", "reorder"

    def to_dict(self) -> dict:
        return {
            "is_match": self.is_match,
            "confidence": self.confidence,
            "matched_headers": self.matched_headers,
            "unmatched_headers_1": self.unmatched_headers_1,
            "unmatched_headers_2": self.unmatched_headers_2,
            "matched_rows": self.matched_rows,
            "unmatched_rows_1": self.unmatched_rows_1,
            "unmatched_rows_2": self.unmatched_rows_2,
            "transformation_needed": self.transformation_needed,
        }


class TableNormalizer:
    """
    Normalisateur de tableaux pour permettre la comparaison de structures hétérogènes.

    Fonctionnalités:
    - Normalisation des en-têtes (synonymes, abréviations)
    - Transposition automatique si nécessaire
    - Matching sémantique des lignes
    - Gestion des colonnes réordonnées
    """

    # Dictionnaire de synonymes pour les en-têtes bancaires
    HEADER_SYNONYMS = {
        # Ratios de capital
        "cet1": ["ratio cet1", "common equity tier 1", "cet 1", "ratio de fonds propres cet1"],
        "tier1": ["ratio tier 1", "tier 1 capital", "t1", "fonds propres de catégorie 1"],
        "total_capital": ["ratio de capital total", "total capital ratio", "capital total"],
        # Ratios de liquidité
        "lcr": ["ratio de liquidité à court terme", "liquidity coverage ratio", "ratio lcr"],
        "nsfr": ["ratio de financement stable net", "net stable funding ratio", "ratio nsfr"],
        # Levier
        "levier": ["ratio de levier", "leverage ratio", "levier bâle iii"],
        # Périodes
        "t1": ["q1", "trimestre 1", "premier trimestre", "1er trimestre"],
        "t2": ["q2", "trimestre 2", "deuxième trimestre", "2e trimestre"],
        "t3": ["q3", "trimestre 3", "troisième trimestre", "3e trimestre"],
        "t4": ["q4", "trimestre 4", "quatrième trimestre", "4e trimestre"],
        # Dates
        "31_janvier": ["jan 31", "january 31", "31 jan"],
        "30_avril": ["apr 30", "april 30", "30 avr"],
        "31_juillet": ["jul 31", "july 31", "31 juil"],
        "31_octobre": ["oct 31", "october 31", "31 oct"],
        # Métriques
        "actifs_ponderes": [
            "actifs pondérés en fonction des risques",
            "rwa",
            "risk-weighted assets",
            "apr",
        ],
        "exposition": ["exposition totale", "total exposure", "mesure de l'exposition"],
        # Valeurs
        "variation": ["changement", "change", "delta", "écart", "différence"],
        "precedent": ["période précédente", "prior period", "trimestre précédent"],
    }

    # Patterns numériques à normaliser
    NUMERIC_PATTERNS = [
        (r"\$\s*", ""),  # Supprimer $
        (r"\s*%\s*", ""),  # Supprimer %
        (r"\s*M\s*$", ""),  # Supprimer M (millions)
        (r"\s*G\s*$", ""),  # Supprimer G (milliards)
        (r"\s*B\s*$", ""),  # Supprimer B (billions)
        (r"\(([^)]+)\)", r"-\1"),  # (x) -> -x
        (r",", ""),  # Supprimer virgules
        (r"\s+", ""),  # Supprimer espaces
        # --- Ajouts (anti faux positifs) ---
        # 1) Supprimer marqueurs de notes (unicode exposants + symboles fréquents)
        (r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", ""),  # exposants
        (r"(?<=\S)\s*[\*\u2020\u2021]+$", ""),  # *, †, ‡ en fin
        # 2) Supprimer chiffres "parasites" collés en fin de libellé (note/colonne),
        #    MAIS on doit préserver les chiffres sémantiques (Série 32, Pilier 1, Bâle III, etc.)
        #    -> à appliquer seulement si aucune "cue" sémantique juste avant.
        #    (voir fonction/guard ci-dessous)
        # (r"\s+\d{1,3}\s*$", ""),  # <-- NE PAS activer brut: à faire via guard
        # 3) Normaliser unités/variantes françaises courantes en fin
        (r"\s*(m\$|m\s*\$|mm\$|mn\s*\$|millions?)\s*$", ""),  # M$, m $, millions
        (r"\s*(g\$|md\$|milliards?)\s*$", ""),  # G$, Md$, milliards
        (r"\s*(k\$|milliers?)\s*$", ""),
    ]

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialiser le normalisateur.

        Args:
            similarity_threshold: Seuil de similarité pour le matching (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._synonym_index = self._build_synonym_index()

    def _build_synonym_index(self) -> dict[str, str]:
        """Construire un index inversé des synonymes."""
        index = {}
        for canonical, synonyms in self.HEADER_SYNONYMS.items():
            index[canonical.lower()] = canonical
            for syn in synonyms:
                index[syn.lower()] = canonical
        return index

    def normalize_table(self, table_data: dict, table_id: str = "table") -> NormalizedTable:
        """
        Normaliser un tableau pour la comparaison.

        Args:
            table_data: Données du tableau {"headers": [...], "rows": [[...], ...]}
            table_id: Identifiant du tableau

        Returns:
            NormalizedTable avec structure normalisée
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        # Normaliser les en-têtes
        normalized_headers, header_mapping = self._normalize_headers(headers)

        # Extraire les clés de lignes (première colonne)
        row_keys = []
        normalized_rows = []

        for row in rows:
            if not row:
                continue

            # La première colonne est la clé
            row_key = self._normalize_text(str(row[0])) if row else ""
            row_keys.append(row_key)

            # Créer un dictionnaire {header_normalisé: valeur}
            row_dict = {}
            for i, header in enumerate(normalized_headers):
                if i < len(row):
                    row_dict[header] = str(row[i]) if row[i] else ""
                else:
                    row_dict[header] = ""

            normalized_rows.append(row_dict)

        return NormalizedTable(
            table_id=table_id,
            original_headers=headers,
            normalized_headers=normalized_headers,
            header_mapping=header_mapping,
            rows=normalized_rows,
            row_keys=row_keys,
            is_transposed=False,
            original_structure={"num_rows": len(rows), "num_cols": len(headers)},
        )

    def _normalize_headers(self, headers: list[str]) -> Tuple[list[str], dict[str, str]]:
        """Normaliser les en-têtes en utilisant les synonymes."""
        normalized = []
        mapping = {}

        for header in headers:
            if not header:
                normalized.append("")
                continue

            header_str = str(header).strip()
            header_lower = header_str.lower()

            # Chercher dans l'index des synonymes
            if header_lower in self._synonym_index:
                norm = self._synonym_index[header_lower]
            else:
                # Chercher une correspondance partielle
                norm = self._find_partial_match(header_lower)
                if not norm:
                    norm = self._normalize_text(header_str)

            normalized.append(norm)
            mapping[header_str] = norm

        return normalized, mapping

    def _find_partial_match(self, text: str) -> Optional[str]:
        """Trouver une correspondance partielle dans les synonymes."""
        text = text.lower()

        for canonical, synonyms in self.HEADER_SYNONYMS.items():
            if canonical in text:
                return canonical
            for syn in synonyms:
                if syn in text or text in syn:
                    return canonical

        return None

    def _normalize_text(self, text: str) -> str:
        """Normaliser un texte (minuscules, sans accents, sans caractères spéciaux)."""
        if not text:
            return ""

        # Minuscules
        text = text.lower().strip()

        # Remplacer les caractères accentués
        replacements = {
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "à": "a",
            "â": "a",
            "ä": "a",
            "ù": "u",
            "û": "u",
            "ü": "u",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ö": "o",
            "ç": "c",
            "œ": "oe",
            "æ": "ae",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Supprimer les caractères spéciaux sauf underscore
        text = re.sub(r"[^a-z0-9_\s]", "", text)

        # Remplacer les espaces par underscore
        text = re.sub(r"\s+", "_", text)

        return text

    def normalize_numeric_value(self, value: str) -> Optional[float]:
        """Normaliser une valeur numérique."""
        if not value:
            return None

        text = str(value).strip()

        # Appliquer les patterns de nettoyage
        for pattern, replacement in self.NUMERIC_PATTERNS:
            text = re.sub(pattern, replacement, text)

        try:
            return float(text)
        except ValueError:
            return None

    def transpose_table(self, table: NormalizedTable) -> NormalizedTable:
        """
        Transposer un tableau (échanger lignes et colonnes).

        Utilisé quand un tableau est renversé par rapport à l'autre.
        """
        if not table.rows or not table.normalized_headers:
            return table

        # Les anciennes colonnes (headers) deviennent les nouvelles lignes
        # Les anciens row_keys deviennent les nouveaux headers

        new_headers = ["row_id"] + table.row_keys
        new_rows = []
        new_row_keys = []

        # Pour chaque ancien header (sauf le premier qui était row_key)
        for i, old_header in enumerate(table.normalized_headers[1:], 1):
            new_row = {}
            new_row["row_id"] = old_header
            new_row_keys.append(old_header)

            for j, row in enumerate(table.rows):
                col_name = table.row_keys[j] if j < len(table.row_keys) else f"col_{j}"
                # Récupérer la valeur de l'ancienne colonne i dans l'ancienne ligne j
                old_col_header = (
                    table.normalized_headers[i] if i < len(table.normalized_headers) else ""
                )
                new_row[col_name] = row.get(old_col_header, "")

            new_rows.append(new_row)

        return NormalizedTable(
            table_id=f"{table.table_id}_transposed",
            original_headers=table.original_headers,
            normalized_headers=new_headers,
            header_mapping=table.header_mapping,
            rows=new_rows,
            row_keys=new_row_keys,
            is_transposed=True,
            original_structure=table.original_structure,
        )

    def match_tables(self, table1: NormalizedTable, table2: NormalizedTable) -> TableMatchResult:
        """
        Déterminer si deux tableaux représentent les mêmes données.

        Args:
            table1: Premier tableau normalisé
            table2: Second tableau normalisé

        Returns:
            TableMatchResult avec les correspondances trouvées
        """
        # Matcher les en-têtes
        matched_headers = []
        unmatched_1 = list(table1.normalized_headers)
        unmatched_2 = list(table2.normalized_headers)

        for h1 in table1.normalized_headers:
            best_match = None
            best_score = 0

            for h2 in unmatched_2:
                score = self._header_similarity(h1, h2)
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = h2

            if best_match:
                matched_headers.append((h1, best_match))
                if h1 in unmatched_1:
                    unmatched_1.remove(h1)
                unmatched_2.remove(best_match)

        # Matcher les lignes (par row_key)
        matched_rows = []
        unmatched_rows_1 = list(table1.row_keys)
        unmatched_rows_2 = list(table2.row_keys)

        for key1 in table1.row_keys:
            best_match = None
            best_score = 0

            for key2 in unmatched_rows_2:
                score = SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = key2

            if best_match:
                matched_rows.append((key1, best_match))
                if key1 in unmatched_rows_1:
                    unmatched_rows_1.remove(key1)
                unmatched_rows_2.remove(best_match)

        # Calculer la confiance globale
        header_match_ratio = len(matched_headers) / max(len(table1.normalized_headers), 1)
        row_match_ratio = len(matched_rows) / max(len(table1.row_keys), 1)
        confidence = (header_match_ratio + row_match_ratio) / 2

        # Déterminer si transposition nécessaire
        transformation = "none"
        if confidence < 0.5:
            # Essayer avec transposition
            transposed = self.transpose_table(table2)
            transposed_result = self._simple_match(table1, transposed)
            if transposed_result > confidence:
                transformation = "transpose"
                confidence = transposed_result

        is_match = confidence >= 0.6

        return TableMatchResult(
            is_match=is_match,
            confidence=confidence,
            matched_headers=matched_headers,
            unmatched_headers_1=unmatched_1,
            unmatched_headers_2=unmatched_2,
            matched_rows=matched_rows,
            unmatched_rows_1=unmatched_rows_1,
            unmatched_rows_2=unmatched_rows_2,
            transformation_needed=transformation,
        )

    def _header_similarity(self, h1: str, h2: str) -> float:
        """Calculer la similarité entre deux en-têtes."""
        if h1 == h2:
            return 1.0

        # Vérifier si même synonyme canonique
        norm1 = self._synonym_index.get(h1.lower(), h1)
        norm2 = self._synonym_index.get(h2.lower(), h2)

        if norm1 == norm2:
            return 0.95

        # Similarité textuelle
        return SequenceMatcher(None, h1.lower(), h2.lower()).ratio()

    def _simple_match(self, table1: NormalizedTable, table2: NormalizedTable) -> float:
        """Calcul simple de matching pour comparaison."""
        header_matches = sum(
            1
            for h1 in table1.normalized_headers
            for h2 in table2.normalized_headers
            if self._header_similarity(h1, h2) >= self.similarity_threshold
        )

        row_matches = sum(
            1
            for k1 in table1.row_keys
            for k2 in table2.row_keys
            if SequenceMatcher(None, k1.lower(), k2.lower()).ratio() >= self.similarity_threshold
        )

        max_headers = max(len(table1.normalized_headers), len(table2.normalized_headers), 1)
        max_rows = max(len(table1.row_keys), len(table2.row_keys), 1)

        return (header_matches / max_headers + row_matches / max_rows) / 2

    def align_tables_for_comparison(
        self, table1: NormalizedTable, table2: NormalizedTable, match_result: TableMatchResult
    ) -> Tuple[NormalizedTable, NormalizedTable]:
        """
        Aligner deux tableaux pour permettre une comparaison directe.

        Args:
            table1: Premier tableau
            table2: Second tableau
            match_result: Résultat du matching

        Returns:
            Tuple des deux tableaux alignés
        """
        if match_result.transformation_needed == "transpose":
            table2 = self.transpose_table(table2)

        # Réordonner les colonnes de table2 pour correspondre à table1
        header_map = {h2: h1 for h1, h2 in match_result.matched_headers}

        aligned_rows_2 = []
        for row in table2.rows:
            aligned_row = {}
            for h2, value in row.items():
                # Utiliser le header mappé si disponible
                h1 = header_map.get(h2, h2)
                aligned_row[h1] = value
            aligned_rows_2.append(aligned_row)

        aligned_table2 = NormalizedTable(
            table_id=table2.table_id,
            original_headers=table2.original_headers,
            normalized_headers=table1.normalized_headers,  # Utiliser les headers de table1
            header_mapping=table2.header_mapping,
            rows=aligned_rows_2,
            row_keys=table2.row_keys,
            is_transposed=table2.is_transposed,
            original_structure=table2.original_structure,
        )

        return table1, aligned_table2


def normalize_table(table_data: dict, table_id: str = "table") -> NormalizedTable:
    """Fonction utilitaire pour normaliser un tableau."""
    normalizer = TableNormalizer()
    return normalizer.normalize_table(table_data, table_id)


def compare_normalized_tables(
    table1_data: dict, table2_data: dict, table1_id: str = "table1", table2_id: str = "table2"
) -> Tuple[NormalizedTable, NormalizedTable, TableMatchResult]:
    """
    Comparer deux tableaux avec normalisation automatique.

    Returns:
        Tuple (table1_normalisé, table2_normalisé, résultat_matching)
    """
    normalizer = TableNormalizer()

    t1 = normalizer.normalize_table(table1_data, table1_id)
    t2 = normalizer.normalize_table(table2_data, table2_id)

    match_result = normalizer.match_tables(t1, t2)

    if match_result.is_match:
        t1, t2 = normalizer.align_tables_for_comparison(t1, t2, match_result)

    return t1, t2, match_result
