"""
Classification des tableaux selon les 6 catégories définies dans la spécification.
Types: A (Ratios), B (Crédit), C (Marché), D (Liquidité), E (Hypothécaire), F (Financement)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from difflib import SequenceMatcher
from enum import Enum

logger = logging.getLogger(__name__)


class TableType(str, Enum):
    """Types de tableaux selon la spécification AGENT_PROMPT."""

    A_RATIOS_REGLEMENTAIRES = "A_RATIOS_REGLEMENTAIRES"
    B_RISQUE_CREDIT = "B_RISQUE_CREDIT"
    C_RISQUE_MARCHE = "C_RISQUE_MARCHE"
    D_LIQUIDITE = "D_LIQUIDITE"
    E_PRETS_HYPOTHECAIRES = "E_PRETS_HYPOTHECAIRES"
    F_FINANCEMENT = "F_FINANCEMENT"
    UNKNOWN = "UNKNOWN"


@dataclass
class TableClassification:
    """Résultat de classification d'un tableau."""

    table_type: TableType
    confidence: float
    matched_keywords: List[str]
    first_column_indicators: List[str]
    detection_method: str  # "exact", "semantic", "indicators"

    def to_dict(self) -> dict:
        return {
            "type_tableau": self.table_type.value,
            "confiance": round(self.confidence, 2),
            "mots_cles_matches": self.matched_keywords,
            "indicateurs_colonne1": self.first_column_indicators,
            "methode_detection": self.detection_method,
        }


# Mots-clés pour chaque type de tableau (FR + EN + Codes Pilier 3)
TABLE_TYPE_KEYWORDS = {
    TableType.A_RATIOS_REGLEMENTAIRES: {
        "primary": [
            "cet1",
            "tier 1",
            "tier 2",
            "ratio",
            "fonds propres",
            "capital réglementaire",
            "tlac",
            # Codes Pilier 3 - Capital
            "ov1",
            "ov2",
            "cc1",
            "cc2",
            "cca",
            "km1",
            "km2",
            "gsib1",
            # Codes Pilier 3 - Leverage
            "lr1",
            "lr2",
        ],
        "secondary": [
            "bâle",
            "levier",
            "regulatory capital",
            "capital ratio",
            "absorption des pertes",
            "rwa",
            "risk-weighted",
        ],
        "titles": [
            "ratios de fonds propres",
            "capital réglementaire",
            "structure de fonds propres",
            "ratios réglementaires",
            "capacité totale d'absorption",
            "ov1 - overview of rwa",
            "cc1 - composition of regulatory capital",
            "lr1 - leverage ratio",
            "km1 - key metrics",
        ],
    },
    TableType.B_RISQUE_CREDIT: {
        "primary": [
            "exposition",
            "risque de crédit",
            "crédit",
            "exposure",
            "credit risk",
            # Codes Pilier 3 - Credit Risk
            "cr1",
            "cr2",
            "cr3",
            "cr4",
            "cr5",
            "cr6",
            "cr7",
            "cr8",
            "cr9",
            "cr10",
            # Codes Pilier 3 - Counterparty Credit Risk
            "ccr1",
            "ccr2",
            "ccr3",
            "ccr4",
            "ccr5",
            "ccr6",
            "ccr7",
            "ccr8",
            # Codes Pilier 3 - Securitization
            "sec1",
            "sec2",
            "sec3",
            "sec4",
            # IFRS 9 - Expected Credit Loss
            "ifrs 9",
            "ecl",
            "expected credit loss",
            "perte de crédit attendue",
            "stage 1",
            "stage 2",
            "stage 3",
        ],
        "secondary": [
            "région",
            "géographique",
            "prêts",
            "provisions",
            "défaut",
            "credit quality",
            "defaulted",
            "allowance",
            "provision",
            "impairment",
            "dépréciation",
        ],
        "titles": [
            "expositions au risque de crédit",
            "expositions par région",
            "risque de crédit par région",
            "credit exposure",
            "cr1 - credit quality of assets",
            "ifrs 9 expected credit loss",
            "ecl by stage",
            "perte de crédit attendueccr1 - analysis by approach",
            "sec1 - securitization exposures",
        ],
    },
    TableType.C_RISQUE_MARCHE: {
        "primary": [
            "risque de marché",
            "taux d'intérêt",
            "market risk",
            "var",
            "sensibilité",
            # Codes Pilier 3 - Market Risk
            "mr1",
            "mr2",
            "mr3",
            "mr4",
            # Codes Pilier 3 - Interest Rate Risk
            "irrbb1",
        ],
        "secondary": [
            "duration",
            "gap",
            "trading",
            "négociation",
            "devise",
            "standardized approach",
        ],
        "titles": [
            "mesures du risque de marché",
            "sensibilité aux taux",
            "risque de marché",
            "liens entre le risque de marché",
            "mr1 - market risk under standardized approach",
            "irrbb1 - interest rate risk",
        ],
    },
    TableType.D_LIQUIDITE: {
        "primary": [
            "liquidité",
            "lcr",
            "nsfr",
            "actifs liquides",
            "liquidity",
            # Codes Pilier 3 - Liquidity
            "liq1",
            "liq2",
            "liqa",
        ],
        "secondary": [
            "court terme",
            "long terme",
            "haute qualité",
            "hqla",
            "coverage ratio",
            "stable funding",
        ],
        "titles": [
            "ratio de liquidité",
            "ratio structurel de liquidité",
            "actifs liquides de haute qualité",
            "lcr",
            "nsfr",
            "liq1 - liquidity coverage ratio",
            "liq2 - net stable funding ratio",
        ],
    },
    TableType.E_PRETS_HYPOTHECAIRES: {
        "primary": ["hypothécaire", "habitation", "ltv", "prêt/valeur", "mortgage"],
        "secondary": ["domiciliaire", "résidentiel", "immobilier", "garantie"],
        "titles": [
            "prêts hypothécaires",
            "marges de crédit sur valeur domiciliaire",
            "ratios prêt/valeur",
            "prêts à l'habitation",
        ],
    },
    TableType.F_FINANCEMENT: {
        "primary": [
            "échéance",
            "financement",
            "dette",
            "maturity",
            "funding",
            # Codes Pilier 3 - Operational Risk
            "or1",
            "or2",
            "or3",
            # Codes Pilier 3 - Remuneration
            "rem1",
            "rem2",
            "rem3",
        ],
        "secondary": [
            "gros",
            "wholesale",
            "contractuel",
            "obligations",
            "operational risk",
            "remuneration",
        ],
        "titles": [
            "échéances contractuelles",
            "financement de gros",
            "échéances du financement",
            "funding maturity",
            "or1 - historical losses",
            "rem1 - remuneration awarded",
        ],
    },
}


# Indicateurs standardisés pour la première colonne
STANDARD_INDICATORS = {
    "ratios_capital": [
        "cet1",
        "tier 1",
        "tier 2",
        "ratio de levier",
        "tlac",
        "ratio tlac",
        "fonds propres",
        "capital réglementaire",
        "coussin",
        "minimum requis",
    ],
    "elements_bilan": [
        "trésorerie",
        "dépôts",
        "valeurs mobilières",
        "prêts",
        "actifs",
        "passifs",
        "fonds propres",
        "dette",
        "titres",
    ],
    "regions": [
        "atlantique",
        "québec",
        "ontario",
        "alberta",
        "colombie-britannique",
        "ailleurs au canada",
        "états-unis",
        "europe",
        "royaume-uni",
        "amérique latine",
        "asie-pacifique",
        "autres pays",
    ],
    "periodes": [
        "au 30 avril",
        "au 31 janvier",
        "au 31 juillet",
        "au 31 octobre",
        "trimestre",
        "exercice",
    ],
}


# Mapping bilingue FR <-> EN
BILINGUAL_MAPPING = {
    # Termes de capital
    "fonds propres": "equity capital",
    "capital réglementaire": "regulatory capital",
    "ratio de levier": "leverage ratio",
    "absorption des pertes": "loss absorption",
    # Termes de risque
    "risque de crédit": "credit risk",
    "risque de marché": "market risk",
    "risque de liquidité": "liquidity risk",
    "risque opérationnel": "operational risk",
    "exposition": "exposure",
    # Termes réglementaires
    "exigence minimale": "minimum requirement",
    "coussin de conservation": "conservation buffer",
    "coussin contracyclique": "countercyclical buffer",
    # Termes géographiques
    "états-unis": "united states",
    "royaume-uni": "united kingdom",
    "amérique latine": "latin america",
    "asie-pacifique": "asia pacific",
    # Autres
    "prêts hypothécaires": "mortgage loans",
    "actifs liquides": "liquid assets",
    "financement de gros": "wholesale funding",
}


class TableClassifier:
    """
    Classifie les tableaux selon les 6 catégories (A-F) de la spécification.
    Utilise une approche multi-méthode: exact, sémantique, indicateurs.
    """

    # Seuils de confiance
    TITLE_MATCH_THRESHOLD = 0.85
    KEYWORD_MATCH_THRESHOLD = 0.70
    INDICATOR_MATCH_THRESHOLD = 0.80

    def __init__(self, custom_thresholds: dict = None):
        """
        Initialise le classificateur.

        Args:
            custom_thresholds: Seuils personnalisés {
                "title_match": float,
                "keyword_match": float,
                "indicator_match": float
            }
        """
        if custom_thresholds:
            self.TITLE_MATCH_THRESHOLD = custom_thresholds.get(
                "title_match", self.TITLE_MATCH_THRESHOLD
            )
            self.KEYWORD_MATCH_THRESHOLD = custom_thresholds.get(
                "keyword_match", self.KEYWORD_MATCH_THRESHOLD
            )
            self.INDICATOR_MATCH_THRESHOLD = custom_thresholds.get(
                "indicator_match", self.INDICATOR_MATCH_THRESHOLD
            )

        # Créer le mapping inverse EN -> FR
        self.en_to_fr = {v.lower(): k.lower() for k, v in BILINGUAL_MAPPING.items()}
        self.fr_to_en = {k.lower(): v.lower() for k, v in BILINGUAL_MAPPING.items()}

    def classify_table(
        self, table_title: str, first_column: List[str] = None, table_content: str = None
    ) -> TableClassification:
        """
        Classifie un tableau selon les 6 catégories.

        Args:
            table_title: Titre du tableau
            first_column: Liste des valeurs de la première colonne
            table_content: Contenu textuel du tableau (optionnel)

        Returns:
            TableClassification avec type et confiance
        """
        normalized_title = self._normalize_text(table_title)
        first_column = first_column or []

        # Méthode 1: Matching exact du titre
        result = self._match_by_title(normalized_title)
        if result and result.confidence >= self.TITLE_MATCH_THRESHOLD:
            result.first_column_indicators = self._extract_indicators(first_column)
            return result

        # Méthode 2: Matching par mots-clés
        result = self._match_by_keywords(normalized_title, first_column)
        if result and result.confidence >= self.KEYWORD_MATCH_THRESHOLD:
            return result

        # Méthode 3: Matching par indicateurs de première colonne
        if first_column:
            result = self._match_by_indicators(first_column)
            if result and result.confidence >= self.INDICATOR_MATCH_THRESHOLD:
                result.detection_method = "indicators"
                return result

        # Fallback: UNKNOWN avec basse confiance
        return TableClassification(
            table_type=TableType.UNKNOWN,
            confidence=0.0,
            matched_keywords=[],
            first_column_indicators=self._extract_indicators(first_column),
            detection_method="none",
        )

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour la comparaison."""
        if not text:
            return ""
        text = text.lower()
        # Normaliser les accents et caractères spéciaux
        replacements = {
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "à": "a",
            "â": "a",
            "ä": "a",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ö": "o",
            "ù": "u",
            "û": "u",
            "ü": "u",
            "ç": "c",
            "'": " ",
            "-": " ",
            "_": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Supprimer espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _match_by_title(self, title: str) -> Optional[TableClassification]:
        """Match par similarité de titre."""
        best_match = None
        best_score = 0.0
        best_keywords = []

        for table_type, keywords in TABLE_TYPE_KEYWORDS.items():
            for ref_title in keywords["titles"]:
                normalized_ref = self._normalize_text(ref_title)

                # Score par similarité de séquence
                seq_score = SequenceMatcher(None, title, normalized_ref).ratio()

                # Score par inclusion de sous-chaîne
                inclusion_score = 0.0
                if normalized_ref in title:
                    inclusion_score = len(normalized_ref) / len(title) if title else 0
                elif title in normalized_ref:
                    inclusion_score = len(title) / len(normalized_ref) if normalized_ref else 0

                # Prendre le meilleur des deux
                score = max(seq_score, inclusion_score)

                if score > best_score:
                    best_score = score
                    best_match = table_type
                    best_keywords = [ref_title]

        if best_match and best_score >= 0.4:  # Seuil abaissé
            return TableClassification(
                table_type=best_match,
                confidence=best_score,
                matched_keywords=best_keywords,
                first_column_indicators=[],
                detection_method="title_matching",
            )
        return None

    def _match_by_keywords(
        self, title: str, first_column: List[str]
    ) -> Optional[TableClassification]:
        """Match par mots-clés dans le titre et la première colonne."""
        combined_text = title + " " + " ".join(self._normalize_text(str(c)) for c in first_column)

        scores = {}
        matched_kws = {}

        for table_type, keywords in TABLE_TYPE_KEYWORDS.items():
            primary_matches = [
                kw for kw in keywords["primary"] if self._normalize_text(kw) in combined_text
            ]
            secondary_matches = [
                kw for kw in keywords["secondary"] if self._normalize_text(kw) in combined_text
            ]

            # Score pondéré: primary = 3 points, secondary = 1 point
            # Au moins 1 mot-clé primaire donne un score de base
            if primary_matches:
                base_score = 0.5
                additional = min(len(primary_matches) * 0.15 + len(secondary_matches) * 0.05, 0.5)
                scores[table_type] = base_score + additional
                matched_kws[table_type] = primary_matches + secondary_matches
            elif secondary_matches and len(secondary_matches) >= 2:
                scores[table_type] = 0.3 + min(len(secondary_matches) * 0.1, 0.4)
                matched_kws[table_type] = secondary_matches

        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] >= 0.3:  # Au moins 30% de confiance
                return TableClassification(
                    table_type=best_type,
                    confidence=scores[best_type],
                    matched_keywords=matched_kws.get(best_type, []),
                    first_column_indicators=self._extract_indicators(first_column),
                    detection_method="keyword_matching",
                )
        return None

    def _match_by_indicators(self, first_column: List[str]) -> Optional[TableClassification]:
        """Match par indicateurs de la première colonne."""
        normalized_col = [self._normalize_text(str(c)) for c in first_column]
        combined = " ".join(normalized_col)

        # Chercher des indicateurs dans chaque catégorie
        indicator_scores = {
            TableType.A_RATIOS_REGLEMENTAIRES: 0,
            TableType.B_RISQUE_CREDIT: 0,
            TableType.D_LIQUIDITE: 0,
            TableType.E_PRETS_HYPOTHECAIRES: 0,
        }

        for indicator in STANDARD_INDICATORS["ratios_capital"]:
            if self._normalize_text(indicator) in combined:
                indicator_scores[TableType.A_RATIOS_REGLEMENTAIRES] += 1

        for indicator in STANDARD_INDICATORS["regions"]:
            if self._normalize_text(indicator) in combined:
                indicator_scores[TableType.B_RISQUE_CREDIT] += 1

        # LCR/NSFR indiquent liquidité
        if "lcr" in combined or "nsfr" in combined or "liquidite" in combined:
            indicator_scores[TableType.D_LIQUIDITE] += 3

        # Hypothécaire
        if "hypothecaire" in combined or "habitation" in combined or "ltv" in combined:
            indicator_scores[TableType.E_PRETS_HYPOTHECAIRES] += 3

        if indicator_scores:
            best_type = max(indicator_scores, key=indicator_scores.get)
            max_score = indicator_scores[best_type]
            if max_score >= 2:
                # Normaliser le score
                confidence = min(1.0, max_score / 5.0)
                return TableClassification(
                    table_type=best_type,
                    confidence=confidence,
                    matched_keywords=[],
                    first_column_indicators=self._extract_indicators(first_column),
                    detection_method="indicators",
                )
        return None

    def _extract_indicators(self, first_column: List[str]) -> List[str]:
        """
        Extrait les indicateurs valides de la première colonne.
        Ignore: valeurs numériques, dates, références de notes.
        """
        indicators = []

        for cell in first_column:
            if not cell:
                continue
            text = str(cell).strip()

            # Ignorer les valeurs numériques pures
            if re.match(r"^[\d\s,.\-$%()]+$", text):
                continue

            # Ignorer les références de notes (1), (2), etc.
            if re.match(r"^\(\d+\)$", text):
                continue

            # Ignorer les textes trop courts
            if len(text) < 3:
                continue

            # Nettoyer les références de notes en fin de texte
            text = re.sub(r"\s*\(\d+\)\s*$", "", text)

            if text:
                indicators.append(text)

        return indicators

    def normalize_term(self, term: str, target_language: str = "fr") -> str:
        """
        Normalise un terme selon le mapping bilingue.

        Args:
            term: Terme à normaliser
            target_language: "fr" ou "en"

        Returns:
            Terme normalisé dans la langue cible
        """
        normalized = self._normalize_text(term)

        if target_language == "fr":
            return self.en_to_fr.get(normalized, term)
        else:
            return self.fr_to_en.get(normalized, term)

    def calculate_indicator_similarity(
        self, indicators1: List[str], indicators2: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calcule la similarité entre deux listes d'indicateurs.

        Returns:
            Tuple (score_similarité, indicateurs_communs, indicateurs_différents)
        """
        norm1 = set(self._normalize_text(i) for i in indicators1)
        norm2 = set(self._normalize_text(i) for i in indicators2)

        if not norm1 and not norm2:
            return 1.0, [], []

        if not norm1 or not norm2:
            return 0.0, [], list(norm1 | norm2)

        common = norm1 & norm2
        all_indicators = norm1 | norm2

        similarity = len(common) / len(all_indicators)

        return (similarity, list(common), list(norm1.symmetric_difference(norm2)))


def classify_tables_batch(
    tables: List[dict], classifier: TableClassifier = None
) -> List[Tuple[dict, TableClassification]]:
    """
    Classifie un lot de tableaux.

    Args:
        tables: Liste de tableaux avec 'title', 'rows', etc.
        classifier: Instance de TableClassifier (optionnel)

    Returns:
        Liste de tuples (table, classification)
    """
    if classifier is None:
        classifier = TableClassifier()

    results = []
    for table in tables:
        title = table.get("title", table.get("id", ""))

        # Extraire la première colonne
        first_column = []
        rows = table.get("rows", [])
        for row in rows:
            if row and len(row) > 0:
                first_column.append(row[0])

        classification = classifier.classify_table(title, first_column)
        results.append((table, classification))

    return results
