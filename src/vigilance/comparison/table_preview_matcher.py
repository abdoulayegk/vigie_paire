"""
Matcher de tableaux pour l'interface de preview cote-a-cote.

Ce module associe les tableaux entre T1 et T2 pour l'affichage
de la comparaison visuelle avec annotations des changements.

Pipeline:
1. Extraire les images de tableaux avec TableImageExtractor
2. Matcher les tableaux par titre ou similarite des indicateurs
3. Detecter les lignes ajoutees/supprimees par tableau
4. Retourner les paires pour affichage cote-a-cote
"""

import logging
import unicodedata
from dataclasses import dataclass, field

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple, Set, Any

logger = logging.getLogger(__name__)
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None

try:
    from vigilance.comparison.robust_table_matcher import match_best as robust_match_best
    from vigilance.config import get_matching_thresholds
    from vigilance.utils.genai import get_openai_api_key

    ROBUST_MATCHER_AVAILABLE = True
except ImportError:
    ROBUST_MATCHER_AVAILABLE = False

# Import de TableImage (import conditionnel)
try:
    from vigilance.extraction.table_image_extractor import (
        TableImage,
        TableImageExtractor,
        ExtractionResult,
    )

    TABLE_IMAGE_AVAILABLE = True
except ImportError:
    TABLE_IMAGE_AVAILABLE = False
    logger.warning("TableImageExtractor non disponible")


@dataclass
class RowAnnotation:
    """Annotation pour une ligne de tableau."""

    indicator: str  # Texte de l'indicateur
    y_position: float  # Position Y estimee dans l'image
    change_type: str  # "added" ou "removed"
    verified: bool = False  # Si l'utilisateur a verifie

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "y_position": self.y_position,
            "change_type": self.change_type,
            "verified": self.verified,
        }


@dataclass
class TablePair:
    """Paire de tableaux T1/T2 pour comparaison cote-a-cote."""

    # Images des tableaux (peuvent etre None si tableau absent)
    table_t1: Optional[Any] = None  # TableImage
    table_t2: Optional[Any] = None  # TableImage

    # Titre du tableau
    title: str = ""

    # Changements detectes
    rows_added: List[str] = field(default_factory=list)
    rows_removed: List[str] = field(default_factory=list)

    # Annotations avec positions
    annotations_t1: List[RowAnnotation] = field(default_factory=list)
    annotations_t2: List[RowAnnotation] = field(default_factory=list)

    # Flags pour tableaux entiers
    is_new_table: bool = False  # Tableau entierement nouveau (absent de T1)
    is_removed_table: bool = False  # Tableau entierement supprime (absent de T2)

    # Numero de page
    page_t1: Optional[int] = None
    page_t2: Optional[int] = None

    # Bounding boxes des tableaux pour highlighting sur pages PDF completes
    # Format: (x0, y0, x1, y1) en coordonnees PDF
    bbox_t1: Optional[Tuple[float, float, float, float]] = None
    bbox_t2: Optional[Tuple[float, float, float, float]] = None

    @property
    def has_changes(self) -> bool:
        """Verifier si cette paire a des changements."""
        return (
            len(self.rows_added) > 0
            or len(self.rows_removed) > 0
            or self.is_new_table
            or self.is_removed_table
        )

    @property
    def total_changes(self) -> int:
        """Nombre total de changements."""
        return len(self.rows_added) + len(self.rows_removed)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "rows_added": self.rows_added,
            "rows_removed": self.rows_removed,
            "annotations_t1": [a.to_dict() for a in self.annotations_t1],
            "annotations_t2": [a.to_dict() for a in self.annotations_t2],
            "is_new_table": self.is_new_table,
            "is_removed_table": self.is_removed_table,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "bbox_t1": self.bbox_t1,
            "bbox_t2": self.bbox_t2,
            "has_changes": self.has_changes,
            "total_changes": self.total_changes,
        }


@dataclass
class PreviewMatchResult:
    """Resultat du matching pour preview."""

    table_pairs: List[TablePair] = field(default_factory=list)
    total_tables_t1: int = 0
    total_tables_t2: int = 0
    tables_matched: int = 0
    tables_added: int = 0
    tables_removed: int = 0

    # Champs de diagnostic
    extraction_errors_t1: List[str] = field(default_factory=list)
    extraction_errors_t2: List[str] = field(default_factory=list)
    page_ranges_t1: List[Tuple[int, int]] = field(default_factory=list)
    page_ranges_t2: List[Tuple[int, int]] = field(default_factory=list)
    pages_processed_t1: int = 0
    pages_processed_t2: int = 0
    extraction_method_t1: str = ""
    extraction_method_t2: str = ""

    @property
    def total_pairs(self) -> int:
        return len(self.table_pairs)

    @property
    def pairs_with_changes(self) -> List[TablePair]:
        return [p for p in self.table_pairs if p.has_changes]

    @property
    def has_extraction_errors(self) -> bool:
        """Verifier s'il y a des erreurs d'extraction."""
        return len(self.extraction_errors_t1) > 0 or len(self.extraction_errors_t2) > 0

    def to_dict(self) -> dict:
        return {
            "table_pairs": [p.to_dict() for p in self.table_pairs],
            "total_tables_t1": self.total_tables_t1,
            "total_tables_t2": self.total_tables_t2,
            "tables_matched": self.tables_matched,
            "tables_added": self.tables_added,
            "tables_removed": self.tables_removed,
            "total_pairs": self.total_pairs,
            "pairs_with_changes": len(self.pairs_with_changes),
            "extraction_errors_t1": self.extraction_errors_t1,
            "extraction_errors_t2": self.extraction_errors_t2,
            "page_ranges_t1": self.page_ranges_t1,
            "page_ranges_t2": self.page_ranges_t2,
            "pages_processed_t1": self.pages_processed_t1,
            "pages_processed_t2": self.pages_processed_t2,
            "extraction_method_t1": self.extraction_method_t1,
            "extraction_method_t2": self.extraction_method_t2,
            "has_extraction_errors": self.has_extraction_errors,
        }


class TablePreviewMatcher:
    """
    Matcher de tableaux pour la preview cote-a-cote.

    Associe les tableaux entre T1 et T2 (quel que soit le numéro de page),
    detecte les changements de lignes (uniquement sur la première colonne = indicateurs),
    et prepare les annotations pour l'affichage.
    """

    # Seuil de similarite Jaccard pour matcher les tableaux par indicateurs
    MATCH_THRESHOLD = 0.5
    # Seuil de recouvrement: intersection / min(len1, len2) pour accepter un match
    OVERLAP_THRESHOLD = 0.7
    # Seuil de similarite pour le match par titre (phase 1b)
    TITLE_SIMILARITY_THRESHOLD = 0.8

    def __init__(
        self,
        match_threshold: float = 0.5,
        overlap_threshold: float = 0.7,
        title_similarity_threshold: float = 0.8,
        normalize_indicators: bool = True,
        use_robust_matcher: bool = True,
    ):
        """
        Initialiser le matcher.

        Args:
            match_threshold: Seuil Jaccard pour matcher les tableaux par indicateurs
            overlap_threshold: Seuil recouvrement (intersection/min) pour accepter un match
            title_similarity_threshold: Seuil pour le match par similarite de titre (phase 1b)
            normalize_indicators: Normaliser les indicateurs avant comparaison
            use_robust_matcher: Utiliser robust_table_matcher + Vision hybride si disponible
        """
        self.match_threshold = match_threshold
        self.overlap_threshold = overlap_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.normalize_indicators = normalize_indicators
        self.use_robust_matcher = use_robust_matcher and ROBUST_MATCHER_AVAILABLE

    def match_tables(
        self,
        tables_t1: List[Any],  # List[TableImage]
        tables_t2: List[Any],  # List[TableImage]
    ) -> PreviewMatchResult:
        """
        Matcher les tableaux entre T1 et T2 (independant du numero de page).

        Le matching se fait par titre (exact puis similarite) puis par similarite
        des indicateurs (premiere colonne uniquement). La comparaison des changements
        (lignes ajoutees/supprimees) est strictement limitee a la premiere colonne
        (indicateurs) de chaque tableau.

        Args:
            tables_t1: Liste des TableImage de T1
            tables_t2: Liste des TableImage de T2

        Returns:
            PreviewMatchResult avec les paires de tableaux
        """
        result = PreviewMatchResult(total_tables_t1=len(tables_t1), total_tables_t2=len(tables_t2))

        if self.use_robust_matcher:
            return self._match_tables_robust(tables_t1, tables_t2, result)

        # Index des tableaux deja matches (legacy flow)
        matched_t1_indices: Set[int] = set()
        matched_t2_indices: Set[int] = set()

        # Phase 1: Match exact par titre
        for i, t1 in enumerate(tables_t1):
            t1_title = self._normalize_text(self._get_title(t1))
            if not t1_title:
                continue

            for j, t2 in enumerate(tables_t2):
                if j in matched_t2_indices:
                    continue

                t2_title = self._normalize_text(self._get_title(t2))

                if t1_title == t2_title and t1_title:
                    # Verifier compatibilite des sections
                    if not self._sections_compatible(t1, t2):
                        continue
                    # Match exact
                    pair = self._create_pair(t1, t2, i, j)
                    result.table_pairs.append(pair)
                    matched_t1_indices.add(i)
                    matched_t2_indices.add(j)
                    result.tables_matched += 1
                    break

        # Phase 1b: Match par similarite de titre (petites variations)
        unmatched_t1_after_1 = [
            (i, t) for i, t in enumerate(tables_t1) if i not in matched_t1_indices
        ]
        for i, t1 in unmatched_t1_after_1:
            t1_title = self._normalize_text(self._get_title(t1))
            if not t1_title or len(t1_title) < 10:
                continue
            for j, t2 in enumerate(tables_t2):
                if j in matched_t2_indices:
                    continue
                t2_title = self._normalize_text(self._get_title(t2))
                if not t2_title:
                    continue
                sim = self._title_similarity(t1_title, t2_title)
                if sim >= self.title_similarity_threshold:
                    # Verifier compatibilite des sections
                    if not self._sections_compatible(t1, t2):
                        continue
                    pair = self._create_pair(t1, t2, i, j)
                    result.table_pairs.append(pair)
                    matched_t1_indices.add(i)
                    matched_t2_indices.add(j)
                    result.tables_matched += 1
                    break

        # Phase 2: Match par similarite des indicateurs (premiere colonne)
        unmatched_t1 = [(i, t) for i, t in enumerate(tables_t1) if i not in matched_t1_indices]
        unmatched_t2 = [(j, t) for j, t in enumerate(tables_t2) if j not in matched_t2_indices]

        for i, t1 in unmatched_t1:
            best_match = None
            best_score = 0.0

            indicators_t1 = self._get_indicators(t1)
            if not indicators_t1:
                continue

            for j, t2 in unmatched_t2:
                if j in matched_t2_indices:
                    continue

                indicators_t2 = self._get_indicators(t2)
                if not indicators_t2:
                    continue

                # Verifier compatibilite des sections avant le calcul des scores
                if not self._sections_compatible(t1, t2):
                    continue

                jaccard = self._jaccard_similarity(indicators_t1, indicators_t2)
                overlap = self._indicator_overlap(indicators_t1, indicators_t2)
                t1_title = self._normalize_text(self._get_title(t1))
                t2_title = self._normalize_text(self._get_title(t2))
                title_sim = self._title_similarity(t1_title, t2_title) if (t1_title and t2_title) else 1.0
                title_similarity_min = 0.75
                if t1_title and t2_title and title_sim < title_similarity_min:
                    continue
                # Accepter si Jaccard suffisant OU recouvrement suffisant
                score = (
                    max(jaccard, overlap)
                    if (jaccard >= self.match_threshold or overlap >= self.overlap_threshold)
                    else 0.0
                )

                if score > best_score:
                    best_score = score
                    best_match = (j, t2)

            if best_match and best_score > 0:
                j, t2 = best_match
                pair = self._create_pair(t1, t2, i, j)
                result.table_pairs.append(pair)
                matched_t1_indices.add(i)
                matched_t2_indices.add(j)
                result.tables_matched += 1

        # Phase 3: Tableaux supprimes (T1 sans match)
        for i, t1 in enumerate(tables_t1):
            if i not in matched_t1_indices:
                pair = TablePair(
                    table_t1=t1,
                    table_t2=None,
                    title=self._get_title(t1),
                    is_removed_table=True,
                    page_t1=getattr(t1, "page_number", None),
                    bbox_t1=getattr(t1, "bbox", None),
                )
                result.table_pairs.append(pair)
                result.tables_removed += 1

        # Phase 4: Tableaux ajoutes (T2 sans match)
        for j, t2 in enumerate(tables_t2):
            if j not in matched_t2_indices:
                pair = TablePair(
                    table_t1=None,
                    table_t2=t2,
                    title=self._get_title(t2),
                    is_new_table=True,
                    page_t2=getattr(t2, "page_number", None),
                    bbox_t2=getattr(t2, "bbox", None),
                )
                result.table_pairs.append(pair)
                result.tables_added += 1

        logger.info(
            f"Preview matching: {result.tables_matched} matched, "
            f"{result.tables_added} added, {result.tables_removed} removed"
        )

        return result

    def _to_robust_input(self, table: Any, idx: int) -> dict:
        """Convertir TableImage en format robust_table_matcher."""
        page = getattr(table, "page_number", 0)
        tid = getattr(table, "table_index", idx)
        table_id = f"p{page}_{tid}"
        return {
            "table_id": table_id,
            "title": getattr(table, "title", "") or "",
            "headers": getattr(table, "headers", []) or [],
            "first_column_indicators": getattr(table, "first_column_indicators", []) or [],
            "section": getattr(table, "section", "") or "",
            "page": page,
            "image_base64": getattr(table, "image_base64", "") or "",
        }

    def _match_tables_robust(
        self, tables_t1: List[Any], tables_t2: List[Any], result: PreviewMatchResult
    ) -> PreviewMatchResult:
        """Matching via robust_table_matcher + Vision hybride."""
        th = {}
        if ROBUST_MATCHER_AVAILABLE:
            try:
                th = get_matching_thresholds()
            except Exception:
                pass
        use_vision = th.get("use_vision_validation", False)
        api_key = get_openai_api_key() if use_vision else None

        t1_inputs = [(i, t, self._to_robust_input(t, i)) for i, t in enumerate(tables_t1)]
        t2_by_id: Dict[str, Any] = {}
        t2_inputs_list: List[dict] = []
        for j, t2 in enumerate(tables_t2):
            inp = self._to_robust_input(t2, j)
            t2_by_id[inp["table_id"]] = (j, t2)
            t2_inputs_list.append(inp)

        matched_t2_indices: Set[int] = set()

        for i, t1, t1_inp in t1_inputs:
            t2_candidates = [
                inp for inp in t2_inputs_list
                if self._sections_compatible(t1, t2_by_id[inp["table_id"]][1])
            ]
            if not t2_candidates:
                pair = TablePair(
                    table_t1=t1,
                    table_t2=None,
                    title=self._get_title(t1),
                    is_removed_table=True,
                    page_t1=getattr(t1, "page_number", None),
                    bbox_t1=getattr(t1, "bbox", None),
                )
                result.table_pairs.append(pair)
                result.tables_removed += 1
                continue

            match_result = robust_match_best(
                t1_inp,
                t2_candidates,
                use_vision_validation=use_vision,
                api_key=api_key,
            )
            best = match_result.get("best", {})
            decision = best.get("decision", "NO_MATCH")
            best_t2_id = best.get("t2_table_id")

            if decision == "MATCH" and best_t2_id and best_t2_id in t2_by_id:
                j, t2 = t2_by_id[best_t2_id]
                matched_t2_indices.add(j)
                pair = self._create_pair(t1, t2, i, j)
                result.table_pairs.append(pair)
                result.tables_matched += 1
            else:
                pair = TablePair(
                    table_t1=t1,
                    table_t2=None,
                    title=self._get_title(t1),
                    is_removed_table=True,
                    page_t1=getattr(t1, "page_number", None),
                    bbox_t1=getattr(t1, "bbox", None),
                )
                result.table_pairs.append(pair)
                result.tables_removed += 1

        for j, t2 in enumerate(tables_t2):
            if j not in matched_t2_indices:
                pair = TablePair(
                    table_t1=None,
                    table_t2=t2,
                    title=self._get_title(t2),
                    is_new_table=True,
                    page_t2=getattr(t2, "page_number", None),
                    bbox_t2=getattr(t2, "bbox", None),
                )
                result.table_pairs.append(pair)
                result.tables_added += 1

        logger.info(
            "Preview matching (robust): %d matched, %d added, %d removed",
            result.tables_matched,
            result.tables_added,
            result.tables_removed,
        )
        return result

    def _create_pair(self, t1: Any, t2: Any, idx_t1: int, idx_t2: int) -> TablePair:
        """
        Creer une paire de tableaux avec detection des changements.

        La comparaison est uniquement sur les indicateurs (premiere colonne de chaque
        tableau). rows_added et rows_removed sont derives de first_column_indicators
        (valeurs presentes dans T2 mais pas T1, et dans T1 mais pas T2).
        """

        # Extraire indicateurs (premiere colonne uniquement)
        indicators_t1 = self._get_indicators(t1)
        indicators_t2 = self._get_indicators(t2)

        # Normaliser pour comparaison (canonicalisation unifiee des indicateurs)
        norm_t1 = {normalize_indicator_for_comparison(ind): ind for ind in indicators_t1}
        norm_t2 = {normalize_indicator_for_comparison(ind): ind for ind in indicators_t2}

        # Detecter ajouts et suppressions
        rows_added = []
        rows_removed = []

        for norm, original in norm_t2.items():
            if norm and norm not in norm_t1:
                rows_added.append(original)

        for norm, original in norm_t1.items():
            if norm and norm not in norm_t2:
                rows_removed.append(original)

        # Creer annotations avec positions
        annotations_t1 = self._create_annotations(
            rows_removed, indicators_t1, "removed", getattr(t1, "height", 400)
        )

        annotations_t2 = self._create_annotations(
            rows_added, indicators_t2, "added", getattr(t2, "height", 400)
        )

        return TablePair(
            table_t1=t1,
            table_t2=t2,
            title=self._get_title(t2) or self._get_title(t1),
            rows_added=rows_added,
            rows_removed=rows_removed,
            annotations_t1=annotations_t1,
            annotations_t2=annotations_t2,
            page_t1=getattr(t1, "page_number", None),
            page_t2=getattr(t2, "page_number", None),
            bbox_t1=getattr(t1, "bbox", None),
            bbox_t2=getattr(t2, "bbox", None),
        )

    def _create_annotations(
        self,
        rows_to_mark: List[str],
        all_indicators: List[str],
        change_type: str,
        image_height: int,
    ) -> List[RowAnnotation]:
        """Creer les annotations avec positions estimees."""
        annotations = []

        if not all_indicators:
            return annotations

        # Estimer hauteur par ligne
        header_height = 50
        available_height = image_height - header_height
        row_height = available_height / len(all_indicators)

        for i, indicator in enumerate(all_indicators):
            if indicator in rows_to_mark:
                y_pos = header_height + (i * row_height) + (row_height / 2)
                annotations.append(
                    RowAnnotation(
                        indicator=indicator,
                        y_position=y_pos,
                        change_type=change_type,
                        verified=False,
                    )
                )

        return annotations

    def _get_title(self, table: Any) -> str:
        """Obtenir le titre d'un tableau."""
        return getattr(table, "title", "") or ""

    def _get_indicators(self, table: Any) -> List[str]:
        """Obtenir les indicateurs (premiere colonne) d'un tableau."""
        return getattr(table, "first_column_indicators", []) or []

    def _normalize_text(self, text: str) -> str:
        """Normaliser un texte pour comparaison."""
        if not text:
            return ""

        # Supprimer accents
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Minuscules et strip
        return text.lower().strip()

    def _jaccard_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Calculer similarite Jaccard entre deux listes (indicateurs, premiere colonne)."""
        norm1 = {normalize_indicator_for_comparison(s) for s in set1 if s}
        norm2 = {normalize_indicator_for_comparison(s) for s in set2 if s}

        if not norm1 and not norm2:
            return 0.0

        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)

        return intersection / union if union > 0 else 0.0

    def _indicator_overlap(self, set1: List[str], set2: List[str]) -> float:
        """Taux de recouvrement: intersection / min(len1, len2). Utile quand une liste a plus de lignes."""
        norm1 = {normalize_indicator_for_comparison(s) for s in set1 if s}
        norm2 = {normalize_indicator_for_comparison(s) for s in set2 if s}

        if not norm1 or not norm2:
            return 0.0

        intersection = len(norm1 & norm2)
        min_len = min(len(norm1), len(norm2))
        return intersection / min_len if min_len > 0 else 0.0

    def _title_similarity(self, norm_title1: str, norm_title2: str) -> float:
        """Similarite textuelle entre deux titres normalises (sous-chaîne ou ratio)."""
        if not norm_title1 or not norm_title2:
            return 0.0
        if norm_title1 == norm_title2:
            return 1.0
        # Une sous-chaîne significative contenue dans l'autre
        if len(norm_title1) >= 15 and norm_title1 in norm_title2:
            return 0.9
        if len(norm_title2) >= 15 and norm_title2 in norm_title1:
            return 0.9
        # Ratio SequenceMatcher
        return SequenceMatcher(None, norm_title1, norm_title2).ratio()

    def _get_section(self, table: Any) -> str:
        """Obtenir la section d'un tableau."""
        if isinstance(table, dict):
            return str(table.get("section", "") or "")
        return getattr(table, "section", "") or ""

    def _sections_compatible(self, table1: Any, table2: Any) -> bool:
        """
        Verifier si deux tableaux appartiennent a des sections compatibles.

        Empeche le matching entre sections differentes comme:
        - Gestion du risque vs Gestion du capital
        - Liquidite vs Credit

        Returns:
            True si les sections sont compatibles.
            Hard-block uniquement quand les deux sections sont connues et differentes.
        """
        section1 = self._get_section(table1)
        section2 = self._get_section(table2)
        if canonicalize_section is not None:
            try:
                section1 = canonicalize_section(section1)
            except Exception:
                section1 = self._normalize_text(section1)
            try:
                section2 = canonicalize_section(section2)
            except Exception:
                section2 = self._normalize_text(section2)
        else:
            section1 = self._normalize_text(section1)
            section2 = self._normalize_text(section2)

        section1 = (section1 or "").strip().lower()
        section2 = (section2 or "").strip().lower()
        known1 = section1 not in UNKNOWN_SECTIONS
        known2 = section2 not in UNKNOWN_SECTIONS
        if known1 and known2:
            return section1 == section2
        return True


def _deduplicate_tables(tables: list) -> list:
    """
    Dedupliquer une liste de tableaux extraits.

    Supprime les doublons bases sur:
    - Meme page + titre similaire
    - Meme page + bbox qui se chevauchent significativement

    Garde le tableau avec le plus d'indicateurs (premiere colonne).

    Args:
        tables: Liste de TableImage

    Returns:
        Liste deduplicee de TableImage
    """
    if not tables:
        return []

    # Grouper par page
    by_page: dict = {}
    for table in tables:
        page = getattr(table, "page_number", 0)
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(table)

    deduplicated = []

    for page, page_tables in by_page.items():
        if len(page_tables) == 1:
            deduplicated.append(page_tables[0])
            continue

        # Pour plusieurs tableaux sur la meme page, filtrer les doublons
        kept = []
        for table in page_tables:
            is_duplicate = False

            for kept_table in kept:
                # Comparer par titre
                title1 = (getattr(table, "title", "") or "").lower().strip()
                title2 = (getattr(kept_table, "title", "") or "").lower().strip()

                if title1 and title2 and title1 == title2:
                    # Meme titre, garder celui avec plus d'indicateurs
                    ind1 = len(getattr(table, "first_column_indicators", []) or [])
                    ind2 = len(getattr(kept_table, "first_column_indicators", []) or [])
                    if ind1 > ind2:
                        kept.remove(kept_table)
                        kept.append(table)
                    is_duplicate = True
                    break

                # Comparer par bbox (chevauchement > 70%)
                bbox1 = getattr(table, "bbox", None)
                bbox2 = getattr(kept_table, "bbox", None)

                if bbox1 and bbox2:
                    overlap = _bbox_overlap(bbox1, bbox2)
                    if overlap > 0.7:
                        # Chevauchement significatif, garder celui avec plus d'indicateurs
                        ind1 = len(getattr(table, "first_column_indicators", []) or [])
                        ind2 = len(getattr(kept_table, "first_column_indicators", []) or [])
                        if ind1 > ind2:
                            kept.remove(kept_table)
                            kept.append(table)
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(table)

        deduplicated.extend(kept)

    return deduplicated


def _bbox_overlap(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculer le taux de chevauchement entre deux bounding boxes.

    Args:
        bbox1, bbox2: Tuples (x0, y0, x1, y1)

    Returns:
        Ratio de chevauchement (0-1) base sur la plus petite bbox
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Calculer l'intersection
    x0_inter = max(x0_1, x0_2)
    y0_inter = max(y0_1, y0_2)
    x1_inter = min(x1_1, x1_2)
    y1_inter = min(y1_1, y1_2)

    if x1_inter <= x0_inter or y1_inter <= y0_inter:
        return 0.0  # Pas de chevauchement

    area_inter = (x1_inter - x0_inter) * (y1_inter - y0_inter)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    min_area = min(area1, area2)
    if min_area <= 0:
        return 0.0

    return area_inter / min_area


def extract_and_match_tables(
    pdf_path_t1: str,
    pdf_path_t2: str,
    page_ranges_t1: Optional[List[Tuple[int, int]]] = None,
    page_ranges_t2: Optional[List[Tuple[int, int]]] = None,
    output_dir: str = "./temp_tables",
) -> PreviewMatchResult:
    """
    Fonction utilitaire pour extraire et matcher les tableaux.

    Args:
        pdf_path_t1: Chemin vers le PDF T1
        pdf_path_t2: Chemin vers le PDF T2
        page_ranges_t1: Plages de pages pour T1
        page_ranges_t2: Plages de pages pour T2
        output_dir: Repertoire pour les images temporaires

    Returns:
        PreviewMatchResult avec les paires de tableaux
    """
    if not TABLE_IMAGE_AVAILABLE:
        logger.error("TableImageExtractor non disponible")
        return PreviewMatchResult()

    import tempfile
    import os

    # Creer repertoire temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialiser le resultat avec diagnostics
        match_result = PreviewMatchResult()
        match_result.page_ranges_t1 = page_ranges_t1 or []
        match_result.page_ranges_t2 = page_ranges_t2 or []

        # Extraire tableaux T1
        extractor_t1 = TableImageExtractor(output_dir=os.path.join(tmpdir, "t1"), dpi=150)

        # Extraire les tableaux pour chaque plage de pages T1
        all_tables_t1 = []
        extraction_methods_t1 = set()
        total_pages_t1 = 0

        try:
            if page_ranges_t1:
                for start, end in page_ranges_t1:
                    logger.info(f"Extraction T1: pages {start}-{end}")
                    result = extractor_t1.extract_table_images(
                        pdf_path_t1, start_page=start, end_page=end
                    )
                    all_tables_t1.extend(result.table_images)
                    match_result.extraction_errors_t1.extend(result.errors)
                    total_pages_t1 += result.pages_processed

                    # Collecter les methodes d'extraction utilisees
                    for table in result.table_images:
                        if hasattr(table, "extraction_method"):
                            extraction_methods_t1.add(table.extraction_method)
            else:
                # Si pas de plages specifiees, extraire tout le PDF
                logger.info(f"Extraction T1: tout le PDF")
                result = extractor_t1.extract_table_images(pdf_path_t1)
                all_tables_t1 = result.table_images
                match_result.extraction_errors_t1.extend(result.errors)
                total_pages_t1 = result.pages_processed

                for table in result.table_images:
                    if hasattr(table, "extraction_method"):
                        extraction_methods_t1.add(table.extraction_method)
        except Exception as e:
            error_msg = f"Erreur extraction T1: {e}"
            logger.error(error_msg)
            match_result.extraction_errors_t1.append(error_msg)

        match_result.pages_processed_t1 = total_pages_t1
        match_result.extraction_method_t1 = (
            ", ".join(extraction_methods_t1) if extraction_methods_t1 else "pdfplumber"
        )

        # Extraire tableaux T2
        extractor_t2 = TableImageExtractor(output_dir=os.path.join(tmpdir, "t2"), dpi=150)

        # Extraire les tableaux pour chaque plage de pages T2
        all_tables_t2 = []
        extraction_methods_t2 = set()
        total_pages_t2 = 0

        try:
            if page_ranges_t2:
                for start, end in page_ranges_t2:
                    logger.info(f"Extraction T2: pages {start}-{end}")
                    result = extractor_t2.extract_table_images(
                        pdf_path_t2, start_page=start, end_page=end
                    )
                    all_tables_t2.extend(result.table_images)
                    match_result.extraction_errors_t2.extend(result.errors)
                    total_pages_t2 += result.pages_processed

                    # Collecter les methodes d'extraction utilisees
                    for table in result.table_images:
                        if hasattr(table, "extraction_method"):
                            extraction_methods_t2.add(table.extraction_method)
            else:
                # Si pas de plages specifiees, extraire tout le PDF
                logger.info(f"Extraction T2: tout le PDF")
                result = extractor_t2.extract_table_images(pdf_path_t2)
                all_tables_t2 = result.table_images
                match_result.extraction_errors_t2.extend(result.errors)
                total_pages_t2 = result.pages_processed

                for table in result.table_images:
                    if hasattr(table, "extraction_method"):
                        extraction_methods_t2.add(table.extraction_method)
        except Exception as e:
            error_msg = f"Erreur extraction T2: {e}"
            logger.error(error_msg)
            match_result.extraction_errors_t2.append(error_msg)

        match_result.pages_processed_t2 = total_pages_t2
        match_result.extraction_method_t2 = (
            ", ".join(extraction_methods_t2) if extraction_methods_t2 else "pdfplumber"
        )

        # Log pour debug
        logger.info(
            f"Tableaux extraits T1: {len(all_tables_t1)} (pages {total_pages_t1}, methodes: {match_result.extraction_method_t1}), "
            f"T2: {len(all_tables_t2)} (pages {total_pages_t2}, methodes: {match_result.extraction_method_t2})"
        )

        if not all_tables_t1 and not all_tables_t2:
            logger.warning("Aucun tableau extrait des PDFs. Verifiez les plages de pages.")
            if page_ranges_t1:
                logger.warning(f"Plages T1 utilisees: {page_ranges_t1}")
            if page_ranges_t2:
                logger.warning(f"Plages T2 utilisees: {page_ranges_t2}")

        # Dedupliquer les tableaux avant matching
        # (evite les doublons quand plusieurs extractions par page)
        all_tables_t1 = _deduplicate_tables(all_tables_t1)
        all_tables_t2 = _deduplicate_tables(all_tables_t2)

        logger.info(f"Apres deduplication: T1={len(all_tables_t1)}, T2={len(all_tables_t2)}")

        # Matcher les tableaux
        matcher = TablePreviewMatcher()
        matched_result = matcher.match_tables(all_tables_t1, all_tables_t2)

        # Copier les diagnostics dans le resultat final
        match_result.table_pairs = matched_result.table_pairs
        match_result.total_tables_t1 = matched_result.total_tables_t1
        match_result.total_tables_t2 = matched_result.total_tables_t2
        match_result.tables_matched = matched_result.tables_matched
        match_result.tables_added = matched_result.tables_added
        match_result.tables_removed = matched_result.tables_removed

        return match_result
