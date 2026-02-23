"""
Detecteur de continuite de tableaux multi-pages.

Ce module detecte et fusionne les tableaux qui s'etendent sur plusieurs pages,
un cas frequent dans les rapports bancaires.

Indicateurs de continuite:
- Headers repetes en haut de page
- Absence de ligne "Total" ou "Sous-total"
- Numerotation de lignes continue
"""

from dataclasses import dataclass
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class TableFragment:
    """Fragment d'un tableau potentiellement multi-pages."""

    page_number: int
    table_index: int  # Index sur la page
    title: str = ""
    headers: list[str] = None
    rows: list[list[str]] = None
    first_column_labels: list[str] = None
    has_total_row: bool = False
    has_header_row: bool = True
    is_likely_continuation: bool = False
    raw_data: Optional[dict] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = []
        if self.rows is None:
            self.rows = []
        if self.first_column_labels is None:
            self.first_column_labels = []


@dataclass
class MergedTable:
    """Tableau fusionne depuis plusieurs fragments."""

    fragments: list[TableFragment]
    merged_title: str
    merged_headers: list[str]
    merged_rows: list[list[str]]
    merged_first_column_labels: list[str]
    page_range: tuple[int, int]  # (start_page, end_page)
    confidence: float = 0.0


class TableContinuityDetector:
    """
    Detecteur de tableaux multi-pages.

    Analyse les fragments de tableaux et decide lesquels doivent etre fusionnes.
    """

    # Patterns indiquant une ligne finale (total, sous-total, etc.)
    TOTAL_PATTERNS = [
        r"^total\b",
        r"^sous[- ]?total\b",
        r"^somme\b",
        r"^total des",
        r"^net\b",
        r"^solde\b",
    ]

    # Patterns indiquant une continuation probable
    CONTINUATION_INDICATORS = [
        "suite",
        "continued",
        "cont'd",
        "(suite)",
    ]

    def __init__(
        self,
        header_similarity_threshold: float = 0.85,
        max_page_gap: int = 1,
    ):
        """
        Initialiser le detecteur.

        Args:
            header_similarity_threshold: Seuil de similarite pour considerer
                que deux ensembles de headers sont identiques
            max_page_gap: Ecart maximum de pages pour considerer une continuation
        """
        self.header_similarity_threshold = header_similarity_threshold
        self.max_page_gap = max_page_gap

        # Compiler les patterns regex
        self.total_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.TOTAL_PATTERNS]

    def detect_continuation(
        self,
        fragment_prev: TableFragment,
        fragment_next: TableFragment,
    ) -> tuple[bool, float, str]:
        """
        Detecter si fragment_next est la continuation de fragment_prev.

        Returns:
            Tuple (is_continuation, confidence, reason)
        """
        # Verifier l'ecart de pages
        page_gap = fragment_next.page_number - fragment_prev.page_number
        if page_gap > self.max_page_gap or page_gap < 1:
            return False, 0.0, "Page gap too large or invalid"

        # Signal 1: fragment_prev n'a pas de ligne Total
        if fragment_prev.has_total_row:
            return False, 0.0, "Previous fragment has total row"

        # Signal 2: Headers identiques ou tres similaires
        headers_match = self._compare_headers(fragment_prev.headers, fragment_next.headers)

        # Signal 3: fragment_next est sur le haut de la page (index 0)
        is_top_of_page = fragment_next.table_index == 0

        # Signal 4: Titre de continuation explicite
        has_continuation_marker = self._has_continuation_marker(fragment_next.title)

        # Calculer le score de continuation
        score = 0.0
        reasons = []

        if not fragment_prev.has_total_row:
            score += 0.30
            reasons.append("no_total_row")

        if headers_match >= self.header_similarity_threshold:
            score += 0.35
            reasons.append(f"headers_match({headers_match:.2f})")

        if is_top_of_page:
            score += 0.20
            reasons.append("top_of_page")

        if has_continuation_marker:
            score += 0.15
            reasons.append("continuation_marker")

        is_continuation = score >= 0.50

        return is_continuation, score, ", ".join(reasons)

    def merge_fragments(
        self,
        fragments: list[TableFragment],
    ) -> list[MergedTable]:
        """
        Fusionner les fragments en tableaux complets.

        Args:
            fragments: Liste de fragments tries par page et index

        Returns:
            Liste de MergedTable
        """
        if not fragments:
            return []

        # Trier par page puis par index
        sorted_fragments = sorted(fragments, key=lambda f: (f.page_number, f.table_index))

        # Analyser chaque fragment
        for fragment in sorted_fragments:
            fragment.has_total_row = self._has_total_row(fragment.rows)

        # Grouper les fragments en tableaux
        merged_tables: list[MergedTable] = []
        current_group: list[TableFragment] = []

        for i, fragment in enumerate(sorted_fragments):
            if not current_group:
                current_group = [fragment]
                continue

            # Verifier si ce fragment continue le precedent
            is_cont, conf, reason = self.detect_continuation(current_group[-1], fragment)

            if is_cont:
                fragment.is_likely_continuation = True
                current_group.append(fragment)
                logger.debug(
                    f"Fragment p{fragment.page_number}t{fragment.table_index} "
                    f"est une continuation: {reason}"
                )
            else:
                # Finaliser le groupe actuel
                merged_tables.append(self._create_merged_table(current_group))
                current_group = [fragment]

        # Finaliser le dernier groupe
        if current_group:
            merged_tables.append(self._create_merged_table(current_group))

        logger.info(
            f"Continuite detectee: {len(sorted_fragments)} fragments -> "
            f"{len(merged_tables)} tableaux"
        )

        return merged_tables

    def _compare_headers(
        self,
        headers1: list[str],
        headers2: list[str],
    ) -> float:
        """Comparer deux ensembles de headers."""
        if not headers1 or not headers2:
            return 0.5  # Score neutre si pas de headers

        # Normaliser
        h1_norm = [h.lower().strip() for h in headers1]
        h2_norm = [h.lower().strip() for h in headers2]

        # Compter les correspondances exactes
        matches = sum(1 for h in h1_norm if h in h2_norm)
        total = max(len(h1_norm), len(h2_norm))

        return matches / total if total > 0 else 0.0

    def _has_total_row(self, rows: list[list[str]]) -> bool:
        """Verifier si les dernieres lignes contiennent un Total."""
        if not rows:
            return False

        # Verifier les 3 dernieres lignes
        for row in rows[-3:]:
            if not row:
                continue
            first_cell = str(row[0]).lower().strip()
            for pattern in self.total_regex:
                if pattern.search(first_cell):
                    return True

        return False

    def _has_continuation_marker(self, title: str) -> bool:
        """Verifier si le titre indique une continuation."""
        if not title:
            return False

        title_lower = title.lower()
        return any(marker in title_lower for marker in self.CONTINUATION_INDICATORS)

    def _create_merged_table(
        self,
        fragments: list[TableFragment],
    ) -> MergedTable:
        """Creer un MergedTable depuis une liste de fragments."""
        if not fragments:
            raise ValueError("Cannot merge empty fragment list")

        # Utiliser le titre du premier fragment
        merged_title = fragments[0].title

        # Utiliser les headers du premier fragment
        merged_headers = fragments[0].headers.copy()

        # Fusionner les lignes
        merged_rows = []
        for fragment in fragments:
            merged_rows.extend(fragment.rows)

        # Fusionner les labels de 1ere colonne
        merged_labels = []
        for fragment in fragments:
            merged_labels.extend(fragment.first_column_labels)

        # Calculer la confiance (basee sur le nombre de fragments)
        if len(fragments) == 1:
            confidence = 1.0
        else:
            # Plus de fragments = moins de confiance (necessite verification)
            confidence = max(0.70, 1.0 - 0.10 * (len(fragments) - 1))

        return MergedTable(
            fragments=fragments,
            merged_title=merged_title,
            merged_headers=merged_headers,
            merged_rows=merged_rows,
            merged_first_column_labels=merged_labels,
            page_range=(fragments[0].page_number, fragments[-1].page_number),
            confidence=confidence,
        )


def create_fragment_from_extracted_table(
    table_data: dict,
    page_number: int,
    table_index: int,
) -> TableFragment:
    """
    Creer un TableFragment depuis les donnees extraites.

    Args:
        table_data: Dictionnaire avec les donnees du tableau
        page_number: Numero de page
        table_index: Index du tableau sur la page

    Returns:
        TableFragment pour l'analyse de continuite
    """
    return TableFragment(
        page_number=page_number,
        table_index=table_index,
        title=table_data.get("title", ""),
        headers=table_data.get("headers", []),
        rows=table_data.get("rows", []),
        first_column_labels=table_data.get("first_column_indicators", []),
        raw_data=table_data,
    )
