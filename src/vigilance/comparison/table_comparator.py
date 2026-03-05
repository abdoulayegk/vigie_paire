"""
Comparateur de tableaux pour detecter les changements structurels et de contenu entre trimestres.
Intègre la normalisation sémantique pour gérer les structures hétérogènes.

Enhanced with multi-signal matching strategy from vigie_paire/brgc.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional, Tuple

# Try to import rapidfuzz for faster fuzzy matching
try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Import canonical indicator normalizer (accent-stripped, note-free, lowercase)
try:
    from ..utils.indicator_cleaner import (
        normalize_indicator_for_comparison as normalize_fr,
    )
except ImportError:
    # Fallback if module not yet available
    normalize_fr = lambda x, **kwargs: x.lower().strip() if x else ""
FRENCH_STOP_WORDS: set[str] = set()

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None

logger = logging.getLogger(__name__)
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}


# ==================== Multi-Signal Matching Configuration ====================

# Weights for multi-signal scoring (must sum to 1.0)
WEIGHT_CONTENT_OVERLAP = 0.40  # Jaccard overlap of first column labels
WEIGHT_FUZZY_LABELS = 0.40  # Fuzzy similarity of first column labels
WEIGHT_POSITION = 0.10  # Relative position within section
WEIGHT_STRUCTURE = 0.10  # Structural similarity (row count)

# Matching thresholds (valeurs par defaut, surchargeables via bank_profiles)
TABLE_MATCH_THRESHOLD_DEFAULT = 0.65  # Minimum score to consider a match
TABLE_MATCH_THRESHOLD = TABLE_MATCH_THRESHOLD_DEFAULT  # Alias pour compatibilite
AMBIGUITY_THRESHOLD = 0.05  # Score difference to flag as ambiguous
MIN_LABEL_LENGTH = 3  # Minimum label length for matching


def _get_table_match_threshold() -> float:
    """Charger le seuil depuis bank_profiles (generique pour les 6 banques)."""
    try:
        from ..config import get_matching_thresholds

        t = get_matching_thresholds()
        return float(
            t.get(
                "table_comparator_threshold",
                t.get("minimum_match", TABLE_MATCH_THRESHOLD_DEFAULT),
            )
        )
    except Exception:
        return TABLE_MATCH_THRESHOLD_DEFAULT


def _normalize_section(section: Optional[str]) -> str:
    raw = (section or "").strip()
    if not raw:
        return "unknown_section"
    if canonicalize_section is not None:
        try:
            normalized = canonicalize_section(raw)
            if normalized:
                return normalized
        except Exception:
            pass
    fallback = normalize_fr(raw).replace(" ", "_")
    return fallback or "unknown_section"


def _is_known_section(section: Optional[str]) -> bool:
    return _normalize_section(section) not in UNKNOWN_SECTIONS


@dataclass
class TableChange:
    """Represents a change detected in a table."""

    change_type: str  # "new_row", "removed_row", "new_column", "removed_column", "value_change", "footnote_change"
    table_id: str
    page_number: int
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    row_identifier: Optional[str] = None
    column_identifier: Optional[str] = None
    significance: str = "MINOR"  # "MAJOR", "MODERATE", "MINOR"
    category: Optional[str] = (
        None  # "REGULATORY", "RISK_EMERGING", "ESG", "INDICATOR", "OTHER"
    )

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "table_id": self.table_id,
            "page_number": self.page_number,
            "description": self.description,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "row_identifier": self.row_identifier,
            "column_identifier": self.column_identifier,
            "significance": self.significance,
            "category": self.category,
        }


@dataclass
class TableMeta:
    """Metadata for a table used in multi-signal matching."""

    table_key: str
    section_norm: Optional[str] = None
    table_title: str = ""
    first_col_labels: list = field(default_factory=list)
    first_col_normalized: list = field(default_factory=list)
    fingerprint: str = ""  # SHA1 of normalized first column
    page: int = 0
    row_count: int = 0
    position_in_section: float = 0.0

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.first_col_labels and not self.first_col_normalized:
            self.first_col_normalized = [
                normalize_fr(label)
                for label in self.first_col_labels
                if len(str(label).strip()) >= MIN_LABEL_LENGTH
            ]
        if self.first_col_normalized and not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute SHA1 fingerprint of normalized first column."""
        content = "|".join(sorted(self.first_col_normalized))
        return hashlib.sha1(content.encode()).hexdigest()[:12]


@dataclass
class MatchResult:
    """Result of matching two tables using multi-signal approach."""

    table_a: TableMeta
    table_b: Optional[TableMeta]
    score: float
    is_match: bool
    is_ambiguous: bool
    match_method: str
    score_breakdown: dict = field(default_factory=dict)


# ==================== Multi-Signal Scoring Functions ====================


def score_content_overlap(a: TableMeta, b: TableMeta) -> float:
    """Calculate Jaccard overlap of normalized first-column labels."""
    set_a = set(a.first_col_normalized)
    set_b = set(b.first_col_normalized)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def score_fuzzy_labels(a: TableMeta, b: TableMeta) -> float:
    """Calculate average fuzzy similarity between label lists."""
    if not a.first_col_normalized or not b.first_col_normalized:
        return 0.0

    total_score = 0.0
    comparisons = 0

    # Compare up to 15 labels for performance
    for label_a in a.first_col_normalized[:15]:
        best_score: float = 0.0
        for label_b in b.first_col_normalized[:15]:
            if RAPIDFUZZ_AVAILABLE:
                score = rapidfuzz_fuzz.token_set_ratio(label_a, label_b)
            else:
                # Fallback to SequenceMatcher
                score = SequenceMatcher(None, label_a, label_b).ratio() * 100
            best_score = max(best_score, score)
        total_score += best_score
        comparisons += 1

    return (total_score / comparisons / 100.0) if comparisons > 0 else 0.0


def score_position(a: TableMeta, b: TableMeta) -> float:
    """Score based on relative position similarity within section."""
    diff = abs(a.position_in_section - b.position_in_section)
    return max(0.0, 1.0 - diff)


def score_structure(a: TableMeta, b: TableMeta) -> float:
    """Score based on structural similarity (row count)."""
    if a.row_count == 0 and b.row_count == 0:
        return 1.0
    max_rows = max(a.row_count, b.row_count)
    min_rows = min(a.row_count, b.row_count)
    if max_rows == 0:
        return 0.0
    ratio = min_rows / max_rows
    # Allow up to 20% difference without penalty
    return min(1.0, ratio / 0.8)


def compute_table_match_score(a: TableMeta, b: TableMeta) -> Tuple[float, dict]:
    """Calculate combined match score using multi-signal approach.

    Returns:
        Tuple of (combined_score, score_breakdown_dict)
    """
    section_a = _normalize_section(a.section_norm)
    section_b = _normalize_section(b.section_norm)

    # Hard business gate: no cross-section and no unknown section matching.
    if section_a in UNKNOWN_SECTIONS or section_b in UNKNOWN_SECTIONS:
        return 0.0, {"blocked": "unknown_section"}
    if section_a != section_b:
        return 0.0, {"blocked": "cross_section_forbidden"}

    # Level 2: Fingerprint exact match (fast path)
    if a.fingerprint and b.fingerprint and a.fingerprint == b.fingerprint:
        return 1.0, {"method": "fingerprint_exact", "section": section_a}

    # Level 3: Multi-signal scoring
    overlap = score_content_overlap(a, b)
    fuzzy = score_fuzzy_labels(a, b)
    position = score_position(a, b)
    structure = score_structure(a, b)

    combined = (
        WEIGHT_CONTENT_OVERLAP * overlap
        + WEIGHT_FUZZY_LABELS * fuzzy
        + WEIGHT_POSITION * position
        + WEIGHT_STRUCTURE * structure
    )

    breakdown = {
        "content_overlap": round(overlap, 3),
        "fuzzy_labels": round(fuzzy, 3),
        "position": round(position, 3),
        "structure": round(structure, 3),
        "section": section_a,
        "weights": {
            "content_overlap": WEIGHT_CONTENT_OVERLAP,
            "fuzzy_labels": WEIGHT_FUZZY_LABELS,
            "position": WEIGHT_POSITION,
            "structure": WEIGHT_STRUCTURE,
        },
    }

    return combined, breakdown


def find_best_match(
    table_a: TableMeta,
    candidates_b: list[TableMeta],
) -> MatchResult:
    """Find the best matching table from candidates using multi-signal scoring.

    Args:
        table_a: The table to match
        candidates_b: List of candidate tables to compare against

    Returns:
        MatchResult with best match info and ambiguity detection
    """
    if not candidates_b:
        return MatchResult(
            table_a=table_a,
            table_b=None,
            score=0.0,
            is_match=False,
            is_ambiguous=False,
            match_method="no_candidates",
        )

    scores = []
    for candidate in candidates_b:
        score, breakdown = compute_table_match_score(table_a, candidate)
        scores.append((score, candidate, breakdown))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)

    best_score, best_candidate, best_breakdown = scores[0]

    # Check for ambiguity (second best is close to best)
    is_ambiguous = False
    if len(scores) > 1:
        second_score = scores[1][0]
        if best_score - second_score < AMBIGUITY_THRESHOLD:
            is_ambiguous = True

    threshold = _get_table_match_threshold()
    is_match = best_score >= threshold

    return MatchResult(
        table_a=table_a,
        table_b=best_candidate if is_match else None,
        score=round(best_score, 4),
        is_match=is_match,
        is_ambiguous=is_ambiguous,
        match_method="multi_signal",
        score_breakdown=best_breakdown,
    )


class TableComparator:
    """
    Compares tables between two quarterly reports to detect meaningful changes.
    Filters out noise like minor numeric variations.

    Supports:
    - Semantic header matching (synonyms, abbreviations)
    - Structural normalization (transposed tables)
    - Fuzzy row matching
    - Multi-level header handling
    """

    # Thresholds for considering changes significant
    NUMERIC_CHANGE_THRESHOLD = 0.05  # 5% change is considered significant
    TEXT_SIMILARITY_THRESHOLD = (
        0.90  # Below this, text is considered changed (Basel III/IFRS spec: 90%)
    )
    HEADER_MATCH_THRESHOLD = 0.75  # For semantic header matching
    ROW_KEY_MATCH_THRESHOLD = 0.85  # For fuzzy row key matching

    # Keywords indicating regulatory/important changes
    REGULATORY_KEYWORDS = [
        "bâle",
        "bsif",
        "nfp",
        "cet1",
        "levier",
        "lcr",
        "nsfr",
        "réglementaire",
        "exigence",
        "coussin",
        "minimum",
    ]

    RISK_KEYWORDS = [
        "risque",
        "provision",
        "perte",
        "créance",
        "défaut",
        "exposition",
        "crédit",
        "marché",
        "liquidité",
        "opérationnel",
    ]

    ESG_KEYWORDS = [
        "esg",
        "climat",
        "environnement",
        "carbone",
        "émission",
        "durable",
        "social",
        "gouvernance",
    ]

    def __init__(
        self,
        ignore_numeric_only: bool = True,
        use_semantic_matching: bool = True,
        auto_detect_transposition: bool = True,
    ):
        """
        Initialize table comparator.

        Args:
            ignore_numeric_only: If True, ignore changes that are only numeric updates
            use_semantic_matching: If True, use semantic matching for headers and rows
            auto_detect_transposition: If True, auto-detect and handle transposed tables
        """
        self.ignore_numeric_only = ignore_numeric_only
        self.use_semantic_matching = use_semantic_matching
        self.auto_detect_transposition = auto_detect_transposition
        self._normalizer = None

    def _get_normalizer(self):
        """Get the table normalizer (lazy loading)."""
        if self._normalizer is None:
            try:
                from .table_normalizer import TableNormalizer

                self._normalizer = TableNormalizer(
                    similarity_threshold=self.HEADER_MATCH_THRESHOLD
                )
            except ImportError:
                logger.warning("TableNormalizer not available")
                self._normalizer = False  # Mark as unavailable
        return self._normalizer if self._normalizer else None

    def compare_tables(
        self,
        table1_data: dict,
        table2_data: dict,
        table1_id: str = "table1",
        table2_id: str = "table2",
    ) -> list[TableChange]:
        """
        Compare two tables and return list of changes.

        Args:
            table1_data: First table (older) with 'headers' and 'rows'
            table2_data: Second table (newer) with 'headers' and 'rows'
            table1_id: Identifier for first table
            table2_id: Identifier for second table

        Returns:
            List of TableChange objects
        """
        changes = []

        # Try semantic matching if available
        normalizer = self._get_normalizer() if self.use_semantic_matching else None

        if normalizer and self.auto_detect_transposition:
            # Use normalized comparison
            changes = self._compare_with_normalization(
                table1_data, table2_data, table1_id, table2_id, normalizer
            )
        else:
            # Fallback to direct comparison
            changes = self._compare_direct(
                table1_data, table2_data, table1_id, table2_id
            )

        # Classify changes
        for change in changes:
            change.category = self._classify_change(change)
            change.significance = self._assess_significance(change)

        return changes

    def _compare_with_normalization(
        self,
        table1_data: dict,
        table2_data: dict,
        table1_id: str,
        table2_id: str,
        normalizer,
    ) -> list[TableChange]:
        """Compare tables using normalization for structural differences."""
        changes = []

        # Normalize tables
        norm_t1 = normalizer.normalize_table(table1_data, table1_id)
        norm_t2 = normalizer.normalize_table(table2_data, table2_id)

        # Check if tables match structurally
        match_result = normalizer.match_tables(norm_t1, norm_t2)

        page_num = table2_data.get("page_number", 0)

        # If transposition detected, log it
        if match_result.transformation_needed == "transpose":
            logger.info(f"Table transposition detected for {table2_id}")
            norm_t1, norm_t2 = normalizer.align_tables_for_comparison(
                norm_t1, norm_t2, match_result
            )
            changes.append(
                TableChange(
                    change_type="structural_change",
                    table_id=table2_id,
                    page_number=page_num,
                    description="Structure du tableau modifiée (transposition détectée)",
                    significance="MODERATE",
                )
            )

        # Report unmatched headers as new/removed columns
        for header in match_result.unmatched_headers_2:
            changes.append(
                TableChange(
                    change_type="new_column",
                    table_id=table2_id,
                    page_number=page_num,
                    description=f"Nouvelle colonne ajoutée: {header}",
                    new_value=header,
                    column_identifier=header,
                )
            )

        for header in match_result.unmatched_headers_1:
            changes.append(
                TableChange(
                    change_type="removed_column",
                    table_id=table2_id,
                    page_number=page_num,
                    description=f"Colonne supprimée: {header}",
                    old_value=header,
                    column_identifier=header,
                )
            )

        # Report unmatched rows
        for row_key in match_result.unmatched_rows_2:
            row_data = norm_t2.get_row_by_key(row_key)
            changes.append(
                TableChange(
                    change_type="new_row",
                    table_id=table2_id,
                    page_number=page_num,
                    description=f"Nouvelle ligne ajoutée: {row_key}",
                    new_value=str(row_data) if row_data else row_key,
                    row_identifier=row_key,
                )
            )

        for row_key in match_result.unmatched_rows_1:
            row_data = norm_t1.get_row_by_key(row_key)
            changes.append(
                TableChange(
                    change_type="removed_row",
                    table_id=table2_id,
                    page_number=page_num,
                    description=f"Ligne supprimée: {row_key}",
                    old_value=str(row_data) if row_data else row_key,
                    row_identifier=row_key,
                )
            )

        # Compare matched rows for value changes
        if not self.ignore_numeric_only:
            for key1, key2 in match_result.matched_rows:
                row1 = norm_t1.get_row_by_key(key1)
                row2 = norm_t2.get_row_by_key(key2)

                if row1 and row2:
                    row_changes = self._compare_row_values(
                        row1, row2, key1, table2_id, page_num
                    )
                    changes.extend(row_changes)

        return changes

    def _compare_row_values(
        self, row1: dict, row2: dict, row_key: str, table_id: str, page_num: int
    ) -> list[TableChange]:
        """
        Compare values between two matched rows.

        NOTE: Ne compare que la premiere colonne (indicateur). Les changements
        de chiffres dans les autres colonnes sont ignores.
        """
        changes = []

        # Ne comparer que la premiere colonne (indicateur)
        # La premiere colonne correspond a la premiere clé du dictionnaire
        if not row1 or not row2:
            return changes

        # Obtenir la premiere clé (premiere colonne normalisee)
        first_col = None
        if row1:
            first_col = next(iter(row1.keys()), None)
        elif row2:
            first_col = next(iter(row2.keys()), None)

        if not first_col:
            return changes

        # Comparer uniquement la premiere colonne
        val1 = row1.get(first_col, "")
        val2 = row2.get(first_col, "")

        if val1 != val2:
            # Check text similarity pour l'indicateur uniquement
            similarity = SequenceMatcher(
                None, str(val1).lower(), str(val2).lower()
            ).ratio()
            if similarity < self.TEXT_SIMILARITY_THRESHOLD:
                changes.append(
                    TableChange(
                        change_type="value_change",
                        table_id=table_id,
                        page_number=page_num,
                        description=f"Indicateur modifié: {row_key}",
                        old_value=str(val1),
                        new_value=str(val2),
                        row_identifier=row_key,
                        column_identifier=first_col,
                    )
                )

        return changes

    def _compare_direct(
        self, table1_data: dict, table2_data: dict, table1_id: str, table2_id: str
    ) -> list[TableChange]:
        """Direct comparison without normalization (original behavior)."""
        changes = []

        headers1 = table1_data.get("headers", [])
        headers2 = table2_data.get("headers", [])
        rows1 = table1_data.get("rows", [])
        rows2 = table2_data.get("rows", [])
        page_num = table2_data.get("page_number", 0)

        # Compare headers (column changes)
        header_changes = self._compare_headers(headers1, headers2, table2_id, page_num)
        changes.extend(header_changes)

        # Compare rows
        row_changes = self._compare_rows(rows1, rows2, headers2, table2_id, page_num)
        changes.extend(row_changes)

        return changes

    def _compare_headers(
        self, headers1: list, headers2: list, table_id: str, page_num: int
    ) -> list[TableChange]:
        """Compare table headers to detect column changes."""
        changes = []

        # Flatten headers if nested
        flat1 = self._flatten_headers(headers1)
        flat2 = self._flatten_headers(headers2)

        # Find new columns
        for col in flat2:
            if col and col not in flat1:
                changes.append(
                    TableChange(
                        change_type="new_column",
                        table_id=table_id,
                        page_number=page_num,
                        description=f"Nouvelle colonne ajoutée: {col}",
                        new_value=col,
                        column_identifier=col,
                    )
                )

        # Find removed columns
        for col in flat1:
            if col and col not in flat2:
                changes.append(
                    TableChange(
                        change_type="removed_column",
                        table_id=table_id,
                        page_number=page_num,
                        description=f"Colonne supprimée: {col}",
                        old_value=col,
                        column_identifier=col,
                    )
                )

        return changes

    def _compare_rows(
        self,
        rows1: list[list],
        rows2: list[list],
        headers: list,
        table_id: str,
        page_num: int,
    ) -> list[TableChange]:
        """Compare table rows to detect row-level changes."""
        changes = []

        # Create row dictionaries using first column as key
        def get_row_key(row):
            if row and len(row) > 0:
                key = str(row[0]).strip()
                return key if key else None
            return None

        rows1_dict = {}
        for row in rows1:
            key = get_row_key(row)
            if key:
                rows1_dict[key] = row

        rows2_dict = {}
        for row in rows2:
            key = get_row_key(row)
            if key:
                rows2_dict[key] = row

        # Find new rows
        for key, row in rows2_dict.items():
            if key not in rows1_dict:
                changes.append(
                    TableChange(
                        change_type="new_row",
                        table_id=table_id,
                        page_number=page_num,
                        description=f"Nouvelle ligne ajoutée: {key}",
                        new_value=str(row),
                        row_identifier=key,
                    )
                )

        # Find removed rows
        for key, row in rows1_dict.items():
            if key not in rows2_dict:
                changes.append(
                    TableChange(
                        change_type="removed_row",
                        table_id=table_id,
                        page_number=page_num,
                        description=f"Ligne supprimée: {key}",
                        old_value=str(row),
                        row_identifier=key,
                    )
                )

        # Find modified rows (only if not ignoring numeric-only changes)
        if not self.ignore_numeric_only:
            for key in rows1_dict:
                if key in rows2_dict:
                    row1 = rows1_dict[key]
                    row2 = rows2_dict[key]

                    if self._rows_differ_significantly(row1, row2):
                        changes.append(
                            TableChange(
                                change_type="value_change",
                                table_id=table_id,
                                page_number=page_num,
                                description=f"Valeurs modifiées pour: {key}",
                                old_value=str(row1),
                                new_value=str(row2),
                                row_identifier=key,
                            )
                        )

        return changes

    def _flatten_headers(self, headers: list) -> list[str]:
        """Flatten nested header structure to list of strings."""
        flat = []

        if not headers:
            return flat

        for item in headers:
            if isinstance(item, list):
                flat.extend(self._flatten_headers(item))
            elif isinstance(item, dict):
                flat.append(str(item.get("value", item)))
            else:
                flat.append(str(item) if item else "")

        return [h.strip() for h in flat if h and h.strip()]

    def _rows_differ_significantly(self, row1: list, row2: list) -> bool:
        """
        Check if two rows differ significantly.

        NOTE: Ne compare que la premiere colonne (indicateur). Les changements
        de chiffres dans les autres colonnes sont ignores.
        """
        # Ne comparer que la premiere colonne (indicateur)
        if not row1 or not row2:
            return len(row1) != len(row2)

        # Comparer uniquement la premiere colonne
        cell1 = row1[0] if len(row1) > 0 else ""
        cell2 = row2[0] if len(row2) > 0 else ""

        str1 = str(cell1).strip() if cell1 else ""
        str2 = str(cell2).strip() if cell2 else ""

        # Check text similarity pour l'indicateur uniquement
        similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        if similarity < self.TEXT_SIMILARITY_THRESHOLD:
            return True

        return False

    def _is_numeric(self, value: str) -> bool:
        """Check if value is primarily numeric."""
        if not value:
            return False

        # Remove formatting
        cleaned = value.replace(",", "").replace(" ", "").replace("$", "")
        cleaned = cleaned.replace("%", "").replace("M", "").replace("G", "")
        cleaned = cleaned.replace("(", "").replace(")", "").replace("-", "")

        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _classify_change(self, change: TableChange) -> str:
        """Classify change into category."""
        text_to_check = (
            f"{change.description} {change.old_value or ''} {change.new_value or ''}"
        )
        text_lower = text_to_check.lower()

        # Check for regulatory content
        for keyword in self.REGULATORY_KEYWORDS:
            if keyword in text_lower:
                return "REGULATORY"

        # Check for ESG content
        for keyword in self.ESG_KEYWORDS:
            if keyword in text_lower:
                return "ESG"

        # Check for risk content
        for keyword in self.RISK_KEYWORDS:
            if keyword in text_lower:
                return "RISK_EMERGING"

        return (
            "INDICATOR" if change.change_type in ["new_row", "removed_row"] else "OTHER"
        )

    def _assess_significance(self, change: TableChange) -> str:
        """Assess significance of a change."""
        # New/removed rows and columns are significant
        if change.change_type in [
            "new_row",
            "removed_row",
            "new_column",
            "removed_column",
        ]:
            # Regulatory changes are major
            if change.category == "REGULATORY":
                return "MAJOR"
            # ESG and risk changes are moderate to major
            if change.category in ["ESG", "RISK_EMERGING"]:
                return "MODERATE"
            return "MODERATE"

        # Value changes are typically minor
        return "MINOR"


def compare_tables(table1: dict, table2: dict) -> list[TableChange]:
    """
    Convenience function to compare two tables.

    Args:
        table1: First table data
        table2: Second table data

    Returns:
        List of detected changes
    """
    comparator = TableComparator()
    return comparator.compare_tables(table1, table2)


def match_tables_multi_signal(
    tables_a: list[dict],
    tables_b: list[dict],
    section_key: Optional[str] = None,
) -> list[MatchResult]:
    """
    Match tables between two documents using multi-signal approach.

    This is the enhanced matching function using 4-level strategy:
    1. Section scope constraint
    2. First-column fingerprint (SHA1)
    3. Multi-signal scoring (overlap + fuzzy + position + structure)
    4. Threshold + ambiguity detection

    Args:
        tables_a: List of tables from T1 (older document)
        tables_b: List of tables from T2 (newer document)
        section_key: Optional section to constrain matching

    Returns:
        List of MatchResult objects with match info
    """
    normalized_scope = _normalize_section(section_key) if section_key else None

    # Convert to TableMeta objects
    metas_a = []
    for i, t in enumerate(tables_a):
        section = _normalize_section(t.get("section"))
        if section in UNKNOWN_SECTIONS:
            continue
        if normalized_scope and section != normalized_scope:
            continue
        first_col = _extract_first_column(t)
        meta = TableMeta(
            table_key=t.get("table_id", f"t1_{i}"),
            section_norm=section,
            table_title=t.get("title", ""),
            first_col_labels=first_col,
            page=t.get("page_number", 0),
            row_count=len(t.get("rows", [])),
            position_in_section=i / max(len(tables_a), 1),
        )
        metas_a.append(meta)

    metas_b = []
    for i, t in enumerate(tables_b):
        section = _normalize_section(t.get("section"))
        if section in UNKNOWN_SECTIONS:
            continue
        if normalized_scope and section != normalized_scope:
            continue
        first_col = _extract_first_column(t)
        meta = TableMeta(
            table_key=t.get("table_id", f"t2_{i}"),
            section_norm=section,
            table_title=t.get("title", ""),
            first_col_labels=first_col,
            page=t.get("page_number", 0),
            row_count=len(t.get("rows", [])),
            position_in_section=i / max(len(tables_b), 1),
        )
        metas_b.append(meta)

    # Find best match for each table in A
    results = []
    matched_b = set()

    for meta_a in metas_a:
        # Only consider unmatched candidates
        candidates = [m for m in metas_b if m.table_key not in matched_b]
        result = find_best_match(meta_a, candidates)

        if result.is_match and result.table_b:
            matched_b.add(result.table_b.table_key)

        results.append(result)

    # Log matching stats
    matched_count = sum(1 for r in results if r.is_match)
    ambiguous_count = sum(1 for r in results if r.is_ambiguous)
    logger.info(
        f"Table matching: {matched_count}/{len(metas_a)} matched, "
        f"{ambiguous_count} ambiguous, {len(metas_b) - matched_count} unmatched in T2"
    )

    return results


def _extract_first_column(table: dict) -> list[str]:
    """Extract first column labels from a table."""
    indicators = table.get("first_column_indicators", [])
    if isinstance(indicators, list):
        normalized = [str(item).strip() for item in indicators if str(item).strip()]
        if normalized:
            return normalized

    rows = table.get("rows", [])
    first_col = []

    for row in rows:
        if isinstance(row, list) and len(row) > 0:
            cell = row[0]
            if cell and str(cell).strip():
                first_col.append(str(cell).strip())
        elif isinstance(row, dict):
            # Handle dict-based rows
            first_key = next(iter(row.keys()), None)
            if first_key:
                val = row[first_key]
                if val and str(val).strip():
                    first_col.append(str(val).strip())

    return first_col
