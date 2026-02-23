"""
Module de comparaison pour detecter les changements entre les rapports trimestriels.
Gere la comparaison des tableaux, du texte et des notes de bas de page avec filtrage du bruit.

Includes:
- TableComparator: Comparaison de tableaux avec matching sémantique
- TableNormalizer: Normalisation structurelle pour tables hétérogènes
- TableClassifier: Classification des tableaux (A-F) selon la spécification
- IndicatorMatcher: Matching par indicateurs de première colonne
- TextComparator: Comparaison de contenu textuel
- NoiseFilter: Filtrage du bruit (changements numériques uniquement)
"""

from .table_comparator import TableComparator, compare_tables
from .text_comparator import TextComparator
from .footnote_comparator import FootnoteComparator
from .noise_filter import NoiseFilter
from .change_detector import ChangeDetector, Change

# Import du comparateur structurel
try:
    from .structural_comparator import (
        StructuralTableComparator,
        StructuralChange,
        TableStructuralChanges,
        StructuralComparisonResult,
        compare_tables_structural,
        get_structural_changes_summary,
        # Nouveaux exports pour analyse GenAI
        StructuralChangeAnalyzed,
        AnalyzedComparisonResult,
        StructuralChangeAnalyzer,
        generate_markdown_report,
        analyze_and_format_structural_changes,
        analyze_and_format_structural_changes_multi_section,
        group_changes_by_page,
        group_changes_by_table,
    )
except ImportError:
    StructuralTableComparator = None
    StructuralChange = None
    TableStructuralChanges = None
    StructuralComparisonResult = None
    StructuralChangeAnalyzed = None
    AnalyzedComparisonResult = None
    StructuralChangeAnalyzer = None

# Import du normalisateur de tableaux
try:
    from .table_normalizer import (
        TableNormalizer,
        NormalizedTable,
        TableMatchResult,
        normalize_table,
        compare_normalized_tables,
    )
except ImportError:
    TableNormalizer = None
    NormalizedTable = None
    TableMatchResult = None

# Import du classificateur de tableaux
try:
    from .table_classifier import (
        TableClassifier,
        TableType,
        TableClassification,
        TABLE_TYPE_KEYWORDS,
        BILINGUAL_MAPPING,
        classify_tables_batch,
    )
except ImportError:
    TableClassifier = None
    TableType = None
    TableClassification = None

# Import du matcher par indicateurs
try:
    from .indicator_matcher import (
        IndicatorMatcher,
        MatchResult,
        LearnedPattern,
        find_best_match,
        detect_renamed_tables,
    )
except ImportError:
    IndicatorMatcher = None
    MatchResult = None
    LearnedPattern = None
    normalize_table = None
    compare_normalized_tables = None

# Comparateur d'indicateurs JSON par table
try:
    from .indicator_comparator import compare_indicator_exports, run_strict_intra_section_compare
except ImportError:
    compare_indicator_exports = None
    run_strict_intra_section_compare = None

# Metriques de matching (Phase 5 - jeu de verite)
try:
    from .matching_metrics import (
        BankMatchingMetrics,
        evaluate_predictions,
        load_ground_truth,
    )
except ImportError:
    BankMatchingMetrics = None
    evaluate_predictions = None
    load_ground_truth = None

# Detection des deplacements cross-tableaux
try:
    from .displacement_detector import (
        DisplacedIndicator,
        detect_cross_table_displacements,
        AddedItem,
        RemovedItem,
    )
except ImportError:
    DisplacedIndicator = None
    detect_cross_table_displacements = None
    AddedItem = None
    RemovedItem = None

__all__ = [
    "TableComparator",
    "compare_tables",
    "TextComparator",
    "FootnoteComparator",
    "NoiseFilter",
    "ChangeDetector",
    "Change",
    "TableNormalizer",
    "NormalizedTable",
    "TableMatchResult",
    "normalize_table",
    "compare_normalized_tables",
    # Structural comparator
    "StructuralTableComparator",
    "StructuralChange",
    "TableStructuralChanges",
    "StructuralComparisonResult",
    "compare_tables_structural",
    "get_structural_changes_summary",
    # Structural analysis with GenAI
    "StructuralChangeAnalyzed",
    "AnalyzedComparisonResult",
    "StructuralChangeAnalyzer",
    "generate_markdown_report",
    "analyze_and_format_structural_changes",
    "group_changes_by_page",
    "group_changes_by_table",
    "compare_indicator_exports",
    "run_strict_intra_section_compare",
    # Displacement detection
    "DisplacedIndicator",
    "detect_cross_table_displacements",
    "AddedItem",
    "RemovedItem",
    # Matching metrics (Phase 5)
    "BankMatchingMetrics",
    "evaluate_predictions",
    "load_ground_truth",
]
