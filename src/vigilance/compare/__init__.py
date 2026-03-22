"""
Surface publique du module de comparaison des tableaux T1/T2.

Ce module expose les symboles essentiels utilisés par le pipeline de comparaison
(``comparison_runner.py``) et par tout code externe qui souhaite utiliser le
moteur de comparaison sans importer directement les sous-modules internes.

Symboles exportés
-----------------
run_strict_intra_section_compare (table_pairing_engine):
    Point d'entrée principal du pipeline de pairing. Associe les tableaux du
    trimestre précédent (T1) à ceux du trimestre courant (T2) en respectant
    la contrainte de section (un tableau de gestion du capital ne peut être
    apparié qu'avec un autre tableau de gestion du capital).

FootnoteComparator, compare_footnotes (footnote_comparator):
    Classe et fonction utilitaire pour détecter les changements dans les notes
    de bas de page entre deux tableaux appariés.

MatchDecision, match_decision, match_tables_intra_section (indicator_comparator):
    Moteur de pairing alternatif (legacy) avec scoring multi-signaux détaillé.
    Toujours utilisé par ``comparison_runner.py`` pour certaines décisions de
    matching et pour la structure ``MatchDecision``.
"""

from vigilance.compare.indicator_comparator import (
    MatchDecision,
    match_decision,
    match_tables_intra_section,
)
from vigilance.compare.footnote_comparator import FootnoteComparator, compare_footnotes
from vigilance.compare.table_pairing_engine import (
    run_recall_first_compare,
    run_strict_intra_section_compare,
)

__all__ = [
    "FootnoteComparator",
    "MatchDecision",
    "compare_footnotes",
    "match_decision",
    "match_tables_intra_section",
    "run_recall_first_compare",
    "run_strict_intra_section_compare",
]
