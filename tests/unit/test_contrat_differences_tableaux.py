"""Contrat de compatibilite de la facade des differences de tableaux."""

from vigilance import comparison_diff_gpt as facade
from vigilance.differences_tableaux import comparaison_deterministe
from vigilance.differences_tableaux import comparaison_paire
from vigilance.differences_tableaux import filtrage_artefacts


def test_facade_reexporte_les_points_entree_historiques() -> None:
    """Les imports historiques pointent vers les implementations extraites."""
    assert facade.diff_table_pair_gpt is comparaison_paire.diff_table_pair_gpt
    assert (
        facade._deterministic_indicator_diff
        is comparaison_deterministe._deterministic_indicator_diff
    )
    assert facade._inspect_diff_artifacts_gpt is filtrage_artefacts._inspect_diff_artifacts_gpt
