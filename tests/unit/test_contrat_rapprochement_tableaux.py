"""Contrat de compatibilite de la facade de rapprochement des tableaux."""

from vigilance import comparison_matching as facade
from vigilance.rapprochement_tableaux import moteur_rapprochement
from vigilance.rapprochement_tableaux import correction_reponses


def test_facade_reexporte_les_points_entree_historiques() -> None:
    """Les appelants historiques continuent de viser les implementations extraites."""
    assert facade._run_table_matching is moteur_rapprochement._run_table_matching
    assert facade._run_matching_stage is moteur_rapprochement._run_matching_stage
    assert (
        facade._build_matching_repair_response_model
        is correction_reponses._build_matching_repair_response_model
    )
