"""Contrat de compatibilite de la facade de comparaison des sections."""

from vigilance.text_analysis import comparison as facade
from vigilance.text_analysis.comparaison_sections import preparation_lots
from vigilance.text_analysis.comparaison_sections import resolution_alignements


def test_facade_reexporte_les_composants_historiques() -> None:
    """Les helpers publics historiques restent accessibles depuis la facade."""
    assert facade.ComparisonBatch is preparation_lots.ComparisonBatch
    assert facade._build_comparison_batches is preparation_lots._build_comparison_batches
    assert (
        facade._attach_alignment_metadata
        is resolution_alignements._attach_alignment_metadata
    )
    assert callable(facade._compare_texts_single_call)
    assert callable(facade._compare_section_texts)
