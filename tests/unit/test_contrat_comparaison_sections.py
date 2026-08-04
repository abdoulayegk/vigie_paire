"""Contrat de l'API publique de comparaison_sections."""

from vigie.analyse_texte.comparaison_sections import (
    ComparisonBatch,
    _attach_alignment_metadata,
    _build_comparison_batches,
    _compare_section_texts,
    _compare_texts_single_call,
)
from vigie.analyse_texte.comparaison_sections import preparation_lots
from vigie.analyse_texte.comparaison_sections import resolution_alignements
from vigie.analyse_texte.comparaison_sections.comparaison_section import (
    _compare_section_texts as compare_section_impl,
)
from vigie.analyse_texte.comparaison_sections.execution_llm import (
    _compare_texts_single_call as compare_single_impl,
)
from vigie.analyse_texte.comparaison_sections.modeles import (
    ComparisonBatch as ComparisonBatchModel,
)


def test_package_expose_les_composants_canoniques() -> None:
    """Les entrees publiques pointent vers les modules de responsabilite."""
    assert ComparisonBatch is ComparisonBatchModel
    assert _build_comparison_batches is preparation_lots._build_comparison_batches
    assert (
        _attach_alignment_metadata
        is resolution_alignements._attach_alignment_metadata
    )
    assert _compare_texts_single_call is compare_single_impl
    assert _compare_section_texts is compare_section_impl
