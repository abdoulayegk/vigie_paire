"""Comparaison semantique des sections textuelles bancaires."""

from vigie.analyse_texte.comparaison_sections.comparaison_section import (
    _compare_section_texts,
)
from vigie.analyse_texte.comparaison_sections.execution_llm import (
    _compare_alignment_batch,
    _compare_alignment_batches,
    _compare_texts_single_call,
)
from vigie.analyse_texte.comparaison_sections.modeles import (
    ChunkComparisonLLMChange,
    ChunkComparisonLLMResponse,
    ComparisonBatch,
)
from vigie.analyse_texte.comparaison_sections.preparation_lots import (
    _build_comparison_batches,
    _exact_diff_change_for_strong_alignment,
)
from vigie.analyse_texte.comparaison_sections.resolution_alignements import (
    _attach_alignment_metadata,
    _deduplicate_alignment_changes,
    _format_sub_items_breakdown,
    _materialize_semantic_alignment_decisions,
)

__all__ = [
    "ChunkComparisonLLMChange",
    "ChunkComparisonLLMResponse",
    "ComparisonBatch",
    "_attach_alignment_metadata",
    "_build_comparison_batches",
    "_compare_alignment_batch",
    "_compare_alignment_batches",
    "_compare_section_texts",
    "_compare_texts_single_call",
    "_deduplicate_alignment_changes",
    "_exact_diff_change_for_strong_alignment",
    "_format_sub_items_breakdown",
    "_materialize_semantic_alignment_decisions",
]
