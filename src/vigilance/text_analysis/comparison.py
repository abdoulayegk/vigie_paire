"""Facade historique de la comparaison semantique des sections.

Les composants vivent dans ``vigilance.text_analysis.comparaison_sections``.
Les wrappers conservent les points de monkeypatch du pipeline et des tests.
"""

from __future__ import annotations

from vigilance.text_analysis.chunk_alignment import (  # noqa: F401
    ChunkAlignment,
    _align_chunks_hybrid,
    _format_alignments_for_prompt,
    _sequence_similarity,
)
from vigilance.text_analysis.chunking import TextChunk, _chunk_subsection_text  # noqa: F401
from vigilance.text_analysis.comparaison_sections import comparaison_section as _section_mod
from vigilance.text_analysis.comparaison_sections import execution_llm as _execution_mod
from vigilance.text_analysis.comparaison_sections.comparaison_section import (
    _compare_section_texts as _compare_section_texts_impl,
)
from vigilance.text_analysis.comparaison_sections.execution_llm import (  # noqa: F401
    _compare_alignment_batch,
    _compare_alignment_batches,
    _compare_texts_single_call as _compare_texts_single_call_impl,
)
from vigilance.text_analysis.comparaison_sections.modeles import (  # noqa: F401
    ChunkComparisonLLMChange,
    ChunkComparisonLLMResponse,
    ComparisonBatch,
    _CHUNK_COMPARISON_VALIDATION_RETRY_MESSAGE,
    _COMPARISON_BATCH_SIZES,
    _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD,
    _MAX_COMPARISON_LLM_WORKERS,
)
from vigilance.text_analysis.comparaison_sections.preparation_lots import (  # noqa: F401
    _atomic_unit_metadata,
    _batch_size_for_alignment_type,
    _build_comparison_batches,
    _exact_diff_change_for_strong_alignment,
    _prepare_subsection_alignments,
    _reindex_changes,
    _split_exact_diff_alignments,
)
from vigilance.text_analysis.comparaison_sections.resolution_alignements import (  # noqa: F401
    _SEMANTIC_ALIGNMENT_DECISIONS,
    _attach_alignment_metadata,
    _coerce_text_to_chunk,
    _deduplicate_alignment_changes,
    _format_sub_items_breakdown,
    _materialize_semantic_alignment_decisions,
    _normalize_for_alignment_contains,
    _resolved_alignment_confidence,
    _resolved_alignment_decision,
)
from vigilance.text_analysis.comparaison_sections.traitement_fragments_orphelins import (  # noqa: F401
    _annotate_section_rescue,
    _changes_from_orphan_chunks,
    _chunk_subsection_bodies,
    _display_heading_for_alignment,
    _heading_slug,
    _is_matched_alignment,
    _process_alignment_group,
    _unmatched_subsection_chunk_changes,
)
from vigilance.text_analysis.constants import _SECTION_LABELS  # noqa: F401
from vigilance.text_analysis.models import TextAnalysisQualityError  # noqa: F401
from vigilance.text_analysis.normalization import (  # noqa: F401
    _sanitize_explanation,
    _sanitize_semantic_text,
)
from vigilance.text_analysis.openai_client import (  # noqa: F401
    _call_structured_completion_with_correction,
)
from vigilance.text_analysis.subsection_matching import (  # noqa: F401
    OrphanSubsection,
    _normalize_heading,
    _pair_subsections,
    _parse_subsections,
    _resolve_orphan_subsections,
    _synthetic_subsection_rename_change,
)


def _compare_texts_single_call(*args, **kwargs):
    """Delegue l appel unique en propageant le client monkeypatche."""
    _execution_mod._call_structured_completion_with_correction = (
        globals()["_call_structured_completion_with_correction"]
    )
    return _compare_texts_single_call_impl(*args, **kwargs)


def _compare_section_texts(*args, **kwargs):
    """Delegue une section en propageant les points d injection historiques."""
    _execution_mod._compare_texts_single_call = globals()["_compare_texts_single_call"]
    _section_mod._align_chunks_hybrid = globals()["_align_chunks_hybrid"]
    _section_mod._chunk_subsection_bodies = globals()["_chunk_subsection_bodies"]
    _section_mod._resolve_orphan_subsections = globals()["_resolve_orphan_subsections"]
    return _compare_section_texts_impl(*args, **kwargs)
