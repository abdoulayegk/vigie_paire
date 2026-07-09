"""Façade de compatibilité pour le pipeline texte modulaire.

La logique métier vit maintenant dans ``vigilance.text_analysis``. Ce module
conserve les anciens imports, y compris plusieurs helpers privés encore utilisés
par les tests et certains monkeypatchs historiques.
"""

from __future__ import annotations

import time

from vigilance.extraction.section_locator import locate_sections_in_pdf
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.text_analysis import comparison as _comparison_mod
from vigilance.text_analysis import pipeline as _pipeline_mod
from vigilance.text_analysis import sections as _sections_mod
from vigilance.text_analysis import subsection_matching as _subsection_mod
from vigilance.text_analysis import triage as _triage_mod
from vigilance.text_analysis.chunk_alignment import (
    ChunkAlignment,
    ChunkCandidate,
    _align_chunks_tfidf,
    _format_alignments_for_prompt,
)
from vigilance.text_analysis.chunking import TextChunk, _chunk_subsection_text, _format_chunks_for_prompt
from vigilance.text_analysis.comparison import ComparisonBatch, _build_comparison_batches
from vigilance.text_analysis.comparison import _compare_section_texts as _comparison_compare_section_texts
from vigilance.text_analysis.comparison import _compare_texts_single_call as _comparison_compare_texts_single_call
from vigilance.text_analysis.constants import (
    UNIFIED_TEXT_SCHEMA_VERSION,
    _CANONICAL_TO_TEXT_KEY,
    _MODEL_MAX_OUTPUT_TOKENS,
    _OPENAI_TIMEOUT_SECONDS,
    _SECTION_LABELS,
    _SUBSECTION_SPLIT_RE,
    _T4_TEXT_TARGET_SECTIONS,
    _TARGET_SECTIONS_BY_BANK,
    _THEME_BY_SECTION,
    _TRIAGE_BATCH_SIZE,
    _TRIAGE_LENGTH_RETRIES,
    _TRIAGE_SEMANTIC_TEXT_LIMIT,
    _TRIAGE_SOURCE_SNIPPET_LIMIT,
    _TRIAGE_TRANSPORT_RETRIES,
)
from vigilance.text_analysis.extraction import (
    _build_section_audit,
    _classify_block_type,
    _docling_bbox_to_norm,
    _exclusion_reason_for_block,
    _extract_audits_for_pdf,
    _extract_docling_page_blocks,
    _repeated_text_counts,
    _table_regions_for_pages,
)
from vigilance.text_analysis.markdown import (
    _build_block_page_index,
    _build_text_extraction_markdown,
    _extract_section_text_from_markdown,
    _find_page_for_fragment,
    _format_page_marker,
    _format_page_suffix,
    _get_page_number_offset_for_period,
    _markdown_blocks_for_section,
    _parse_page_index_from_markdown,
    _rewrite_page_markers_for_display,
    _section_page_range_from_index,
    _strip_page_markers,
)
from vigilance.text_analysis.models import (
    PDFBlock,
    ResolvedSection,
    SectionAudit,
    SemanticUnit,
    TextAnalysisQualityError,
)
from vigilance.text_analysis.normalization import (
    _bbox_overlap_ratio,
    _block_overlaps_table,
    _contains_dense_numeric_line,
    _count_numeric_values,
    _infer_table_footnote_bboxes,
    _json_dumps,
    _looks_like_footnote,
    _looks_like_narrative_paragraph,
    _looks_like_table_footnote_text,
    _looks_like_table_or_financial_grid,
    _normalized_block_text,
    _sanitize_explanation,
    _sanitize_semantic_text,
)
from vigilance.text_analysis.openai_client import (
    _append_concise_triage_retry_message,
    _build_json_repair_messages,
    _build_openai_client,
    _call_json_completion,
    _call_structured_completion,
    _call_structured_completion_with_correction,
    _classify_openai_transport_error,
    _max_output_tokens_for_model,
    _parse_json_object_response,
    _preview_response_text,
    _strip_markdown_fences,
    _truncate_prompt_text,
)
from vigilance.text_analysis.pipeline import (
    _effective_text_allowed_sections,
    _prepare_period_extraction,
    _resolve_text_project_root,
)
from vigilance.text_analysis.sections import (
    _allowed_target_sections as _sections_allowed_target_sections,
    _next_section_by_key,
    _resolve_sections as _sections_resolve_sections,
    _section_window_for_page,
    _sorted_sections,
)
from vigilance.text_analysis.subsection_matching import (
    _gpt_match_orphan_headings as _subsection_gpt_match_orphan_headings,
    _normalize_heading,
    _pair_subsections,
    _parse_subsections,
    _synthetic_subsection_change,
    _synthetic_subsection_rename_change,
)
from vigilance.text_analysis.summary import (
    _STRONG_AMF_THEMES_FOR_MODERE_RETENTION,
    _build_global_summary,
    _is_new_major_or_allowed_moderate,
    _is_non_cosmetic_change,
    _retained_change_sort_key,
)
from vigilance.text_analysis.triage import (
    _FEW_SHOT_TRIAGE_AMF,
    _default_triage,
    _derive_legacy_fields,
    _triage_section_changes as _triage_triage_section_changes,
)


def _compat_target(name: str, facade_wrapper: object, original: object) -> object:
    current = globals()[name]
    return original if current is facade_wrapper else current


def _sync_section_hooks() -> None:
    _sections_mod.locate_sections_in_pdf = locate_sections_in_pdf
    _sections_mod.canonicalize_section = canonicalize_section


def _resolve_sections(*args, **kwargs):
    _sync_section_hooks()
    return _sections_resolve_sections(*args, **kwargs)


def _allowed_target_sections(*args, **kwargs):
    return _sections_allowed_target_sections(*args, **kwargs)


def _gpt_match_orphan_headings(*args, **kwargs):
    _subsection_mod._call_json_completion = _call_json_completion
    return _subsection_gpt_match_orphan_headings(*args, **kwargs)


def _compare_texts_single_call(*args, **kwargs):
    _comparison_mod._call_json_completion = _call_json_completion
    return _comparison_compare_texts_single_call(*args, **kwargs)


def _compare_section_texts(*args, **kwargs):
    _comparison_mod._call_json_completion = _call_json_completion
    _comparison_mod._compare_texts_single_call = _compat_target(
        "_compare_texts_single_call",
        _FACADE_COMPARE_TEXTS_SINGLE_CALL,
        _comparison_compare_texts_single_call,
    )
    _comparison_mod._gpt_match_orphan_headings = _compat_target(
        "_gpt_match_orphan_headings",
        _FACADE_GPT_MATCH_ORPHAN_HEADINGS,
        _subsection_gpt_match_orphan_headings,
    )
    return _comparison_compare_section_texts(*args, **kwargs)


def _triage_section_changes(*args, **kwargs):
    _triage_mod._call_structured_completion_with_correction = _call_structured_completion_with_correction
    _triage_mod._truncate_prompt_text = _truncate_prompt_text
    return _triage_triage_section_changes(*args, **kwargs)


def _sync_pipeline_hooks() -> None:
    _pipeline_mod._build_openai_client = _build_openai_client
    _pipeline_mod._resolve_sections = globals()["_resolve_sections"]
    _pipeline_mod._extract_audits_for_pdf = globals()["_extract_audits_for_pdf"]
    _pipeline_mod._compare_section_texts = globals()["_compare_section_texts"]
    _pipeline_mod._triage_section_changes = globals()["_triage_section_changes"]
    _pipeline_mod._extract_section_text_from_markdown = globals()["_extract_section_text_from_markdown"]
    _pipeline_mod._find_page_for_fragment = globals()["_find_page_for_fragment"]


def run_text_extraction_pipeline(*args, **kwargs):
    """Exécuter l'extraction texte via le package modulaire."""
    _sync_pipeline_hooks()
    return _pipeline_mod.run_text_extraction_pipeline(*args, **kwargs)


def run_text_analysis_pipeline(*args, **kwargs):
    """Exécuter l'analyse texte complète via le package modulaire."""
    _sync_pipeline_hooks()
    return _pipeline_mod.run_text_analysis_pipeline(*args, **kwargs)


_FACADE_COMPARE_TEXTS_SINGLE_CALL = _compare_texts_single_call
_FACADE_GPT_MATCH_ORPHAN_HEADINGS = _gpt_match_orphan_headings
