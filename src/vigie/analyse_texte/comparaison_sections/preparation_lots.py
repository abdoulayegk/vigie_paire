"""Preparation des alignements et lots de comparaison."""

from __future__ import annotations

import re
from typing import Any

from vigie.analyse_texte.chunk_alignment import (
    ChunkAlignment,
    _align_chunks_hybrid,
    _sequence_similarity,
)
from vigie.analyse_texte.chunking import TextChunk, _chunk_subsection_text
from vigie.analyse_texte.comparaison_sections.modeles import (
    ComparisonBatch,
    _COMPARISON_BATCH_SIZES,
    _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD,
)
from vigie.analyse_texte.constants import _SECTION_LABELS
from vigie.analyse_texte.normalization import _sanitize_semantic_text


def _prepare_subsection_alignments(
    *,
    section_key: str,
    subsection_heading_t1: str,
    subsection_heading_t2: str,
    body_t1: str,
    body_t2: str,
    client: Any | None = None,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-4o",
) -> list[ChunkAlignment]:
    """Prépare une paire de sous-sections en alignements hybrides locaux."""
    section_title = _SECTION_LABELS.get(section_key, section_key)
    chunks_t1 = _chunk_subsection_text(
        body_t1,
        subsection_heading=subsection_heading_t1,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    chunks_t2 = _chunk_subsection_text(
        body_t2,
        subsection_heading=subsection_heading_t2,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    # Après exclusion des tableaux, cellules et renvois non narratifs, une
    # sous-section peut légitimement ne plus avoir de contenu comparable. Ce
    # n'est pas une erreur de qualité : elle ne doit simplement pas produire
    # un ajout/retrait artificiel.
    if not chunks_t1 or not chunks_t2:
        return []
    return _align_chunks_hybrid(
        chunks_t1,
        chunks_t2,
        client=client,
        embedding_model=embedding_model,
    )


def _atomic_unit_metadata(
    chunk_t1: TextChunk | None,
    chunk_t2: TextChunk | None,
) -> dict[str, Any]:
    """Expose la filiation des unités sans modifier leur preuve source."""
    return {
        "unit_role_t1": chunk_t1.unit_role if chunk_t1 else None,
        "unit_role_t2": chunk_t2.unit_role if chunk_t2 else None,
        "parent_chunk_id_t1": chunk_t1.parent_chunk_id if chunk_t1 else None,
        "parent_chunk_id_t2": chunk_t2.parent_chunk_id if chunk_t2 else None,
        "atomic_marker_t1": chunk_t1.atomic_marker if chunk_t1 else None,
        "atomic_marker_t2": chunk_t2.atomic_marker if chunk_t2 else None,
        "parent_context_t1": chunk_t1.parent_context if chunk_t1 else "",
        "parent_context_t2": chunk_t2.parent_context if chunk_t2 else "",
    }


def _exact_diff_change_for_strong_alignment(
    *,
    alignment: ChunkAlignment,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    change_index: int,
) -> dict[str, Any] | None:
    """Compare localement un alignement très solide sans arbitrage GPT supplémentaire."""
    if alignment.alignment_type != "matched_strong":
        return None
    if alignment.chunk_t1 is None or alignment.chunk_t2 is None:
        return None
    text_t1 = alignment.chunk_t1.text
    text_t2 = alignment.chunk_t2.text
    comparison_t1 = alignment.chunk_t1.comparison_text or text_t1
    comparison_t2 = alignment.chunk_t2.comparison_text or text_t2
    similarity = _sequence_similarity(comparison_t1, comparison_t2)
    if similarity < _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD:
        return None
    normalized_t1 = re.sub(r"\s+", " ", comparison_t1).strip()
    normalized_t2 = re.sub(r"\s+", " ", comparison_t2).strip()
    if normalized_t1 == normalized_t2:
        diff_type = "unchanged"
        summary = "Passages alignés identiques après normalisation."
    else:
        diff_type = "modified"
        summary = "Passages fortement alignés avec une différence locale exacte."
    return {
        "change_id": f"{section_key}_{heading_slug}_change_{change_index:03d}",
        "section_key": section_key,
        "subsection_heading": heading_label,
        "diff_type": diff_type,
        "source_scope": "chunk",
        "alignment_id": alignment.alignment_id,
        "alignment_type": alignment.alignment_type,
        "chunk_id_t1": alignment.chunk_t1.chunk_id,
        "chunk_id_t2": alignment.chunk_t2.chunk_id,
        "semantic_text_t1": _sanitize_semantic_text(text_t1),
        "semantic_text_t2": _sanitize_semantic_text(text_t2),
        "source_text_t1": text_t1,
        "source_text_t2": text_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": text_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": text_t2[:400]},
        "change_summary": summary,
        "alignment_decision": "same_disclosure",
        "alignment_confidence": "high",
        "alignment_rationale": (
            f"Alignement hybride fort (tfidf={alignment.tfidf_score:.2f}, "
            f"embedding={alignment.embedding_score:.2f}, sequence={similarity:.2f}); "
            "diff exacte locale sans arbitrage GPT supplémentaire."
        ),
        "alignment_reason": alignment.reason,
        "tfidf_score": alignment.tfidf_score,
        "embedding_score": alignment.embedding_score,
        **_atomic_unit_metadata(alignment.chunk_t1, alignment.chunk_t2),
    }


def _batch_size_for_alignment_type(alignment_type: str) -> int:
    return _COMPARISON_BATCH_SIZES.get(alignment_type, 1)


def _build_comparison_batches(
    *,
    alignments: list[ChunkAlignment],
    heading_label: str,
    heading_slug: str,
) -> list[ComparisonBatch]:
    """Découpe les alignements en lots homogènes et ordonnés."""
    batches: list[ComparisonBatch] = []
    current_type = ""
    current: list[ChunkAlignment] = []

    def flush_current() -> None:
        nonlocal current, current_type
        if not current:
            return
        batch_index = len(batches)
        batches.append(
            ComparisonBatch(
                batch_id=f"b{batch_index:02d}",
                alignment_type=current_type,
                alignments=current,
                heading_label=heading_label,
                heading_slug=heading_slug,
                idx_offset=batch_index * 1000,
            )
        )
        current = []
        current_type = ""

    for alignment in alignments:
        alignment_type = alignment.alignment_type
        max_size = _batch_size_for_alignment_type(alignment_type)
        if current and (alignment_type != current_type or len(current) >= max_size):
            flush_current()
        current_type = alignment_type
        current.append(alignment)
    flush_current()
    return batches


def _split_exact_diff_alignments(
    alignments: list[ChunkAlignment],
) -> tuple[list[ChunkAlignment], list[ChunkAlignment]]:
    """Sépare les alignements hybrides assez solides pour un diff exact local."""
    exact: list[ChunkAlignment] = []
    remaining: list[ChunkAlignment] = []
    for alignment in alignments:
        # matched_strong may come from TF-IDF-only fallback (embedding_score=0)
        # or from embeddings; both deserve the local exact-diff fast path when
        # the sequences are near-identical.
        strong_signal = (
            alignment.embedding_score >= 0.85 or alignment.tfidf_score >= 0.85
        )
        if (
            alignment.alignment_type == "matched_strong"
            and strong_signal
            and alignment.chunk_t1 is not None
            and alignment.chunk_t2 is not None
            and _sequence_similarity(alignment.chunk_t1.text, alignment.chunk_t2.text)
            >= _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD
        ):
            exact.append(alignment)
        else:
            remaining.append(alignment)
    return exact, remaining


def _reindex_changes(
    changes: list[dict[str, Any]],
    *,
    section_key: str,
    heading_slug: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Réindexe les changements fusionnés après appels LLM parallèles."""
    reindexed: list[dict[str, Any]] = []
    for local_idx, change in enumerate(changes, start=1):
        updated = dict(change)
        updated["change_id"] = f"{section_key}_{heading_slug}_change_{idx_offset + local_idx:03d}"
        reindexed.append(updated)
    return reindexed
