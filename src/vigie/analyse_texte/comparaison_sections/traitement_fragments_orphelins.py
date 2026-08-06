"""Traitement des fragments sans correspondance et des recuperations de section."""

from __future__ import annotations

import re
from typing import Any

from vigie.analyse_texte.chunk_alignment import ChunkAlignment
from vigie.analyse_texte.chunking import TextChunk, _chunk_subsection_text
from vigie.analyse_texte.comparaison_sections.execution_llm import (
    _compare_alignment_batches,
)
from vigie.analyse_texte.comparaison_sections.preparation_lots import (
    _atomic_unit_metadata,
    _build_comparison_batches,
    _exact_diff_change_for_strong_alignment,
    _reindex_changes,
    _split_exact_diff_alignments,
)
from vigie.analyse_texte.comparaison_sections.resolution_alignements import (
    _deduplicate_alignment_changes,
)
from vigie.analyse_texte.constants import _SECTION_LABELS
from vigie.analyse_texte.normalization import _sanitize_semantic_text
from vigie.analyse_texte.subsection_matching import _normalize_heading


def _heading_slug(heading: str) -> str:
    return re.sub(r"[^\w]+", "_", _normalize_heading(heading))[:40].strip("_") or "unknown"


def _display_heading_for_alignment(alignment: ChunkAlignment) -> str:
    """Heading affiché : H1 → H2 quand le match croise deux sous-sections."""
    heading_t1 = str(alignment.chunk_t1.subsection_heading if alignment.chunk_t1 else "").strip()
    heading_t2 = str(alignment.chunk_t2.subsection_heading if alignment.chunk_t2 else "").strip()
    if heading_t1 and heading_t2 and heading_t1 != heading_t2:
        return f"{heading_t1} → {heading_t2}"
    return heading_t1 or heading_t2 or "unknown"


def _annotate_section_rescue(alignment: ChunkAlignment) -> ChunkAlignment:
    """Marque un match Phase B comme récupération cross-sous-section."""
    alignment.reason = "section_rescue"
    return alignment


def _is_matched_alignment(alignment: ChunkAlignment) -> bool:
    return alignment.alignment_type not in {"possible_added", "possible_removed"}


def _chunk_subsection_bodies(
    *,
    section_key: str,
    heading: str,
    body: str,
    client: Any,
    embedding_model: str,
    semantic_model: str,
) -> list[TextChunk]:
    if not str(body or "").strip():
        return []
    section_title = _SECTION_LABELS.get(section_key, section_key)
    return _chunk_subsection_text(
        body,
        subsection_heading=heading,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )


def _process_alignment_group(
    *,
    client: Any,
    model: str,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    alignments: list[ChunkAlignment],
    idx_offset: int,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Exact-diff + LLM pour un groupe d'alignements déjà résolus."""
    if not alignments:
        return []
    exact_alignments, llm_alignments = _split_exact_diff_alignments(alignments)
    exact_changes: list[dict[str, Any]] = []
    for index, alignment in enumerate(exact_alignments, start=1):
        change = _exact_diff_change_for_strong_alignment(
            alignment=alignment,
            section_key=section_key,
            heading_label=heading_label,
            heading_slug=heading_slug,
            change_index=index,
        )
        if change is None:
            llm_alignments.append(alignment)
            continue
        exact_changes.append(change)

    batches = _build_comparison_batches(
        alignments=llm_alignments,
        heading_label=heading_label,
        heading_slug=heading_slug,
    )
    llm_changes = _compare_alignment_batches(
        client=client,
        model=model,
        section_key=section_key,
        batches=batches,
        bank_code=bank_code,
    )
    group_changes = _deduplicate_alignment_changes([*exact_changes, *llm_changes])
    return _reindex_changes(
        group_changes,
        section_key=section_key,
        heading_slug=heading_slug,
        idx_offset=idx_offset,
    )


def _changes_from_orphan_chunks(
    *,
    section_key: str,
    diff_type: str,
    chunks: list[TextChunk],
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Ajouts/retraits déterministes pour les orphelins restants après Phase B."""
    changes: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        heading = chunk.subsection_heading or "unknown"
        slug = _heading_slug(heading)
        text_t1 = chunk.text if diff_type == "removed" else ""
        text_t2 = chunk.text if diff_type == "added" else ""
        change_index = idx_offset + chunk_index
        changes.append(
            {
                "change_id": f"{section_key}_{slug}_change_{change_index:03d}",
                "section_key": section_key,
                "subsection_heading": heading,
                "diff_type": diff_type,
                "source_scope": "chunk",
                "alignment_id": f"unmatched_{chunk.chunk_id}",
                "alignment_type": f"unmatched_{diff_type}",
                "alignment_reason": "section_orphan_after_rescue",
                "chunk_id_t1": chunk.chunk_id if diff_type == "removed" else None,
                "chunk_id_t2": chunk.chunk_id if diff_type == "added" else None,
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
                "change_summary": (
                    f"Passage de sous-section {'ajouté' if diff_type == 'added' else 'supprimé'}: {heading}"
                ),
                **_atomic_unit_metadata(
                    chunk if diff_type == "removed" else None,
                    chunk if diff_type == "added" else None,
                ),
            }
        )
    return changes


def _unmatched_subsection_chunk_changes(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body: str,
    idx_offset: int,
    client: Any,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Produit des ajouts/retraits par chunk pour une sous-section sans paire."""
    chunks = _chunk_subsection_bodies(
        section_key=section_key,
        heading=heading,
        body=body,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    return _changes_from_orphan_chunks(
        section_key=section_key,
        diff_type=diff_type,
        chunks=chunks,
        idx_offset=idx_offset,
    )
