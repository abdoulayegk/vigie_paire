"""Chunking déterministe des corps de sous-sections markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from vigie.analyse_texte.atomic_units import AtomicCandidate, split_atomic_candidate
from vigie.analyse_texte.boundary_repair import RepairableBlock, repair_block_boundaries
from vigie.analyse_texte.list_items import parse_list_item_line
from vigie.analyse_texte.normalization import _is_not_applicable_marker
from vigie.analyse_texte.semantic_chunking import (
    SemanticChunkingError,
    _requires_semantic_partition,
    _semantic_partition_paragraphs,
)

_HEADING_LINE_RE = re.compile(r"^\s*#{2,6}\s+")
_MARKDOWN_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")


@dataclass(slots=True)
class TextChunk:
    """Unité de comparaison déterministe à l'intérieur d'une sous-section."""

    chunk_id: str
    kind: str
    text: str
    subsection_heading: str
    hierarchy_path: str
    order: int
    comparison_text: str = ""
    unit_role: str = "standalone"
    parent_chunk_id: str | None = None
    atomic_marker: str | None = None
    parent_context: str = ""


def _is_bullet_line(line: str) -> bool:
    """Indique si une ligne démarre une puce ou un item numéroté simple."""
    return parse_list_item_line(line) is not None


def _is_heading_line(line: str) -> bool:
    """Indique si une ligne est un titre markdown à exclure du chunking."""
    return bool(_HEADING_LINE_RE.match(str(line or "")))


def _strip_list_markers(text: str) -> str:
    """Retire les puces de présentation sans séparer les éléments de liste."""
    stripped_lines: list[str] = []
    for line in str(text or "").splitlines():
        parsed = parse_list_item_line(line)
        cleaned = parsed.text if parsed is not None else line.strip()
        if cleaned:
            stripped_lines.append(cleaned)
    return "\n".join(stripped_lines)


def _is_narrative_comparison_candidate(text: str) -> bool:
    """Indique si le bloc doit rejoindre la comparaison.

    Le filtrage de contenu est effectué en amont, avec les métadonnées Docling
    et les zones géométriques des tableaux. Le chunker ne doit donc jamais
    écarter un passage pour sa longueur, un symbole financier ou un renvoi à un
    tableau : tous peuvent être des divulgations pertinentes. Les seules
    exceptions sont une table Markdown structurellement identifiable et le
    marqueur autonome « s.o. ».
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _is_not_applicable_marker(value):
        return False
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return False
    if any(_MARKDOWN_TABLE_DIVIDER_RE.match(line) for line in lines):
        return False
    return not (len(lines) >= 2 and all(_MARKDOWN_TABLE_ROW_RE.match(line) for line in lines))


def _candidate_kind(lines: list[str]) -> str:
    """Classe un bloc candidat comme paragraphe ou liste."""
    for line in lines:
        if line.strip():
            return "list" if _is_bullet_line(line) else "paragraph"
    return "paragraph"


def _split_repairable_blocks(text: str) -> list[RepairableBlock]:
    """Découpe le Markdown tout en conservant les barrières structurelles."""
    candidates: list[RepairableBlock] = []
    current: list[str] = []
    hard_boundary_before = False

    def flush_current() -> None:
        """Ajoute le bloc courant et conserve sa frontière structurelle."""
        nonlocal hard_boundary_before
        if not current:
            return
        cleaned = "\n".join(line.rstrip() for line in current).strip()
        if cleaned:
            candidates.append(
                RepairableBlock(
                    kind=_candidate_kind(current),
                    text=cleaned,
                    hard_boundary_before=hard_boundary_before,
                )
            )
            hard_boundary_before = False
        current.clear()

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if _is_heading_line(line):
            flush_current()
            hard_boundary_before = True
            continue
        if not line.strip():
            flush_current()
            continue
        current.append(line)

    flush_current()
    return candidates


def _group_adjacent_lists(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Regroupe les blocs de liste consécutifs pour garder une liste en un chunk."""
    grouped: list[tuple[str, str]] = []
    for kind, text in candidates:
        if kind == "list" and grouped and grouped[-1][0] == "list":
            previous_kind, previous_text = grouped[-1]
            grouped[-1] = (previous_kind, f"{previous_text}\n\n{text}")
            continue
        grouped.append((kind, text))
    return grouped


def _word_count(text: str) -> int:
    """Compte les mots d'un bloc candidat."""
    return len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", str(text or "")))


def _looks_like_short_label(text: str) -> bool:
    """Détecte un micro-titre extrait comme paragraphe isolé."""
    value = str(text or "").strip()
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ'’-]*", value)
    return (
        bool(words)
        and len(words) <= 5
        and "\n" not in value
        and not re.search(r"[.!?;:]\s*$", value)
        and not re.search(r"\d", value)
        and value[0].isupper()
    )


def _merge_short_labels_with_following(
    candidates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Attache un micro-titre au paragraphe narratif qu'il introduit."""
    merged: list[tuple[str, str]] = []
    index = 0
    while index < len(candidates):
        kind, candidate_text = candidates[index]
        if index + 1 < len(candidates):
            next_kind, next_text = candidates[index + 1]
            if (
                kind == "paragraph"
                and next_kind == "paragraph"
                and _looks_like_short_label(candidate_text)
                and _word_count(next_text) >= 8
            ):
                merged.append(("paragraph", f"{candidate_text}\n\n{next_text}"))
                index += 2
                continue
        merged.append((kind, candidate_text))
        index += 1
    return merged


def _looks_like_list_context(text: str) -> bool:
    """Reconnaît une introduction autonome qui annonce une liste."""
    value = str(text or "").strip()
    return _word_count(value) >= 5 and value.endswith(":")


def _chunk_subsection_text(
    text: str,
    *,
    subsection_heading: str = "",
    section_title: str = "",
    min_chars: int = 0,
    client: Any | None = None,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-5.4",
) -> list[TextChunk]:
    """Découpe un corps en unités d'idée et listes cohérentes.

    Les paragraphes simples restent intacts. Tous les paragraphes complexes de
    la sous-section partagent un lot d'embeddings, puis seules les frontières
    ambiguës sont arbitrées par LLM. Il n'existe aucun fallback en cas d'échec.
    ``min_chars`` reste accepté uniquement pour compatibilité d'appel.
    """
    _ = min_chars
    repair_result = repair_block_boundaries(_split_repairable_blocks(text))
    candidates = [(block.kind, block.text) for block in repair_result.blocks]
    candidates = _merge_short_labels_with_following(candidates)
    candidates = _group_adjacent_lists(candidates)
    filtered_candidates: list[AtomicCandidate] = []
    for parent_index, (kind, candidate_text) in enumerate(candidates):
        parent_key = f"p{parent_index:02d}"
        atomic_candidates = split_atomic_candidate(
            kind=kind,
            text=candidate_text,
            parent_key=parent_key,
        )
        if (
            atomic_candidates is not None
            and kind == "list"
            and parent_index > 0
            and candidates[parent_index - 1][0] == "paragraph"
            and _looks_like_list_context(candidates[parent_index - 1][1])
            and filtered_candidates
        ):
            context_text = candidates[parent_index - 1][1]
            parent_key = f"p{parent_index - 1:02d}"
            previous = filtered_candidates[-1]
            if previous.text == context_text and previous.unit_role == "standalone":
                filtered_candidates[-1] = replace(
                    previous,
                    kind="list_context",
                    unit_role="context",
                    parent_key=parent_key,
                    parent_context=context_text,
                )
                atomic_candidates = [
                    replace(
                        candidate,
                        parent_key=parent_key,
                        parent_context=context_text,
                    )
                    for candidate in atomic_candidates
                ]
        if atomic_candidates is None:
            # Une liste mono-item conserve le comportement historique. Les
            # marqueurs de présentation ne doivent pas influencer la similarité.
            normalized_candidate = _strip_list_markers(candidate_text) if kind == "list" else candidate_text
            atomic_candidates = [
                AtomicCandidate(
                    kind=kind,
                    text=normalized_candidate,
                    comparison_text=normalized_candidate,
                )
            ]
        filtered_candidates.extend(
            candidate
            for candidate in atomic_candidates
            if _is_narrative_comparison_candidate(candidate.comparison_text)
        )

    complex_indexes = [
        index
        for index, candidate in enumerate(filtered_candidates)
        if candidate.kind == "paragraph" and _requires_semantic_partition(candidate.comparison_text)
    ]
    partitions_by_index: dict[int, list[str]] = {}
    if complex_indexes:
        if client is None:
            raise SemanticChunkingError(
                "Un client OpenAI est requis pour découper les paragraphes complexes; aucun fallback n'est autorisé."
            )
        complex_paragraphs = [filtered_candidates[index].comparison_text for index in complex_indexes]
        partitions = _semantic_partition_paragraphs(
            complex_paragraphs,
            client=client,
            embedding_model=embedding_model,
            semantic_model=semantic_model,
        )
        partitions_by_index = dict(zip(complex_indexes, partitions, strict=True))

    split_candidates: list[AtomicCandidate] = []
    for index, candidate in enumerate(filtered_candidates):
        parts = partitions_by_index.get(index)
        if parts is None:
            split_candidates.append(candidate)
            continue
        split_candidates.extend(
            AtomicCandidate(
                kind=candidate.kind,
                text=part,
                comparison_text=part,
                unit_role=candidate.unit_role,
                parent_key=candidate.parent_key,
                marker=candidate.marker,
                parent_context=candidate.parent_context,
            )
            for part in parts
            if _is_narrative_comparison_candidate(part)
        )

    subsection = str(subsection_heading or "").strip()
    section = str(section_title or "").strip()
    hierarchy_path = f"{section} > {subsection}" if section and subsection else section or subsection
    # Prefixed IDs stay unique when orphans from several subsections are
    # later merged for a section-wide rescue pass.
    id_prefix = _chunk_id_prefix(subsection)

    chunk_ids = [f"{id_prefix}c{index:02d}" if id_prefix else f"c{index:02d}" for index in range(len(split_candidates))]
    context_ids = {
        candidate.parent_key: chunk_ids[index]
        for index, candidate in enumerate(split_candidates)
        if candidate.unit_role == "context" and candidate.parent_key
    }
    chunks: list[TextChunk] = []
    for index, candidate in enumerate(split_candidates):
        parent_chunk_id = None
        if candidate.unit_role == "item" and candidate.parent_key:
            parent_chunk_id = context_ids.get(
                candidate.parent_key,
                f"{id_prefix}parent_{candidate.parent_key}",
            )
        chunks.append(
            TextChunk(
                chunk_id=chunk_ids[index],
                kind=candidate.kind,
                text=candidate.text,
                subsection_heading=subsection,
                hierarchy_path=hierarchy_path,
                order=index,
                comparison_text=candidate.comparison_text,
                unit_role=candidate.unit_role,
                parent_chunk_id=parent_chunk_id,
                atomic_marker=candidate.marker,
                parent_context=candidate.parent_context,
            )
        )
    return chunks


def _chunk_id_prefix(subsection_heading: str) -> str:
    """Slug stable pour préfixer les chunk_id d'une sous-section."""
    slug = re.sub(r"[^\w]+", "_", str(subsection_heading or "").strip().lower(), flags=re.UNICODE)
    slug = slug.strip("_")[:40].strip("_")
    return f"{slug}_" if slug else ""


def _format_chunks_for_prompt(chunks: list[TextChunk]) -> str:
    """Formate les chunks pour le prompt GPT sans modifier leur texte source."""
    blocks: list[str] = []
    for chunk in chunks:
        path = f" | {chunk.hierarchy_path}" if chunk.hierarchy_path else ""
        blocks.append(f"[{chunk.chunk_id} | {chunk.kind}{path}]\n{chunk.text}")
    return "\n\n".join(blocks)
