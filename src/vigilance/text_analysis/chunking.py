"""Chunking déterministe des corps de sous-sections markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vigilance.text_analysis.list_items import parse_list_item_line
from vigilance.text_analysis.normalization import _is_not_applicable_marker
from vigilance.text_analysis.semantic_chunking import (
    SemanticChunkingError,
    _requires_semantic_partition,
    _semantic_partition_paragraphs,
)


_HEADING_LINE_RE = re.compile(r"^\s*#{2,6}\s+")
_MARKDOWN_TABLE_DIVIDER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
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


def _split_candidate_blocks(text: str) -> list[tuple[str, str]]:
    """Découpe un texte en blocs candidats par lignes vides, sans titres markdown."""
    candidates: list[tuple[str, str]] = []
    current: list[str] = []

    def flush_current() -> None:
        if not current:
            return
        cleaned = "\n".join(line.rstrip() for line in current).strip()
        if cleaned:
            candidates.append((_candidate_kind(current), cleaned))
        current.clear()

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if _is_heading_line(line):
            flush_current()
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


def _chunk_subsection_text(
    text: str,
    *,
    subsection_heading: str = "",
    section_title: str = "",
    min_chars: int = 0,
    client: Any | None = None,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-4o",
) -> list[TextChunk]:
    """Découpe un corps en unités d'idée et listes cohérentes.

    Les paragraphes simples restent intacts. Tous les paragraphes complexes de
    la sous-section partagent un lot d'embeddings, puis seules les frontières
    ambiguës sont arbitrées par LLM. Il n'existe aucun fallback en cas d'échec.
    ``min_chars`` reste accepté uniquement pour compatibilité d'appel.
    """
    _ = min_chars
    candidates = _split_candidate_blocks(text)
    candidates = _merge_short_labels_with_following(candidates)
    candidates = _group_adjacent_lists(candidates)
    filtered_candidates: list[tuple[str, str]] = []
    for kind, candidate_text in candidates:
        # Une liste est une seule unité sémantique : ses items restent groupés.
        # Les marqueurs ``[]``/``[x]`` ne sont que de la mise en forme et ne
        # doivent jamais entraîner l'exclusion de son contenu narratif.
        normalized_candidate = _strip_list_markers(candidate_text) if kind == "list" else candidate_text
        if not _is_narrative_comparison_candidate(normalized_candidate):
            continue
        filtered_candidates.append((kind, normalized_candidate))

    complex_indexes = [
        index
        for index, (kind, candidate_text) in enumerate(filtered_candidates)
        if kind == "paragraph" and _requires_semantic_partition(candidate_text)
    ]
    partitions_by_index: dict[int, list[str]] = {}
    if complex_indexes:
        if client is None:
            raise SemanticChunkingError(
                "Un client OpenAI est requis pour découper les paragraphes complexes; aucun fallback n'est autorisé."
            )
        complex_paragraphs = [filtered_candidates[index][1] for index in complex_indexes]
        partitions = _semantic_partition_paragraphs(
            complex_paragraphs,
            client=client,
            embedding_model=embedding_model,
            semantic_model=semantic_model,
        )
        partitions_by_index = dict(zip(complex_indexes, partitions, strict=True))

    split_candidates: list[tuple[str, str]] = []
    for index, (kind, candidate_text) in enumerate(filtered_candidates):
        parts = partitions_by_index.get(index, [candidate_text])
        split_candidates.extend(
            (kind, part) for part in parts if _is_narrative_comparison_candidate(part)
        )

    subsection = str(subsection_heading or "").strip()
    section = str(section_title or "").strip()
    hierarchy_path = f"{section} > {subsection}" if section and subsection else section or subsection
    # Prefixed IDs stay unique when orphans from several subsections are
    # later merged for a section-wide rescue pass.
    id_prefix = _chunk_id_prefix(subsection)

    return [
        TextChunk(
            chunk_id=f"{id_prefix}c{index:02d}" if id_prefix else f"c{index:02d}",
            kind=kind,
            text=chunk_text,
            subsection_heading=subsection,
            hierarchy_path=hierarchy_path,
            order=index,
        )
        for index, (kind, chunk_text) in enumerate(split_candidates)
    ]


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
