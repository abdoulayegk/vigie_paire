"""Chunking déterministe des corps de sous-sections markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass


_DEFAULT_MIN_CHARS = 120
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•‰]\s+|\d{1,3}[.)]\s+)")
_HEADING_LINE_RE = re.compile(r"^\s*#{2,6}\s+")


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
    return bool(_BULLET_LINE_RE.match(str(line or "")))


def _is_heading_line(line: str) -> bool:
    """Indique si une ligne est un titre markdown à exclure du chunking."""
    return bool(_HEADING_LINE_RE.match(str(line or "")))


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


def _merge_short_candidates(
    candidates: list[tuple[str, str]],
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[tuple[str, str]]:
    """Fusionne les blocs trop courts avec le précédent, sinon avec le suivant."""
    merged: list[tuple[str, str]] = []
    pending = list(candidates)
    min_chars = max(0, int(min_chars))

    for index, (kind, text) in enumerate(pending):
        if min_chars and len(text.strip()) < min_chars:
            if merged:
                previous_kind, previous_text = merged[-1]
                merged[-1] = (previous_kind, f"{previous_text}\n\n{text}".strip())
                continue
            if index + 1 < len(pending):
                next_kind, next_text = pending[index + 1]
                pending[index + 1] = (next_kind, f"{text}\n\n{next_text}".strip())
                continue
        merged.append((kind, text.strip()))

    return [(kind, text) for kind, text in merged if text]


def _chunk_subsection_text(
    text: str,
    *,
    subsection_heading: str = "",
    section_title: str = "",
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[TextChunk]:
    """Découpe un corps de sous-section en chunks stables paragraphes/listes."""
    candidates = _split_candidate_blocks(text)
    candidates = _group_adjacent_lists(candidates)
    candidates = _merge_short_candidates(candidates, min_chars=min_chars)

    subsection = str(subsection_heading or "").strip()
    section = str(section_title or "").strip()
    hierarchy_path = f"{section} > {subsection}" if section and subsection else section or subsection

    return [
        TextChunk(
            chunk_id=f"c{index:02d}",
            kind=kind,
            text=chunk_text,
            subsection_heading=subsection,
            hierarchy_path=hierarchy_path,
            order=index,
        )
        for index, (kind, chunk_text) in enumerate(candidates)
    ]


def _format_chunks_for_prompt(chunks: list[TextChunk]) -> str:
    """Formate les chunks pour le prompt GPT sans modifier leur texte source."""
    blocks: list[str] = []
    for chunk in chunks:
        path = f" | {chunk.hierarchy_path}" if chunk.hierarchy_path else ""
        blocks.append(f"[{chunk.chunk_id} | {chunk.kind}{path}]\n{chunk.text}")
    return "\n\n".join(blocks)
