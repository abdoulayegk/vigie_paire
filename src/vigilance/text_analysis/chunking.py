"""Chunking déterministe des corps de sous-sections markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BULLET_LINE_RE = re.compile(r"^\s*(?:\[\s*(?:x|X)?\s*\]\s*|[-*•‰]\s+|\d{1,3}[.)]\s+)")
_HEADING_LINE_RE = re.compile(r"^\s*#{2,6}\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])")
_SEMANTIC_TRANSITION_RE = re.compile(
    r"^(?:En plus|Toutefois|Par ailleurs|De plus|En outre|Néanmoins|Cependant|"
    r"Le cadre couvre également|Cette approche|À cet égard|Dans ce contexte)\b",
    flags=re.IGNORECASE,
)
_LONG_PARAGRAPH_TRIGGER_WORDS = 300
_TARGET_CHUNK_WORDS = 220
_HARD_MAX_CHUNK_WORDS = 300
_NON_NARRATIVE_EXACT_RE = re.compile(
    r"^(?:s\.?\s*o\.?(?:\s+sans\s+objet)?|sans\s+objet)$",
    flags=re.IGNORECASE,
)
_TABLE_REFERENCE_RE = re.compile(r"^le\s+tableau\s+ci[-\s]dessus\b", flags=re.IGNORECASE)
_EXCLUDED_NARRATIVE_SYMBOLS = frozenset("[]$%")
_LEADING_LIST_MARKER_RE = re.compile(
    r"^\s*(?:\[\s*(?:x|X)?\s*\]\s*|[-*•‰]\s+|\d{1,3}[.)]\s+)",
)


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


def _strip_list_markers(text: str) -> str:
    """Retire les puces de présentation sans séparer les éléments de liste."""
    return "\n".join(
        _LEADING_LIST_MARKER_RE.sub("", line).strip()
        for line in str(text or "").splitlines()
        if _LEADING_LIST_MARKER_RE.sub("", line).strip()
    )


def _is_narrative_comparison_candidate(text: str) -> bool:
    """Écarte les unités qui ne peuvent pas porter un changement sémantique.

    Le pipeline narratif ne compare ni les tableaux, ni leurs cellules isolées,
    ni les renvois à un tableau. Ces exclusions sont des critères de qualité de
    l'entrée; le jugement de pertinence du texte narratif restant demeure confié
    au triage GPT.
    """
    value = str(text or "").strip()
    if not value:
        return False
    value_without_list_marker = _LEADING_LIST_MARKER_RE.sub("", value)
    if _NON_NARRATIVE_EXACT_RE.fullmatch(value_without_list_marker):
        return False
    if _TABLE_REFERENCE_RE.match(value_without_list_marker):
        return False
    if "|" in value or "\t" in value:
        return False
    if any(symbol in value for symbol in _EXCLUDED_NARRATIVE_SYMBOLS):
        return False

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9]*", value)
    if len(words) < 2:
        return False

    # Les libellés de cellules et de colonnes (« Crédit », « Marché »,
    # « Financement spécialisé… ») ne portent normalement aucune ponctuation
    # de phrase. Une phrase courte reste admise si elle est bien terminée.
    if not re.search(r"[.!?;:]\s*$", value) and len(words) < 12:
        return False
    return True


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
    """Compte les mots pour borner uniquement les paragraphes exceptionnellement longs."""
    return len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", str(text or "")))


def _split_long_paragraph(text: str) -> list[str]:
    """Découpe un long paragraphe aux frontières de phrase les plus naturelles.

    La taille n'est qu'un garde-fou : une transition explicite est privilégiée,
    puis une fin de phrase est utilisée seulement pour éviter qu'un paragraphe
    anormalement long devienne une unique unité de comparaison.
    """
    paragraph = str(text or "").strip()
    if _word_count(paragraph) <= _LONG_PARAGRAPH_TRIGGER_WORDS:
        return [paragraph] if paragraph else []

    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph) if sentence.strip()]
    if len(sentences) <= 1:
        return [paragraph]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = _word_count(sentence)
        starts_new_idea = bool(_SEMANTIC_TRANSITION_RE.match(sentence))
        should_split = bool(current) and (
            (current_words >= 100 and starts_new_idea)
            or current_words + sentence_words > _HARD_MAX_CHUNK_WORDS
            or current_words >= _TARGET_CHUNK_WORDS
        )
        if should_split:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _chunk_subsection_text(
    text: str,
    *,
    subsection_heading: str = "",
    section_title: str = "",
    min_chars: int = 0,
) -> list[TextChunk]:
    """Découpe un corps en paragraphes autonomes et listes cohérentes.

    ``min_chars`` est conservé pour compatibilité d'appel, mais les paragraphes
    ne sont plus fusionnés automatiquement : une ligne vide définit un chunk.
    """
    _ = min_chars
    candidates = _split_candidate_blocks(text)
    candidates = _group_adjacent_lists(candidates)
    split_candidates: list[tuple[str, str]] = []
    for kind, candidate_text in candidates:
        # Une liste est une seule unité sémantique : ses items restent groupés.
        # Les marqueurs ``[]``/``[x]`` ne sont que de la mise en forme et ne
        # doivent jamais entraîner l'exclusion de son contenu narratif.
        normalized_candidate = _strip_list_markers(candidate_text) if kind == "list" else candidate_text
        if not _is_narrative_comparison_candidate(normalized_candidate):
            continue
        if kind == "paragraph":
            split_candidates.extend(
                (kind, part)
                for part in _split_long_paragraph(normalized_candidate)
                if _is_narrative_comparison_candidate(part)
            )
        else:
            split_candidates.append((kind, normalized_candidate))

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
        for index, (kind, chunk_text) in enumerate(split_candidates)
    ]


def _format_chunks_for_prompt(chunks: list[TextChunk]) -> str:
    """Formate les chunks pour le prompt GPT sans modifier leur texte source."""
    blocks: list[str] = []
    for chunk in chunks:
        path = f" | {chunk.hierarchy_path}" if chunk.hierarchy_path else ""
        blocks.append(f"[{chunk.chunk_id} | {chunk.kind}{path}]\n{chunk.text}")
    return "\n\n".join(blocks)
