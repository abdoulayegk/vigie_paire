"""Alignement local TF-IDF des chunks d'une même sous-section."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vigilance.text_analysis.chunking import TextChunk


_DEFAULT_TOP_K = 5
_CANDIDATE_EXCERPT_CHARS = 300
_SHORT_CANDIDATE_LIMIT = 2
_AMBIGUOUS_FULL_CANDIDATE_LIMIT = 2
_AMBIGUOUS_SHORT_CANDIDATE_LIMIT = 3
_STRONG_THRESHOLD = 0.85
_WEAK_THRESHOLD = 0.35
_MARGIN_THRESHOLD = 0.10
_PDF_MARKER_RE = re.compile(r"\[(?:pdf|p)\.?\s*\d+(?:\s*[-–]\s*\d+)?\]", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", flags=re.UNICODE)
_STOPWORDS = {
    "a",
    "afin",
    "ainsi",
    "and",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cet",
    "cette",
    "dans",
    "de",
    "des",
    "du",
    "en",
    "et",
    "for",
    "il",
    "in",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "notre",
    "nous",
    "of",
    "on",
    "ou",
    "par",
    "pour",
    "que",
    "qui",
    "sur",
    "the",
    "to",
    "un",
    "une",
}


@dataclass(slots=True)
class ChunkCandidate:
    """Candidat TF-IDF d'un chunk source vers un chunk cible."""

    source_chunk_id: str
    target_chunk_id: str
    score: float
    target_chunk: TextChunk


@dataclass(slots=True)
class ChunkAlignment:
    """Alignement provisoire entre chunks T1/T2 avant validation LLM."""

    alignment_id: str
    alignment_type: str
    chunk_t1: TextChunk | None
    chunk_t2: TextChunk | None
    similarity_score: float
    candidates_t1_for_t2: list[ChunkCandidate]
    candidates_t2_for_t1: list[ChunkCandidate]
    reason: str


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokenize_for_tfidf(text: str) -> list[str]:
    """Tokenise un chunk pour un TF-IDF local simple et déterministe."""
    cleaned = _PDF_MARKER_RE.sub(" ", str(text or "").lower())
    cleaned = _strip_accents(cleaned)
    tokens = []
    for match in _TOKEN_RE.finditer(cleaned):
        token = match.group(0).strip("'_")
        if len(token) <= 2 or token in _STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _clamp_similarity_score(score: float) -> float:
    """Contraint les scores cosine dans l'intervalle public attendu."""
    return max(0.0, min(1.0, float(score)))


def _tfidf_similarity_matrix_from_texts(texts: list[str]) -> list[list[float]]:
    """Calcule une matrice cosine TF-IDF locale avec scikit-learn."""
    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        analyzer=_tokenize_for_tfidf,
        smooth_idf=True,
        use_idf=True,
        sublinear_tf=False,
        norm="l2",
    )
    try:
        vectors = vectorizer.fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return [[0.0 for _ in texts] for _ in texts]
    similarities = cosine_similarity(vectors)
    return [
        [_clamp_similarity_score(score) for score in row]
        for row in similarities.tolist()
    ]


def _tfidf_similarity_matrix(chunks: list[TextChunk]) -> list[list[float]]:
    """Calcule une matrice cosine TF-IDF locale pour des chunks."""
    return _tfidf_similarity_matrix_from_texts([chunk.text for chunk in chunks])


def _candidate_lookup(
    source_chunks: list[TextChunk],
    target_chunks: list[TextChunk],
    similarity_matrix: list[list[float]],
    *,
    source_offset: int,
    target_offset: int,
    top_k: int,
) -> dict[str, list[ChunkCandidate]]:
    lookup: dict[str, list[ChunkCandidate]] = {}
    limit = max(0, int(top_k))
    for source_index, source in enumerate(source_chunks):
        candidates = [
            ChunkCandidate(
                source_chunk_id=source.chunk_id,
                target_chunk_id=target.chunk_id,
                score=similarity_matrix[source_offset + source_index][target_offset + target_index],
                target_chunk=target,
            )
            for target_index, target in enumerate(target_chunks)
        ]
        candidates.sort(key=lambda candidate: (-candidate.score, abs(source.order - candidate.target_chunk.order)))
        lookup[source.chunk_id] = candidates[:limit] if limit else candidates
    return lookup


def _clear_margin(candidates: list[ChunkCandidate]) -> bool:
    if len(candidates) < 2:
        return True
    return candidates[0].score - candidates[1].score >= _MARGIN_THRESHOLD


def _alignment_type_for_score(
    *,
    score: float,
    candidates_t1_for_t2: list[ChunkCandidate],
    candidates_t2_for_t1: list[ChunkCandidate],
) -> str:
    if score >= _STRONG_THRESHOLD:
        return "matched_strong"
    if score >= _WEAK_THRESHOLD and _clear_margin(candidates_t1_for_t2) and _clear_margin(candidates_t2_for_t1):
        return "matched_weak"
    return "ambiguous"


def _align_chunks_tfidf(
    chunks_t1: list[TextChunk],
    chunks_t2: list[TextChunk],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> list[ChunkAlignment]:
    """Aligne provisoirement des chunks T1/T2 par TF-IDF local 1-to-1."""
    all_chunks = [*chunks_t1, *chunks_t2]
    similarity_matrix = _tfidf_similarity_matrix(all_chunks)
    offset_t1 = 0
    offset_t2 = len(chunks_t1)

    candidates_t2_to_t1 = _candidate_lookup(
        chunks_t2,
        chunks_t1,
        similarity_matrix,
        source_offset=offset_t2,
        target_offset=offset_t1,
        top_k=top_k,
    )
    candidates_t1_to_t2 = _candidate_lookup(
        chunks_t1,
        chunks_t2,
        similarity_matrix,
        source_offset=offset_t1,
        target_offset=offset_t2,
        top_k=top_k,
    )

    scored_pairs: list[tuple[float, int, TextChunk, TextChunk]] = []
    for index_t1, chunk_t1 in enumerate(chunks_t1):
        for index_t2, chunk_t2 in enumerate(chunks_t2):
            scored_pairs.append(
                (
                    similarity_matrix[offset_t1 + index_t1][offset_t2 + index_t2],
                    abs(chunk_t1.order - chunk_t2.order),
                    chunk_t1,
                    chunk_t2,
                )
            )
    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2].order, item[3].order))

    used_t1: set[str] = set()
    used_t2: set[str] = set()
    alignments: list[ChunkAlignment] = []
    for score, _distance, chunk_t1, chunk_t2 in scored_pairs:
        if score < _WEAK_THRESHOLD:
            continue
        if chunk_t1.chunk_id in used_t1 or chunk_t2.chunk_id in used_t2:
            continue
        candidates_for_t2 = candidates_t2_to_t1.get(chunk_t2.chunk_id, [])
        candidates_for_t1 = candidates_t1_to_t2.get(chunk_t1.chunk_id, [])
        alignment_type = _alignment_type_for_score(
            score=score,
            candidates_t1_for_t2=candidates_for_t2,
            candidates_t2_for_t1=candidates_for_t1,
        )
        alignments.append(
            ChunkAlignment(
                alignment_id="",
                alignment_type=alignment_type,
                chunk_t1=chunk_t1,
                chunk_t2=chunk_t2,
                similarity_score=score,
                candidates_t1_for_t2=candidates_for_t2,
                candidates_t2_for_t1=candidates_for_t1,
                reason="tfidf_one_to_one",
            )
        )
        used_t1.add(chunk_t1.chunk_id)
        used_t2.add(chunk_t2.chunk_id)

    for chunk_t2 in chunks_t2:
        if chunk_t2.chunk_id in used_t2:
            continue
        alignments.append(
            ChunkAlignment(
                alignment_id="",
                alignment_type="possible_added",
                chunk_t1=None,
                chunk_t2=chunk_t2,
                similarity_score=0.0,
                candidates_t1_for_t2=candidates_t2_to_t1.get(chunk_t2.chunk_id, []),
                candidates_t2_for_t1=[],
                reason="unmatched_t2",
            )
        )

    for chunk_t1 in chunks_t1:
        if chunk_t1.chunk_id in used_t1:
            continue
        alignments.append(
            ChunkAlignment(
                alignment_id="",
                alignment_type="possible_removed",
                chunk_t1=chunk_t1,
                chunk_t2=None,
                similarity_score=0.0,
                candidates_t1_for_t2=[],
                candidates_t2_for_t1=candidates_t1_to_t2.get(chunk_t1.chunk_id, []),
                reason="unmatched_t1",
            )
        )

    alignments.sort(
        key=lambda alignment: (
            alignment.chunk_t2.order if alignment.chunk_t2 else 10_000 + (alignment.chunk_t1.order if alignment.chunk_t1 else 0),
            alignment.chunk_t1.order if alignment.chunk_t1 else 10_000,
        )
    )
    for index, alignment in enumerate(alignments):
        alignment.alignment_id = f"a{index:02d}"
    return alignments


def _candidate_text(candidate: ChunkCandidate, *, full: bool) -> str:
    text = re.sub(r"\s+", " ", candidate.target_chunk.text).strip()
    if full or len(text) <= _CANDIDATE_EXCERPT_CHARS:
        return candidate.target_chunk.text
    return f"{text[:_CANDIDATE_EXCERPT_CHARS].rstrip()}..."


def _format_candidate_lines(
    candidates: list[ChunkCandidate],
    *,
    label: str,
    full_limit: int = 0,
    short_limit: int = _SHORT_CANDIDATE_LIMIT,
) -> str:
    """Formate un nombre limité de candidats TF-IDF pour validation LLM."""
    if not candidates:
        return f"{label}: aucun candidat local."
    lines = [f"{label}:"]
    limit = max(0, full_limit) + max(0, short_limit)
    for index, candidate in enumerate(candidates[:limit]):
        is_full = index < full_limit
        path = f" | {candidate.target_chunk.hierarchy_path}" if candidate.target_chunk.hierarchy_path else ""
        mode = "complet" if is_full else f"extrait { _CANDIDATE_EXCERPT_CHARS } caractères"
        lines.append(
            "\n".join(
                [
                    f"- {candidate.target_chunk_id} score={candidate.score:.2f} ({mode})",
                    f"[{candidate.target_chunk.chunk_id} | {candidate.target_chunk.kind}{path}]",
                    _candidate_text(candidate, full=is_full),
                ]
            )
        )
    return "\n".join(lines)


def _format_chunk_for_prompt(chunk: TextChunk | None, *, empty_label: str) -> str:
    if chunk is None:
        return empty_label
    path = f" | {chunk.hierarchy_path}" if chunk.hierarchy_path else ""
    return f"[{chunk.chunk_id} | {chunk.kind}{path}]\n{chunk.text}"


def _format_alignments_for_prompt(alignments: list[ChunkAlignment]) -> tuple[str, str]:
    """Formate les alignements en deux vues T1/T2 pour le prompt existant."""
    blocks_t1: list[str] = []
    blocks_t2: list[str] = []
    for alignment in alignments:
        score = f"{alignment.similarity_score:.2f}" if alignment.similarity_score else "n/a"
        header = (
            f"[{alignment.alignment_id} | {alignment.alignment_type} | score={score} | "
            f"reason={alignment.reason}]"
        )
        if alignment.alignment_type == "ambiguous":
            candidate_kwargs = {
                "full_limit": _AMBIGUOUS_FULL_CANDIDATE_LIMIT,
                "short_limit": _AMBIGUOUS_SHORT_CANDIDATE_LIMIT,
            }
        else:
            candidate_kwargs = {"full_limit": 0, "short_limit": _SHORT_CANDIDATE_LIMIT}
        if alignment.chunk_t1 is None:
            t1_body = "\n".join(
                [
                    "Aucun match T1 retenu en 1-to-1 local.",
                    _format_candidate_lines(
                        alignment.candidates_t1_for_t2,
                        label="Meilleurs candidats T1 à vérifier",
                        **candidate_kwargs,
                    ),
                ]
            )
        else:
            t1_body = _format_chunk_for_prompt(alignment.chunk_t1, empty_label="")
            if alignment.alignment_type in {"matched_weak", "ambiguous", "possible_removed"}:
                t1_body = "\n".join(
                    [
                        t1_body,
                        _format_candidate_lines(
                            alignment.candidates_t2_for_t1,
                            label="Meilleurs candidats T2 à vérifier",
                            **candidate_kwargs,
                        ),
                    ]
                )

        if alignment.chunk_t2 is None:
            t2_body = "\n".join(
                [
                    "Aucun match T2 retenu en 1-to-1 local.",
                    _format_candidate_lines(
                        alignment.candidates_t2_for_t1,
                        label="Meilleurs candidats T2 à vérifier",
                        **candidate_kwargs,
                    ),
                ]
            )
        else:
            t2_body = _format_chunk_for_prompt(alignment.chunk_t2, empty_label="")
            if alignment.alignment_type in {"matched_weak", "ambiguous", "possible_added"}:
                t2_body = "\n".join(
                    [
                        t2_body,
                        _format_candidate_lines(
                            alignment.candidates_t1_for_t2,
                            label="Meilleurs candidats T1 à vérifier",
                            **candidate_kwargs,
                        ),
                    ]
                )

        blocks_t1.append(f"{header}\n{t1_body}")
        blocks_t2.append(f"{header}\n{t2_body}")
    return "\n\n".join(blocks_t1), "\n\n".join(blocks_t2)
