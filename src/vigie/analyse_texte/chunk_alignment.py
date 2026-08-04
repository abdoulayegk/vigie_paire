"""Alignement local hybride TF-IDF + embeddings des chunks d'une sous-section."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vigie.analyse_texte.atomic_alignment import (
    atomic_marker_match_priority,
    atomic_roles_compatible,
    atomic_similarity_text,
)
from vigie.analyse_texte.chunking import TextChunk
from vigie.analyse_texte.openai_client import _embed_texts


logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_CANDIDATE_EXCERPT_CHARS = 300
_SHORT_CANDIDATE_LIMIT = 2
_AMBIGUOUS_FULL_CANDIDATE_LIMIT = 2
_AMBIGUOUS_SHORT_CANDIDATE_LIMIT = 3
_STRONG_THRESHOLD = 0.85
_WEAK_THRESHOLD = 0.35
_MARGIN_THRESHOLD = 0.10
_EMBEDDING_STRONG_THRESHOLD = 0.85
_EMBEDDING_WEAK_THRESHOLD = 0.55
_EMBEDDING_VERY_STRONG_THRESHOLD = 0.92
_HYBRID_MARGIN_THRESHOLD = 0.08
_CHUNK_EMBEDDING_TRUNCATE_CHARS = 4000
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# A long paragraph can cross the chunking threshold in only one version after
# a small insertion/removal.  We only reassemble adjacent chunks when the
# complete passages are near-verbatim; this deliberately rejects similar
# boilerplate describing distinct transactions.
_ADJACENT_GROUP_SEQUENCE_THRESHOLD = 0.82
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
    """Candidat lexical et/ou embedding d'un chunk source vers un chunk cible."""

    source_chunk_id: str
    target_chunk_id: str
    score: float
    target_chunk: TextChunk
    tfidf_score: float = 0.0
    embedding_score: float = 0.0
    marker_match: bool = False


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
    tfidf_score: float = 0.0
    embedding_score: float = 0.0


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
    return _tfidf_similarity_matrix_from_texts(
        [atomic_similarity_text(chunk) for chunk in chunks]
    )


def _candidate_lookup(
    source_chunks: list[TextChunk],
    target_chunks: list[TextChunk],
    similarity_matrix: list[list[float]],
    *,
    source_offset: int,
    target_offset: int,
    top_k: int,
    score_kind: str = "tfidf",
) -> dict[str, list[ChunkCandidate]]:
    lookup: dict[str, list[ChunkCandidate]] = {}
    limit = max(0, int(top_k))
    for source_index, source in enumerate(source_chunks):
        candidates = []
        for target_index, target in enumerate(target_chunks):
            if not atomic_roles_compatible(source, target):
                continue
            score = similarity_matrix[source_offset + source_index][target_offset + target_index]
            candidates.append(
                ChunkCandidate(
                    source_chunk_id=source.chunk_id,
                    target_chunk_id=target.chunk_id,
                    score=score,
                    target_chunk=target,
                    tfidf_score=score if score_kind == "tfidf" else 0.0,
                    embedding_score=score if score_kind == "embedding" else 0.0,
                    marker_match=atomic_marker_match_priority(source, target),
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -int(candidate.marker_match),
                abs(source.order - candidate.target_chunk.order),
            )
        )
        lookup[source.chunk_id] = candidates[:limit] if limit else candidates
    return lookup


def _cosine_similarity_vectors(left: list[float], right: list[float]) -> float:
    """Cosine similarity entre deux vecteurs d'embedding."""
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return _clamp_similarity_score(dot / (left_norm * right_norm))


def _truncate_for_embedding(text: str, *, limit: int = _CHUNK_EMBEDDING_TRUNCATE_CHARS) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit]


def _embedding_similarity_matrix(
    chunks: list[TextChunk],
    *,
    client: Any,
    embedding_model: str,
) -> list[list[float]]:
    """Calcule une matrice cosine d'embeddings pour des chunks locaux."""
    if not chunks:
        return []
    embeddings = _embed_texts(
        client,
        [
            _truncate_for_embedding(atomic_similarity_text(chunk))
            for chunk in chunks
        ],
        model=embedding_model,
    )
    matrix: list[list[float]] = []
    for left in embeddings:
        matrix.append([_cosine_similarity_vectors(left, right) for right in embeddings])
    return matrix


def _merge_candidate_lookups(
    *lookups: dict[str, list[ChunkCandidate]],
    top_k: int,
) -> dict[str, list[ChunkCandidate]]:
    """Union des shortlists TF-IDF et embeddings, score = max des signaux."""
    merged: dict[str, list[ChunkCandidate]] = {}
    source_ids = set()
    for lookup in lookups:
        source_ids.update(lookup)
    for source_id in source_ids:
        by_target: dict[str, ChunkCandidate] = {}
        for lookup in lookups:
            for candidate in lookup.get(source_id, []):
                existing = by_target.get(candidate.target_chunk_id)
                if existing is None:
                    by_target[candidate.target_chunk_id] = ChunkCandidate(
                        source_chunk_id=candidate.source_chunk_id,
                        target_chunk_id=candidate.target_chunk_id,
                        score=max(candidate.tfidf_score, candidate.embedding_score, candidate.score),
                        target_chunk=candidate.target_chunk,
                        tfidf_score=candidate.tfidf_score,
                        embedding_score=candidate.embedding_score,
                        marker_match=candidate.marker_match,
                    )
                    continue
                tfidf_score = max(existing.tfidf_score, candidate.tfidf_score)
                embedding_score = max(existing.embedding_score, candidate.embedding_score)
                by_target[candidate.target_chunk_id] = ChunkCandidate(
                    source_chunk_id=existing.source_chunk_id,
                    target_chunk_id=existing.target_chunk_id,
                    score=max(tfidf_score, embedding_score),
                    target_chunk=existing.target_chunk,
                    tfidf_score=tfidf_score,
                    embedding_score=embedding_score,
                    marker_match=existing.marker_match or candidate.marker_match,
                )
        ranked = sorted(
            by_target.values(),
            key=lambda candidate: (
                -candidate.score,
                -candidate.embedding_score,
                -candidate.tfidf_score,
                -int(candidate.marker_match),
                abs(candidate.target_chunk.order),
            ),
        )
        merged[source_id] = ranked[: max(0, int(top_k))] if top_k else ranked
    return merged


def _pair_score_from_matrices(
    *,
    index_t1: int,
    index_t2: int,
    offset_t1: int,
    offset_t2: int,
    tfidf_matrix: list[list[float]],
    embedding_matrix: list[list[float]] | None,
) -> tuple[float, float, float]:
    tfidf_score = tfidf_matrix[offset_t1 + index_t1][offset_t2 + index_t2]
    embedding_score = 0.0
    if embedding_matrix is not None:
        embedding_score = embedding_matrix[offset_t1 + index_t1][offset_t2 + index_t2]
    return max(tfidf_score, embedding_score), tfidf_score, embedding_score


def _signals_agree_on_best(
    *,
    chunk_t1: TextChunk,
    chunk_t2: TextChunk,
    candidates_t1_to_t2: dict[str, list[ChunkCandidate]],
    candidates_t2_to_t1: dict[str, list[ChunkCandidate]],
) -> bool:
    """True when TF-IDF and embedding shortlists share the same top target."""
    forward = candidates_t1_to_t2.get(chunk_t1.chunk_id, [])
    backward = candidates_t2_to_t1.get(chunk_t2.chunk_id, [])
    if not forward or not backward:
        return False
    best_forward = forward[0]
    best_backward = backward[0]
    if best_forward.target_chunk_id != chunk_t2.chunk_id:
        return False
    if best_backward.target_chunk_id != chunk_t1.chunk_id:
        return False
    # Agreement is strong when both lexical and embedding scores support the pair,
    # or when one signal is decisive enough on its own.
    if best_forward.tfidf_score >= _STRONG_THRESHOLD and best_forward.embedding_score >= _EMBEDDING_STRONG_THRESHOLD:
        return True
    if best_forward.embedding_score >= _EMBEDDING_VERY_STRONG_THRESHOLD:
        return True
    if best_forward.tfidf_score >= _STRONG_THRESHOLD and best_forward.embedding_score <= 0.0:
        return True
    if (
        best_forward.tfidf_score >= _WEAK_THRESHOLD
        and best_forward.embedding_score >= _EMBEDDING_STRONG_THRESHOLD
    ):
        return True
    return best_forward.tfidf_score >= _STRONG_THRESHOLD or best_forward.embedding_score >= _EMBEDDING_STRONG_THRESHOLD


def _numeric_fact_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"\b\S*\d\S*\b", str(text or ""))}


def _hybrid_alignment_type(
    *,
    hybrid_score: float,
    tfidf_score: float,
    embedding_score: float,
    candidates_t1_for_t2: list[ChunkCandidate],
    candidates_t2_for_t1: list[ChunkCandidate],
    signals_agree: bool,
    chunk_t1: TextChunk,
    chunk_t2: TextChunk,
) -> str:
    clear_margin = _clear_margin(candidates_t1_for_t2) and _clear_margin(candidates_t2_for_t1)
    strong_embedding = embedding_score >= _EMBEDDING_STRONG_THRESHOLD
    strong_tfidf = tfidf_score >= _STRONG_THRESHOLD
    # Boilerplate proche avec faits chiffrés distincts: laisser GPT trancher.
    if _numeric_fact_tokens(atomic_similarity_text(chunk_t1)) != _numeric_fact_tokens(
        atomic_similarity_text(chunk_t2)
    ):
        if hybrid_score >= _WEAK_THRESHOLD:
            return "ambiguous"
    if (
        hybrid_score >= max(_WEAK_THRESHOLD, _EMBEDDING_WEAK_THRESHOLD)
        and clear_margin
        and signals_agree
        and (strong_embedding or strong_tfidf)
    ):
        return "matched_strong"
    if hybrid_score >= _WEAK_THRESHOLD and clear_margin and (strong_embedding or strong_tfidf or signals_agree):
        return "matched_weak"
    if hybrid_score >= _WEAK_THRESHOLD:
        return "ambiguous"
    return "ambiguous"


def _clear_margin(candidates: list[ChunkCandidate]) -> bool:
    if len(candidates) < 2:
        return True
    margin = candidates[0].score - candidates[1].score
    # Embedding-aware pairs can be close; keep a slightly softer hybrid margin.
    threshold = _HYBRID_MARGIN_THRESHOLD if candidates[0].embedding_score > 0 else _MARGIN_THRESHOLD
    return margin >= threshold


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


def _group_adjacent_chunks(chunks: list[TextChunk]) -> TextChunk:
    """Creates one synthetic chunk from contiguous source chunks.

    The synthetic identifier remains traceable in the artifact (for example
    ``c01+c02``), while its text is exactly the source text presented to the
    downstream comparison and highlight stages.
    """
    if not chunks:
        raise ValueError("At least one chunk is required to build a group")
    ordered = sorted(chunks, key=lambda chunk: chunk.order)
    first = ordered[0]
    return TextChunk(
        chunk_id="+".join(chunk.chunk_id for chunk in ordered),
        kind=first.kind,
        text=" ".join(chunk.text.strip() for chunk in ordered if chunk.text.strip()),
        subsection_heading=first.subsection_heading,
        hierarchy_path=first.hierarchy_path,
        order=first.order,
        comparison_text=" ".join(
            atomic_similarity_text(chunk)
            for chunk in ordered
            if atomic_similarity_text(chunk)
        ),
        unit_role="grouped",
        parent_chunk_id=(
            first.parent_chunk_id
            if all(chunk.parent_chunk_id == first.parent_chunk_id for chunk in ordered)
            else None
        ),
        parent_context=first.parent_context,
    )


def _sequence_similarity(text_t1: str, text_t2: str) -> float:
    """Returns a conservative verbatim similarity for adjacent-group rescue."""
    normalized_t1 = re.sub(r"\s+", " ", str(text_t1 or "")).strip()
    normalized_t2 = re.sub(r"\s+", " ", str(text_t2 or "")).strip()
    if not normalized_t1 or not normalized_t2:
        return 0.0
    return SequenceMatcher(None, normalized_t1, normalized_t2, autojunk=False).ratio()


def _reassemble_adjacent_one_to_many(alignments: list[ChunkAlignment]) -> list[ChunkAlignment]:
    """Rescues a split/unsplit paragraph before it becomes two false changes.

    The first TF-IDF pass intentionally remains one-to-one.  This post-pass
    examines only an unmatched chunk directly beside a retained pair and
    replaces the two records with one synthetic pair when the reassembled
    passage is almost verbatim.  It addresses threshold-driven splits without
    collapsing distinct but templated disclosures (for example separate debt
    issuances) into a single change.

    Adjacency is scoped to the same subsection heading: ``order`` is local to
    each subsection, so pooling orphans section-wide (Phase B rescue) must not
    treat unrelated chunks with coincidentally consecutive orders as neighbors.
    A proposal is kept only when it improves on the current pair's verbatim
    similarity, so a near-perfect 1:1 match is never diluted by a weaker group.
    """
    proposals: list[tuple[float, ChunkAlignment, ChunkAlignment, TextChunk | None, TextChunk | None]] = []
    paired = [alignment for alignment in alignments if alignment.chunk_t1 and alignment.chunk_t2]

    for matched in paired:
        assert matched.chunk_t1 is not None and matched.chunk_t2 is not None
        matched_verbatim = _sequence_similarity(matched.chunk_t1.text, matched.chunk_t2.text)
        for unmatched in alignments:
            if unmatched.alignment_type == "possible_removed" and unmatched.chunk_t1:
                if unmatched.chunk_t1.subsection_heading != matched.chunk_t1.subsection_heading:
                    continue
                if abs(unmatched.chunk_t1.order - matched.chunk_t1.order) != 1:
                    continue
                grouped_t1 = _group_adjacent_chunks([unmatched.chunk_t1, matched.chunk_t1])
                score = _sequence_similarity(grouped_t1.text, matched.chunk_t2.text)
                if score >= _ADJACENT_GROUP_SEQUENCE_THRESHOLD and score > matched_verbatim:
                    proposals.append((score, unmatched, matched, grouped_t1, matched.chunk_t2))
            elif unmatched.alignment_type == "possible_added" and unmatched.chunk_t2:
                if unmatched.chunk_t2.subsection_heading != matched.chunk_t2.subsection_heading:
                    continue
                if abs(unmatched.chunk_t2.order - matched.chunk_t2.order) != 1:
                    continue
                grouped_t2 = _group_adjacent_chunks([matched.chunk_t2, unmatched.chunk_t2])
                score = _sequence_similarity(matched.chunk_t1.text, grouped_t2.text)
                if score >= _ADJACENT_GROUP_SEQUENCE_THRESHOLD and score > matched_verbatim:
                    proposals.append((score, unmatched, matched, matched.chunk_t1, grouped_t2))

    # A chunk can be adjacent to two possible pairs.  Keep the strongest
    # monotonic proposal only; all other records remain available normally.
    proposals.sort(
        key=lambda proposal: (
            -proposal[0],
            proposal[3].order if proposal[3] else 10_000,
            proposal[4].order if proposal[4] else 10_000,
        )
    )
    consumed: set[int] = set()
    replacements: list[ChunkAlignment] = []
    for score, unmatched, matched, chunk_t1, chunk_t2 in proposals:
        unmatched_key = id(unmatched)
        matched_key = id(matched)
        if unmatched_key in consumed or matched_key in consumed:
            continue
        consumed.update({unmatched_key, matched_key})
        replacements.append(
            ChunkAlignment(
                alignment_id="",
                alignment_type="matched_grouped",
                chunk_t1=chunk_t1,
                chunk_t2=chunk_t2,
                similarity_score=score,
                candidates_t1_for_t2=matched.candidates_t1_for_t2,
                candidates_t2_for_t1=matched.candidates_t2_for_t1,
                reason="adjacent_many_to_one_reassembled",
            )
        )

    if not replacements:
        return alignments
    return [
        *[alignment for alignment in alignments if id(alignment) not in consumed],
        *replacements,
    ]


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

    scored_pairs: list[tuple[float, bool, int, TextChunk, TextChunk]] = []
    for index_t1, chunk_t1 in enumerate(chunks_t1):
        for index_t2, chunk_t2 in enumerate(chunks_t2):
            if not atomic_roles_compatible(chunk_t1, chunk_t2):
                continue
            scored_pairs.append(
                (
                    similarity_matrix[offset_t1 + index_t1][offset_t2 + index_t2],
                    atomic_marker_match_priority(chunk_t1, chunk_t2),
                    abs(chunk_t1.order - chunk_t2.order),
                    chunk_t1,
                    chunk_t2,
                )
            )
    scored_pairs.sort(
        key=lambda item: (
            -item[0],
            -int(item[1]),
            item[2],
            item[3].order,
            item[4].order,
        )
    )

    used_t1: set[str] = set()
    used_t2: set[str] = set()
    alignments: list[ChunkAlignment] = []
    for score, _marker_match, _distance, chunk_t1, chunk_t2 in scored_pairs:
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
                tfidf_score=score,
                embedding_score=0.0,
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

    alignments = _reassemble_adjacent_one_to_many(alignments)
    alignments.sort(
        key=lambda alignment: (
            alignment.chunk_t2.order if alignment.chunk_t2 else 10_000 + (alignment.chunk_t1.order if alignment.chunk_t1 else 0),
            alignment.chunk_t1.order if alignment.chunk_t1 else 10_000,
        )
    )
    for index, alignment in enumerate(alignments):
        alignment.alignment_id = f"a{index:02d}"
    return alignments


def _align_chunks_hybrid(
    chunks_t1: list[TextChunk],
    chunks_t2: list[TextChunk],
    *,
    client: Any | None = None,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    top_k: int = _DEFAULT_TOP_K,
) -> list[ChunkAlignment]:
    """Aligne des chunks T1/T2 par union TF-IDF + embeddings, 1-to-1 réciproque.

    Sans client embeddings, retombe sur l'alignement TF-IDF local existant.
    """
    if client is None:
        return _align_chunks_tfidf(chunks_t1, chunks_t2, top_k=top_k)

    all_chunks = [*chunks_t1, *chunks_t2]
    tfidf_matrix = _tfidf_similarity_matrix(all_chunks)
    offset_t1 = 0
    offset_t2 = len(chunks_t1)
    try:
        embedding_matrix = _embedding_similarity_matrix(
            all_chunks,
            client=client,
            embedding_model=embedding_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embeddings chunks indisponibles, repli TF-IDF: %s", exc)
        return _align_chunks_tfidf(chunks_t1, chunks_t2, top_k=top_k)

    tfidf_t2_to_t1 = _candidate_lookup(
        chunks_t2,
        chunks_t1,
        tfidf_matrix,
        source_offset=offset_t2,
        target_offset=offset_t1,
        top_k=top_k,
        score_kind="tfidf",
    )
    tfidf_t1_to_t2 = _candidate_lookup(
        chunks_t1,
        chunks_t2,
        tfidf_matrix,
        source_offset=offset_t1,
        target_offset=offset_t2,
        top_k=top_k,
        score_kind="tfidf",
    )
    embedding_t2_to_t1 = _candidate_lookup(
        chunks_t2,
        chunks_t1,
        embedding_matrix,
        source_offset=offset_t2,
        target_offset=offset_t1,
        top_k=top_k,
        score_kind="embedding",
    )
    embedding_t1_to_t2 = _candidate_lookup(
        chunks_t1,
        chunks_t2,
        embedding_matrix,
        source_offset=offset_t1,
        target_offset=offset_t2,
        top_k=top_k,
        score_kind="embedding",
    )
    candidates_t2_to_t1 = _merge_candidate_lookups(tfidf_t2_to_t1, embedding_t2_to_t1, top_k=top_k)
    candidates_t1_to_t2 = _merge_candidate_lookups(tfidf_t1_to_t2, embedding_t1_to_t2, top_k=top_k)

    scored_pairs: list[
        tuple[float, float, float, bool, int, TextChunk, TextChunk]
    ] = []
    for index_t1, chunk_t1 in enumerate(chunks_t1):
        for index_t2, chunk_t2 in enumerate(chunks_t2):
            if not atomic_roles_compatible(chunk_t1, chunk_t2):
                continue
            hybrid_score, tfidf_score, embedding_score = _pair_score_from_matrices(
                index_t1=index_t1,
                index_t2=index_t2,
                offset_t1=offset_t1,
                offset_t2=offset_t2,
                tfidf_matrix=tfidf_matrix,
                embedding_matrix=embedding_matrix,
            )
            scored_pairs.append(
                (
                    hybrid_score,
                    tfidf_score,
                    embedding_score,
                    atomic_marker_match_priority(chunk_t1, chunk_t2),
                    abs(chunk_t1.order - chunk_t2.order),
                    chunk_t1,
                    chunk_t2,
                )
            )
    scored_pairs.sort(
        key=lambda item: (
            -item[0],
            -item[2],
            -item[1],
            -int(item[3]),
            item[4],
            item[5].order,
            item[6].order,
        )
    )

    used_t1: set[str] = set()
    used_t2: set[str] = set()
    alignments: list[ChunkAlignment] = []
    for (
        hybrid_score,
        tfidf_score,
        embedding_score,
        _marker_match,
        _distance,
        chunk_t1,
        chunk_t2,
    ) in scored_pairs:
        credible = hybrid_score >= _WEAK_THRESHOLD or embedding_score >= _EMBEDDING_WEAK_THRESHOLD
        if not credible:
            continue
        if chunk_t1.chunk_id in used_t1 or chunk_t2.chunk_id in used_t2:
            continue

        candidates_for_t2 = candidates_t2_to_t1.get(chunk_t2.chunk_id, [])
        candidates_for_t1 = candidates_t1_to_t2.get(chunk_t1.chunk_id, [])
        reciprocal = (
            bool(candidates_for_t1)
            and bool(candidates_for_t2)
            and candidates_for_t1[0].target_chunk_id == chunk_t2.chunk_id
            and candidates_for_t2[0].target_chunk_id == chunk_t1.chunk_id
        )
        # Priorité haute: seulement les paires réciproques consomment un slot 1→1.
        # Une paire non réciproque reste disponible pour un meilleur match plus bas
        # dans le classement, puis finira en added/removed avec ses candidats.
        if not reciprocal:
            continue

        signals_agree = _signals_agree_on_best(
            chunk_t1=chunk_t1,
            chunk_t2=chunk_t2,
            candidates_t1_to_t2=candidates_t1_to_t2,
            candidates_t2_to_t1=candidates_t2_to_t1,
        )
        alignment_type = _hybrid_alignment_type(
            hybrid_score=hybrid_score,
            tfidf_score=tfidf_score,
            embedding_score=embedding_score,
            candidates_t1_for_t2=candidates_for_t2,
            candidates_t2_for_t1=candidates_for_t1,
            signals_agree=signals_agree,
            chunk_t1=chunk_t1,
            chunk_t2=chunk_t2,
        )
        reason = "hybrid_tfidf_embedding"

        alignments.append(
            ChunkAlignment(
                alignment_id="",
                alignment_type=alignment_type,
                chunk_t1=chunk_t1,
                chunk_t2=chunk_t2,
                similarity_score=hybrid_score,
                candidates_t1_for_t2=candidates_for_t2,
                candidates_t2_for_t1=candidates_for_t1,
                reason=reason,
                tfidf_score=tfidf_score,
                embedding_score=embedding_score,
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

    alignments = _reassemble_adjacent_one_to_many(alignments)
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
                    (
                        f"- {candidate.target_chunk_id} score={candidate.score:.2f} "
                        f"(tfidf={candidate.tfidf_score:.2f}, "
                        f"embedding={candidate.embedding_score:.2f}, {mode})"
                    ),
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
        tfidf = f"{alignment.tfidf_score:.2f}"
        embedding = f"{alignment.embedding_score:.2f}"
        header = (
            f"[{alignment.alignment_id} | {alignment.alignment_type} | score={score} | "
            f"tfidf={tfidf} | embedding={embedding} | reason={alignment.reason}]"
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
