"""Appariement des sous-sections et récupération des éléments orphelins.

Les titres, le contenu et les embeddings servent à présélectionner les paires
entre trimestres; un arbitrage structuré traite les cas ambigus et les
changements synthétiques. La stratégie privilégie les correspondances sûres.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from vigie.analyse_texte.chunk_alignment import _tfidf_similarity_matrix_from_texts
from vigie.analyse_texte.constants import _SUBSECTION_SPLIT_RE
from vigie.analyse_texte.normalization import _sanitize_semantic_text
from vigie.analyse_texte.openai_client import _call_structured_completion_with_correction, _embed_texts

logger = logging.getLogger(__name__)

_ORPHAN_TOP_K = 3
_ORPHAN_GPT_BATCH_SIZE = 5
_ORPHAN_BODY_EXCERPT_CHARS = 500
_ORPHAN_EMBEDDING_TRUNCATE_CHARS = 8000
_ORPHAN_MIN_BODY_CHARS_FOR_BODY_MATCH = 100
_EMBEDDING_STRONG_CANDIDATE_THRESHOLD = 0.82
_DETERMINISTIC_HEADING_STRONG = 0.82
_DETERMINISTIC_HEADING_CONTAIN_MIN = 0.75
_DETERMINISTIC_HEADING_HYBRID_MIN = 0.55
_DETERMINISTIC_EMBEDDING_STRONG = 0.90
_DETERMINISTIC_EMBEDDING_HYBRID = 0.85
_DETERMINISTIC_TFIDF_BODY_STRONG = 0.35
_DETERMINISTIC_TFIDF_HYBRID = 0.50
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_ORPHAN_MATCH_VALIDATION_RETRY_MESSAGE = (
    "Corrige la réponse et renvoie le batch COMPLET en respectant strictement le schéma. "
    "Chaque match doit inclure heading_t1, heading_t2, confidence parmi high|medium|low, "
    "et reason. Retourne les headings exactement comme fournis et ne crée pas de champ supplémentaire."
)
_ORPHAN_MATCH_LENGTH_RETRY_MESSAGE = (
    "La réponse précédente a dépassé la limite de sortie. Renvoie le même batch de "
    "correspondances orphelines, mais avec des champs reason très courts (1 phrase max) "
    "et uniquement les paires high/medium confirmées. Ne répète pas les extraits de texte."
)


class OrphanMatchLLMItem(BaseModel):
    """Match brut validé à la frontière LLM pour les sous-sections orphelines."""

    model_config = ConfigDict(extra="forbid")

    heading_t1: str
    heading_t2: str
    confidence: Literal["high", "medium", "low"]
    reason: str

    @field_validator("heading_t1", "heading_t2", "confidence", "reason", mode="before")
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        return str(value or "").strip()


class OrphanMatchLLMResponse(BaseModel):
    """Réponse structurée du LLM pour l'appariement de sous-sections orphelines."""

    model_config = ConfigDict(extra="forbid")

    matches: list[OrphanMatchLLMItem]


@dataclass(slots=True)
class OrphanSubsection:
    """Sous-section orpheline d'un trimestre."""

    heading: str
    body: str


@dataclass(slots=True)
class OrphanCandidate:
    """Candidat d'appariement entre deux sous-sections orphelines."""

    heading_t1: str
    body_t1: str
    heading_t2: str
    body_t2: str
    tfidf_score: float
    heading_score: float
    embedding_score: float = 0.0


def _normalize_heading(heading: str) -> str:
    """Normalise un heading ### pour le pairing T1/T2 (insensible à la casse et aux préfixes de tableaux)."""
    h = heading.lower()
    h = re.sub(r"\s*\[(?:pdf|p)\.?\s*\d+(?:\s*[-–]\s*\d+)?\]\s*", " ", h, flags=re.IGNORECASE)
    h = re.sub(r"\b[tT]\d{2,3}\b\s*", "", h)  # strip T22, T25, T125, etc.
    h = re.sub(r"[^\w\s]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _heading_similarity(heading_t1: str, heading_t2: str) -> float:
    """Score de similarité entre deux titres normalisés."""
    left = _normalize_heading(heading_t1)
    right = _normalize_heading(heading_t2)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _heading_one_contains_other(heading_t1: str, heading_t2: str) -> bool:
    """Indique si un titre normalisé contient l'autre."""
    left = _normalize_heading(heading_t1)
    right = _normalize_heading(heading_t2)
    if not left or not right:
        return False
    return left in right or right in left


def _classify_deterministic_orphan_tier(candidate: OrphanCandidate) -> str | None:
    """Retourne le tier déterministe (A/B/C) si la paire est auto-confirmable."""
    heading_score = candidate.heading_score
    tfidf_score = candidate.tfidf_score
    embedding_score = candidate.embedding_score

    if heading_score >= _DETERMINISTIC_HEADING_STRONG or (
        _heading_one_contains_other(candidate.heading_t1, candidate.heading_t2)
        and heading_score >= _DETERMINISTIC_HEADING_CONTAIN_MIN
    ):
        return "deterministic_heading"
    if embedding_score >= _DETERMINISTIC_EMBEDDING_STRONG and tfidf_score >= _DETERMINISTIC_TFIDF_BODY_STRONG:
        return "deterministic_embedding"
    if heading_score >= _DETERMINISTIC_HEADING_HYBRID_MIN and (
        tfidf_score >= _DETERMINISTIC_TFIDF_HYBRID or embedding_score >= _DETERMINISTIC_EMBEDDING_HYBRID
    ):
        return "deterministic_hybrid"
    return None


def _deterministic_tier_rank(match_source: str) -> int:
    return {
        "deterministic_heading": 3,
        "deterministic_embedding": 2,
        "deterministic_hybrid": 1,
    }.get(match_source, 0)


def _build_orphan_match_record(
    *,
    candidate: OrphanCandidate,
    match_source: str,
    reason: str,
    llm_confidence: str | None = None,
) -> dict[str, Any]:
    return {
        "heading_t1": candidate.heading_t1,
        "heading_t2": candidate.heading_t2,
        "confidence": llm_confidence or "high",
        "llm_confidence": llm_confidence,
        "reason": reason,
        "match_source": match_source,
        "tfidf_score": round(candidate.tfidf_score, 4),
        "embedding_score": round(candidate.embedding_score, 4),
        "heading_score": round(candidate.heading_score, 4),
    }


def _deterministic_confirm_orphan_matches(
    candidates: list[OrphanCandidate],
    *,
    allowed_t1: set[str] | None = None,
    allowed_t2: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Confirme déterministement les paires orphelines fortes (1→1 greedy)."""
    if not candidates:
        return []

    scored: list[tuple[int, float, float, float, str, OrphanCandidate]] = []
    for candidate in candidates:
        if allowed_t1 is not None and candidate.heading_t1 not in allowed_t1:
            continue
        if allowed_t2 is not None and candidate.heading_t2 not in allowed_t2:
            continue
        tier = _classify_deterministic_orphan_tier(candidate)
        if tier is None:
            continue
        scored.append(
            (
                _deterministic_tier_rank(tier),
                candidate.embedding_score,
                candidate.tfidf_score,
                candidate.heading_score,
                tier,
                candidate,
            )
        )

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[5].heading_t1, item[5].heading_t2))
    used_t1: set[str] = set()
    used_t2: set[str] = set()
    matches: list[dict[str, Any]] = []
    for _rank, _embedding, _tfidf, _heading, tier, candidate in scored:
        if candidate.heading_t1 in used_t1 or candidate.heading_t2 in used_t2:
            continue
        matches.append(
            _build_orphan_match_record(
                candidate=candidate,
                match_source=tier,
                reason=f"deterministic_{tier.removeprefix('deterministic_')}",
            )
        )
        used_t1.add(candidate.heading_t1)
        used_t2.add(candidate.heading_t2)
    return matches


def _deterministic_match_orphan_headings(
    orphans_t1: list[str],
    orphans_t2: list[str],
) -> list[dict[str, Any]]:
    """Apparie déterministement les titres orphelins courts (sans corps substantiel)."""
    if not orphans_t1 or not orphans_t2:
        return []

    scored: list[tuple[float, str, str]] = []
    for heading_t1 in orphans_t1:
        for heading_t2 in orphans_t2:
            heading_score = _heading_similarity(heading_t1, heading_t2)
            if heading_score >= _DETERMINISTIC_HEADING_STRONG or (
                _heading_one_contains_other(heading_t1, heading_t2)
                and heading_score >= _DETERMINISTIC_HEADING_CONTAIN_MIN
            ):
                scored.append((heading_score, heading_t1, heading_t2))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_t1: set[str] = set()
    used_t2: set[str] = set()
    matches: list[dict[str, Any]] = []
    for heading_score, heading_t1, heading_t2 in scored:
        if heading_t1 in used_t1 or heading_t2 in used_t2:
            continue
        matches.append(
            {
                "heading_t1": heading_t1,
                "heading_t2": heading_t2,
                "confidence": "high",
                "llm_confidence": None,
                "reason": "deterministic_heading",
                "match_source": "deterministic_heading",
                "tfidf_score": None,
                "embedding_score": None,
                "heading_score": round(heading_score, 4),
            }
        )
        used_t1.add(heading_t1)
        used_t2.add(heading_t2)
    return matches


def _orphan_body_excerpt(body: str, *, limit: int = _ORPHAN_BODY_EXCERPT_CHARS) -> str:
    """Retourne un extrait début/fin d'un body pour le prompt LLM."""
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 20) // 2)
    return f"début: {text[:half].rstrip()} ... fin: {text[-half:].lstrip()}"


def _has_substantial_body(orphan: OrphanSubsection) -> bool:
    """Indique si le body est assez long pour TF-IDF + embeddings."""
    return len(str(orphan.body or "").strip()) >= _ORPHAN_MIN_BODY_CHARS_FOR_BODY_MATCH


def _truncate_for_embedding(text: str, *, limit: int = _ORPHAN_EMBEDDING_TRUNCATE_CHARS) -> str:
    """Tronque un body avant encodage embedding."""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit]


def _cosine_similarity_vectors(left: list[float], right: list[float]) -> float:
    """Cosine similarity entre deux vecteurs d'embedding."""
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _shortlist_orphan_candidates(
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
    *,
    top_k: int = _ORPHAN_TOP_K,
) -> list[OrphanCandidate]:
    """Shortlist TF-IDF sklearn des paires orphelines possibles (pas de verdict final)."""
    if not orphans_t1 or not orphans_t2:
        return []

    all_bodies = [orphan.body for orphan in orphans_t1] + [orphan.body for orphan in orphans_t2]
    similarity_matrix = _tfidf_similarity_matrix_from_texts(all_bodies)
    offset_t2 = len(orphans_t1)

    candidates: list[OrphanCandidate] = []
    for index_t1, orphan_t1 in enumerate(orphans_t1):
        scored: list[tuple[float, float, OrphanSubsection]] = []
        for index_t2, orphan_t2 in enumerate(orphans_t2):
            tfidf_score = similarity_matrix[index_t1][offset_t2 + index_t2]
            heading_score = _heading_similarity(orphan_t1.heading, orphan_t2.heading)
            scored.append((tfidf_score, heading_score, orphan_t2))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].heading))
        for tfidf_score, heading_score, orphan_t2 in scored[: max(1, top_k)]:
            candidates.append(
                OrphanCandidate(
                    heading_t1=orphan_t1.heading,
                    body_t1=orphan_t1.body,
                    heading_t2=orphan_t2.heading,
                    body_t2=orphan_t2.body,
                    tfidf_score=tfidf_score,
                    heading_score=heading_score,
                )
            )
    return candidates


def _candidate_key(candidate: OrphanCandidate) -> tuple[str, str]:
    return candidate.heading_t1, candidate.heading_t2


def _attach_embedding_scores(
    *,
    client: Any,
    candidates: list[OrphanCandidate],
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
    embedding_model: str,
) -> tuple[list[OrphanCandidate], dict[tuple[str, str], float]]:
    """Calcule les scores embedding pour les paires shortlistees."""
    if not candidates:
        return [], {}

    unique_t1 = {orphan.heading: orphan.body for orphan in orphans_t1}
    unique_t2 = {orphan.heading: orphan.body for orphan in orphans_t2}
    ordered_headings = list(unique_t1) + list(unique_t2)
    ordered_bodies = [_truncate_for_embedding(unique_t1[heading]) for heading in unique_t1] + [
        _truncate_for_embedding(unique_t2[heading]) for heading in unique_t2
    ]
    embeddings = _embed_texts(client, ordered_bodies, model=embedding_model)
    embedding_by_heading = {heading: embeddings[index] for index, heading in enumerate(ordered_headings)}

    pair_scores: dict[tuple[str, str], float] = {}
    enriched: list[OrphanCandidate] = []
    for candidate in candidates:
        embedding_score = _cosine_similarity_vectors(
            embedding_by_heading.get(candidate.heading_t1, []),
            embedding_by_heading.get(candidate.heading_t2, []),
        )
        pair_scores[_candidate_key(candidate)] = embedding_score
        enriched.append(
            OrphanCandidate(
                heading_t1=candidate.heading_t1,
                body_t1=candidate.body_t1,
                heading_t2=candidate.heading_t2,
                body_t2=candidate.body_t2,
                tfidf_score=candidate.tfidf_score,
                heading_score=candidate.heading_score,
                embedding_score=embedding_score,
            )
        )
    return enriched, pair_scores


def _format_orphan_candidate_for_prompt(candidate: OrphanCandidate) -> str:
    strength = (
        "candidate_strong" if candidate.embedding_score >= _EMBEDDING_STRONG_CANDIDATE_THRESHOLD else "candidate_review"
    )
    return (
        f"- T1: {candidate.heading_t1}\n"
        f"  T2: {candidate.heading_t2}\n"
        f"  tfidf_score={candidate.tfidf_score:.2f} "
        f"embedding_score={candidate.embedding_score:.2f} "
        f"heading_score={candidate.heading_score:.2f} "
        f"signal={strength}\n"
        f"  body_chars_T1={len(candidate.body_t1.strip())} "
        f"body_chars_T2={len(candidate.body_t2.strip())}\n"
        f"  extrait_T1: {_orphan_body_excerpt(candidate.body_t1)}\n"
        f"  extrait_T2: {_orphan_body_excerpt(candidate.body_t2)}"
    )


def _gpt_arbitrate_orphan_batch(
    *,
    client: Any,
    model: str,
    section_key: str,
    candidates: list[OrphanCandidate],
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
) -> list[dict[str, Any]]:
    """Arbitre via GPT un lot de paires orphelines ambigues."""
    if not candidates:
        return []

    candidate_lines = "\n\n".join(_format_orphan_candidate_for_prompt(candidate) for candidate in candidates)
    unmatched_t1 = "\n".join(f"- {orphan.heading}" for orphan in orphans_t1) or "- aucun"
    unmatched_t2 = "\n".join(f"- {orphan.heading}" for orphan in orphans_t2) or "- aucun"

    raw = _call_structured_completion_with_correction(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es expert en rapports bancaires réglementaires canadiens. "
                    "Tu identifies les correspondances entre sous-sections orphelines "
                    "d'un trimestre à l'autre en te basant sur le titre ET le contenu."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Format de réponse: {"matches": [{"heading_t1": "...", "heading_t2": "...", '
                    '"confidence": "high|medium|low", "reason": "..."}]}\n'
                    "Règles strictes:\n"
                    "- Correspondances 1→1 uniquement\n"
                    "- N'inclure que confidence high ou medium\n"
                    "- Les scores TF-IDF/embedding sont des signaux, jamais des verdicts\n"
                    "- Même si embedding_score est très élevé, tu dois confirmer par le contenu\n"
                    "- Matcher seulement si c'est la même sous-section renommée/reformulée\n"
                    "- Si le texte a été reformulé mais reste le même sujet substantiel, matcher\n"
                    "- Si c'est un vrai ajout ou retrait, ne pas matcher\n"
                    "- Retourner les headings EXACTEMENT comme fournis\n\n"
                    f"Section: {section_key}\n\n"
                    "Candidats shortlistés (TF-IDF + embeddings):\n"
                    f"{candidate_lines}\n\n"
                    "Orphelins T1 encore sans paire:\n"
                    f"{unmatched_t1}\n\n"
                    "Orphelins T2 encore sans paire:\n"
                    f"{unmatched_t2}"
                ),
            },
        ],
        response_format=OrphanMatchLLMResponse,
        max_retries=1,
        validation_retry_message=_ORPHAN_MATCH_VALIDATION_RETRY_MESSAGE,
        length_retry_message=_ORPHAN_MATCH_LENGTH_RETRY_MESSAGE,
    )

    orphans_t1_set = {orphan.heading for orphan in orphans_t1}
    orphans_t2_set = {orphan.heading for orphan in orphans_t2}
    candidate_lookup = {_candidate_key(candidate): candidate for candidate in candidates}
    valid_matches: list[tuple[int, float, float, float, str, str, str, str, OrphanCandidate]] = []
    for item in raw.matches:
        conf = item.confidence
        heading_t1 = item.heading_t1
        heading_t2 = item.heading_t2
        if conf not in {"high", "medium"}:
            continue
        if heading_t1 not in orphans_t1_set or heading_t2 not in orphans_t2_set:
            continue
        candidate = candidate_lookup.get((heading_t1, heading_t2))
        if candidate is None:
            continue
        confidence_rank = 2 if conf == "high" else 1
        valid_matches.append(
            (
                confidence_rank,
                candidate.embedding_score,
                candidate.tfidf_score,
                candidate.heading_score,
                heading_t1,
                heading_t2,
                conf,
                item.reason or "llm_arbitration",
                candidate,
            )
        )

    valid_matches.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5]))
    used_t1: set[str] = set()
    used_t2: set[str] = set()
    matches: list[dict[str, Any]] = []
    for _rank, _embedding, _tfidf, _heading, heading_t1, heading_t2, conf, reason, candidate in valid_matches:
        if heading_t1 in used_t1 or heading_t2 in used_t2:
            continue
        matches.append(
            _build_orphan_match_record(
                candidate=candidate,
                match_source="llm_embedding_confirmed",
                reason=reason,
                llm_confidence=conf,
            )
        )
        used_t1.add(heading_t1)
        used_t2.add(heading_t2)
    return matches


def _batch_orphan_candidates(
    candidates: list[OrphanCandidate],
    *,
    batch_size: int = _ORPHAN_GPT_BATCH_SIZE,
) -> list[list[OrphanCandidate]]:
    """Découpe les candidats orphelins par groupes de headings T1."""
    if not candidates:
        return []

    by_t1: dict[str, list[OrphanCandidate]] = {}
    for candidate in candidates:
        by_t1.setdefault(candidate.heading_t1, []).append(candidate)

    batches: list[list[OrphanCandidate]] = []
    current_batch: list[OrphanCandidate] = []
    current_t1_count = 0
    for heading_t1 in sorted(by_t1):
        group = by_t1[heading_t1]
        if current_t1_count >= batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_t1_count = 0
        current_batch.extend(group)
        current_t1_count += 1
    if current_batch:
        batches.append(current_batch)
    return batches


def _gpt_arbitrate_orphan_subsections(
    *,
    client: Any,
    model: str,
    section_key: str,
    candidates: list[OrphanCandidate],
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
) -> list[dict[str, Any]]:
    """Arbitre via GPT les paires orphelines ambigues avant added/removed."""
    if not candidates or (not orphans_t1 and not orphans_t2):
        return []

    all_matches: list[dict[str, Any]] = []
    matched_t1: set[str] = set()
    matched_t2: set[str] = set()
    gpt_failed = False

    for batch in _batch_orphan_candidates(candidates):
        remaining_t1 = [orphan for orphan in orphans_t1 if orphan.heading not in matched_t1]
        remaining_t2 = [orphan for orphan in orphans_t2 if orphan.heading not in matched_t2]
        batch_candidates = [
            candidate
            for candidate in batch
            if candidate.heading_t1 not in matched_t1 and candidate.heading_t2 not in matched_t2
        ]
        if not batch_candidates or not remaining_t1 or not remaining_t2:
            continue
        try:
            batch_matches = _gpt_arbitrate_orphan_batch(
                client=client,
                model=model,
                section_key=section_key,
                candidates=batch_candidates,
                orphans_t1=remaining_t1,
                orphans_t2=remaining_t2,
            )
        except Exception as exc:
            gpt_failed = True
            logger.warning(
                "Orphan subsection GPT arbitration failed for %s — skipping batch (%d candidates): %s",
                section_key,
                len(batch_candidates),
                exc,
                exc_info=True,
            )
            continue

        for match in batch_matches:
            heading_t1 = match["heading_t1"]
            heading_t2 = match["heading_t2"]
            if heading_t1 in matched_t1 or heading_t2 in matched_t2:
                continue
            all_matches.append(match)
            matched_t1.add(heading_t1)
            matched_t2.add(heading_t2)

    if gpt_failed and not all_matches:
        logger.warning(
            "Orphan subsection GPT arbitration failed for %s — skipping",
            section_key,
        )
    return all_matches


def _resolve_orphan_subsections(
    *,
    client: Any,
    model: str,
    section_key: str,
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Résout les orphelins via TF-IDF sklearn, embeddings, déterministe puis LLM."""
    # Un titre sans corps narratif n'est pas une sous-section comparable. Cette
    # garde protège aussi les artefacts markdown historiques, générés avant le
    # filtrage des titres de tableaux et parents structurels.
    orphans_t1 = [orphan for orphan in orphans_t1 if str(orphan.body or "").strip()]
    orphans_t2 = [orphan for orphan in orphans_t2 if str(orphan.body or "").strip()]
    if not orphans_t1 or not orphans_t2:
        return []

    body_orphans_t1 = [orphan for orphan in orphans_t1 if _has_substantial_body(orphan)]
    body_orphans_t2 = [orphan for orphan in orphans_t2 if _has_substantial_body(orphan)]
    shortlist = _shortlist_orphan_candidates(body_orphans_t1, body_orphans_t2)
    embedding_failed = False
    enriched_candidates = shortlist
    if shortlist:
        try:
            enriched_candidates, _ = _attach_embedding_scores(
                client=client,
                candidates=shortlist,
                orphans_t1=body_orphans_t1,
                orphans_t2=body_orphans_t2,
                embedding_model=embedding_model,
            )
        except Exception:
            embedding_failed = True
            logger.warning("Orphan embedding match failed for %s — falling back to TF-IDF + LLM", section_key)

    allowed_t1 = {orphan.heading for orphan in body_orphans_t1}
    allowed_t2 = {orphan.heading for orphan in body_orphans_t2}
    deterministic_matches = _deterministic_confirm_orphan_matches(
        enriched_candidates,
        allowed_t1=allowed_t1,
        allowed_t2=allowed_t2,
    )
    matched_t1 = {match["heading_t1"] for match in deterministic_matches}
    matched_t2 = {match["heading_t2"] for match in deterministic_matches}

    remaining_candidates = [
        candidate
        for candidate in enriched_candidates
        if candidate.heading_t1 not in matched_t1 and candidate.heading_t2 not in matched_t2
    ]
    remaining_body_t1 = [orphan for orphan in body_orphans_t1 if orphan.heading not in matched_t1]
    remaining_body_t2 = [orphan for orphan in body_orphans_t2 if orphan.heading not in matched_t2]

    llm_matches: list[dict[str, Any]] = []
    gpt_failed = False
    if remaining_body_t1 and remaining_body_t2 and remaining_candidates:
        try:
            llm_matches = _gpt_arbitrate_orphan_subsections(
                client=client,
                model=model,
                section_key=section_key,
                candidates=remaining_candidates,
                orphans_t1=remaining_body_t1,
                orphans_t2=remaining_body_t2,
            )
        except Exception as exc:
            gpt_failed = True
            logger.warning(
                "Orphan LLM arbitration failed for %s — no body matches confirmed: %s",
                section_key,
                exc,
                exc_info=True,
            )
            llm_matches = []

    matched_t1.update(match["heading_t1"] for match in llm_matches)
    matched_t2.update(match["heading_t2"] for match in llm_matches)

    if gpt_failed and not llm_matches:
        fallback_matches = _deterministic_confirm_orphan_matches(
            remaining_candidates,
            allowed_t1={orphan.heading for orphan in remaining_body_t1},
            allowed_t2={orphan.heading for orphan in remaining_body_t2},
        )
        for match in fallback_matches:
            if match["heading_t1"] in matched_t1 or match["heading_t2"] in matched_t2:
                continue
            deterministic_matches.append(match)
            matched_t1.add(match["heading_t1"])
            matched_t2.add(match["heading_t2"])

    remaining_t1 = [orphan for orphan in orphans_t1 if orphan.heading not in matched_t1]
    remaining_t2 = [orphan for orphan in orphans_t2 if orphan.heading not in matched_t2]

    title_matches: list[dict[str, Any]] = []
    short_t1 = [orphan.heading for orphan in remaining_t1 if not _has_substantial_body(orphan)]
    if short_t1 and remaining_t2:
        title_matches.extend(
            _deterministic_match_orphan_headings(
                short_t1,
                [orphan.heading for orphan in remaining_t2],
            )
        )
        matched_t1.update(match["heading_t1"] for match in title_matches)
        matched_t2.update(match["heading_t2"] for match in title_matches)
        remaining_t1 = [orphan for orphan in remaining_t1 if orphan.heading not in matched_t1]
        remaining_t2 = [orphan for orphan in remaining_t2 if orphan.heading not in matched_t2]
        short_t1 = [orphan.heading for orphan in remaining_t1 if not _has_substantial_body(orphan)]
        if short_t1 and remaining_t2:
            title_matches.extend(
                _gpt_match_orphan_headings(
                    client=client,
                    model=model,
                    section_key=section_key,
                    orphans_t1=short_t1,
                    orphans_t2=[orphan.heading for orphan in remaining_t2],
                )
            )

    matched_t1.update(match["heading_t1"] for match in title_matches)
    matched_t2.update(match["heading_t2"] for match in title_matches)
    remaining_t1 = [orphan for orphan in orphans_t1 if orphan.heading not in matched_t1]
    remaining_t2 = [orphan for orphan in orphans_t2 if orphan.heading not in matched_t2]

    short_t2 = [orphan.heading for orphan in remaining_t2 if not _has_substantial_body(orphan)]
    if remaining_t1 and short_t2:
        second_pass_title_matches = _deterministic_match_orphan_headings(
            [orphan.heading for orphan in remaining_t1],
            short_t2,
        )
        title_matches.extend(second_pass_title_matches)
        matched_t1.update(match["heading_t1"] for match in second_pass_title_matches)
        matched_t2.update(match["heading_t2"] for match in second_pass_title_matches)
        remaining_t1 = [orphan for orphan in remaining_t1 if orphan.heading not in matched_t1]
        remaining_t2 = [orphan for orphan in remaining_t2 if orphan.heading not in matched_t2]
        short_t2 = [orphan.heading for orphan in remaining_t2 if not _has_substantial_body(orphan)]
        if remaining_t1 and short_t2:
            title_matches.extend(
                _gpt_match_orphan_headings(
                    client=client,
                    model=model,
                    section_key=section_key,
                    orphans_t1=[orphan.heading for orphan in remaining_t1],
                    orphans_t2=short_t2,
                )
            )

    tagged_title_matches = [
        {
            **match,
            "match_source": match.get("match_source") or "title_only",
            "llm_confidence": match.get("llm_confidence") or str(match.get("confidence") or ""),
            "tfidf_score": match.get("tfidf_score"),
            "embedding_score": match.get("embedding_score"),
            "heading_score": match.get("heading_score"),
        }
        for match in title_matches
    ]
    if embedding_failed:
        for match in deterministic_matches + llm_matches:
            match.setdefault("embedding_score", None)
    return [*deterministic_matches, *llm_matches, *tagged_title_matches]


def _parse_subsections(md_text: str) -> list[tuple[str, str]]:
    """Découpe un texte markdown en paires (heading, body).

    Le texte avant le premier ### devient (``__intro__``, body).
    Les headings ## de section ne sont pas inclus.
    """
    parts = _SUBSECTION_SPLIT_RE.split(md_text)
    result: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        result.append(("__intro__", intro))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Les titres sans corps sont exclus du matching. Ils peuvent provenir
        # d'un tableau ignoré, d'un parent structurel ou d'un ancien markdown
        # canonique encore en cache.
        if heading and body:
            result.append((heading, body))
    return result


def _pair_subsections(
    subs_t1: list[tuple[str, str]],
    subs_t2: list[tuple[str, str]],
) -> list[tuple[str | None, str, str | None, str]]:
    """Paire les sous-sections T1 et T2 par heading normalisé.

    Retourne une liste de ``(heading_t1, body_t1, heading_t2, body_t2)``.
    ``None`` pour un heading signifie qu'il n'a pas de contrepartie dans l'autre trimestre.
    """
    norm_to_t2: dict[str, tuple[str, str]] = {_normalize_heading(h): (h, body) for h, body in subs_t2}
    matched_t2_norms: set[str] = set()
    pairs: list[tuple[str | None, str, str | None, str]] = []
    for h1, body1 in subs_t1:
        norm = _normalize_heading(h1)
        if norm in norm_to_t2:
            h2, body2 = norm_to_t2[norm]
            pairs.append((h1, body1, h2, body2))
            matched_t2_norms.add(norm)
        else:
            pairs.append((h1, body1, None, ""))
    for h2, body2 in subs_t2:
        if _normalize_heading(h2) not in matched_t2_norms:
            pairs.append((None, "", h2, body2))
    return pairs


def _gpt_match_orphan_headings(
    *,
    client: Any,
    model: str,
    section_key: str,
    orphans_t1: list[str],
    orphans_t2: list[str],
) -> list[dict[str, Any]]:
    """Matching GPT title-only des sous-sections orphelines (correspondances 1→1).

    Utilise par ``_resolve_orphan_subsections`` quand le matching deterministe
    laisse des titres courts sans corps substantiel.
    """
    if not orphans_t1 or not orphans_t2:
        return []

    deterministic_matches = _deterministic_match_orphan_headings(orphans_t1, orphans_t2)
    matched_t1 = {match["heading_t1"] for match in deterministic_matches}
    matched_t2 = {match["heading_t2"] for match in deterministic_matches}
    remaining_t1 = [heading for heading in orphans_t1 if heading not in matched_t1]
    remaining_t2 = [heading for heading in orphans_t2 if heading not in matched_t2]
    if not remaining_t1 or not remaining_t2:
        return deterministic_matches

    try:
        raw = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es expert en rapports bancaires réglementaires canadiens. "
                        "Tu identifies les correspondances entre sous-sections renommées "
                        "d'un trimestre à l'autre."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Format de réponse: {"matches": [{"heading_t1": "...", "heading_t2": "...", '
                        '"confidence": "high|medium|low", "reason": "..."}]}\n'
                        "Règles strictes:\n"
                        "- Correspondances 1→1 uniquement (un heading T1 ↔ un heading T2)\n"
                        "- N'inclure que confidence high ou medium\n"
                        "- Si tu n'es pas sûr, ne pas inclure la paire\n"
                        "- Retourner les headings EXACTEMENT comme fournis\n\n"
                        f"Section: {section_key}\n\n"
                        "Sous-sections T1 sans correspondance exacte:\n"
                        + "\n".join(f"- {h}" for h in remaining_t1)
                        + "\n\nSous-sections T2 sans correspondance exacte:\n"
                        + "\n".join(f"- {h}" for h in remaining_t2)
                    ),
                },
            ],
            response_format=OrphanMatchLLMResponse,
            max_retries=1,
            validation_retry_message=_ORPHAN_MATCH_VALIDATION_RETRY_MESSAGE,
            length_retry_message=_ORPHAN_MATCH_LENGTH_RETRY_MESSAGE,
        )
        orphans_t1_set = set(remaining_t1)
        orphans_t2_set = set(remaining_t2)
        used_t1: set[str] = set(matched_t1)
        used_t2: set[str] = set(matched_t2)
        llm_matches = []
        for m in raw.matches:
            conf = m.confidence
            h1 = m.heading_t1
            h2 = m.heading_t2
            if conf not in {"high", "medium"}:
                continue
            if h1 not in orphans_t1_set or h2 not in orphans_t2_set:
                continue
            if h1 in used_t1 or h2 in used_t2:
                continue
            llm_matches.append(
                {
                    **m.model_dump(),
                    "match_source": "title_only",
                    "llm_confidence": conf,
                    "tfidf_score": None,
                    "embedding_score": None,
                    "heading_score": None,
                }
            )
            used_t1.add(h1)
            used_t2.add(h2)
        return [*deterministic_matches, *llm_matches]
    except Exception as exc:
        logger.warning(
            "Orphan heading GPT match failed for %s — skipping: %s",
            section_key,
            exc,
            exc_info=True,
        )
        fallback_matches = _deterministic_match_orphan_headings(remaining_t1, remaining_t2)
        existing_pairs = {(match["heading_t1"], match["heading_t2"]) for match in deterministic_matches}
        for match in fallback_matches:
            pair = (match["heading_t1"], match["heading_t2"])
            if pair in existing_pairs:
                continue
            deterministic_matches.append(match)
        return deterministic_matches


def _synthetic_subsection_change(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body_t1: str,
    body_t2: str,
    idx: int,
) -> dict[str, Any]:
    """Crée un enregistrement de changement pour une sous-section entièrement ajoutée ou supprimée."""
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading))[:40].strip("_")
    label = "ajoutée" if diff_type == "added" else "supprimée"
    return {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": heading,
        "diff_type": diff_type,
        "source_scope": "subsection",
        "semantic_text_t1": _sanitize_semantic_text(body_t1),
        "semantic_text_t2": _sanitize_semantic_text(body_t2),
        "source_text_t1": body_t1,
        "source_text_t2": body_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": body_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": body_t2[:400]},
        "change_summary": f"Sous-section {label}: {heading}",
    }


def _synthetic_subsection_rename_change(
    *,
    section_key: str,
    heading_t1: str,
    heading_t2: str,
    idx: int,
) -> dict[str, Any]:
    """Crée un changement explicite pour une sous-section renommée."""
    slug_source = f"{heading_t1}_{heading_t2}"
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(slug_source))[:40].strip("_")
    summary = f"Sous-section renommée: {heading_t1} -> {heading_t2}"
    return {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": f"{heading_t1} → {heading_t2}",
        "previous_subsection_heading": heading_t1,
        "current_subsection_heading": heading_t2,
        "diff_type": "renamed",
        "source_scope": "heading",
        "semantic_text_t1": _sanitize_semantic_text(heading_t1),
        "semantic_text_t2": _sanitize_semantic_text(heading_t2),
        "source_text_t1": heading_t1,
        "source_text_t2": heading_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown_heading",
        "source_resolution_t2": "markdown_heading",
        "evidence_t1": {"pages": [], "snippet": heading_t1},
        "evidence_t2": {"pages": [], "snippet": heading_t2},
        "change_summary": summary,
    }
