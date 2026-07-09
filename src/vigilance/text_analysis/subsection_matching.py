"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import math
import re
from typing import Any

from vigilance.text_analysis.chunk_alignment import _tfidf_similarity_matrix_from_texts
from vigilance.text_analysis.constants import _SUBSECTION_SPLIT_RE
from vigilance.text_analysis.normalization import _sanitize_semantic_text
from vigilance.text_analysis.openai_client import _call_json_completion, _embed_texts

logger = logging.getLogger(__name__)

_ORPHAN_TOP_K = 3
_ORPHAN_BODY_EXCERPT_CHARS = 500
_ORPHAN_EMBEDDING_TRUNCATE_CHARS = 8000
_ORPHAN_MIN_BODY_CHARS_FOR_BODY_MATCH = 100
_EMBEDDING_STRONG_CANDIDATE_THRESHOLD = 0.82
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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
    ordered_bodies = [
        _truncate_for_embedding(unique_t1[heading]) for heading in unique_t1
    ] + [
        _truncate_for_embedding(unique_t2[heading]) for heading in unique_t2
    ]
    embeddings = _embed_texts(client, ordered_bodies, model=embedding_model)
    embedding_by_heading = {
        heading: embeddings[index]
        for index, heading in enumerate(ordered_headings)
    }

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
        "candidate_strong"
        if candidate.embedding_score >= _EMBEDDING_STRONG_CANDIDATE_THRESHOLD
        else "candidate_review"
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

    candidate_lines = "\n\n".join(_format_orphan_candidate_for_prompt(candidate) for candidate in candidates)
    unmatched_t1 = "\n".join(f"- {orphan.heading}" for orphan in orphans_t1) or "- aucun"
    unmatched_t2 = "\n".join(f"- {orphan.heading}" for orphan in orphans_t2) or "- aucun"

    try:
        raw = _call_json_completion(
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
        )
    except Exception:
        logger.warning("Orphan subsection GPT arbitration failed for %s — skipping", section_key)
        return []

    orphans_t1_set = {orphan.heading for orphan in orphans_t1}
    orphans_t2_set = {orphan.heading for orphan in orphans_t2}
    candidate_lookup = {_candidate_key(candidate): candidate for candidate in candidates}
    valid_matches: list[tuple[int, float, float, float, str, str, str, str, OrphanCandidate]] = []
    for item in raw.get("matches") or []:
        conf = str(item.get("confidence") or "").lower()
        heading_t1 = str(item.get("heading_t1") or "")
        heading_t2 = str(item.get("heading_t2") or "")
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
                str(item.get("reason") or "llm_arbitration"),
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
            {
                "heading_t1": heading_t1,
                "heading_t2": heading_t2,
                "confidence": conf,
                "llm_confidence": conf,
                "reason": reason,
                "match_source": "llm_embedding_confirmed",
                "tfidf_score": round(candidate.tfidf_score, 4),
                "embedding_score": round(candidate.embedding_score, 4),
                "heading_score": round(candidate.heading_score, 4),
            }
        )
        used_t1.add(heading_t1)
        used_t2.add(heading_t2)
    return matches


def _resolve_orphan_subsections(
    *,
    client: Any,
    model: str,
    section_key: str,
    orphans_t1: list[OrphanSubsection],
    orphans_t2: list[OrphanSubsection],
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Résout les orphelins via TF-IDF sklearn, embeddings puis LLM final."""
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

    llm_matches: list[dict[str, Any]] = []
    if body_orphans_t1 and body_orphans_t2:
        try:
            llm_matches = _gpt_arbitrate_orphan_subsections(
                client=client,
                model=model,
                section_key=section_key,
                candidates=enriched_candidates,
                orphans_t1=body_orphans_t1,
                orphans_t2=body_orphans_t2,
            )
        except Exception:
            logger.warning("Orphan LLM arbitration failed for %s — no body matches confirmed", section_key)
            llm_matches = []

    matched_t1 = {match["heading_t1"] for match in llm_matches}
    matched_t2 = {match["heading_t2"] for match in llm_matches}
    remaining_t1 = [orphan for orphan in orphans_t1 if orphan.heading not in matched_t1]
    remaining_t2 = [orphan for orphan in orphans_t2 if orphan.heading not in matched_t2]

    title_matches: list[dict[str, Any]] = []
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
            "match_source": "title_only",
            "llm_confidence": str(match.get("confidence") or ""),
            "tfidf_score": None,
            "embedding_score": None,
            "heading_score": None,
        }
        for match in title_matches
    ]
    if embedding_failed and llm_matches:
        for match in llm_matches:
            match.setdefault("embedding_score", None)
    return [*llm_matches, *tagged_title_matches]


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
        if heading:
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
    """Identifie via GPT les sous-sections renommées entre T1 et T2 (1→1 uniquement).

    Conservé pour compatibilité tests/facade. Le pipeline principal utilise
    ``_resolve_orphan_subsections``.
    """
    if not orphans_t1 or not orphans_t2:
        return []
    try:
        raw = _call_json_completion(
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
                        + "\n".join(f"- {h}" for h in orphans_t1)
                        + "\n\nSous-sections T2 sans correspondance exacte:\n"
                        + "\n".join(f"- {h}" for h in orphans_t2)
                    ),
                },
            ],
        )
        orphans_t1_set = set(orphans_t1)
        orphans_t2_set = set(orphans_t2)
        used_t1: set[str] = set()
        used_t2: set[str] = set()
        matches = []
        for m in raw.get("matches") or []:
            conf = str(m.get("confidence") or "").lower()
            h1 = m.get("heading_t1") or ""
            h2 = m.get("heading_t2") or ""
            if conf not in {"high", "medium"}:
                continue
            if h1 not in orphans_t1_set or h2 not in orphans_t2_set:
                continue
            if h1 in used_t1 or h2 in used_t2:
                continue
            matches.append(m)
            used_t1.add(h1)
            used_t2.add(h2)
        return matches
    except Exception:
        logger.warning("Orphan heading GPT match failed for %s — skipping", section_key)
        return []


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
