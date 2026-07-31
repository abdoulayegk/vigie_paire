"""Déduplication sémantique et regroupement des paires similaires pour le triage.

Ce module regroupe les paires de paragraphes quasiment identiques avant l'envoi au LLM
afin d'éviter les requêtes redondantes et de propager les décisions de manière déterministe.
"""

from __future__ import annotations

import logging
from typing import Any

from vigilance.text_analysis.openai_client import _embed_texts

logger = logging.getLogger(__name__)

_TRIAGE_DEDUP_EMBEDDING_THRESHOLD = 0.92
_TRIAGE_EMBEDDING_TRUNCATE_CHARS = 1800


def _compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calcule la similarite cosinus entre deux vecteurs d'embeddings."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _extract_triage_embedding_text(change: dict[str, Any]) -> str:
    """Extrait le texte representatif pour le calcul d'embedding."""
    t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "").strip()
    t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "").strip()
    combined = f"{t1}\n{t2}".strip()
    return combined[:_TRIAGE_EMBEDDING_TRUNCATE_CHARS]


def group_semantic_triage_duplicates(
    changes: list[dict[str, Any]],
    *,
    client: Any,
) -> list[list[int]]:
    """Regroupe les changements ayant des embeddings tres proches.

    Retourne une liste de groupes (indices dans changes). Le premier element de chaque
    groupe est le representant choisi pour le triage.
    """
    if not changes:
        return []
    if len(changes) == 1:
        return [[0]]

    texts = [_extract_triage_embedding_text(c) for c in changes]
    try:
        embeddings = _embed_texts(client, texts)
    except Exception as exc:
        logger.warning("Échec du calcul des embeddings pour la déduplication : %s", exc)
        return [[i] for i in range(len(changes))]

    visited: set[int] = set()
    groups: list[list[int]] = []

    for i in range(len(changes)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        vec_i = embeddings[i] if i < len(embeddings) else []
        if not vec_i:
            groups.append(group)
            continue

        for j in range(i + 1, len(changes)):
            if j in visited:
                continue
            vec_j = embeddings[j] if j < len(embeddings) else []
            if not vec_j:
                continue
            sim = _compute_cosine_similarity(vec_i, vec_j)
            if sim >= _TRIAGE_DEDUP_EMBEDDING_THRESHOLD:
                group.append(j)
                visited.add(j)

        groups.append(group)

    return groups
