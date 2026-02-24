"""Embedding service with batch, cache, and stats for table/indicator matching."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for embedding API usage and caching."""

    api_calls: int = 0
    cache_hits: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "batch_sizes": list(self.batch_sizes),
            "errors": self.errors,
        }


class EmbeddingService:
    """
    Central embedding service: batch requests, in-memory cache, error handling.
    Used by both table matching and indicator matching.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 50,
    ) -> None:
        self._client = None
        if api_key and str(api_key).strip():
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=str(api_key).strip())
            except Exception as e:
                logger.warning("EmbeddingService init failed: %s", e)
        self.model = model
        self.batch_size = min(max(batch_size, 1), 100)
        self._cache: dict[str, list[float]] = {}
        self.stats = EmbeddingStats()

    @property
    def available(self) -> bool:
        """True if API client is ready."""
        return self._client is not None

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """
        Batch embed with cache. Returns list of embeddings or None for failed/empty.
        """
        if not texts:
            return []
        result: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            t = (t or "").strip() or " "
            k = self._cache_key(t)
            if k in self._cache:
                result[i] = self._cache[k]
                self.stats.cache_hits += 1
            else:
                to_fetch.append((i, t))

        if not self._client:
            return result

        for chunk_start in range(0, len(to_fetch), self.batch_size):
            chunk = to_fetch[chunk_start : chunk_start + self.batch_size]
            indices = [c[0] for c in chunk]
            batch_texts = [c[1] for c in chunk]
            try:
                resp = self._client.embeddings.create(
                    input=batch_texts, model=self.model
                )
                self.stats.api_calls += 1
                self.stats.batch_sizes.append(len(batch_texts))
                for j, d in enumerate(resp.data):
                    if j < len(indices):
                        emb = list(d.embedding)
                        idx = indices[j]
                        result[idx] = emb
                        self._cache[self._cache_key(batch_texts[j])] = emb
            except Exception as e:
                logger.warning("Embedding API error: %s", e)
                self.stats.errors += 1

        return result

    def get_pairwise_cosine(
        self, texts_a: list[str], texts_b: list[str]
    ) -> np.ndarray:
        """
        NxM cosine similarity matrix. Uses batch + cache.
        Returns zeros for failed/empty inputs.
        """
        if not texts_a or not texts_b:
            return np.zeros((len(texts_a), len(texts_b)))

        emb_a = self.get_embeddings(texts_a)
        emb_b = self.get_embeddings(texts_b)

        valid_a = [(i, e) for i, e in enumerate(emb_a) if e is not None]
        valid_b = [(i, e) for i, e in enumerate(emb_b) if e is not None]

        if not valid_a or not valid_b:
            return np.zeros((len(texts_a), len(texts_b)))

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("sklearn not available for cosine_similarity")
            return np.zeros((len(texts_a), len(texts_b)))

        ea = np.array([e for _, e in valid_a], dtype=np.float64)
        eb = np.array([e for _, e in valid_b], dtype=np.float64)
        sim = cosine_similarity(ea, eb)

        out = np.zeros((len(texts_a), len(texts_b)))
        for ii, (i, _) in enumerate(valid_a):
            for jj, (j, _) in enumerate(valid_b):
                out[i, j] = float(sim[ii, jj])
        return out

    def get_single_pair_cosine(self, text_a: str, text_b: str) -> float:
        """Convenience: cosine between two strings. Returns 0.0 on failure."""
        mat = self.get_pairwise_cosine([text_a], [text_b])
        return float(mat[0, 0]) if mat.size else 0.0
