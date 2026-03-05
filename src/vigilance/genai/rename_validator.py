"""Post-matching GenAI validator for indicator rename pairs.

Validates whether proposed (removed, added) pairs represent the same regulatory
concept (true rename) or different concepts (false positive). Used to filter
fuzzy-paired renames before output.

- Circuit breaker: after N consecutive failures, stops calling the API.
- Batching: validates multiple pairs per request to limit cost.
- Conservative fallback: on API error or circuit open, keeps all renames (no rejection).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_CIRCUIT_BREAKER_THRESHOLD = 3

_SYSTEM_PROMPT = """Tu es un expert en rapports reglementaires bancaires (Bale III, BSIF).

TACHE: Tu recois des paires de libelles (colonne 1 de tableaux) extraits de rapports trimestriels T1 et T2.
Pour chaque paire (ancien_libelle, nouveau_libelle), determine si ils designent le MEME concept reglementaire ou financier
(vrai renommage / reformulation) ou deux concepts DIFFERENTS (add + remove, pas un renommage).

REGLE: Reponds UNIQUEMENT en JSON valide. Pas de texte avant ou apres.
Format pour N paires:
{"results": [{"same_concept": true|false, "confidence": 0.0-1.0}, ...]}

Confidence: 1.0 = certain, 0.5 = incertain. Si same_concept=false, confidence doit refleter la certitude que ce ne sont PAS le meme concept.
"""


def _build_user_prompt(pairs: list[tuple[str, str]]) -> str:
    lines = ["Paires a evaluer (index, ancien_libelle, nouveau_libelle):"]
    for i, (old_label, new_label) in enumerate(pairs):
        lines.append(f"  {i}: \"{old_label}\" -> \"{new_label}\"")
    lines.append("\nReponds en JSON: {\"results\": [{\"same_concept\": bool, \"confidence\": float}, ...]}")
    return "\n".join(lines)


class RenameValidator:
    """Validate rename pairs using GPT-4o to reduce false positives.

    Call validate_batch(pairs) with list of (removed_label, added_label).
    Returns list of dicts with same_concept, confidence per pair.
    On error or circuit open: returns all same_concept=True, confidence=0 (conservative).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout: int = 30,
        circuit_breaker_threshold: int = _CIRCUIT_BREAKER_THRESHOLD,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._client: Any | None = None
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "calls": 0,
            "pairs_validated": 0,
            "rejected": 0,
            "accepted": 0,
            "errors": 0,
        }

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from vigilance.utils.genai import get_openai_api_key

        key = self._api_key or get_openai_api_key()
        if not key:
            raise RuntimeError("No OpenAI API key available for rename validator")

        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=self.timeout)
        return self._client

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures >= self._circuit_breaker_threshold
                and not self._circuit_open
            ):
                self._circuit_open = True
                logger.warning(
                    "Rename validator circuit breaker OPEN after %d consecutive failures.",
                    self._consecutive_failures,
                )

    def validate_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Validate a batch of rename pairs. Returns one result dict per pair.

        Each result has: same_concept (bool), confidence (float).
        On error or circuit open: returns all same_concept=True, confidence=0.
        """
        if not pairs:
            return []

        if self._circuit_open:
            return [
                {"same_concept": True, "confidence": 0.0} for _ in pairs
            ]

        try:
            client = self._ensure_client()
        except Exception as exc:
            logger.warning("Rename validator: client init failed: %s", exc)
            self._record_failure()
            self.stats["errors"] += 1
            return [{"same_concept": True, "confidence": 0.0} for _ in pairs]

        prompt = _build_user_prompt(pairs)
        start = time.monotonic()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=1024,
            )
        except Exception as exc:
            logger.debug("Rename validator API error: %s", exc)
            self._record_failure()
            self.stats["errors"] += 1
            return [{"same_concept": True, "confidence": 0.0} for _ in pairs]

        self._record_success()
        self.stats["calls"] += 1
        self.stats["pairs_validated"] += len(pairs)
        self.stats["total_latency"] = self.stats.get("total_latency", 0) + (
            time.monotonic() - start
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("Rename validator: invalid JSON: %s", exc)
            self.stats["errors"] += 1
            return [{"same_concept": True, "confidence": 0.0} for _ in pairs]

        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(pairs):
            logger.debug(
                "Rename validator: expected %d results, got %s",
                len(pairs),
                type(results).__name__,
            )
            return [{"same_concept": True, "confidence": 0.0} for _ in pairs]

        out: list[dict[str, Any]] = []
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                out.append({"same_concept": True, "confidence": 0.0})
                continue
            same = item.get("same_concept")
            if same is False:
                self.stats["rejected"] += 1
            else:
                self.stats["accepted"] += 1
            try:
                conf = float(item.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                conf = 0.5
            out.append({
                "same_concept": bool(same) if same is not None else True,
                "confidence": conf,
            })
        return out


def validate_rename_pairs(
    pairs: list[tuple[str, str]],
    *,
    api_key: str | None = None,
    batch_size: int = 10,
    confidence_min: float = 0.8,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Any]]:
    """Validate rename pairs and filter out those that are not true renames.

    Returns:
        (accepted_pairs, rejected_pairs, stats)
    """
    if not pairs:
        return [], [], {"calls": 0, "accepted": 0, "rejected": 0, "errors": 0}

    validator = RenameValidator(api_key=api_key)
    accepted: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        results = validator.validate_batch(batch)
        for (old_l, new_l), res in zip(batch, results):
            if (
                res.get("same_concept") is True
                and res.get("confidence", 0) >= confidence_min
            ):
                accepted.append((old_l, new_l))
            else:
                rejected.append((old_l, new_l))

    return accepted, rejected, dict(validator.stats)
