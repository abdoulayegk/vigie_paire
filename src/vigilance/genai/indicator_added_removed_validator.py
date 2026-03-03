"""Post-matching GenAI validator for indicator added/removed.

Validates whether indicators marked as "added" (in T2) or "removed" (from T1)
actually exist in the opposite table with different wording. Reduces false positives
when fuzzy pairing fails to match reformulated indicators.

- Circuit breaker: after N consecutive failures, stops calling the API.
- Batching: validates multiple indicators per request to limit cost.
- Conservative fallback: on API error or circuit open, keeps all indicators (no filter).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_CIRCUIT_BREAKER_THRESHOLD = 3

_SYSTEM_PROMPT_ADDED = """Tu es un expert en rapports reglementaires bancaires (Bale III, BSIF).

TACHE: Tu recois un libelle d'indicateur marque "ajoute" dans T2 (rapport trimestriel posterieur).
Tu recois aussi la liste complete des indicateurs de T1 (rapport anterieur).

Question: Ce libelle "ajoute" represente-t-il un concept DEJA present dans T1 (reformulation,
variante orthographique, leger changement de libelle) ou est-ce une VRAIE nouveaute?

Si DEJA present (meme concept, formulation differente): same_concept=true.
Si VRAIE nouveaute: same_concept=false.

REGLE: Reponds UNIQUEMENT en JSON valide. Pas de texte avant ou apres.
Format pour N indicateurs:
{"results": [{"same_concept": true|false, "confidence": 0.0-1.0}, ...]}

Confidence: 1.0 = certain, 0.5 = incertain.
"""

_SYSTEM_PROMPT_REMOVED = """Tu es un expert en rapports reglementaires bancaires (Bale III, BSIF).

TACHE: Tu recois un libelle d'indicateur marque "supprime" de T1 (rapport trimestriel anterieur).
Tu recois aussi la liste complete des indicateurs de T2 (rapport posterieur).

Question: Ce libelle "supprime" represente-t-il un concept DEJA present dans T2 (reformulation,
variante orthographique, leger changement de libelle) ou a-t-il vraiment disparu?

Si DEJA present dans T2 (meme concept, formulation differente): same_concept=true.
Si vraiment disparu: same_concept=false.

REGLE: Reponds UNIQUEMENT en JSON valide. Pas de texte avant ou apres.
Format pour N indicateurs:
{"results": [{"same_concept": true|false, "confidence": 0.0-1.0}, ...]}

Confidence: 1.0 = certain, 0.5 = incertain.
"""


def _build_user_prompt_added(
    indicators: list[str], opposite_indicators: list[str]
) -> str:
    lines = [
        "Indicateurs marques AJOUTES (a evaluer):",
        *[f"  {i}: \"{ind}\"" for i, ind in enumerate(indicators)],
        "",
        "Indicateurs de T1 (rapport anterieur):",
        *[f"  - {ind}" for ind in opposite_indicators[:80]],
    ]
    if len(opposite_indicators) > 80:
        lines.append(f"  ... et {len(opposite_indicators) - 80} autres")
    lines.append(
        "\nReponds en JSON: {\"results\": [{\"same_concept\": bool, \"confidence\": float}, ...]}"
    )
    return "\n".join(lines)


def _build_user_prompt_removed(
    indicators: list[str], opposite_indicators: list[str]
) -> str:
    lines = [
        "Indicateurs marques SUPPRIMES (a evaluer):",
        *[f"  {i}: \"{ind}\"" for i, ind in enumerate(indicators)],
        "",
        "Indicateurs de T2 (rapport posterieur):",
        *[f"  - {ind}" for ind in opposite_indicators[:80]],
    ]
    if len(opposite_indicators) > 80:
        lines.append(f"  ... et {len(opposite_indicators) - 80} autres")
    lines.append(
        "\nReponds en JSON: {\"results\": [{\"same_concept\": bool, \"confidence\": float}, ...]}"
    )
    return "\n".join(lines)


class IndicatorAddedRemovedValidator:
    """Validate added/removed indicators using GPT-4o to reduce false positives.

    Call validate_batch_added or validate_batch_removed.
    Returns list of dicts with same_concept, confidence per indicator.
    On error or circuit open: returns all same_concept=False, confidence=0 (keep all).
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
            "indicators_validated": 0,
            "filtered_added": 0,
            "filtered_removed": 0,
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
            raise RuntimeError(
                "No OpenAI API key available for indicator added/removed validator"
            )

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
                    "Indicator validator circuit breaker OPEN after %d consecutive failures.",
                    self._consecutive_failures,
                )

    def _validate_batch(
        self,
        indicators: list[str],
        opposite_indicators: list[str],
        indicator_type: str,
    ) -> list[dict[str, Any]]:
        """Validate a batch. indicator_type: 'added' or 'removed'."""
        if not indicators:
            return []

        if self._circuit_open:
            return [{"same_concept": False, "confidence": 0.0} for _ in indicators]

        try:
            client = self._ensure_client()
        except Exception as exc:
            logger.warning("Indicator validator: client init failed: %s", exc)
            self._record_failure()
            self.stats["errors"] += 1
            return [{"same_concept": False, "confidence": 0.0} for _ in indicators]

        if indicator_type == "added":
            system = _SYSTEM_PROMPT_ADDED
            prompt = _build_user_prompt_added(indicators, opposite_indicators)
        else:
            system = _SYSTEM_PROMPT_REMOVED
            prompt = _build_user_prompt_removed(indicators, opposite_indicators)

        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=1024,
            )
        except Exception as exc:
            logger.debug("Indicator validator API error: %s", exc)
            self._record_failure()
            self.stats["errors"] += 1
            return [{"same_concept": False, "confidence": 0.0} for _ in indicators]

        self._record_success()
        self.stats["calls"] += 1
        self.stats["indicators_validated"] += len(indicators)
        self.stats["total_latency"] = self.stats.get("total_latency", 0) + (
            time.monotonic() - start
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("Indicator validator: invalid JSON: %s", exc)
            self.stats["errors"] += 1
            return [{"same_concept": False, "confidence": 0.0} for _ in indicators]

        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(indicators):
            logger.debug(
                "Indicator validator: expected %d results, got %s",
                len(indicators),
                type(results).__name__,
            )
            return [{"same_concept": False, "confidence": 0.0} for _ in indicators]

        out: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                out.append({"same_concept": False, "confidence": 0.0})
                continue
            same = item.get("same_concept")
            if same is True:
                if indicator_type == "added":
                    self.stats["filtered_added"] += 1
                else:
                    self.stats["filtered_removed"] += 1
            try:
                conf = float(item.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                conf = 0.5
            out.append({
                "same_concept": bool(same) if same is not None else False,
                "confidence": conf,
            })
        return out


def validate_indicator_added_removed(
    added: list[str],
    removed: list[str],
    all_t1_indicators: list[str],
    all_t2_indicators: list[str],
    *,
    api_key: str | None = None,
    batch_size: int = 8,
    confidence_min: float = 0.8,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate added/removed indicators and filter false positives.

    Returns:
        (filtered_added, filtered_removed, stats)
        Indicators with same_concept=true and confidence >= confidence_min are removed.
    """
    validator = IndicatorAddedRemovedValidator(api_key=api_key)
    to_remove_added: set[str] = set()
    to_remove_removed: set[str] = set()

    for i in range(0, len(added), batch_size):
        batch = added[i : i + batch_size]
        results = validator._validate_batch(
            batch, all_t1_indicators, "added"
        )
        for ind, res in zip(batch, results):
            if (
                res.get("same_concept") is True
                and res.get("confidence", 0) >= confidence_min
            ):
                to_remove_added.add(ind)

    for i in range(0, len(removed), batch_size):
        batch = removed[i : i + batch_size]
        results = validator._validate_batch(
            batch, all_t2_indicators, "removed"
        )
        for ind, res in zip(batch, results):
            if (
                res.get("same_concept") is True
                and res.get("confidence", 0) >= confidence_min
            ):
                to_remove_removed.add(ind)

    filtered_added = [x for x in added if x not in to_remove_added]
    filtered_removed = [x for x in removed if x not in to_remove_removed]
    return filtered_added, filtered_removed, dict(validator.stats)
