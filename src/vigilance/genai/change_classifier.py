"""Post-matching GenAI classifier for table-level change relevance.

Called AFTER deterministic matching (Hungarian / indicator diff) to classify
each detected change into regulatory relevance categories. Does not alter
matching results -- only enriches the payload with ``genai_analysis``.

Features:
- Circuit breaker: after N consecutive failures, stops calling the API.
- Thread-pool batching: classifies multiple tables concurrently.
- Rate limiting: token-bucket to avoid exceeding API quotas.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

_VALID_RELEVANCE = {"REGLEMENTAIRE", "NON_SIGNIFICATIF", "STRUCTUREL", "NOUVELLE_DIVULGATION"}
_VALID_RISK = {"FAIBLE", "MODERE", "ELEVE"}

_SYSTEM_PROMPT = """\
Tu es un classificateur de changements reglementaires pour les rapports \
trimestriels bancaires (Bale III / BSIF).

Tu recevras un objet JSON decrivant les changements detectes entre deux rapports \
trimestriels pour un tableau. Ta tache : classifier la *pertinence reglementaire* \
du changement.

REGLES (strictes) :
- Ignorer les changements de valeurs numeriques (montants, pourcentages, dates).
- Ignorer les changements purement cosmetiques ou de mise en page (reordonnancement \
  de colonnes, espacement, alignement).
- Se concentrer UNIQUEMENT sur l'impact du changement sur la divulgation reglementaire.
- Si le changement concerne les ratios de capital, CET1, TLAC, LCR, NSFR, ratio de \
  levier, actifs ponderes en fonction des risques, liquidite, risque de credit, \
  risque de marche, risque operationnel, ou toute exigence de divulgation BSIF / \
  Bale III --> pertinence = REGLEMENTAIRE.
- Si le changement est purement de mise en forme sans impact semantique --> NON_SIGNIFICATIF.
- Si le changement est une reorganisation structurelle (scission/fusion de tableau, \
  deplacement de section) SANS changement semantique --> STRUCTUREL.
- Si le changement introduit une nouvelle ligne ou section de divulgation non \
  presente auparavant --> NOUVELLE_DIVULGATION.

FORMAT DE SORTIE (JSON uniquement, aucun texte supplementaire) :
{
  "relevance": "REGLEMENTAIRE | NON_SIGNIFICATIF | STRUCTUREL | NOUVELLE_DIVULGATION",
  "risk_level": "FAIBLE | MODERE | ELEVE",
  "confidence": <float 0.0-1.0>,
  "justification": "<max 3 phrases, en francais>"
}

IMPORTANT : Toutes les reponses doivent etre en francais.
"""

_USER_PROMPT_TEMPLATE = """\
Classifie le changement de tableau suivant :

```json
{table_payload}
```
"""

_CIRCUIT_BREAKER_THRESHOLD = 3


_EN_TO_FR_RELEVANCE: dict[str, str] = {
    "REGULATORY": "REGLEMENTAIRE",
    "NON_MATERIAL": "NON_SIGNIFICATIF",
    "STRUCTURAL": "STRUCTUREL",
    "NEW_DISCLOSURE": "NOUVELLE_DIVULGATION",
}

_EN_TO_FR_RISK: dict[str, str] = {
    "LOW": "FAIBLE",
    "MODERATE": "MODERE",
    "HIGH": "ELEVE",
}


def _sanitize_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Clamp and validate the LLM response to known enum values.

    Accepts both French and English labels from the LLM and normalises to French.
    """
    relevance = str(raw.get("relevance", "")).upper().strip()
    relevance = _EN_TO_FR_RELEVANCE.get(relevance, relevance)
    if relevance not in _VALID_RELEVANCE:
        relevance = "NON_CLASSIFIE"

    risk = str(raw.get("risk_level", "")).upper().strip()
    risk = _EN_TO_FR_RISK.get(risk, risk)
    if risk not in _VALID_RISK:
        risk = "MODERE"

    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    justification = str(raw.get("justification", ""))[:500]

    return {
        "relevance": relevance,
        "risk_level": risk,
        "confidence": round(confidence, 2),
        "justification": justification,
        "source": "gpt-4o",
    }


_UNCLASSIFIED: dict[str, Any] = {
    "relevance": "NON_CLASSIFIE",
    "risk_level": "MODERE",
    "confidence": 0.0,
    "justification": "",
    "source": "fallback",
}


class _SimpleRateLimiter:
    """Token-bucket limiter: at most *rate* calls per second."""

    def __init__(self, rate: float = 2.0) -> None:
        self._min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


class GenAIChangeClassifier:
    """Classify table-level change payloads using GPT-4o.

    Instantiate once per analysis run; the OpenAI client is created lazily.
    Includes a circuit breaker that disables API calls after
    ``_CIRCUIT_BREAKER_THRESHOLD`` consecutive failures, and a
    ``classify_batch`` method for concurrent classification.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout: int = 45,
        rate_limit: float = 2.0,
        max_workers: int = 4,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_workers = min(max_workers, 8)
        self._client: Any | None = None
        self._limiter = _SimpleRateLimiter(rate=rate_limit)
        self.stats = {"calls": 0, "errors": 0, "total_latency": 0.0}

        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_lock = threading.Lock()

    # -- Circuit breaker helpers --

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD
                and not self._circuit_open
            ):
                self._circuit_open = True
                logger.warning(
                    "GenAI classifier circuit breaker OPEN after %d consecutive "
                    "failures. Remaining tables will get UNCLASSIFIED.",
                    self._consecutive_failures,
                )

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    # -- Client --

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from vigilance.utils.genai import get_openai_api_key

        key = self._api_key or get_openai_api_key()
        if not key:
            raise RuntimeError("No OpenAI API key available for GenAI classifier")

        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=self.timeout)
        return self._client

    # -- Payload --

    def _build_compact_payload(self, comp: dict[str, Any]) -> str:
        """Extract only classification-relevant fields from the full comparison."""
        payload = {
            "section": comp.get("section", ""),
            "table_title": comp.get("table_title")
            or comp.get("title_t1")
            or comp.get("title_t2")
            or "",
            "table_status": comp.get("table_status", ""),
            "added_indicators": [
                str(i) for i in (comp.get("added_indicators") or [])
            ][:30],
            "removed_indicators": [
                str(i) for i in (comp.get("removed_indicators") or [])
            ][:30],
            "renamed_indicators": [
                {
                    "old": str(r.get("old_name", "")),
                    "new": str(r.get("new_name", "")),
                }
                for r in (comp.get("renamed_indicators") or [])
            ][:20],
        }
        fn = comp.get("footnotes_diff") or comp.get("footnotes_counts")
        if fn:
            payload["footnotes_diff_summary"] = fn

        return json.dumps(payload, ensure_ascii=False, indent=None)

    # -- Single classification --

    def classify_table_change(self, comp: dict[str, Any]) -> dict[str, Any]:
        """Classify a single table comparison dict. Returns sanitized result.

        Respects the circuit breaker: if open, returns UNCLASSIFIED immediately.
        """
        if self._circuit_open:
            return dict(_UNCLASSIFIED)

        try:
            client = self._ensure_client()
        except Exception as exc:
            logger.warning("GenAI classifier: client init failed: %s", exc)
            self._record_failure()
            return dict(_UNCLASSIFIED)

        compact = self._build_compact_payload(comp)
        user_msg = _USER_PROMPT_TEMPLATE.format(table_payload=compact)

        self._limiter.acquire()
        self.stats["calls"] += 1
        t0 = time.monotonic()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            elapsed = time.monotonic() - t0
            self.stats["total_latency"] += elapsed

            raw_text = response.choices[0].message.content or "{}"
            raw = json.loads(raw_text)
            result = _sanitize_response(raw)
            self._record_success()
            logger.info(
                "GenAI classifier: %s -> %s (%.1fs)",
                (comp.get("table_title") or comp.get("title_t1") or "?")[:40],
                result["relevance"],
                elapsed,
            )
            return result

        except Exception as exc:
            self.stats["errors"] += 1
            self._record_failure()
            logger.warning("GenAI classifier call failed: %s", exc)
            return dict(_UNCLASSIFIED)

    # -- Batch classification (concurrent) --

    def classify_batch(
        self, comparisons: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify a list of table comparisons concurrently.

        Uses a ThreadPoolExecutor capped at ``self.max_workers`` threads.
        The rate limiter and circuit breaker are shared across threads.

        Returns a list of results in the same order as the input.
        """
        n = len(comparisons)
        if n == 0:
            return []

        results: list[dict[str, Any] | None] = [None] * n
        effective_workers = min(self.max_workers, n)

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_idx = {
                pool.submit(self.classify_table_change, comp): idx
                for idx, comp in enumerate(comparisons)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = dict(_UNCLASSIFIED)

        return [r if r is not None else dict(_UNCLASSIFIED) for r in results]
