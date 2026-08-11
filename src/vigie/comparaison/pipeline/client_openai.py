"""Transport OpenAI et collecte des metriques d utilisation."""

from __future__ import annotations

import json
import time
from typing import Any

import openai

from vigie.comparaison.io import _extract_usage_metrics
from vigie.support.utils.genai import get_openai_api_key

OPENAI_COMPARISON_TIMEOUT_SECONDS = 120.0


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int | None = None,
    temperature: float = 0.0,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
    response_model: type | None = None,
) -> dict[str, Any]:
    """Appeler l'API OpenAI avec sortie JSON.

    Quand *response_model* est une sous-classe de ``pydantic.BaseModel``, l'appel
    utilise les **Structured Outputs** OpenAI pour garantir la conformite au schema.
    Le modele valide est reconverti en dict pour que les appelants gardent une
    interface identique.

    ``max_completion_tokens=None`` (defaut) laisse le modele s'arreter naturellement
    sans plafond artificiel — privilegier la qualite complete plutot que la vitesse.
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = openai.OpenAI(
        api_key=api_key,
        timeout=OPENAI_COMPARISON_TIMEOUT_SECONDS,
        # Les retries sont geres par la boucle applicative ci-dessous afin
        # d'eviter de multiplier les tentatives avec celles du SDK.
        max_retries=0,
    )
    last_error: Exception | None = None
    use_structured = response_model is not None
    for attempt in range(api_retry_max + 1):
        if attempt > 0:
            time.sleep(1.5 * (2 ** (attempt - 1)))
        try:
            if use_structured:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": response_model,
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.beta.chat.completions.parse(**kwargs)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Structured Output parsing returned None")
                data = parsed.model_dump()
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("OpenAI response is not a JSON object")
            if usage_recorder is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
                usage_recorder.append(
                    {
                        "model": model,
                        "call_kind": call_kind,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message
                and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")


def _call_openai_embeddings(
    *,
    model: str,
    inputs: list[str],
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison_embeddings",
) -> list[list[float]]:
    """Encoder les vues de tableaux par lots pour la recuperation hybride RBC."""
    if not inputs:
        return []
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = openai.OpenAI(
        api_key=api_key,
        timeout=OPENAI_COMPARISON_TIMEOUT_SECONDS,
    )
    vectors: list[list[float]] = []
    for start in range(0, len(inputs), 96):
        response = client.embeddings.create(model=model, input=inputs[start : start + 96])
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend([list(item.embedding) for item in ordered])
        if usage_recorder is not None:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
            usage_recorder.append(
                {
                    "model": model,
                    "call_kind": call_kind,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
    return vectors
