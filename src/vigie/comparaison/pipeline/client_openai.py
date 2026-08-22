"""Transport OpenAI et collecte des metriques d utilisation."""

from __future__ import annotations

from typing import Any

from vigie.llm import ReasoningProfile, complete_openai_json, embed

OPENAI_COMPARISON_TIMEOUT_SECONDS = 120.0


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int | None = None,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
    response_model: type | None = None,
    profile: ReasoningProfile | str = "default",
) -> dict[str, Any]:
    """Appeler l'API OpenAI avec sortie JSON.

    Quand *response_model* est une sous-classe de ``pydantic.BaseModel``, l'appel
    utilise les **Structured Outputs** OpenAI pour garantir la conformite au schema.
    Le modele valide est reconverti en dict pour que les appelants gardent une
    interface identique.

    ``max_completion_tokens=None`` (defaut) laisse le modele s'arreter naturellement
    sans plafond artificiel — privilegier la qualite complete plutot que la vitesse.
    """
    return complete_openai_json(
        model=model,
        messages=messages,
        profile=profile,
        max_completion_tokens=max_completion_tokens,
        api_retry_max=api_retry_max,
        usage_recorder=usage_recorder,
        call_kind=call_kind,
        response_model=response_model,
    )


def _call_openai_embeddings(
    *,
    model: str,
    inputs: list[str],
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison_embeddings",
) -> list[list[float]]:
    """Encoder les vues de tableaux par lots pour la recuperation hybride RBC."""
    role = "embedding_large" if "large" in str(model) else "embedding_small"
    return embed(
        inputs,
        role=role,
        usage_recorder=usage_recorder,
        call_kind=call_kind,
    )
