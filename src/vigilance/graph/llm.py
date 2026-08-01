"""Factory de clients LLM LangChain avec résilience native et tentatives automatiques."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI


def get_llm(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 3,
    **kwargs: Any,
) -> ChatOpenAI:
    """Instancie et retourne un client ChatOpenAI LangChain configuré.

    Intègre nativement la gestion des retries (backoff exponentiel) et le support
    des Structured Outputs via Pydantic.
    """
    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

    kwargs_clean = dict(kwargs)
    kwargs_clean.pop("api_key", None)

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
        **kwargs_clean,
    )
    return llm
