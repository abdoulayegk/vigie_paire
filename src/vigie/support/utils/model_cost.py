"""Utilitaires d'estimation des couts de tokens OpenAI pour les modeles connus."""

from __future__ import annotations

from typing import Any

_OPENAI_MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
}


def estimate_openai_cost_usd(
    model: str | None,
    *,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
) -> float:
    """Estime le cout API pour les noms de modeles connus.

    Les modeles inconnus retournent ``0.0`` afin que l'observabilite reste
    sure meme lorsque le modele actif n'est pas encore tarifie localement.

    Args:
        model: Nom du modele OpenAI (ex. ``"gpt-4o"``).
        prompt_tokens: Nombre de tokens en entree.
        completion_tokens: Nombre de tokens en sortie.

    Returns:
        Cout estime en USD, arrondi a 6 decimales.
    """
    model_name = str(model or "").strip()
    pricing = _OPENAI_MODEL_PRICING_USD_PER_MILLION.get(model_name)
    if not pricing:
        return 0.0
    try:
        prompt = max(0, int(prompt_tokens or 0))
        completion = max(0, int(completion_tokens or 0))
    except (TypeError, ValueError):
        return 0.0
    cost = (prompt / 1_000_000.0) * pricing["input"] + (completion / 1_000_000.0) * pricing["output"]
    return round(cost, 6)
