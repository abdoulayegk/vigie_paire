"""Helpers for estimating OpenAI token costs for known model routes."""

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
    """Estimate API cost for known model names.

    Unknown models return ``0.0`` so observability remains safe even when the
    active model route is not priced locally yet.
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
    cost = (
        (prompt / 1_000_000.0) * pricing["input"]
        + (completion / 1_000_000.0) * pricing["output"]
    )
    return round(cost, 6)
