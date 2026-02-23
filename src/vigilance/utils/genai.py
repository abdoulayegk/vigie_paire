"""GenAI environment helpers."""

from __future__ import annotations

import os


def get_openai_api_key() -> str | None:
    """Read OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY") or None


def is_genai_configured() -> bool:
    """Return True when a GenAI API key is available."""
    return bool(get_openai_api_key())
