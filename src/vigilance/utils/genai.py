"""GenAI environment helpers."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_openai_api_key() -> str | None:
    """Read OpenAI API key from environment (.env is loaded at import if dotenv available)."""
    raw = os.getenv("OPENAI_API_KEY") or ""
    return raw.strip() if isinstance(raw, str) else None


def is_genai_configured() -> bool:
    """Return True when a GenAI API key is available."""
    return bool(get_openai_api_key())
