"""Utilitaires d'environnement pour l'IA generative."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_openai_api_key() -> str | None:
    """Lit la cle API OpenAI depuis l'environnement (.env charge a l'import si dotenv est disponible).

    Returns:
        La cle API nettoyee, ou None si absente.
    """
    raw = os.getenv("OPENAI_API_KEY") or ""
    return raw.strip() if isinstance(raw, str) else None


def is_genai_configured() -> bool:
    """Retourne True lorsqu'une cle API GenAI est disponible."""
    return bool(get_openai_api_key())
