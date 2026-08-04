"""Etat runtime du validateur leger, sans extraction ni appels LLM."""

from __future__ import annotations

import getpass
import os
import re

_VALIDATOR_MODE = False
_USERNAME: str | None = None
_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def set_validator_mode(enabled: bool) -> None:
    """Activer ou desactiver le mode validateur pour le processus courant."""
    global _VALIDATOR_MODE
    _VALIDATOR_MODE = bool(enabled)


def is_validator_mode() -> bool:
    """Indiquer si Dash fonctionne en consultation/validation sans pipeline."""
    return _VALIDATOR_MODE


def set_username(username: str | None) -> None:
    """Definir l'identifiant utilise pour le sidecar de validation."""
    global _USERNAME
    _USERNAME = _sanitize_username(username) if username else None


def current_username() -> str | None:
    """Retourner l'identifiant analyste en mode validateur, sinon ``None``."""
    if _USERNAME:
        return _USERNAME
    if not _VALIDATOR_MODE:
        return None
    try:
        return _sanitize_username(getpass.getuser())
    except Exception:
        fallback = os.environ.get("USERNAME") or os.environ.get("USER") or "anonymous"
        return _sanitize_username(fallback)


def _sanitize_username(value: str) -> str:
    """Convertir un nom d'utilisateur en fragment de nom de fichier sur."""
    cleaned = _INVALID_CHARS.sub("_", value.strip())
    return cleaned or "anonymous"
