"""Configuration globale du mode reader (.exe VigieRegDesjardins).

Ce module expose des drapeaux et helpers consultes par le reste de l'app
pour adapter son comportement quand on tourne en mode "viewer" :

- ``is_reader_mode()`` -> True si on tourne dans le .exe (lecture seule).
- ``current_username()`` -> identifiant analyste utilise pour suffixer
  ``comparison.review_state.<username>.json`` (strategie multi-utilisateurs (b)).
- ``set_reader_mode()`` / ``set_username()`` -> setters appeles uniquement
  par ``reader.py`` au demarrage. Le mode normal (``app.py``) ne les touche
  jamais et conserve l'ancien comportement.
"""

from __future__ import annotations

import getpass
import os
import re

_READER_MODE: bool = False
_USERNAME: str | None = None


def set_reader_mode(enabled: bool) -> None:
    """Active ou desactive le mode reader (.exe). Appele par ``reader.py``."""
    global _READER_MODE
    _READER_MODE = bool(enabled)


def is_reader_mode() -> bool:
    """Retourne ``True`` si le serveur Dash tourne en mode reader (lecture seule)."""
    return _READER_MODE


def set_username(username: str | None) -> None:
    """Force l'identifiant analyste (sinon ``current_username`` derive depuis ``getpass``)."""
    global _USERNAME
    _USERNAME = _sanitize_username(username) if username else None


def current_username() -> str | None:
    """Retourne l'identifiant analyste pour les sidecar review_state.

    En mode reader, defaut sur ``getpass.getuser()`` si aucun nom n'a ete
    explicitement defini. En mode normal, retourne ``None`` -> comportement
    historique (fichier partage ``comparison.review_state.json``).
    """
    if _USERNAME:
        return _USERNAME
    if _READER_MODE:
        try:
            return _sanitize_username(getpass.getuser())
        except Exception:
            return _sanitize_username(os.environ.get("USERNAME") or os.environ.get("USER") or "anonymous")
    return None


_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_username(value: str) -> str:
    cleaned = _INVALID_CHARS.sub("_", value.strip())
    return cleaned or "anonymous"
