"""État d'exécution partagé pour la revue analyste."""

from __future__ import annotations

import getpass
import os
import re

_REVIEW_MODE = False
_ANALYST: str | None = None
_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def set_review_mode(enabled: bool) -> None:
    """Active ou désactive le mode de revue analyste."""
    global _REVIEW_MODE
    _REVIEW_MODE = bool(enabled)


def is_review_mode() -> bool:
    """Indique si l'application fonctionne en mode de revue analyste."""
    return _REVIEW_MODE


def set_analyst(analyst: str | None) -> None:
    """Définit l'identifiant utilisé pour le fichier individuel de revue."""
    global _ANALYST
    _ANALYST = _sanitize_analyst(analyst) if analyst else None


def current_analyst() -> str | None:
    """Retourne l'identifiant de l'analyste lorsque la revue est active."""
    if _ANALYST:
        return _ANALYST
    if not _REVIEW_MODE:
        return None
    try:
        return _sanitize_analyst(getpass.getuser())
    except Exception:
        fallback = os.environ.get("USERNAME") or os.environ.get("USER") or "anonymous"
        return _sanitize_analyst(fallback)


def configure_from_environment() -> None:
    """Configure la revue depuis les variables d'environnement multiplateformes."""
    raw_mode = os.environ.get("VIGIE_MODE_REVUE", "").strip().lower()
    analyst = os.environ.get("VIGIE_ANALYSTE", "").strip()
    resultats = os.environ.get("VIGIE_RESULTATS_DIR", "").strip()
    set_review_mode(raw_mode in {"1", "true", "yes", "on"} or bool(analyst or resultats))
    set_analyst(analyst or None)


def _sanitize_analyst(value: str) -> str:
    """Convertit un nom d'analyste en fragment de nom de fichier sûr."""
    cleaned = _INVALID_CHARS.sub("_", value.strip())
    return cleaned or "anonymous"
