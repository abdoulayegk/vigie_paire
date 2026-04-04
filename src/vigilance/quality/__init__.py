"""Utilitaires de porte de qualite pour les sorties d'extraction."""

from __future__ import annotations

from typing import Any


def run_quality_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Wrapper a import differe pour eviter le chargement du module a l'import du package."""
    from .quality_gate import run_quality_gate as _impl

    return _impl(*args, **kwargs)


__all__ = ["run_quality_gate"]
