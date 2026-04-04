"""Drapeaux de fonctionnalite pour l'extraction et le comportement du cache."""

from __future__ import annotations


def extraction_cache_mode_tag() -> str:
    """Retourne un tag utilise dans les cles de cache pour invalider le cache lors d'un changement de mode d'extraction."""
    return "v1"
