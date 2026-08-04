"""Règles d'alignement propres aux unités atomiques parent-enfants."""

from __future__ import annotations

from vigie.analyse_texte.chunking import TextChunk


def atomic_similarity_text(chunk: TextChunk) -> str:
    """Retourne le contenu sémantique sans marqueur de liste."""
    return str(chunk.comparison_text or chunk.text).strip()


def _normalized_atomic_marker(chunk: TextChunk) -> str:
    marker = str(chunk.atomic_marker or "").strip().lower()
    return marker.strip("().")


def atomic_marker_match_priority(source: TextChunk, target: TextChunk) -> bool:
    """Le marqueur départage seulement des contenus déjà comparables."""
    source_marker = _normalized_atomic_marker(source)
    target_marker = _normalized_atomic_marker(target)
    return bool(
        source.unit_role == "item"
        and target.unit_role == "item"
        and source_marker
        and source_marker != "-"
        and source_marker == target_marker
    )


def atomic_roles_compatible(source: TextChunk, target: TextChunk) -> bool:
    """Empêche qu'une introduction soit associée à un item de sa liste."""
    return {source.unit_role, target.unit_role} != {"context", "item"}
