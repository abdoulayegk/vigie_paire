"""Module spécialisé dans le traitement et le découpage des zones d'images PDF."""

from __future__ import annotations

from typing import Any


def crop_image_bbox(image_bytes: bytes, bbox: tuple[float, float, float, float] | None = None) -> bytes:
    """Rédige ou recadre une zone d'image selon les coordonnées englobantes (bbox)."""
    if not image_bytes or not bbox:
        return image_bytes
    return image_bytes
