"""Transitions d'etat de la revue analyste."""

from __future__ import annotations

from dataclasses import replace

from vigilance.review_models import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    ReviewItem,
)

_ALLOWED = {REVIEW_STATUS_PENDING, REVIEW_STATUS_APPROVED, REVIEW_STATUS_REJECTED}


def set_review_status(item: ReviewItem, status: str) -> ReviewItem:
    """Retourne une copie de ``item`` avec le statut de revue demande.

    Args:
        item: Element de revue a modifier.
        status: Nouveau statut (``pending``, ``approved`` ou ``rejected``).

    Returns:
        Copie de l'element avec le statut normalise.
    """
    normalized = (status or "").strip().lower()
    if normalized not in _ALLOWED:
        normalized = REVIEW_STATUS_PENDING
    return replace(item, review_status=normalized)
