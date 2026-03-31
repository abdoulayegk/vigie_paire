"""Review state transitions."""

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
    """Return a copy of ``item`` with the requested review status."""
    normalized = (status or "").strip().lower()
    if normalized not in _ALLOWED:
        normalized = REVIEW_STATUS_PENDING
    return replace(item, review_status=normalized)
