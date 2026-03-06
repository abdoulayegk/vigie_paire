"""Feature flags for extraction and cache behavior."""

from __future__ import annotations


def extraction_cache_mode_tag() -> str:
    """Return a tag used in cache keys to invalidate cache when extraction mode changes."""
    return "v1"
