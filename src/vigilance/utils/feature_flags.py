"""Feature flags for extraction and fallback behavior."""

from __future__ import annotations

import os


def is_vision_fallback_enabled() -> bool:
    """Return True if GPT-4 Vision fallback for table extraction is enabled."""
    return os.getenv("VIGILANCE_VISION_FALLBACK", "").lower() in ("1", "true", "yes")


def extraction_cache_mode_tag() -> str:
    """Return a tag used in cache keys to invalidate cache when extraction mode changes."""
    return "v1"
