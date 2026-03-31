"""Unit tests for UI detection fallback logic."""

from __future__ import annotations

from vigilance.ui_detection import _fallback_sections


def test_fallback_sections_returns_single_unknown_range() -> None:
    """Fallback returns one unknown_section spanning all pages when no sections detected."""
    sections = _fallback_sections(12)

    assert sections == [
        {
            "type": "unknown_section",
            "label": "Unknown Section",
            "start_page": 1,
            "end_page": 12,
        }
    ]
