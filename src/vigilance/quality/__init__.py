"""Quality gate utilities for extraction outputs."""

from __future__ import annotations

from typing import Any


def run_quality_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy import wrapper to avoid importing the module during package import."""
    from .quality_gate import run_quality_gate as _impl

    return _impl(*args, **kwargs)


__all__ = ["run_quality_gate"]
