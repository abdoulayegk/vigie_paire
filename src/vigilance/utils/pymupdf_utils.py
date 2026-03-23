"""Helpers for configuring PyMuPDF / MuPDF process-wide behavior."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_MUPDF_CONFIGURED = False


def configure_mupdf_runtime(fitz: Any) -> None:
    """Reduce noisy MuPDF stderr warnings while keeping Python exceptions intact."""
    global _MUPDF_CONFIGURED
    if _MUPDF_CONFIGURED:
        return
    tools = getattr(fitz, "TOOLS", None)
    if tools is None:
        _MUPDF_CONFIGURED = True
        return
    try:
        display_warnings = getattr(tools, "mupdf_display_warnings", None)
        if callable(display_warnings):
            display_warnings(False)
        reset_warnings = getattr(tools, "reset_mupdf_warnings", None)
        if callable(reset_warnings):
            reset_warnings()
    except Exception as exc:
        logger.debug("Impossible de configurer MuPDF proprement: %s", exc)
    _MUPDF_CONFIGURED = True

