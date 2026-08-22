"""Constantes de reglage de l'extraction Vision.

Extraites de l'ancien monolithe Vision sans modification. Isolees ici pour
etre partagees par l'extracteur et les mixins sans import circulaire.
"""

from __future__ import annotations

from typing import Any

_EXTRACTION_METHOD = "vision_full_gpt4o"
OPENAI_VISION_TIMEOUT_SECONDS = 120.0
_RECROP_EXTENSION_INCREMENT = 0.06
_DEFAULT_MAX_COMPLETION_TOKENS = 120000
# Current extraction routing uses 120k by default and allows a 128k rescue pass when truncation is detected.
_MAX_COMPLETION_TOKENS_API_LIMIT = 128000
_RESCUE_MAX_COMPLETION_TOKENS = 128000
_MAX_COMPLETION_TOKENS_SAFE_FALLBACK = 16384
_MODEL_ROLE = "extraction_primary"
_QUALITY_PASS_CACHE_VERSION = "v2"


def resolve_vision_timeout(vision_cfg: dict[str, Any] | None) -> float:
    """Resout le timeout client Vision depuis la configuration d'extraction."""
    return float((vision_cfg or {}).get("api_timeout_sec", OPENAI_VISION_TIMEOUT_SECONDS))
