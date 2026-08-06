"""Constantes de reglage de l'extraction Vision.

Extraites de l'ancien monolithe Vision sans modification. Isolees ici pour
etre partagees par l'extracteur et les mixins sans import circulaire.
"""

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
