"""Modules issus du decoupage de ``vision_full_extractor.py``.

Le decoupage est mene par etapes, sans changement de comportement.
``vision_full_extractor`` reste la facade publique et re-exporte tout ce qui
etait accessible avant, y compris les symboles prives utilises par les tests.
"""

from .prompts import (
    _CONSENSUS_PROMPT_VARIANTS,
    _DEFAULT_REFERENCE_TEXT_MAX_CHARS,
    _PROMPT_BASE,
    _PROMPT_BASE_PRECISION,
    _PROMPT_JSON_STRICT,
    _PROMPT_RESCUE_SUFFIX,
    _PROMPT_VARIANT_EXHAUSTIVE,
    _PROMPT_VARIANT_PRECISION,
    _build_content,
    _build_precision_prompt,
    _build_prompt,
    _build_repair_prompt,
)
from .schema import (
    VisionFootnoteItem,
    VisionFullResponseSchema,
    VisionResponseCommonSchema,
    VisionSchemaContractError,
    _build_openai_json_schema,
    _validate_openai_strict_schema_contract,
)

__all__ = [
    "VisionFootnoteItem",
    "VisionFullResponseSchema",
    "VisionResponseCommonSchema",
    "VisionSchemaContractError",
    "_CONSENSUS_PROMPT_VARIANTS",
    "_DEFAULT_REFERENCE_TEXT_MAX_CHARS",
    "_PROMPT_BASE",
    "_PROMPT_BASE_PRECISION",
    "_PROMPT_JSON_STRICT",
    "_PROMPT_RESCUE_SUFFIX",
    "_PROMPT_VARIANT_EXHAUSTIVE",
    "_PROMPT_VARIANT_PRECISION",
    "_build_content",
    "_build_openai_json_schema",
    "_build_precision_prompt",
    "_build_prompt",
    "_build_repair_prompt",
    "_validate_openai_strict_schema_contract",
]
