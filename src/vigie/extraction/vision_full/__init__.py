"""Extraction Vision complete des tableaux bancaires."""

from vigie.extraction.vision_full.constants import OPENAI_VISION_TIMEOUT_SECONDS
from vigie.extraction.vision_full.extractor import VisionFullExtractor
from vigie.extraction.vision_full.parsing import _parse_vision_result
from vigie.extraction.vision_full.prompts import (
    _PROMPT_BASE,
    _PROMPT_BASE_PRECISION,
)
from vigie.extraction.vision_full.quality_grading import (
    _grade_extraction_quality,
    _select_targeted_rescue_variant,
)
from vigie.extraction.vision_full.quality_heuristics import (
    _extract_native_text_indicators,
    _normalize_footnote_marker_id,
    _structural_indicator_count,
    _viable_indicator_count,
)
from vigie.extraction.vision_full.result import VisionFullResult
from vigie.extraction.vision_full.schema import (
    VisionFootnoteItem,
    VisionFullResponseSchema,
    VisionResponseCommonSchema,
    VisionSchemaContractError,
    _build_openai_json_schema,
    _validate_openai_strict_schema_contract,
)

__all__ = [
    "OPENAI_VISION_TIMEOUT_SECONDS",
    "VisionFootnoteItem",
    "VisionFullExtractor",
    "VisionFullResponseSchema",
    "VisionFullResult",
    "VisionResponseCommonSchema",
    "VisionSchemaContractError",
    "_PROMPT_BASE",
    "_PROMPT_BASE_PRECISION",
    "_build_openai_json_schema",
    "_extract_native_text_indicators",
    "_grade_extraction_quality",
    "_normalize_footnote_marker_id",
    "_parse_vision_result",
    "_select_targeted_rescue_variant",
    "_structural_indicator_count",
    "_validate_openai_strict_schema_contract",
    "_viable_indicator_count",
]
