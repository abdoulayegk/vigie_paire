"""Package des utilitaires modulaires Vision (prompts, schémas Pydantic, découpage d'images)."""

from __future__ import annotations

from vigilance.extraction.vision.vision_crop import crop_image_bbox
from vigilance.extraction.vision.vision_prompts import (
    build_vision_system_prompt,
    build_vision_user_prompt,
)
from vigilance.extraction.vision.vision_schema import (
    VisionFootnoteItem,
    VisionIndicatorItem,
    VisionTableResponse,
)

__all__ = [
    "build_vision_system_prompt",
    "build_vision_user_prompt",
    "VisionIndicatorItem",
    "VisionFootnoteItem",
    "VisionTableResponse",
    "crop_image_bbox",
]
