"""Post-matching Vision validation: GPT-4o confirms same regulatory concept for table pairs.

Reduces false positives by rejecting pairs where Vision determines different concepts.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_VALIDATE_PROMPT = """Tu es un expert en rapports reglementaires bancaires canadiens.

TACHE: Ces deux images montrent des tableaux extraits de rapports trimestriels (T1 a gauche, T2 a droite).

Question: Ces deux tableaux correspondent-ils au MEME concept reglementaire (ex: meme ratio, meme categorie d'exposition, meme divulgation)?

Si OUI (meme concept, meme theme, eventuellement renumeroes ou legerement reformules): reponds par un JSON valide.
Si NON (themes differents, tableaux non homologues): reponds par un JSON valide.

Reponse JSON stricte (rien d'autre):
{"same_concept": true ou false, "confidence": 0.0-1.0}
"""


def _normalize_bbox(bbox: Any) -> list[float] | None:
    """Normalize bbox to [l,t,r,b] 0-1."""
    if bbox is None:
        return None
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        if isinstance(bbox, dict):
            if "l" in bbox and "t" in bbox:
                return [
                    float(bbox["l"]),
                    float(bbox["t"]),
                    float(bbox["r"]),
                    float(bbox["b"]),
                ]
            if "x0" in bbox and "y0" in bbox:
                return [
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                ]
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _create_side_by_side(
    crop1_bytes: bytes,
    crop2_bytes: bytes,
    gap: int = 20,
) -> bytes | None:
    """Combine two crop images side-by-side (T1 left, T2 right)."""
    try:
        from PIL import Image

        img1 = Image.open(io.BytesIO(crop1_bytes))
        img2 = Image.open(io.BytesIO(crop2_bytes))
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")
        max_h = max(img1.height, img2.height)
        total_w = img1.width + gap + img2.width
        combined = Image.new("RGB", (total_w, max_h), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width + gap, 0))
        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.debug("Side-by-side creation failed: %s", e)
        return None


def validate_pair_same_concept(
    pdf_path_t1: str,
    page_t1: int,
    bbox_t1: Any,
    pdf_path_t2: str,
    page_t2: int,
    bbox_t2: Any,
    api_key: str,
    bottom_extension: float = 0.12,
) -> tuple[bool, float]:
    """
    Call GPT-4o to validate if two table crops represent the same regulatory concept.

    Returns (same_concept: bool, confidence: float).
    On error, returns (True, 0.0) to avoid rejecting valid pairs.
    """
    from ..utils.pdf_crop import crop_table_region_to_bytes

    b1 = _normalize_bbox(bbox_t1)
    b2 = _normalize_bbox(bbox_t2)
    if not b1 or not b2:
        return True, 0.0

    try:
        crop1 = crop_table_region_to_bytes(
            pdf_path_t1, page_t1, b1, dpi=300, bottom_extension=bottom_extension
        )
        crop2 = crop_table_region_to_bytes(
            pdf_path_t2, page_t2, b2, dpi=300, bottom_extension=bottom_extension
        )
    except Exception as e:
        logger.debug("Crop failed for pair validation: %s", e)
        return True, 0.0

    if not crop1 or not crop2:
        return True, 0.0

    combined = _create_side_by_side(crop1, crop2)
    if not combined:
        return True, 0.0

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        image_b64 = base64.standard_b64encode(combined).decode("ascii")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VALIDATE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=256,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        same = bool(data.get("same_concept", True))
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return same, conf
    except Exception as e:
        logger.warning("Vision pair validation API error: %s", e)
        return True, 0.0
