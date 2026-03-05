"""Post-matching Vision validation for added tables.

Reduces false 'table added' positives by asking GPT-4o whether a table crop
represents a real new regulatory disclosure or a duplicate/artifact.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_VALIDATE_PROMPT = """Tu es un expert en rapports reglementaires bancaires canadiens.

TACHE: Cette image montre un tableau extrait d'un rapport trimestriel (T2).

Question: Ce tableau represente-t-il une VRAIE nouvelle divulgation reglementaire ou financiere
(nouveau ratio, nouvelle categorie d'exposition, nouveau tableau de divulgation)?

Ou est-ce plutot:
- un doublon ou une variante d'un tableau deja present dans le rapport precedent,
- un artefact d'extraction (en-tete repete, partie de tableau, bruit),
- une ligne ou section non pertinente?

Si VRAIE nouveaute: reponds par un JSON valide.
Si doublon/artefact/bruit: reponds par un JSON valide.

Reponse JSON stricte (rien d'autre):
{"is_real_new": true ou false, "confidence": 0.0-1.0}
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
            if "x0" in bbox and "y0" in bbox and "x1" in bbox:
                y1 = bbox.get("y1") or bbox.get("b")
                if y1 is None:
                    return None
                return [
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(y1),
                ]
    except (TypeError, ValueError, KeyError):
        pass
    return None


def validate_added_table(
    pdf_path: str,
    page: int,
    bbox: Any,
    api_key: str,
    *,
    bottom_extension: float = 0.12,
    title: str = "",
) -> tuple[bool, float]:
    """
    Call GPT-4o to validate if an added table crop represents a real new disclosure.

    Returns (is_real_new: bool, confidence: float).
    On error, returns (True, 0.0) to avoid removing valid added tables (conservative).
    """
    from ..utils.pdf_crop import crop_table_region_to_bytes

    bbox_norm = _normalize_bbox(bbox)
    if not bbox_norm:
        return True, 0.0

    if not pdf_path or page is None or page < 1:
        return True, 0.0

    try:
        crop = crop_table_region_to_bytes(
            pdf_path,
            page,
            bbox_norm,
            dpi=300,
            bottom_extension=bottom_extension,
        )
    except Exception as e:
        logger.debug("Crop failed for added table validation: %s", e)
        return True, 0.0

    if not crop:
        return True, 0.0

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        image_b64 = base64.standard_b64encode(crop).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _VALIDATE_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
        if title:
            content.insert(0, {"type": "text", "text": f"Titre du tableau: {title}\n\n"})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=256,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        is_real = bool(data.get("is_real_new", True))
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return is_real, conf
    except Exception as e:
        logger.warning("Vision added table validation API error: %s", e)
        return True, 0.0
