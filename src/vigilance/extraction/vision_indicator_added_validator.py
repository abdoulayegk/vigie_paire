"""Post-matching Vision validation for indicator added/removed.

Reduces false positives by asking GPT-4o Vision whether a row (cropped from T2 or T1)
represents an indicator already present in the opposite table. Uses row_bboxes from
extract_row_bboxes_from_pdf when available; falls back to GenAI otherwise.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_VALIDATE_ADDED_PROMPT = """Tu es un expert en rapports reglementaires bancaires canadiens.

TACHE: L'image a gauche montre une LIGNE d'un tableau (indicateur + valeurs) extraite du rapport T2.
L'image a droite montre le TABLEAU COMPLET du rapport T1 (meme section, meme type de divulgation).

Question: Cette ligne (gauche) represente-t-elle un indicateur DEJA present dans le tableau (droite)?
(meme concept reglementaire ou financier, eventuellement reformule ou renumero)

Si OUI (deja present, reformulation): reponds par un JSON valide.
Si NON (vraie nouveaute): reponds par un JSON valide.

Reponse JSON stricte (rien d'autre):
{"same_concept": true ou false, "confidence": 0.0-1.0}
"""

_VALIDATE_REMOVED_PROMPT = """Tu es un expert en rapports reglementaires bancaires canadiens.

TACHE: L'image a gauche montre une LIGNE d'un tableau (indicateur + valeurs) extraite du rapport T1.
L'image a droite montre le TABLEAU COMPLET du rapport T2 (meme section, meme type de divulgation).

Question: Cette ligne (gauche) represente-t-elle un indicateur DEJA present dans le tableau (droite)?
(meme concept reglementaire ou financier, eventuellement reformule ou renumero)

Si OUI (deja present dans T2): reponds par un JSON valide.
Si NON (vraiment disparu): reponds par un JSON valide.

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


def _bbox_norm_to_pdf_coords(
    bbox_norm: list[float], page_width: float, page_height: float
) -> tuple[float, float, float, float]:
    """Convert [l,t,r,b] 0-1 to (x0,y0,x1,y1) in PDF points."""
    l_val, t_val, r_val, b_val = bbox_norm[:4]
    return (
        l_val * page_width,
        t_val * page_height,
        r_val * page_width,
        b_val * page_height,
    )


def _get_page_dimensions(pdf_path: str, page_number: int) -> tuple[float, float] | None:
    """Return (width, height) in PDF points or None."""
    try:
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(pdf_path)
        try:
            if page_number < 1 or page_number > len(doc):
                return None
            page = doc[page_number - 1]
            rect = page.rect
            return rect.width, rect.height
        finally:
            doc.close()
    except Exception as e:
        logger.debug("Failed to get page dimensions: %s", e)
        return None


def _create_side_by_side(
    crop1_bytes: bytes,
    crop2_bytes: bytes,
    gap: int = 20,
) -> bytes | None:
    """Combine two crop images side-by-side (row left, table right)."""
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


def _find_row_bbox_for_indicator(
    row_bboxes: list[tuple[str, float, float]],
    indicator: str,
    table_bbox_norm: list[float],
    page_width: float,
    page_height: float,
) -> list[float] | None:
    """Find row bbox for indicator and return normalized [l,t,r,b] or None."""
    if not row_bboxes:
        return None
    norm_ind = " ".join(indicator.lower().split()).strip()
    matched: tuple[str, float, float] | None = None
    for rb_ind, y0, y1 in row_bboxes:
        norm_rb = " ".join(str(rb_ind).lower().split()).strip()
        if norm_ind == norm_rb or norm_ind in norm_rb or norm_rb in norm_ind:
            matched = (rb_ind, y0, y1)
            break
    if matched is None:
        for rb_ind, y0, y1 in row_bboxes:
            if len(norm_ind) > 5 and len(rb_ind) > 5:
                w1, w2 = set(norm_ind.split()), set(str(rb_ind).lower().split())
                if w1 & w2 and len(w1 & w2) / max(len(w1), len(w2)) > 0.3:
                    matched = (rb_ind, y0, y1)
                    break
    if matched is None:
        return None
    _, y0, y1 = matched
    l_val, t_val, r_val, b_val = table_bbox_norm
    row_t_norm = y0 / page_height
    row_b_norm = y1 / page_height
    row_bbox_norm = [l_val, row_t_norm, r_val, row_b_norm]
    if row_b_norm <= row_t_norm or row_t_norm < 0 or row_b_norm > 1.01:
        return None
    return row_bbox_norm


def validate_indicator_added_vision(
    indicator: str,
    indicator_type: str,
    pdf_row_path: str,
    page_row: int,
    bbox_table_other_norm: list[float],
    pdf_table_path: str,
    page_table: int,
    api_key: str,
    *,
    row_bboxes: list[tuple[str, float, float]] | None = None,
    table_row_bbox_norm: list[float] | None = None,
    bottom_extension: float = 0.05,
) -> tuple[bool, float]:
    """Validate a single added/removed indicator via Vision.

    Args:
        indicator: The indicator label to validate
        indicator_type: "added" or "removed"
        pdf_row_path: PDF path for the table containing the row
        page_row: Page number (1-based) for the row
        bbox_table_other_norm: Normalized bbox [l,t,r,b] of the opposite table
        pdf_table_path: PDF path for the opposite table
        page_table: Page number for the opposite table
        api_key: OpenAI API key
        row_bboxes: Optional pre-extracted row bboxes (indicator, y0, y1)
        table_row_bbox_norm: Optional pre-normalized table bbox for the row's table
        bottom_extension: Extra height for row crop

    Returns:
        (same_concept: bool, confidence: float, called_api: bool).
        On error or when row bbox not found, returns (False, 0.0, False) to keep the indicator (conservative).
        called_api=True only when the Vision API was actually invoked.
    """
    from ..utils.pdf_crop import crop_table_region_to_bytes

    dims = _get_page_dimensions(pdf_row_path, page_row)
    if not dims:
        return False, 0.0, False
    page_width, page_height = dims

    if table_row_bbox_norm is None:
        return False, 0.0, False

    row_bbox_norm: list[float] | None = None
    if row_bboxes:
        row_bbox_norm = _find_row_bbox_for_indicator(
            row_bboxes,
            indicator,
            table_row_bbox_norm,
            page_width,
            page_height,
        )
    if not row_bbox_norm:
        return False, 0.0, False

    try:
        row_crop = crop_table_region_to_bytes(
            pdf_row_path,
            page_row,
            row_bbox_norm,
            dpi=300,
            bottom_extension=bottom_extension,
        )
        table_crop = crop_table_region_to_bytes(
            pdf_table_path,
            page_table,
            bbox_table_other_norm,
            dpi=300,
            bottom_extension=0.12,
        )
    except Exception as e:
        logger.debug("Crop failed for indicator validation: %s", e)
        return False, 0.0, False

    if not row_crop or not table_crop:
        return False, 0.0, False

    combined = _create_side_by_side(row_crop, table_crop)
    if not combined:
        return False, 0.0, False

    prompt = (
        _VALIDATE_ADDED_PROMPT if indicator_type == "added" else _VALIDATE_REMOVED_PROMPT
    )

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
                        {"type": "text", "text": prompt},
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
        same = bool(data.get("same_concept", False))
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return same, conf, True
    except Exception as e:
        logger.warning("Vision indicator validation API error: %s", e)
        return False, 0.0, False


def try_vision_validate_indicators(
    added: list[str],
    removed: list[str],
    table_t1: Any,
    table_t2: Any,
    pdf_path_t1: str,
    pdf_path_t2: str,
    api_key: str,
    confidence_min: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Try Vision validation for added/removed indicators.

    Uses extract_row_bboxes_from_pdf when available. Falls back to (added, removed, {})
    if Vision cannot run (caller should use GenAI).

    Returns:
        (filtered_added, filtered_removed, stats)
    """
    from .row_bbox_extractor import extract_row_bboxes_from_pdf

    stats: dict[str, Any] = {
        "vision_calls": 0,
        "vision_filtered_added": 0,
        "vision_filtered_removed": 0,
        "vision_fallback_reason": None,
        "could_validate_added": False,
        "could_validate_removed": False,
    }

    bbox_t1 = _normalize_bbox(getattr(table_t1, "bbox", None))
    bbox_t2 = _normalize_bbox(getattr(table_t2, "bbox", None))
    if not bbox_t1 or not bbox_t2:
        stats["vision_fallback_reason"] = "missing_bbox"
        return list(added), list(removed), stats

    dims_t1 = _get_page_dimensions(pdf_path_t1, getattr(table_t1, "page_pdf", 0) or 0)
    dims_t2 = _get_page_dimensions(pdf_path_t2, getattr(table_t2, "page_pdf", 0) or 0)
    if not dims_t1 or not dims_t2:
        stats["vision_fallback_reason"] = "page_dimensions"
        return list(added), list(removed), stats

    w1, h1 = dims_t1
    w2, h2 = dims_t2
    table_bbox_t1_pdf = _bbox_norm_to_pdf_coords(bbox_t1, w1, h1)
    table_bbox_t2_pdf = _bbox_norm_to_pdf_coords(bbox_t2, w2, h2)

    indicators_t1 = list(getattr(table_t1, "first_column_indicators", None) or [])
    indicators_t2 = list(getattr(table_t2, "first_column_indicators", None) or [])

    row_bboxes_t1 = extract_row_bboxes_from_pdf(
        pdf_path_t1,
        getattr(table_t1, "page_pdf", 0) or 0,
        table_bbox_t1_pdf,
        indicators_t1,
    )
    row_bboxes_t2 = extract_row_bboxes_from_pdf(
        pdf_path_t2,
        getattr(table_t2, "page_pdf", 0) or 0,
        table_bbox_t2_pdf,
        indicators_t2,
    )

    to_remove_added: set[str] = set()
    to_remove_removed: set[str] = set()

    for ind in added:
        if not row_bboxes_t2:
            break
        same, conf, called_api = validate_indicator_added_vision(
            ind,
            "added",
            pdf_path_t2,
            getattr(table_t2, "page_pdf", 0) or 0,
            bbox_t1,
            pdf_path_t1,
            getattr(table_t1, "page_pdf", 0) or 0,
            api_key,
            row_bboxes=row_bboxes_t2,
            table_row_bbox_norm=bbox_t2,
        )
        if called_api:
            stats["vision_calls"] += 1
            stats["could_validate_added"] = True
        if same and conf >= confidence_min:
            to_remove_added.add(ind)
            stats["vision_filtered_added"] += 1

    for ind in removed:
        if not row_bboxes_t1:
            break
        same, conf, called_api = validate_indicator_added_vision(
            ind,
            "removed",
            pdf_path_t1,
            getattr(table_t1, "page_pdf", 0) or 0,
            bbox_t2,
            pdf_path_t2,
            getattr(table_t2, "page_pdf", 0) or 0,
            api_key,
            row_bboxes=row_bboxes_t1,
            table_row_bbox_norm=bbox_t1,
        )
        if called_api:
            stats["vision_calls"] += 1
            stats["could_validate_removed"] = True
        if same and conf >= confidence_min:
            to_remove_removed.add(ind)
            stats["vision_filtered_removed"] += 1

    if not row_bboxes_t1 and not row_bboxes_t2 and (added or removed):
        stats["vision_fallback_reason"] = "row_bboxes_empty"

    filtered_added = [x for x in added if x not in to_remove_added]
    filtered_removed = [x for x in removed if x not in to_remove_removed]
    return filtered_added, filtered_removed, stats
