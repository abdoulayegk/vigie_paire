"""Integration hook for Vision first-column fallback. Called from docling_processor."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..utils.pdf_crop import crop_table_image
from .vision_cache import (
    get_vision_cache_dir,
    cache_get,
    cache_put,
    compute_pdf_sha256,
    make_cache_key,
)
from .vision_gating import is_table_extraction_suspect

logger = logging.getLogger(__name__)


def _try_vision_first_column_fallback(
    extracted_table: Any,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    page_num: int,
    table_bbox: list[float],
) -> None:
    """
    Optionally replace first_column_indicators with Vision result when extraction is suspect.
    Mutates extracted_table in place if accepted. Never raises.
    """
    if not is_table_extraction_suspect(extracted_table):
        return

    keep_crop = os.environ.get("ENABLE_TABLE_CROP_DUMP") == "1"
    if keep_crop:
        from .vision_cache import get_vision_crop_dir

        crop_dir = Path(get_vision_crop_dir()) / f"{bank_code}_{quarter}_{year}"
        crop_path = crop_dir / f"{extracted_table.table_id}_p{page_num}.png"
        crop_path = str(crop_path)
    else:
        fd, crop_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    try:
        ok = crop_table_image(pdf_path, page_num, table_bbox, crop_path, dpi=300)
        if not ok:
            logger.debug(
                "Vision fallback: crop failed table_id=%s page=%s",
                extracted_table.table_id,
                page_num,
            )
            return

        pdf_sha = compute_pdf_sha256(pdf_path)
        if not pdf_sha:
            return
        cache_key = make_cache_key(pdf_sha, page_num, table_bbox)
        if not cache_key:
            return

        cache_dir = get_vision_cache_dir()
        cached = cache_get(cache_dir, cache_key)

        valid_from_cache = False
        indicators_raw: list[str] = []
        confidence = 0.0
        cache_hit = False

        if cached:
            raw_indicators = cached.get("indicators")
            if isinstance(raw_indicators, list) and all(isinstance(x, str) for x in raw_indicators):
                try:
                    conf = float(cached.get("confidence", 0.0))
                    confidence = max(0.0, min(1.0, conf))
                    indicators_raw = list(raw_indicators)
                    valid_from_cache = True
                    cache_hit = True
                except (TypeError, ValueError):
                    pass
            if not valid_from_cache:
                logger.warning(
                    "Vision cache: invalid payload for key %s — ignoring, proceeding as cache miss",
                    cache_key[:80] + "..." if len(cache_key) > 80 else cache_key,
                )

        if not valid_from_cache:
            try:
                from .gpt4o_vision_first_column import GPT4oVisionFirstColumnProvider

                from ..utils.genai import get_openai_api_key

                api_key = get_openai_api_key()
                if not api_key:
                    logger.debug("Vision fallback: no OPENAI_API_KEY")
                    return
                provider = GPT4oVisionFirstColumnProvider(api_key=api_key)
                result = provider.extract_first_column(crop_path)
                indicators_raw = result.indicators_raw
                confidence = result.confidence
                cache_put(
                    cache_dir,
                    cache_key,
                    {"indicators": indicators_raw, "confidence": confidence},
                )
            except Exception as e:
                logger.debug("Vision fallback provider error: %s", e)
                return

        dm = getattr(extracted_table, "debug_metrics", None) or {}
        docling_count = dm.get("indicator_count", len(extracted_table.first_column_indicators or []))

        reason_parts = []
        if dm.get("indicator_count") is not None and dm.get("indicator_count", 0) < 3:
            reason_parts.append("low_count")
        if dm.get("duplicate_ratio") is not None and dm.get("duplicate_ratio", 0) > 0.2:
            reason_parts.append("high_dup")
        if dm.get("header_like_ratio") is not None and dm.get("header_like_ratio", 0) > 0.2:
            reason_parts.append("high_header")
        if dm.get("merge_count") is not None and dm.get("merge_count", 0) > 8:
            reason_parts.append("high_merge")
        if dm.get("table_quality_score") is not None and dm.get("table_quality_score", 1) < 0.5:
            reason_parts.append("low_quality")
        reason = ",".join(reason_parts) or "suspect"

        accepted = False
        if confidence >= 0.80 and indicators_raw:
            lo = 0.5 * docling_count if docling_count > 0 else 1
            hi = 2.0 * docling_count if docling_count > 0 else 50
            if lo <= len(indicators_raw) <= hi:
                accepted = True
                from ..utils.indicator_cleaner import normalize_indicator_for_comparison

                extracted_table.first_column_indicators_raw = list(indicators_raw)
                extracted_table.first_column_indicators = [
                    normalize_indicator_for_comparison(x) for x in indicators_raw
                ]
                extracted_table.extraction_method = "vision_fallback_gpt4o"

        logger.info(
            "Vision fallback: table_id=%s page=%s reason=%s cache=%s accepted=%s conf=%.2f",
            extracted_table.table_id,
            page_num,
            reason,
            "hit" if cache_hit else "miss",
            accepted,
            confidence,
        )
    finally:
        if not keep_crop and os.path.exists(crop_path):
            try:
                os.remove(crop_path)
            except Exception:
                pass
