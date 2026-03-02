"""Integration hook for Vision first-column fallback. Called from docling_processor."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..utils.indicator_cleaner import normalize_indicator_for_comparison
from ..utils.indicator_normalizer import get_canonical_text, get_token_sorted_text
from ..utils.matching_normalizer import is_non_indicator_line
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


def _build_vision_indicator_entries(labels: list[str]) -> list[dict[str, Any]]:
    """Build canonical indicator entries from Vision output (same contract as Docling)."""
    entries: list[dict[str, Any]] = []
    for label in labels:
        raw = str(label or "").strip()
        if not raw:
            continue
        canonical = get_canonical_text(raw)
        token_sorted = get_token_sorted_text(raw)
        entries.append(
            {
                "raw_text": raw,
                "canonical_text": canonical,
                "token_sorted": token_sorted,
                "anchor_tokens": token_sorted.split()[:12] if token_sorted else [],
                "line_role": "header_like" if is_non_indicator_line(raw) else "indicator",
            }
        )
    return entries


def _compute_agreement_signals(
    docling_indicators: list[str],
    vision_indicators: list[str],
) -> dict[str, Any]:
    """Compute agreement signals between Docling and Vision extractions."""
    doc_count = len(docling_indicators)
    vis_count = len(vision_indicators)
    if doc_count == 0 and vis_count == 0:
        return {"count_ratio": 1.0, "anchor_overlap": 1.0, "agreement": "both_empty"}

    count_ratio = min(doc_count, vis_count) / max(doc_count, vis_count) if max(doc_count, vis_count) > 0 else 0.0

    doc_tokens = set()
    for ind in docling_indicators:
        doc_tokens.update(get_token_sorted_text(ind).split())
    vis_tokens = set()
    for ind in vision_indicators:
        vis_tokens.update(get_token_sorted_text(ind).split())

    if doc_tokens or vis_tokens:
        anchor_overlap = len(doc_tokens & vis_tokens) / len(doc_tokens | vis_tokens) if (doc_tokens | vis_tokens) else 0.0
    else:
        anchor_overlap = 1.0

    if count_ratio >= 0.75 and anchor_overlap >= 0.60:
        agreement = "agree"
    elif count_ratio < 0.50 or anchor_overlap < 0.30:
        agreement = "strong_disagree"
    else:
        agreement = "disagree"

    return {
        "count_ratio": round(count_ratio, 3),
        "anchor_overlap": round(anchor_overlap, 3),
        "docling_count": doc_count,
        "vision_count": vis_count,
        "agreement": agreement,
    }


_FIRST_COL_WIDTH_RATIO = 0.35


def _narrow_crop_for_first_column(image_path: str) -> None:
    """Re-crop a table image to keep only the leftmost ~35% (first column).

    Reduces token usage and focuses GPT-4o attention on indicator labels.
    Overwrites the file in place. Never raises.
    """
    try:
        from PIL import Image

        img = Image.open(image_path)
        w, h = img.size
        first_col = img.crop((0, 0, int(w * _FIRST_COL_WIDTH_RATIO), h))
        first_col.save(image_path)
    except Exception as e:
        logger.debug("Narrow first-column crop failed (using full image): %s", e)


def _try_vision_first_column_fallback(
    extracted_table: Any,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    page_num: int,
    table_bbox: list[float],
    *,
    force_vision: bool = False,
    vision_min_confidence: float = 0.80,
    vision_count_ratio_min: float = 0.50,
    vision_count_ratio_max: float = 2.00,
) -> None:
    """
    Vision-priority arbitration for first-column indicators.

    When force_vision is False, only runs if extraction is suspect.
    When force_vision is True, always runs Vision and applies arbitration.

    Thresholds can be overridden via config (bank_profiles.yaml matching_thresholds).

    Mutates extracted_table in place. Never raises.
    """
    if not force_vision and not is_table_extraction_suspect(extracted_table):
        return

    keep_crop = os.environ.get("ENABLE_TABLE_CROP_DUMP") == "1"
    crop_path_str: str
    if keep_crop:
        from .vision_cache import get_vision_crop_dir

        crop_dir = Path(get_vision_crop_dir()) / f"{bank_code}_{quarter}_{year}"
        crop_path_str = str(crop_dir / f"{extracted_table.table_id}_p{page_num}.png")
    else:
        fd, crop_path_str = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    try:
        ok = crop_table_image(pdf_path, page_num, table_bbox, crop_path_str, dpi=300)
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
                    "Vision cache: invalid payload for key %s -- ignoring, proceeding as cache miss",
                    cache_key[:80] + "..." if len(cache_key) > 80 else cache_key,
                )

        if not valid_from_cache:
            try:
                from .gpt4o_vision_first_column import GPT4oVisionFirstColumnProvider

                from ..utils.genai import get_openai_api_key

                _narrow_crop_for_first_column(crop_path_str)

                api_key = get_openai_api_key()
                if not api_key:
                    logger.debug("Vision fallback: no OPENAI_API_KEY")
                    return
                provider = GPT4oVisionFirstColumnProvider(api_key=api_key)
                result = provider.extract_first_column(crop_path_str)
                indicators_raw = result.indicators_raw
                confidence = result.confidence
                if indicators_raw:
                    cache_put(
                        cache_dir,
                        cache_key,
                        {"indicators": indicators_raw, "confidence": confidence},
                    )
                else:
                    logger.warning(
                        "Vision fallback: GPT-4o returned 0 indicators for table_id=%s page=%s -- NOT caching empty result",
                        extracted_table.table_id,
                        page_num,
                    )
            except Exception as e:
                logger.debug("Vision fallback provider error: %s", e)
                return

        dm = getattr(extracted_table, "debug_metrics", None) or {}
        docling_raw = list(getattr(extracted_table, "first_column_indicators_raw", None) or [])
        docling_clean = list(getattr(extracted_table, "first_column_indicators", None) or [])
        docling_count = dm.get("indicator_count", len(docling_clean))

        signals = _compute_agreement_signals(docling_clean, indicators_raw)

        reason_parts = []
        if dm.get("indicator_count") is not None and dm.get("indicator_count", 0) < 3:
            reason_parts.append("low_count")
        if dm.get("duplicate_ratio") is not None and dm.get("duplicate_ratio", 0) > 0.2:
            reason_parts.append("high_dup")
        if dm.get("header_like_ratio") is not None and dm.get("header_like_ratio", 0) > 0.2:
            reason_parts.append("high_header")
        if dm.get("line_reconstruction_merges") is not None and dm.get("line_reconstruction_merges", 0) > 8:
            reason_parts.append("high_merge")
        if dm.get("table_quality_score") is not None and dm.get("table_quality_score", 1) < 0.5:
            reason_parts.append("low_quality")
        suspect_reason = ",".join(reason_parts) or "suspect"

        decision = "rejected"
        decision_reason = ""

        if confidence >= vision_min_confidence and indicators_raw:
            lo = vision_count_ratio_min * docling_count if docling_count > 0 else 1
            hi = vision_count_ratio_max * docling_count if docling_count > 0 else 50
            count_ok = lo <= len(indicators_raw) <= hi

            if signals["agreement"] == "agree":
                decision = "accepted_agree"
                decision_reason = "docling_vision_agree"
                _apply_vision_output(extracted_table, indicators_raw)
            elif count_ok:
                decision = "accepted_vision_priority"
                decision_reason = f"vision_priority_on_{signals['agreement']}"
                _apply_vision_output(extracted_table, indicators_raw)
            else:
                decision = "rejected"
                decision_reason = f"count_out_of_range_lo={lo:.0f}_hi={hi:.0f}_vis={len(indicators_raw)}"
        elif confidence < vision_min_confidence and indicators_raw:
            decision = "rejected_low_confidence"
            decision_reason = f"confidence={confidence:.2f}<0.80"
        else:
            decision = "rejected_empty"
            decision_reason = "no_vision_indicators"

        accepted = decision.startswith("accepted")

        vision_entries = _build_vision_indicator_entries(indicators_raw) if indicators_raw else []
        dm["vision_arbitration"] = {
            "decision": decision,
            "decision_reason": decision_reason,
            "suspect_reason": suspect_reason,
            "confidence": round(confidence, 3),
            "cache_hit": cache_hit,
            "agreement_signals": signals,
            "vision_indicator_count": len(indicators_raw),
            "docling_indicator_count": docling_count,
            "vision_indicator_entries_sample": vision_entries[:10],
        }

        logger.info(
            "Vision arbitration: table_id=%s page=%s decision=%s reason=%s cache=%s conf=%.2f agree=%s",
            extracted_table.table_id,
            page_num,
            decision,
            decision_reason,
            "hit" if cache_hit else "miss",
            confidence,
            signals["agreement"],
        )
    finally:
        if not keep_crop and os.path.exists(crop_path_str):
            try:
                os.remove(crop_path_str)
            except Exception:
                pass


def _apply_vision_output(extracted_table: Any, indicators_raw: list[str]) -> None:
    """Apply Vision indicators to table, replacing Docling output."""
    extracted_table.first_column_indicators_raw = list(indicators_raw)
    extracted_table.first_column_indicators = [
        normalize_indicator_for_comparison(x) for x in indicators_raw
    ]
    extracted_table.extraction_method = "vision_fallback_gpt4o"
