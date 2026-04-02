"""PDF proof rendering helpers for Dash callbacks.

Extracted from dash_app/app.py. app.py re-exports all names from this module
so that all existing monkeypatches (setattr on dash_app) continue to work.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from vigilance.extraction.table_annotator import annotate_table_with_changes
from vigilance.ui_detection import get_pdf_preview
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison

from vigilance.dash_app.services.comparison_context import _normalize_pdf_paths_store
from vigilance.utils.proof_rendering import (
    normalize_proof_bbox as _shared_normalize_proof_bbox,
    render_full_proof_bytes,
)

logger = logging.getLogger(__name__)


def _filter_noise(items: list[str]) -> list[str]:
    """Filter out noise lines (dates, units, footnotes) using normalize_indicator_for_comparison."""
    return [
        x for x in items if x and normalize_indicator_for_comparison(str(x).strip())
    ]


def _cached_render_or_crop(
    pdf_path: str,
    page: int,
    scale: float,
    bbox_key: str,
    display_mode: str = "crop",
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
) -> bytes:
    """Return PNG bytes based on display_mode: crop, full (page + bbox highlight), or footnote."""
    from vigilance.utils.pdf_crop import (
        crop_footnote_region_to_bytes,
        crop_table_region_to_bytes,
    )

    bbox: list[float] | None = None
    if bbox_key:
        try:
            parsed = json.loads(bbox_key)
            if isinstance(parsed, list) and len(parsed) == 4:
                bbox = parsed
        except (json.JSONDecodeError, TypeError):
            bbox = None

    if display_mode == "full":
        raw, _, _ = render_full_proof_bytes(
            pdf_path,
            page=page,
            bbox=bbox,
            scale=scale,
            allow_full_page_fallback=True,
        )
        return raw if raw else b""

    if bbox is None:
        return b""

    try:
        if display_mode == "footnote":
            return crop_footnote_region_to_bytes(pdf_path, page, bbox, scale=scale)
        else:  # "crop" (default)
            return crop_table_region_to_bytes(
                pdf_path, page, bbox, scale=scale,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
            )
    except Exception:
        return b""


def _normalize_proof_bbox(bbox: Any) -> list[float] | None:
    """Return [l, t, r, b] in 0..1 when usable for proof rendering."""
    return _shared_normalize_proof_bbox(bbox)


def _proof_render_result(
    image_b64: str | None, status: str, mode_effective: str
) -> dict[str, str | None]:
    return {
        "image_b64": image_b64,
        "status": status,
        "mode_effective": mode_effective,
    }


def _get_proof_render_result_for_item(
    item_dict: dict,
    side: str,
    paths: dict,
    *,
    proof_display_mode: str = "crop",
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
) -> dict[str, str | None]:
    """Return proof image + availability status for one side."""
    display_mode = (proof_display_mode or "crop").strip().lower()
    if display_mode not in {"crop", "full", "footnote"}:
        display_mode = "crop"

    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    if page is None:
        return _proof_render_result(None, "page_missing", display_mode)

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    normalized_paths = _normalize_pdf_paths_store(paths)
    pdf_path = (
        normalized_paths.get("pdf_t1", "")
        if side == "t1"
        else normalized_paths.get("pdf_t2", "")
    )
    if not pdf_path:
        pdf_path = ref
    if not pdf_path:
        return _proof_render_result(None, "render_failed", display_mode)

    bbox = _normalize_proof_bbox(
        item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2")
    )
    if display_mode in {"crop", "footnote"} and bbox is None:
        return _proof_render_result(None, "bbox_missing", display_mode)

    if display_mode == "full" and bbox is None:
        raw_bytes, status, mode_effective = render_full_proof_bytes(
            pdf_path,
            page=page,
            bbox=None,
            scale=1.5,
            allow_full_page_fallback=True,
        )
        if raw_bytes:
            image_b64 = base64.b64encode(raw_bytes).decode("ascii")
            return _proof_render_result(image_b64, "ok", mode_effective)
        if status == "page_missing":
            return _proof_render_result(None, "page_missing", mode_effective)
        return _proof_render_result(None, "render_failed", mode_effective)

    image_b64 = _get_proof_image_b64_for_item(
        item_dict,
        side,
        paths,
        proof_display_mode=display_mode,
        highlight_rects=highlight_rects,
        secondary_highlight_rects=secondary_highlight_rects,
    )
    if image_b64:
        return _proof_render_result(image_b64, "ok", display_mode)
    return _proof_render_result(None, "render_failed", display_mode)


def _get_proof_image_b64_for_item(
    item_dict: dict,
    side: str,
    paths: dict,
    *,
    proof_display_mode: str = "crop",
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
) -> str | None:
    """Get proof image base64. With proof_display_mode='full' skip crop (full page); with 'crop' use bbox; with 'footnote' show only footnote region."""
    display_mode = (proof_display_mode or "crop").strip().lower()

    table_status = (item_dict.get("table_status") or "").strip().lower()
    if table_status == "stable" and display_mode == "crop":
        return _get_proof_image_b64(item_dict, side, paths)

    proof_image_path = item_dict.get("proof_image_path", "") or ""
    if side == "t2" and proof_image_path and display_mode == "crop":
        return None

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    normalized_paths = _normalize_pdf_paths_store(paths)
    path_t1 = normalized_paths.get("pdf_t1", "")
    path_t2 = normalized_paths.get("pdf_t2", "")
    pdf_path = path_t1 if side == "t1" else path_t2
    if not pdf_path:
        pdf_path = ref
    bbox = _normalize_proof_bbox(
        item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2")
    )

    # For "footnote" or "full" modes, always render from PDF (ignore pre-existing images)
    if display_mode in ("footnote", "full"):
        if not pdf_path or page is None:
            return None
        if display_mode == "footnote" and bbox is None:
            return None

        page_effective = max(1, int(page))
        bbox_key = ""
        if bbox:
            bbox_key = json.dumps(bbox)
        try:
            raw_bytes = _cached_render_or_crop(
                str(pdf_path),
                page_effective,
                1.5,
                bbox_key,
                display_mode,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
            )
            if raw_bytes:
                return base64.b64encode(raw_bytes).decode("ascii")
        except Exception as e:
            logger.warning("Render failed for mode %s: %s", display_mode, e)
        return None

    # For "crop" mode, use existing images if available
    base_img_b64: str | None = None

    # If we have any highlights we MUST bypass the physical file cache and re-render from PDF
    skip_cache = bool(highlight_rects or secondary_highlight_rects)

    if (
        not skip_cache
        and side == "t1"
        and proof_image_path
        and Path(proof_image_path).exists()
        and Path(proof_image_path).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(proof_image_path, "rb") as f:
                raw = f.read()
            base_img_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
    if (
        not skip_cache
        and base_img_b64 is None
        and ref
        and Path(ref).exists()
        and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            base_img_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    if base_img_b64 is None:
        if not pdf_path or page is None:
            return None
        if bbox is None:
            return None

        page_effective = max(1, int(page))
        bbox_key = ""
        if bbox:
            bbox_key = json.dumps(bbox)
        try:
            raw_bytes = _cached_render_or_crop(
                str(pdf_path),
                page_effective,
                1.5,
                bbox_key,
                display_mode,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
            )
            base_img_b64 = (
                base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else None
            )
        except Exception as e:
            logger.warning("Render/crop failed: %s", e)
            base_img_b64 = None

    if not base_img_b64:
        return None

    all_t1 = _filter_noise(list(item_dict.get("all_indicators_t1") or []))
    all_t2 = _filter_noise(list(item_dict.get("all_indicators_t2") or []))
    rows_added = _filter_noise(list(item_dict.get("added_indicators") or []))
    rows_removed = _filter_noise(list(item_dict.get("removed_indicators") or []))

    try:
        annotated = annotate_table_with_changes(
            image_base64=base_img_b64,
            rows_added=rows_added,
            rows_removed=rows_removed,
            all_indicators_t1=all_t1,
            all_indicators_t2=all_t2,
            is_for_t1=(side == "t1"),
            row_bboxes_t1=None,
            row_bboxes_t2=None,
        )
        if annotated:
            return base64.b64encode(annotated).decode("ascii")
    except Exception as e:
        logger.warning("Annotation failed, returning non-annotated image: %s", e)

    return base_img_b64


def _get_proof_image_b64(item_dict: dict, side: str, paths: dict) -> str | None:
    """Obtenir l'image base64 pour T1 ou T2 (PDF page ou fichier PNG)."""
    import base64 as _base64

    proof_image_path = item_dict.get("proof_image_path", "") or ""
    if (
        side == "t1"
        and proof_image_path
        and Path(proof_image_path).exists()
        and Path(proof_image_path).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(proof_image_path, "rb") as f:
                raw = f.read()
            return _base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
    if side == "t2" and proof_image_path:
        # Une preuve tableau entier couvre les deux periodes.
        return None

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    normalized_paths = _normalize_pdf_paths_store(paths)
    path_t1 = normalized_paths.get("pdf_t1", "")
    path_t2 = normalized_paths.get("pdf_t2", "")

    pdf_path = path_t1 if side == "t1" else path_t2
    if not pdf_path:
        pdf_path = ref

    if (
        ref
        and Path(ref).exists()
        and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            return _base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    if pdf_path and page is not None:
        page_effective = max(1, int(page))
        raw = get_pdf_preview(pdf_path, page_effective, scale=1.5)
        if raw:
            return _base64.b64encode(raw).decode("ascii")
    return None
