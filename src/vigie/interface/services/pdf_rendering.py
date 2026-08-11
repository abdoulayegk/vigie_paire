"""Rendu de preuves PDF pour les callbacks Dash."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from vigie.extraction.table_annotator import annotate_table_with_changes
from vigie.interface.services.comparison_context import _normalize_pdf_paths_store
from vigie.interface.ui_detection import get_pdf_preview
from vigie.support.utils import pdf_crop
from vigie.support.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigie.support.utils.proof_rendering import (
    PROOF_RENDER_DPI,
    render_full_proof_bytes,
)
from vigie.support.utils.proof_rendering import (
    normalize_proof_bbox as _shared_normalize_proof_bbox,
)

logger = logging.getLogger(__name__)

PROOF_HIGHLIGHT_COLOR_T1: tuple[float, float, float] = (0.86, 0.21, 0.33)
PROOF_HIGHLIGHT_COLOR_T2: tuple[float, float, float] = (0.20, 0.72, 0.36)


def _filter_noise(items: list[str]) -> list[str]:
    """Filtre les lignes de bruit (dates, unites, notes) via la normalisation d'indicateurs."""
    return [x for x in items if x and normalize_indicator_for_comparison(str(x).strip())]


def _cached_render_or_crop(
    pdf_path: str,
    page: int,
    dpi: int,
    bbox_key: str,
    display_mode: str = "crop",
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
    highlight_color: tuple[float, float, float] = PROOF_HIGHLIGHT_COLOR_T2,
) -> bytes:
    """Retourne les octets PNG selon le mode d'affichage : crop, full ou footnote."""
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
            dpi=dpi,
            allow_full_page_fallback=True,
        )
        return raw if raw else b""

    if bbox is None:
        return b""

    try:
        if display_mode == "footnote":
            return pdf_crop.crop_footnote_region_to_bytes(
                pdf_path,
                page,
                bbox,
                dpi=dpi,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
                highlight_color=highlight_color,
                secondary_highlight_color=highlight_color,
            )
        else:  # "crop" (default)
            return pdf_crop.crop_table_region_to_bytes(
                pdf_path,
                page,
                bbox,
                dpi=dpi,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
                highlight_color=highlight_color,
                secondary_highlight_color=highlight_color,
            )
    except Exception:
        return b""


def _normalize_proof_bbox(bbox: Any) -> list[float] | None:
    """Retourne ``[l, t, r, b]`` normalise entre 0 et 1 si exploitable pour le rendu."""
    return _shared_normalize_proof_bbox(bbox)


def _proof_render_result(image_b64: str | None, status: str, mode_effective: str) -> dict[str, str | None]:
    """Construit le dictionnaire de resultat de rendu de preuve."""
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
    """Retourne l'image de preuve et le statut de disponibilite pour un cote.

    Args:
        item_dict: Dictionnaire de l'element de revue contenant les pages et bbox.
        side: Cote du trimestre (``t1`` pour precedent, ``t2`` pour courant).
        paths: Dictionnaire des chemins PDF.
        proof_display_mode: Mode d'affichage (``crop``, ``full`` ou ``footnote``).
        highlight_rects: Rectangles de surbrillance primaires.
        secondary_highlight_rects: Rectangles de surbrillance secondaires.

    Returns:
        Dictionnaire avec ``image_b64``, ``status`` et ``mode_effective``.
    """
    display_mode = (proof_display_mode or "crop").strip().lower()
    if display_mode not in {"crop", "full", "footnote"}:
        display_mode = "crop"

    page = item_dict.get("page_t1" if side == "t1" else "page_t2")
    if page is None:
        return _proof_render_result(None, "page_missing", display_mode)

    ref = item_dict.get("source_ref_t1" if side == "t1" else "source_ref_t2", "")
    normalized_paths = _normalize_pdf_paths_store(paths)
    pdf_path = normalized_paths.get("pdf_t1", "") if side == "t1" else normalized_paths.get("pdf_t2", "")
    if not pdf_path:
        pdf_path = ref
    if not pdf_path:
        return _proof_render_result(None, "render_failed", display_mode)

    bbox = _normalize_proof_bbox(item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2"))
    if display_mode in {"crop", "footnote"} and bbox is None:
        return _proof_render_result(None, "bbox_missing", display_mode)

    if display_mode == "full" and bbox is None:
        raw_bytes, status, mode_effective = render_full_proof_bytes(
            pdf_path,
            page=page,
            bbox=None,
            dpi=PROOF_RENDER_DPI,
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
    """Obtient l'image de preuve en base64 avec annotation des changements.

    En mode ``full``, affiche la page entiere ; en mode ``crop``, decoupe
    selon la bbox ; en mode ``footnote``, affiche uniquement la zone de
    notes de bas de page.
    """
    display_mode = (proof_display_mode or "crop").strip().lower()
    highlight_color = PROOF_HIGHLIGHT_COLOR_T1 if side == "t1" else PROOF_HIGHLIGHT_COLOR_T2

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
    bbox = _normalize_proof_bbox(item_dict.get("bbox_t1") if side == "t1" else item_dict.get("bbox_t2"))

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
                PROOF_RENDER_DPI,
                bbox_key,
                display_mode,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
                highlight_color=highlight_color,
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
                PROOF_RENDER_DPI,
                bbox_key,
                display_mode,
                highlight_rects=highlight_rects,
                secondary_highlight_rects=secondary_highlight_rects,
                highlight_color=highlight_color,
            )
            base_img_b64 = base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else None
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
    """Obtient l'image base64 simplifiee pour T1 ou T2 (PDF page ou fichier PNG)."""
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
            return base64.b64encode(raw).decode("ascii")
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

    if ref and Path(ref).exists() and Path(ref).suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            with open(ref, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode("ascii")
        except Exception:
            pass

    if pdf_path and page is not None:
        page_effective = max(1, int(page))
        raw = get_pdf_preview(pdf_path, page_effective, scale=PROOF_RENDER_DPI / 72.0)
        if raw:
            return base64.b64encode(raw).decode("ascii")
    return None
