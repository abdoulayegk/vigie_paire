"""Utilitaires partages de rendu de preuves utilises par Dash et les pipelines de comparaison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vigie.extraction.pdf_preview import render_pdf_page
from vigie.support.utils.pdf_crop import render_page_with_bbox_highlight_to_bytes

# Resolution par defaut pour les preuves ecran (Dash, coherence visuelle).
# ~108 dpi (scale 1.5) etait trop doux ; 200 dpi reste raisonnable en taille memoire.
PROOF_RENDER_DPI = 200


def normalize_proof_bbox(bbox: Any) -> list[float] | None:
    """Retourne [l, t, r, b] dans 0..1 si exploitable pour le rendu de preuve.

    Args:
        bbox: Bounding box brute (liste, tuple ou autre).

    Returns:
        Liste normalisee ``[l, t, r, b]`` ou ``None`` si invalide.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        normalized = [
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        ]
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = normalized
    if not (0.0 <= left <= 1.0 and 0.0 <= top <= 1.0 and 0.0 <= right <= 1.0 and 0.0 <= bottom <= 1.0):
        return None
    if right <= left or bottom <= top:
        return None
    return normalized


def render_full_proof_bytes(
    pdf_path: str | Path | None,
    *,
    page: Any,
    bbox: Any,
    scale: float = 1.5,
    dpi: int | None = PROOF_RENDER_DPI,
    allow_full_page_fallback: bool = False,
) -> tuple[bytes | None, str, str]:
    """Rend une image de preuve pleine page avec surlignage de bbox si disponible.

    Args:
        pdf_path: Chemin du fichier PDF source.
        page: Numero de page (base 1).
        bbox: Bounding box brute pour le surlignage.
        scale: Echelle PyMuPDF (72 pt) si ``dpi`` est ``None`` (defaut 1.5).
        dpi: Resolution de rendu ; si defini (defaut :data:`PROOF_RENDER_DPI`), prime sur
            ``scale``. Passer ``None`` pour n'utiliser que ``scale``.
        allow_full_page_fallback: Si ``True``, rend la page sans bbox en cas d'absence.

    Returns:
        Tuple ``(image_bytes, status, mode_effective)`` ou status est l'un de :
        ``ok``, ``pdf_missing``, ``page_missing``, ``bbox_missing`` ou ``render_failed``.
        ``mode_effective`` vaut ``full`` ou ``full_without_bbox`` quand un repli
        page seule est utilise car la bbox n'est pas disponible.
    """
    pdf = str(pdf_path or "").strip()
    if not pdf:
        return None, "pdf_missing", "full"

    try:
        page_num = int(page)
    except (TypeError, ValueError):
        return None, "page_missing", "full"
    if page_num < 1:
        return None, "page_missing", "full"

    zoom = (dpi / 72.0) if dpi is not None else scale

    bbox_norm = normalize_proof_bbox(bbox)
    if bbox_norm is None:
        if not allow_full_page_fallback:
            return None, "bbox_missing", "full"
        raw = render_pdf_page(pdf, page_num, scale=zoom, format="png")
        if raw:
            return raw, "ok", "full_without_bbox"
        return None, "render_failed", "full_without_bbox"

    raw = render_page_with_bbox_highlight_to_bytes(
        pdf,
        page_num,
        bbox_norm,
        scale=scale,
        dpi=dpi,
    )
    if raw:
        return raw, "ok", "full"
    if not allow_full_page_fallback:
        return None, "render_failed", "full"
    raw = render_pdf_page(pdf, page_num, scale=zoom, format="png")
    if raw:
        return raw, "ok", "full"
    return None, "render_failed", "full"
