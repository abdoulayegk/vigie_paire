"""Utilitaire de recadrage de tableaux PDF via PyMuPDF. Rend une region recadree en PNG."""

from __future__ import annotations

import logging
from pathlib import Path

from .pymupdf_utils import configure_mupdf_runtime

logger = logging.getLogger(__name__)


def crop_table_image(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    out_path: str,
    dpi: int = 300,
    bottom_extension: float = 0.0,
) -> bool:
    """Recadre une region de tableau d'une page PDF et sauvegarde en PNG.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numero de page (base 1).
        bbox_norm: Bounding box normalisee [l, t, r, b] dans 0..1 (gauche, haut, droite, bas).
        out_path: Chemin de sortie pour le fichier PNG.
        dpi: Resolution de rendu (defaut 300).
        bottom_extension: Extension supplementaire sous la bbox (normalisee 0..1).

    Returns:
        ``True`` en cas de succes, ``False`` en cas d'echec. Ne leve pas d'exception.
    """
    if not _validate_bbox(bbox_norm):
        return False

    try:
        import pymupdf
    except ImportError:
        logger.debug("PyMuPDF not available for crop_table_image")
        return False
    configure_mupdf_runtime(pymupdf)

    try:
        doc = pymupdf.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                return False
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            pad = 0.03
            b_norm_effective = min(1.0, b_norm + bottom_extension)
            x0 = rect.x0 + max(0.0, l_norm - pad) * rect.width
            y0 = rect.y0 + max(0.0, t_norm - pad) * rect.height
            x1 = rect.x0 + min(1.0, r_norm + pad) * rect.width
            y1 = rect.y0 + min(1.0, b_norm_effective + pad) * rect.height
            clip = pymupdf.Rect(x0, y0, x1, y1)
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pix.save(out_path)
            return True
        finally:
            doc.close()
    except Exception:
        return False


def _validate_bbox(bbox_norm: list[float]) -> bool:
    """Valide une bbox normalisee : longueur=4, chaque valeur dans [0,1], x1>x0, y1>y0."""
    if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
        return False
    try:
        l_norm, t_norm, r_norm, b_norm = (
            float(bbox_norm[0]),
            float(bbox_norm[1]),
            float(bbox_norm[2]),
            float(bbox_norm[3]),
        )
    except (TypeError, ValueError):
        return False
    if not (
        0 <= l_norm <= 1 and 0 <= t_norm <= 1 and 0 <= r_norm <= 1 and 0 <= b_norm <= 1
    ):
        return False
    if r_norm <= l_norm or b_norm <= t_norm:
        return False
    return True


def bbox_sanity_profile(bbox_norm: list[float]) -> dict:
    """Calcule un profil de coherence pour une bbox normalisee [l, t, r, b].

    Utilise pour filtrer la Vision : rejette les bbox trop petites, quasi pleine page
    ou avec un ratio d'aspect extreme.

    Args:
        bbox_norm: Bounding box normalisee [l, t, r, b] dans 0..1.

    Returns:
        Dictionnaire contenant les dimensions, l'aire, le ratio d'aspect et
        eventuellement la raison de rejet.
    """
    if not _validate_bbox(bbox_norm):
        return {
            "width_norm": 0.0,
            "height_norm": 0.0,
            "area_norm": 0.0,
            "is_near_full_page": False,
            "aspect_ratio": 0.0,
            "reject_reason": "invalid_bbox",
        }
    l_norm, t_norm, r_norm, b_norm = bbox_norm
    w = r_norm - l_norm
    h = b_norm - t_norm
    area = w * h
    aspect = w / h if h > 0 else 0.0
    # Une table financiere peut legitimement occuper presque toute la largeur
    # (ou la hauteur) sans etre un crop pleine page. Il faut que l'aire soit
    # quasi totale, ou que les deux dimensions le soient simultanement.
    near_full = area >= 0.90 or (w >= 0.95 and h >= 0.95)
    profile: dict = {
        "width_norm": round(w, 6),
        "height_norm": round(h, 6),
        "area_norm": round(area, 6),
        "is_near_full_page": near_full,
        "aspect_ratio": round(aspect, 4),
    }
    return profile


def is_bbox_sane(
    bbox_norm: list[float], cfg: dict | None = None
) -> tuple[bool, str | None, dict]:
    """Verifie si une bbox normalisee est exploitable pour la Vision.

    Args:
        bbox_norm: Bounding box normalisee [l, t, r, b] dans 0..1.
        cfg: Configuration optionnelle (``bbox_min_width``, ``bbox_min_height``,
            ``bbox_min_area``, ``bbox_max_area``, ``bbox_near_full_page_threshold``).

    Returns:
        Tuple ``(sane, reject_reason, profile)``. Si ``sane`` est ``False``,
        ``reject_reason`` indique la cause du rejet.
    """
    profile = bbox_sanity_profile(bbox_norm)
    if profile.get("reject_reason") == "invalid_bbox":
        return False, "invalid_bbox", profile

    cfg = cfg or {}
    min_w = float(cfg.get("bbox_min_width", 0.02))
    min_h = float(cfg.get("bbox_min_height", 0.02))
    min_area = float(cfg.get("bbox_min_area", 0.0005))
    max_area = float(cfg.get("bbox_max_area", 0.95))
    near_full_threshold = float(cfg.get("bbox_near_full_page_threshold", 0.90))

    w = profile["width_norm"]
    h = profile["height_norm"]
    area = profile["area_norm"]

    if w < min_w:
        profile["reject_reason"] = "bbox_too_narrow"
        return False, "bbox_too_narrow", profile
    if h < min_h:
        profile["reject_reason"] = "bbox_too_short"
        return False, "bbox_too_short", profile
    if area < min_area:
        profile["reject_reason"] = "bbox_area_too_small"
        return False, "bbox_area_too_small", profile
    if area > max_area:
        profile["reject_reason"] = "bbox_area_too_large"
        return False, "bbox_area_too_large", profile
    if (
        area >= near_full_threshold
        or (
            w >= near_full_threshold
            and h >= near_full_threshold
        )
    ):
        profile["reject_reason"] = "bbox_near_full_page"
        return False, "bbox_near_full_page", profile
    # Extreme aspect ratio inconsistent with tables (e.g. 20:1 or 1:20)
    if w / h > 20.0 or h / w > 20.0:
        profile["reject_reason"] = "extreme_aspect_ratio"
        return False, "extreme_aspect_ratio", profile

    return True, None, profile


def crop_table_region_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    top_extension: float = 0.0,
    horizontal_padding: float = 0.0,
    dpi: int | None = None,
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
    highlight_color: tuple[float, float, float] = (1, 0, 1),
    secondary_highlight_color: tuple[float, float, float] | None = None,
) -> bytes:
    """Recadre une region de tableau d'une page PDF et retourne les octets PNG.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numero de page (base 1, meme convention que render_pdf_page).
        bbox_norm: Bounding box normalisee [l, t, r, b] dans 0..1.
        scale: Echelle de rendu quand dpi n'est pas defini (defaut 1.5, comme les previsualisations).
        bottom_extension: Hauteur supplementaire sous la bbox (ex. notes de bas de page), normalisee 0..1.
        top_extension: Hauteur supplementaire au-dessus de la bbox (ex. titre ou lignes manquees), normalisee 0..1.
        horizontal_padding: Padding horizontal symetrique (normalise 0..1), borne a la page.
        dpi: Si defini, rendu a cette resolution (72 * zoom) ; ecrase scale. Utiliser 300 pour Vision/OCR.
        highlight_rects: Surlignages primaires -- le changement actif en cours de validation.
        secondary_highlight_rects: Surlignages secondaires -- autres changements du tableau
            (contexte seulement, plus discrets pour que le changement actif reste visuellement dominant).
        highlight_color: Couleur RGB normalisee du surlignage primaire.
        secondary_highlight_color: Couleur RGB normalisee du surlignage secondaire.
            Si absent, utilise la meme couleur que ``highlight_color``.

    Returns:
        Octets PNG de la region recadree. Retourne ``b""`` si bbox invalide, page hors limites,
        import echoue ou toute exception de recadrage (pas de repli pleine page).
    """
    zoom = (dpi / 72.0) if dpi is not None else scale
    secondary_color = secondary_highlight_color or highlight_color

    if not _validate_bbox(bbox_norm):
        return b""

    try:
        import pymupdf
    except ImportError:
        logger.debug("PyMuPDF not available for crop_table_region_to_bytes")
        return b""
    configure_mupdf_runtime(pymupdf)

    try:
        doc = pymupdf.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                return b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            t_norm_effective = max(0.0, t_norm - top_extension)
            b_norm_effective = min(1.0, b_norm + bottom_extension)
            l_norm_effective = max(0.0, l_norm - horizontal_padding)
            r_norm_effective = min(1.0, r_norm + horizontal_padding)
            x0 = rect.x0 + l_norm_effective * rect.width
            y0 = rect.y0 + t_norm_effective * rect.height
            x1 = rect.x0 + r_norm_effective * rect.width
            y1 = rect.y0 + b_norm_effective * rect.height
            clip = pymupdf.Rect(x0, y0, x1, y1)

            # Secondary highlights first, dimmer, so primary renders on top.
            if secondary_highlight_rects:
                for hl_norm in secondary_highlight_rects:
                    if len(hl_norm) == 4:
                        hx0 = rect.x0 + hl_norm[0] * rect.width
                        hy0 = rect.y0 + hl_norm[1] * rect.height
                        hx1 = rect.x0 + hl_norm[2] * rect.width
                        hy1 = rect.y0 + hl_norm[3] * rect.height
                        page.draw_rect(
                            pymupdf.Rect(hx0, hy0, hx1, hy1),
                            color=secondary_color,
                            fill=secondary_color,
                            fill_opacity=0.2,
                        )

            # Primary highlights — the active change being validated.
            if highlight_rects:
                for hl_norm in highlight_rects:
                    if len(hl_norm) == 4:
                        hx0 = rect.x0 + hl_norm[0] * rect.width
                        hy0 = rect.y0 + hl_norm[1] * rect.height
                        hx1 = rect.x0 + hl_norm[2] * rect.width
                        hy1 = rect.y0 + hl_norm[3] * rect.height
                        page.draw_rect(
                            pymupdf.Rect(hx0, hy0, hx1, hy1),
                            color=highlight_color,
                            fill=highlight_color,
                            fill_opacity=0.35,
                        )

            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return b""


def render_page_with_bbox_highlight_to_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float],
    scale: float = 1.5,
    bottom_extension: float = 0.0,
    dpi: int | None = None,
) -> bytes:
    """Rend une page PDF complete en PNG avec un rectangle rouge autour du tableau.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numero de page (base 1).
        bbox_norm: Bounding box normalisee [l, t, r, b] dans 0..1.
        scale: Echelle de rendu quand dpi n'est pas defini.
        bottom_extension: Hauteur supplementaire incluse dans le rectangle rouge (normalisee 0..1).
        dpi: Si defini, rendu a cette resolution (72 * zoom) ; ecrase scale. Utiliser 300 pour Vision/OCR.

    Returns:
        Octets PNG de la page complete avec un rectangle rouge, ou page normale si bbox invalide.
    """
    from vigie.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale

    if not _validate_bbox(bbox_norm):
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

    try:
        import pymupdf
    except ImportError:
        logger.debug(
            "PyMuPDF not available for render_page_with_bbox_highlight_to_bytes"
        )
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""
    configure_mupdf_runtime(pymupdf)

    try:
        doc = pymupdf.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
                return full if full else b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = bbox_norm
            b_norm_effective = min(1.0, b_norm + bottom_extension)

            x0 = rect.x0 + l_norm * rect.width
            y0 = rect.y0 + t_norm * rect.height
            x1 = rect.x0 + r_norm * rect.width
            y1 = rect.y0 + b_norm_effective * rect.height

            # Draw a bright red rectangle with 3px width on the page
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(1, 0, 0), width=3)

            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""


def crop_footnote_region_to_bytes(
    pdf_path: str,
    page_number: int,
    table_bbox_norm: list[float],
    scale: float = 1.5,
    footnote_height: float = 0.25,
    dpi: int | None = None,
    highlight_rects: list[list[float]] | None = None,
    secondary_highlight_rects: list[list[float]] | None = None,
    highlight_color: tuple[float, float, float] = (1, 0, 1),
    secondary_highlight_color: tuple[float, float, float] | None = None,
) -> bytes:
    """Recadre uniquement la region de notes de bas de page sous un tableau.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numero de page (base 1).
        table_bbox_norm: Bounding box normalisee du tableau [l, t, r, b] dans 0..1.
        scale: Echelle de rendu quand dpi n'est pas defini.
        footnote_height: Hauteur de la region de notes en fraction de page (defaut 0.25 = 25 %).
        dpi: Si defini, rendu a cette resolution ; ecrase scale.
        highlight_rects: Surlignages primaires.
        secondary_highlight_rects: Surlignages secondaires.
        highlight_color: Couleur RGB normalisee du surlignage primaire.
        secondary_highlight_color: Couleur RGB normalisee du surlignage secondaire.
            Si absent, utilise la meme couleur que ``highlight_color``.

    Returns:
        Octets PNG de la region de notes sous le tableau.
    """
    from vigie.extraction.pdf_preview import render_pdf_page

    zoom = (dpi / 72.0) if dpi is not None else scale
    secondary_color = secondary_highlight_color or highlight_color

    if not _validate_bbox(table_bbox_norm):
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""

    try:
        import pymupdf
    except ImportError:
        logger.debug("PyMuPDF not available for crop_footnote_region_to_bytes")
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""
    configure_mupdf_runtime(pymupdf)

    try:
        doc = pymupdf.open(pdf_path)
        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
                return full if full else b""
            page = doc[page_idx]
            rect = page.rect
            l_norm, t_norm, r_norm, b_norm = table_bbox_norm

            # Footnote region: from bottom of table to footnote_height below (or page bottom)
            footnote_top = b_norm
            footnote_bottom = min(1.0, b_norm + footnote_height)

            # Use full page width for footnotes (they often span the whole width)
            x0 = rect.x0
            y0 = rect.y0 + footnote_top * rect.height
            x1 = rect.x1
            y1 = rect.y0 + footnote_bottom * rect.height

            clip = pymupdf.Rect(x0, y0, x1, y1)
            if secondary_highlight_rects:
                for hl_norm in secondary_highlight_rects:
                    if len(hl_norm) == 4:
                        hx0 = rect.x0 + hl_norm[0] * rect.width
                        hy0 = rect.y0 + hl_norm[1] * rect.height
                        hx1 = rect.x0 + hl_norm[2] * rect.width
                        hy1 = rect.y0 + hl_norm[3] * rect.height
                        highlight = pymupdf.Rect(hx0, hy0, hx1, hy1)
                        if highlight.intersects(clip):
                            page.draw_rect(
                                highlight,
                                color=secondary_color,
                                fill=secondary_color,
                                fill_opacity=0.2,
                            )
            if highlight_rects:
                for hl_norm in highlight_rects:
                    if len(hl_norm) == 4:
                        hx0 = rect.x0 + hl_norm[0] * rect.width
                        hy0 = rect.y0 + hl_norm[1] * rect.height
                        hx1 = rect.x0 + hl_norm[2] * rect.width
                        hy1 = rect.y0 + hl_norm[3] * rect.height
                        highlight = pymupdf.Rect(hx0, hy0, hx1, hy1)
                        if highlight.intersects(clip):
                            page.draw_rect(
                                highlight,
                                color=highlight_color,
                                fill=highlight_color,
                                fill_opacity=0.35,
                            )
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        full = render_pdf_page(pdf_path, page_number, scale=zoom, format="png")
        return full if full else b""


def crop_page_region_bytes(
    pdf_path: str,
    page_number: int,
    bbox_norm: list[float] | None = None,
    dpi: int = 200,
) -> bytes:
    """Rend la zone spécifiée (ou toute la page) d'un PDF sous forme de bytes PNG.

    Args:
        pdf_path: Chemin du fichier PDF.
        page_number: Numéro de page (base 1).
        bbox_norm: Bounding box normalisée [l, t, r, b] ou None pour la page entière.
        dpi: Résolution de rendu.

    Returns:
        Bytes PNG de l'image.
    """
    if not bbox_norm:
        # Import paresseux : render_pdf_page etait appele sans etre importe, ce qui
        # levait NameError sur cette branche. Charge ici pour eviter tout cycle
        # entre vigie.support.utils et vigie.extraction.
        from vigie.extraction.pdf_preview import render_pdf_page

        return render_pdf_page(pdf_path, page_number, scale=dpi / 72, format="png") or b""

    return crop_table_region_to_bytes(
        pdf_path,
        page_number,
        bbox_norm,
        dpi=dpi,
    )
