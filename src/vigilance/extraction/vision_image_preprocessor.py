"""Pre-traitement d'image pour l'extraction GPT-4o Vision.

Applique l'amelioration du contraste, la nettete et la normalisation
de l'arriere-plan afin d'ameliorer la precision OCR sur les tableaux
de rapports bancaires canadiens.

Active par la variable d'environnement ``VISION_PREPROCESS`` (defaut : active).
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

_PREPROCESS_ENV = "VISION_PREPROCESS"


def _is_enabled() -> bool:
    """Verifier si le pre-traitement Vision est active via la variable d'environnement."""
    val = os.environ.get(_PREPROCESS_ENV)
    if val is None:
        return True
    return val.strip().lower() in ("1", "true", "yes")


def preprocess_for_vision(image_bytes: bytes, *, enabled: bool | None = None) -> bytes:
    """Appliquer le pre-traitement contraste/nettete a une image PNG pour Vision.

    Pipeline (Pillow uniquement, sans dependance OpenCV) :

    1. Conversion en niveaux de gris puis retour en RGB pour supprimer les
       bordures / arriere-plans colores
    2. Amelioration du contraste via autocontrast (similaire a CLAHE)
    3. Masque flou (Unsharp Mask) pour la nettete du texte
    4. Normalisation de l'arriere-plan (pixels quasi-blancs -> blanc pur)

    Args:
        image_bytes: Octets PNG de l'image source.
        enabled: Surcharge explicite. ``None`` (defaut) utilise la variable
                 ``VISION_PREPROCESS``. Passer ``True``/``False`` pour un
                 controle thread-safe sans modifier l'environnement.

    Returns:
        Octets de l'image pre-traitee, ou les octets originaux si le
        pre-traitement est desactive ou en cas d'echec (ne leve jamais
        d'exception).
    """
    if enabled is not None:
        if not enabled:
            return image_bytes
    elif not _is_enabled():
        return image_bytes

    try:
        from PIL import Image, ImageFilter, ImageOps

        src: Image = Image.open(io.BytesIO(image_bytes))

        if src.mode == "RGBA":
            background: Image = Image.new("RGB", src.size, (255, 255, 255))
            background.paste(src, mask=src.split()[3])
            src = background
        elif src.mode != "RGB":
            src = src.convert("RGB")

        gray = src.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=0.5)
        img: Image = gray.convert("RGB")

        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

        img = _normalize_background(img, threshold=240)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.debug("Vision preprocessing failed, returning original image: %s", e)
        return image_bytes


def preprocess_pil_image(img: Image) -> Image:
    """Meme pipeline que ``preprocess_for_vision`` mais accepte/retourne un objet PIL Image.

    Utilise par ``GPT4VisionFallback._encode_image`` qui dispose deja d'un
    objet PIL.

    Args:
        img: Image PIL source.

    Returns:
        Image PIL pre-traitee (ou originale si desactive / en erreur).
    """
    if not _is_enabled():
        return img

    try:
        from PIL import ImageFilter, ImageOps

        if img.mode == "RGBA":
            background = img.copy().convert("RGB")
            from PIL import Image as _Img

            background = _Img.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=0.5)
        img = gray.convert("RGB")

        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

        img = _normalize_background(img, threshold=240)

        return img

    except Exception as e:
        logger.debug("Vision PIL preprocessing failed, returning original: %s", e)
        return img


def _normalize_background(img: Image, threshold: int = 240) -> Image:
    """Remplacer les pixels quasi-blancs par du blanc pur pour supprimer les fonds gris."""
    import numpy as np

    arr = np.array(img)
    mask = np.all(arr >= threshold, axis=-1)
    arr[mask] = 255
    from PIL import Image

    return Image.fromarray(arr)
