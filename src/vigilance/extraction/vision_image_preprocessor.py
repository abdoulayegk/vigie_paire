"""Image preprocessing for GPT-4o Vision extraction.

Applies contrast enhancement, sharpening, and background normalization
to improve OCR accuracy on Canadian banking PDF tables.

Controlled via VISION_PREPROCESS env var (default: enabled).
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
    val = os.environ.get(_PREPROCESS_ENV)
    if val is None:
        return True
    return val.strip().lower() in ("1", "true", "yes")


def preprocess_for_vision(image_bytes: bytes, *, enabled: bool | None = None) -> bytes:
    """Apply contrast/sharpness preprocessing to a PNG image for Vision.

    Pipeline (Pillow-only, no OpenCV dependency):
    1. Convert to grayscale then back to RGB to drop colored borders/backgrounds
    2. CLAHE-like contrast boost via autocontrast
    3. Unsharp mask for text sharpness
    4. Background normalization (near-white pixels -> pure white)

    Args:
        enabled: Explicit override. ``None`` (default) falls back to the
                 ``VISION_PREPROCESS`` env var. Pass ``True``/``False`` for
                 thread-safe control from config without mutating env.

    Returns the original bytes unchanged when preprocessing is disabled or
    if any step fails (never raises).
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
    """Same pipeline as preprocess_for_vision but accepts/returns a PIL Image.

    Used by GPT4VisionFallback._encode_image which already has a PIL object.
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
    """Replace near-white pixels with pure white to remove gray backgrounds."""
    import numpy as np

    arr = np.array(img)
    mask = np.all(arr >= threshold, axis=-1)
    arr[mask] = 255
    from PIL import Image

    return Image.fromarray(arr)
