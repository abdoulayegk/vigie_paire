"""Cache layer for Vision first-column extraction results. JSON files per key."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump this when prompt, schema, parsing logic, or extraction-budget routing changes
# so stale cache entries are automatically bypassed without manual cleanup.
_VISION_CACHE_VERSION = "v3"

DEFAULT_CACHE_DIR = "outputs/vision_cache"
DEFAULT_CROP_DIR = "outputs/debug_crops/vision_fallback"


def get_vision_cache_dir() -> str:
    """Return Vision cache directory (VISION_CACHE_DIR env or default)."""
    return os.environ.get("VISION_CACHE_DIR", DEFAULT_CACHE_DIR)


def get_vision_crop_dir() -> str:
    """Return Vision crop output directory (VISION_CROP_DIR env or default)."""
    return os.environ.get("VISION_CROP_DIR", DEFAULT_CROP_DIR)


def compute_pdf_sha256(pdf_path: str) -> str:
    """Compute SHA256 hash of PDF file contents."""
    h = hashlib.sha256()
    raw = str(pdf_path or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.exists():
        return ""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_cache_key(
    pdf_sha: str,
    page_number: int,
    bbox_norm: list[float],
    max_completion_tokens: int | None = None,
) -> str:
    """
    Create a cache key from PDF hash, page number, and normalized bbox.
    Bbox values are rounded to 4 decimal places for stability.
    Includes _VISION_CACHE_VERSION so prompt/schema changes invalidate stale entries.
    """
    if not bbox_norm or len(bbox_norm) != 4:
        return ""
    rounded = [round(float(v), 4) for v in bbox_norm[:4]]
    key = (
        f"{_VISION_CACHE_VERSION}_{pdf_sha}_{page_number}_"
        f"{rounded[0]}_{rounded[1]}_{rounded[2]}_{rounded[3]}"
    )
    if max_completion_tokens is not None:
        key += f"_mt{int(max_completion_tokens)}"
    return key


def cache_get(cache_dir: str, key: str) -> dict | None:
    """
    Load a cached payload by key. Returns None if not found or invalid.
    """
    if not key or ".." in key or "/" in key or "\\" in key:
        return None
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def cache_put(cache_dir: str, key: str, payload: dict) -> None:
    """
    Store a payload in the cache. Overwrites if key exists.
    """
    if not key or not isinstance(payload, dict) or ".." in key or "/" in key or "\\" in key:
        return
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=None), encoding="utf-8")
    except Exception as e:
        logger.debug("Vision cache put failed: %s", e)
