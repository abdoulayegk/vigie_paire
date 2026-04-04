"""Couche de cache pour les resultats d'extraction Vision (premiere colonne).

Stocke un fichier JSON par cle dans le repertoire de cache configure.
"""

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
    """Retourner le repertoire de cache Vision (``VISION_CACHE_DIR`` ou defaut)."""
    return os.environ.get("VISION_CACHE_DIR", DEFAULT_CACHE_DIR)


def get_vision_crop_dir() -> str:
    """Retourner le repertoire de sortie des crops Vision (``VISION_CROP_DIR`` ou defaut)."""
    return os.environ.get("VISION_CROP_DIR", DEFAULT_CROP_DIR)


def compute_pdf_sha256(pdf_path: str) -> str:
    """Calculer le hash SHA-256 du contenu d'un fichier PDF.

    Args:
        pdf_path: Chemin vers le fichier PDF.

    Returns:
        Hash hexadecimal ou chaine vide si le fichier est absent.
    """
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
    """Creer une cle de cache a partir du hash PDF, du numero de page et du bbox normalise.

    Les valeurs du bbox sont arrondies a 4 decimales pour la stabilite.
    Inclut ``_VISION_CACHE_VERSION`` afin que les changements de prompt /
    schema invalident les entrees obsoletes.

    Args:
        pdf_sha: Hash SHA-256 du PDF.
        page_number: Numero de page (1-indexed).
        bbox_norm: Bbox normalise ``[l, t, r, b]``.
        max_completion_tokens: Limite de tokens (optionnel, inclus dans la cle).

    Returns:
        Chaine de cle de cache, ou chaine vide si le bbox est invalide.
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
    """Charger un payload en cache par cle.

    Args:
        cache_dir: Repertoire de cache.
        key: Cle de cache.

    Returns:
        Dictionnaire du payload ou ``None`` si introuvable / invalide.
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
    """Stocker un payload dans le cache. Ecrase si la cle existe deja.

    Args:
        cache_dir: Repertoire de cache.
        key: Cle de cache.
        payload: Dictionnaire a persister.
    """
    if (
        not key
        or not isinstance(payload, dict)
        or ".." in key
        or "/" in key
        or "\\" in key
    ):
        return
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=None), encoding="utf-8"
        )
    except Exception as e:
        logger.debug("Vision cache put failed: %s", e)
