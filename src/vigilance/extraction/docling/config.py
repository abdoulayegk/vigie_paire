"""Resolution de la configuration d'extraction et synthese qualite Vision.

Extrait de ``docling_processor.py`` sans modification.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# _ENV_TRUE / _ENV_FALSE etaient utilises sans etre importes dans le module
# d'origine : _env_bool levait NameError des qu'une variable d'environnement
# etait reellement positionnee. L'import manquant est retabli ici.
from .._docling_env import _ENV_FALSE, _ENV_TRUE

logger = logging.getLogger("vigilance.extraction.docling_processor")


def _env_bool(*names: str) -> bool | None:
    """Lire une variable d'environnement booleenne parmi les noms fournis."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in _ENV_TRUE:
            return True
        if value in _ENV_FALSE:
            return False
    return None


def _resolve_vision_extraction_enabled(bank_code: str, explicit: bool | None) -> bool:
    """Resoudre l'activation de l'extraction Vision : argument explicite > env > config banque."""
    if explicit is not None:
        return bool(explicit)

    env_choice = _env_bool("VIGILANCE_VISION_EXTRACTION_ENABLED")
    if env_choice is not None:
        return env_choice

    try:
        from ...config import get_vision_extraction_config

        cfg = get_vision_extraction_config(bank_code=bank_code) or {}
        if "enabled" in cfg:
            return bool(cfg.get("enabled"))
    except Exception:
        pass
    return False


def _compute_vision_quality_summary(tables: list[Any]) -> dict[str, Any]:
    """Agréger les métriques de débogage par tableau en un résumé de qualité global.

    Parcourt les ``debug_metrics`` de chaque tableau extrait et calcule des
    statistiques agrégées sur la qualité de l'extraction Vision GPT-4o :
    nombre de tentatives, succès, échecs partiels, troncatures, confiance faible,
    recadrages utilisés, bboxes rejetées, etc.

    Retourne un dictionnaire de compteurs utilisé pour le logging et le diagnostic
    de la qualité d'extraction d'un run complet.
    """
    total = len(tables)
    attempted = 0
    ok = 0
    partial = 0
    failed = 0
    truncated = 0
    low_confidence = 0
    no_reference_text = 0
    recrop_used_count = 0
    bbox_rejected = 0

    for t in tables:
        dm = getattr(t, "debug_metrics", None)
        if not isinstance(dm, dict):
            continue
        if dm.get("vision_extraction_attempted"):
            attempted += 1
        status = dm.get("vision_status", "")
        if status == "ok":
            ok += 1
        elif status == "partial":
            partial += 1
        elif status == "failed":
            failed += 1
        retry_reasons = {str(value).strip() for value in list(dm.get("retry_reasons") or []) if str(value).strip()}
        if "output_budget_truncated" in retry_reasons:
            truncated += 1
        conf = dm.get("vision_extraction_confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < 0.85 and status in ("ok", "partial"):
            low_confidence += 1
        if dm.get("crop_reject_reason"):
            bbox_rejected += 1
        if dm.get("recrop_used"):
            recrop_used_count += 1
        if not dm.get("has_reference_text") and dm.get("vision_extraction_attempted"):
            no_reference_text += 1

    return {
        "total_tables": total,
        "attempted": attempted,
        "ok": ok,
        "partial": partial,
        "failed": failed,
        "truncated": truncated,
        "low_confidence": low_confidence,
        "no_reference_text": no_reference_text,
        "recrop_used": recrop_used_count,
        "bbox_rejected": bbox_rejected,
    }
