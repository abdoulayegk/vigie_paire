"""Chargement du text_comparison.json pour Dash.

Résout automatiquement le chemin depuis les métadonnées d'une comparaison
(bank_code + quarter_current + quarter_previous).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vigie.support.quarter_utils import get_payload_quarter_context
from vigie.analyse_texte.text_comparison.text_comparison_writer import _ACCEPTED_TEXT_COMPARISON_SCHEMAS
from vigie.interface.ui_config import TEXT_COMPARISON_DIR

logger = logging.getLogger(__name__)


def load_text_comparison_for_dash(
    bank_code: str,
    quarter_current: str,
    quarter_previous: str,
    root_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Charge text_comparison.json pour une banque et une période donnée.

    Args:
        bank_code: Code banque (ex. "bns").
        quarter_current: Période courante (ex. "2025_t2").
        quarter_previous: Période précédente (ex. "2025_t1").
        root_dir: Racine des comparaisons texte (défaut: TEXT_COMPARISON_DIR).

    Returns:
        Dictionnaire chargé ou None si le fichier est absent.
    """
    root = Path(root_dir) if root_dir else TEXT_COMPARISON_DIR
    folder = f"{quarter_current}_vs_{quarter_previous}"
    path = root / bank_code.lower() / folder / "text_comparison.json"

    if not path.exists():
        logger.debug("text_comparison.json absent : %s", path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") not in _ACCEPTED_TEXT_COMPARISON_SCHEMAS:
            logger.warning("schema_version inattendu dans %s", path)
            return None
        logger.info("text_comparison.json chargé : %s", path)
        return data
    except Exception as exc:
        logger.warning("Erreur lecture text_comparison.json %s : %s", path, exc)
        return None


def resolve_text_comparison_from_payload(
    comparison_payload: dict[str, Any],
    root_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Résout et charge text_comparison.json depuis le payload brut comparison.json.

    Le fichier comparison.json contient bank_code, year_current, year_previous,
    quarter_current, quarter_previous. On les combine pour former le chemin
    ``{root}/`` ``{bank}/{year_cur}_{q_cur}_vs_{year_prev}_{q_prev}/text_comparison.json``
    (défaut : racine ``outputs/resultats``).

    Args:
        comparison_payload: Payload brut issu de comparison.json (raw_data).
        root_dir: Racine optionnelle.

    Returns:
        Dictionnaire text_comparison ou None.
    """
    if not isinstance(comparison_payload, dict):
        return None

    bank_code = str(
        comparison_payload.get("bank_code")
        or comparison_payload.get("bank")
        or ""
    ).lower()
    if not bank_code:
        return None

    # Supporte à la fois :
    # - le raw comparison.json (year_current/year_previous + quarter_current/previous)
    # - le payload canonique Dash (meta.quarter_context + quarter_to/from)
    quarter_context = get_payload_quarter_context(comparison_payload)
    current_ctx = quarter_context.get("current") or {}
    previous_ctx = quarter_context.get("previous") or {}

    year_cur = str(
        comparison_payload.get("year_current")
        or current_ctx.get("year")
        or ""
    ).strip()
    year_prev = str(
        comparison_payload.get("year_previous")
        or previous_ctx.get("year")
        or ""
    ).strip()
    q_cur = str(
        comparison_payload.get("quarter_current")
        or current_ctx.get("code")
        or ""
    ).lower().strip()
    q_prev = str(
        comparison_payload.get("quarter_previous")
        or previous_ctx.get("code")
        or ""
    ).lower().strip()

    # Former "{year}_{quarter}" si year disponible
    if year_cur and q_cur:
        period_cur = f"{year_cur}_{q_cur}"
    else:
        period_cur = q_cur

    if year_prev and q_prev:
        period_prev = f"{year_prev}_{q_prev}"
    else:
        period_prev = q_prev

    if not period_cur or not period_prev:
        logger.debug(
            "Impossible de résoudre les périodes pour %s (cur=%r prev=%r)",
            bank_code, period_cur, period_prev,
        )
        return None

    return load_text_comparison_for_dash(
        bank_code=bank_code,
        quarter_current=period_cur,
        quarter_previous=period_prev,
        root_dir=root_dir,
    )
