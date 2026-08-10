"""Persistence des decisions analystes pour l'analyse textuelle."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigie.support.quarter_utils import get_payload_quarter_context
from vigie.analyse_texte.text_comparison.text_comparison_excel import generate_text_comparison_excel
from vigie.interface.services.json_io import atomic_write_json
from vigie.interface.ui_config import TEXT_COMPARISON_DIR

logger = logging.getLogger(__name__)

TEXT_REVIEW_STATUSES = {"approved", "rejected", "skipped"}


def _period_from_payload(payload: dict[str, Any], role: str) -> str:
    """Retourne le dossier periode ``YYYY_tN`` pour le role demande."""
    raw_key = "quarter_current" if role == "current" else "quarter_previous"
    raw = str(payload.get(raw_key) or "").lower().strip()
    if raw and raw[:4].isdigit() and "_t" in raw:
        return raw

    ctx = get_payload_quarter_context(payload)
    side = ctx.get(role) or {}
    year = side.get("year")
    code = str(side.get("code") or "").lower().strip()
    if year and code:
        return f"{int(year)}_{code}"
    return raw


def text_comparison_path_from_payload(
    payload: dict[str, Any],
    root_dir: Path | None = None,
) -> Path | None:
    """Resout le chemin ``text_comparison.json`` depuis le payload texte."""
    bank = str(payload.get("bank_code") or payload.get("bank") or "").lower().strip()
    current = _period_from_payload(payload, "current")
    previous = _period_from_payload(payload, "previous")
    if not bank or not current or not previous:
        return None
    root = Path(root_dir) if root_dir else TEXT_COMPARISON_DIR
    return root / bank / f"{current}_vs_{previous}" / "text_comparison.json"


def apply_text_review_decision(
    text_data: dict[str, Any],
    *,
    change_id: str,
    status: str,
    comment: str = "",
    reviewer: str = "analyste",
) -> tuple[dict[str, Any], bool]:
    """Retourne une copie du payload avec la decision appliquee au changement."""
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in TEXT_REVIEW_STATUSES:
        raise ValueError(f"Statut texte non supporte: {status!r}")

    updated = copy.deepcopy(text_data)
    target_id = str(change_id or "").strip()
    if not target_id:
        return updated, False

    decision = {
        "status": normalized_status,
        "comment": str(comment or "").strip(),
        "review_user": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if normalized_status == "rejected":
        decision["nouvelle_idee_override"] = False

    found = False
    for section in updated.get("section_comparisons") or []:
        for bucket in ("all_block_comparisons", "block_comparisons"):
            for change in section.get(bucket) or []:
                if isinstance(change, dict) and str(change.get("change_id") or "") == target_id:
                    change["_analyst_review"] = dict(decision)
                    found = True

    return updated, found


def write_text_review_to_disk(
    text_data: dict[str, Any],
    *,
    regenerate_excel: bool = True,
) -> bool:
    """Ecrit ``text_comparison.json`` et, si demande, regenere l'Excel."""
    path = text_comparison_path_from_payload(text_data)
    if path is None:
        logger.warning("[text_review] chemin text_comparison introuvable")
        return False
    if not path.exists():
        logger.warning("[text_review] text_comparison.json introuvable: %s", path)
        return False

    try:
        atomic_write_json(path, text_data)
        if regenerate_excel:
            generate_text_comparison_excel(text_data, path.with_suffix(".xlsx"))
    except Exception:
        logger.exception("[text_review] echec writeback texte: %s", path)
        return False
    return True
