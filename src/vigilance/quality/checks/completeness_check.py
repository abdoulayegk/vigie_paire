"""Module spécialisé dans les contrôles de complétude des extractions (tableaux, notes, indicateurs)."""

from __future__ import annotations

from typing import Any


def check_extraction_completeness(payload: dict[str, Any]) -> dict[str, Any]:
    """Vérifie la complétude globale d'un dictionnaire d'extractions.

    Returns:
        Un dictionnaire contenant le statut de complétude et la liste des éléments manquants.
    """
    if not isinstance(payload, dict):
        return {"is_complete": False, "missing_elements": ["payload_invalid"]}

    missing = []
    tables = payload.get("tables") or []
    if not tables:
        missing.append("tables")

    return {
        "is_complete": len(missing) == 0,
        "table_count": len(tables),
        "missing_elements": missing,
    }
