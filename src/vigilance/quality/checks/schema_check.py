"""Module spécialisé dans la validation de la conformité des schémas JSON."""

from __future__ import annotations

from typing import Any


def check_schema_compliance(payload: dict[str, Any], required_keys: list[str] | None = None) -> dict[str, Any]:
    """Vérifie la conformité des clés obligatoires d'un payload JSON.

    Returns:
        Dictionnaire avec statut de conformité et clés manquantes.
    """
    keys_to_check = required_keys or ["bank_code", "year", "quarter"]
    if not isinstance(payload, dict):
        return {"is_compliant": False, "missing_keys": keys_to_check}

    missing = [k for k in keys_to_check if k not in payload or payload[k] is None]
    return {
        "is_compliant": len(missing) == 0,
        "missing_keys": missing,
    }
