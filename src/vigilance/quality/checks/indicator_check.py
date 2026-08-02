"""Module spécialisé dans la validation des métriques et indicateurs chiffrés."""

from __future__ import annotations

from typing import Any


def check_indicator_consistency(indicators: list[dict[str, Any]]) -> dict[str, Any]:
    """Valide la cohérence des indicateurs chiffrés d'un tableau.

    Returns:
        Dictionnaire avec statut de cohérence et anomalies détectées.
    """
    anomalies = []
    for idx, ind in enumerate(indicators):
        if not isinstance(ind, dict):
            anomalies.append(f"indicator_{idx}_invalid_type")
            continue
        label = str(ind.get("label") or ind.get("name") or "").strip()
        if not label:
            anomalies.append(f"indicator_{idx}_missing_label")

    return {
        "is_valid": len(anomalies) == 0,
        "indicator_count": len(indicators),
        "anomalies": anomalies,
    }
