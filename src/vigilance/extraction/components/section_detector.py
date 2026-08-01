"""Composant spécialisé dans la détection des titres de sections et périmètres de chapitres."""

from __future__ import annotations

import re


def detect_section_key(title_text: str) -> str:
    """Identifie la clé de section canonique (gestion_capital, gestion_risques, reglementation) à partir d'un titre."""
    if not title_text:
        return "unknown_section"

    text = title_text.lower()
    if any(k in text for k in ("capital", "fonds propres", "tlac", "cet1")):
        return "gestion_capital"
    if any(k in text for k in ("risque", "risk", "liquidité", "crédit", "marché")):
        return "gestion_risques"
    if any(k in text for k in ("réglementation", "regulatory", "bsif", "amf")):
        return "gestion_reglementation"

    return "unknown_section"
