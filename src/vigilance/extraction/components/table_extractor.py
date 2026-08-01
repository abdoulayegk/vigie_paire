"""Composant spécialisé dans la validation et le nettoyage des contours de tableaux."""

from __future__ import annotations

import re
from typing import Any


def validate_table_structure(table_dict: dict[str, Any]) -> bool:
    """Valide qu'un dictionnaire de tableau possède au moins un titre ou des indicateurs chiffrés."""
    if not isinstance(table_dict, dict):
        return False
    title = str(table_dict.get("title") or table_dict.get("page_context_title") or "").strip()
    indicators = table_dict.get("indicators") or table_dict.get("rows") or []
    return bool(title or len(indicators) > 0)
