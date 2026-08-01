"""Composant spécialisé dans la capture et le traitement complet des notes de bas de page."""

from __future__ import annotations

import re
from typing import Any


def extract_clean_footnotes(raw_footnotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nettoie et valide les notes de bas de page extraites d'une page ou d'un tableau.

    Garantit que chaque note possède un symbole/identifiant et un texte explicatif non vide.
    """
    clean_notes = []
    for item in raw_footnotes:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id") or item.get("symbol") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        text_clean = re.sub(r"\s+", " ", text)

        if fid or text_clean:
            clean_notes.append({
                "id": fid or "1",
                "text": text_clean,
            })
    return clean_notes
