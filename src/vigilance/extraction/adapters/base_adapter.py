"""Classe abstraite de base pour les adaptateurs d'extraction par banque."""

from __future__ import annotations

import re
from typing import Any


class BaseBankAdapter:
    """Interface d'adaptation commune pour chaque banque canadienne.

    Isole les tics de rédaction et règles d'édition spécifiques à chaque banque
    (titres de tableaux, marqueurs de notes, délimitation de sections).
    """

    bank_code: str = "base"

    def clean_table_title(self, raw_title: str) -> str:
        """Nettoie le titre brut d'un tableau spécifique à cette banque."""
        if not raw_title:
            return ""
        clean = raw_title.strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    def normalize_footnote_markers(self, text: str) -> str:
        """Normalise les marqueurs de notes de bas de page (ex: exposants, parenthèses)."""
        if not text:
            return ""
        return text.strip()

    def process_extracted_table(self, table_data: dict[str, Any]) -> dict[str, Any]:
        """Applique les ajustements métiers spécifiques à cette banque sur la carte de tableau."""
        res = dict(table_data)
        if "title" in res:
            res["title"] = self.clean_table_title(res["title"])
        return res
