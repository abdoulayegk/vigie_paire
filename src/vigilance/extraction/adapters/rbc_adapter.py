"""Adaptateur d'extraction spécifique à la Banque Royale (RBC)."""

from __future__ import annotations

import re
from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class RBCBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque Royale (RBC).

    Traite la numérotation canonique 'Tableau XX' et les structures de risques RBC.
    """

    bank_code: str = "rbc"

    def clean_table_title(self, raw_title: str) -> str:
        """Nettoie le titre du tableau spécifique à la Banque Royale (ex: Tableau XX)."""
        clean = super().clean_table_title(raw_title)
        clean = re.sub(r"^(?:TABLEAU|Tableau)\s*(\d+)\s*[-:]?\s*", r"Tableau \1 ", clean)
        return clean.strip()
