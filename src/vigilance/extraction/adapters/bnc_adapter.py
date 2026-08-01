"""Adaptateur d'extraction spécifique à la Banque Nationale (BNC)."""

from __future__ import annotations

import re
from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class BNCBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque Nationale (BNC)."""

    bank_code: str = "bnc"

    def clean_table_title(self, raw_title: str) -> str:
        """Nettoie le titre du tableau spécifique à la Banque Nationale."""
        clean = super().clean_table_title(raw_title)
        clean = re.sub(r"^\s*BANQUE NATIONALE\s*[-|]\s*", "", clean, flags=re.IGNORECASE)
        return clean.strip()
