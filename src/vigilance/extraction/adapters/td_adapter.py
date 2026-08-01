"""Adaptateur d'extraction spécifique à la Banque Toronto-Dominion (TD)."""

from __future__ import annotations

import re
from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class TDBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque TD.

    Nettoie les symboles en exposant (¹, ², ³, ⁴) et la section NSFR.
    """

    bank_code: str = "td"

    def clean_table_title(self, raw_title: str) -> str:
        """Nettoie les exposants de notes du titre de tableau TD."""
        clean = super().clean_table_title(raw_title)
        clean = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", clean).strip()
        return clean

    def normalize_footnote_markers(self, text: str) -> str:
        """Normalise les marqueurs de notes en exposant de la Banque TD."""
        clean = super().normalize_footnote_markers(text)
        return re.sub(r"^[¹²³⁴⁵⁶⁷⁸⁹⁰]\s*", "", clean).strip()
