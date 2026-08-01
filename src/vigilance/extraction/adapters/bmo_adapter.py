"""Adaptateur d'extraction spécifique à la Banque de Montréal (BMO)."""

from __future__ import annotations

from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class BMOBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque de Montréal (BMO)."""

    bank_code: str = "bmo"
