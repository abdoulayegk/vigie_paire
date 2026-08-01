"""Adaptateur d'extraction spécifique à la Banque CIBC."""

from __future__ import annotations

from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class CIBCBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque CIBC."""

    bank_code: str = "cibc"
