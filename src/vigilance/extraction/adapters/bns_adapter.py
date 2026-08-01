"""Adaptateur d'extraction spécifique à la Banque Scotia (BNS)."""

from __future__ import annotations

from vigilance.extraction.adapters.base_adapter import BaseBankAdapter


class BNSBankAdapter(BaseBankAdapter):
    """Adaptateur pour la Banque Scotia (BNS)."""

    bank_code: str = "bns"
