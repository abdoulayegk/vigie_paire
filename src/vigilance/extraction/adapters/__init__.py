"""Package des adaptateurs d'extraction spécifiques par banque canadienne."""

from __future__ import annotations

from typing import Type
from vigilance.extraction.adapters.base_adapter import BaseBankAdapter
from vigilance.extraction.adapters.bnc_adapter import BNCBankAdapter
from vigilance.extraction.adapters.bmo_adapter import BMOBankAdapter
from vigilance.extraction.adapters.bns_adapter import BNSBankAdapter
from vigilance.extraction.adapters.cibc_adapter import CIBCBankAdapter
from vigilance.extraction.adapters.rbc_adapter import RBCBankAdapter
from vigilance.extraction.adapters.td_adapter import TDBankAdapter

_BANK_ADAPTER_REGISTRY: dict[str, Type[BaseBankAdapter]] = {
    "rbc": RBCBankAdapter,
    "bmo": BMOBankAdapter,
    "cibc": CIBCBankAdapter,
    "bns": BNSBankAdapter,
    "td": TDBankAdapter,
    "bnc": BNCBankAdapter,
}


def get_bank_adapter(bank_code: str) -> BaseBankAdapter:
    """Retourne l'adaptateur d'extraction spécifique pour la banque demandée.

    Defaults à BaseBankAdapter si la banque n'a pas d'adaptateur sur-mesure.
    """
    code_clean = str(bank_code or "").strip().lower()
    adapter_cls = _BANK_ADAPTER_REGISTRY.get(code_clean, BaseBankAdapter)
    return adapter_cls()
