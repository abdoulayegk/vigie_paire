"""Tests unitaires pour les adaptateurs d'extraction par banque."""

from __future__ import annotations

from vigilance.extraction.adapters import (
    BNCBankAdapter,
    RBCBankAdapter,
    TDBankAdapter,
    get_bank_adapter,
)


def test_factory_returns_correct_adapters() -> None:
    assert isinstance(get_bank_adapter("rbc"), RBCBankAdapter)
    assert isinstance(get_bank_adapter("td"), TDBankAdapter)
    assert isinstance(get_bank_adapter("bnc"), BNCBankAdapter)
    assert get_bank_adapter("unknown").bank_code == "base"


def test_rbc_adapter_clean_title() -> None:
    adapter = get_bank_adapter("rbc")
    assert adapter.clean_table_title("TABLEAU  54   Risques de liquidité") == "Tableau 54 Risques de liquidité"


def test_td_adapter_footnote_markers() -> None:
    adapter = get_bank_adapter("td")
    assert adapter.clean_table_title("RATIO DE LIQUIDITÉ¹") == "RATIO DE LIQUIDITÉ"
    assert adapter.normalize_footnote_markers("¹ Le ratio est conforme") == "Le ratio est conforme"


def test_bnc_adapter_title_cleaning() -> None:
    adapter = get_bank_adapter("bnc")
    assert adapter.clean_table_title("BANQUE NATIONALE - Rapport de gestion") == "Rapport de gestion"
