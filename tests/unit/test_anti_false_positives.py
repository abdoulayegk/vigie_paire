"""Tests de non-regression pour les correctifs anti-faux positifs (TD Bank)."""

from __future__ import annotations

from vigilance.config import get_matching_thresholds
from vigilance.extraction.docling_normalization import _is_footnote_row
from vigilance.utils.indicator_cleaner import (
    is_header_footer_table_title,
    strip_note_refs_from_title,
)
from vigilance.utils.matching_normalizer import is_date_only_line, is_non_indicator_line


def test_is_date_only_line_unit_headers() -> None:
    """is_date_only_line filtre correctement les unites et dates."""
    assert is_date_only_line(
        "(en milliers d'actions / de parts et en millions de dollars canadiens, sauf indication contraire)"
    ) is True
    assert is_date_only_line("(en millions de dollars canadiens) Au 31 janvier 2025") is True
    assert is_date_only_line("Dépôts personnels") is False


def test_strip_note_refs_from_title() -> None:
    """strip_note_refs_from_title retire les refs de notes en fin de titre."""
    assert strip_note_refs_from_title("NOTATIONS DE CREDIT1") == "NOTATIONS DE CREDIT"
    assert strip_note_refs_from_title("INDICATEURS BISM1") == "INDICATEURS BISM"
    assert strip_note_refs_from_title("Tableau 12") == "Tableau 12"
    assert strip_note_refs_from_title("31 janvier 2025 1, 2") == "31 janvier 2025"
    assert strip_note_refs_from_title("Expositions 1, 2, 3") == "Expositions"


def test_is_footnote_row() -> None:
    """_is_footnote_row filtre les lignes (1), [2] mais pas les listes numérotées."""
    assert _is_footnote_row(["(1) Definition"]) is True
    assert _is_footnote_row(["1. Dépôts personnels"]) is False
    assert _is_footnote_row(["[2] Voir note"]) is True
    assert _is_footnote_row(["Dépôts"]) is False


def test_get_matching_thresholds_bank_code_case_insensitive() -> None:
    """get_matching_thresholds applique les overrides TD quelle que soit la casse."""
    td_lower = get_matching_thresholds(bank_code="td")
    td_upper = get_matching_thresholds(bank_code="TD")
    assert td_lower.get("table_rename_threshold") == 0.78
    assert td_upper.get("table_rename_threshold") == 0.78


def test_is_non_indicator_line() -> None:
    """is_non_indicator_line filtre totaux, unites, nombres seuls."""
    assert is_non_indicator_line("1") is True
    assert is_non_indicator_line("26") is True
    assert is_non_indicator_line("Total du passif et des capitaux propres") is True
    assert is_non_indicator_line("Total des elements hors bilan") is True
    assert is_non_indicator_line("en millions de dollars") is True
    assert is_non_indicator_line("%") is True
    assert is_non_indicator_line("Dépôts personnels") is False
    assert is_non_indicator_line("Risque de crédit") is False


def test_is_header_footer_table_title() -> None:
    """is_header_footer_table_title detecte les titres en-tete/pied RBC."""
    assert is_header_footer_table_title("24 Banque Royale du Canada Premier trimestre de 2025") is True
    assert is_header_footer_table_title("Banque Royale du Canada Premier trimestre de 2025 27") is True
    assert is_header_footer_table_title("Gestion des fonds propres") is False
    assert is_header_footer_table_title("Risque de crédit") is False
