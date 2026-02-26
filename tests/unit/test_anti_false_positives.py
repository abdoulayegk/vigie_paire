"""Tests de non-regression pour les correctifs anti-faux positifs (TD Bank)."""

from __future__ import annotations

from vigilance.config import get_matching_thresholds
from vigilance.extraction.docling_normalization import _is_footnote_row
from vigilance.utils.indicator_cleaner import (
    is_header_footer_table_title,
    strip_note_refs_from_title,
)
from vigilance.utils.matching_normalizer import (
    _classify_excluded_line,
    is_date_only_line,
    is_non_indicator_line,
)


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


# ---------------------------------------------------------------------------
# Tests for aggressive unit/period/date exclusion (false-positive suppression)
# ---------------------------------------------------------------------------

class TestClassifyExcludedLineUnitPeriod:
    """_classify_excluded_line excludes unit/period headers aggressively."""

    def test_en_millions_de_dollars_sauf(self) -> None:
        assert _classify_excluded_line(
            "En millions de dollars, sauf les montants par action et les pourcentages au 31 janvier 2025"
        ) == "unit"

    def test_en_millions(self) -> None:
        assert _classify_excluded_line("En millions") == "unit"

    def test_en_alone(self) -> None:
        assert _classify_excluded_line("En") == "unit"

    def test_en_millions_de_dollars_comma(self) -> None:
        assert _classify_excluded_line("En millions de dollars,") == "unit"

    def test_moyenne_du_trimestre_clos(self) -> None:
        assert _classify_excluded_line(
            "moyenne du trimestre clos le 31 janvier 2025"
        ) == "unit"

    def test_en_milliards(self) -> None:
        assert _classify_excluded_line("En milliards de dollars") == "unit"

    def test_en_milliers(self) -> None:
        assert _classify_excluded_line("En milliers") == "unit"

    def test_in_millions(self) -> None:
        assert _classify_excluded_line("In millions of Canadian dollars") == "unit"

    def test_in_billions(self) -> None:
        assert _classify_excluded_line("In billions") == "unit"

    def test_paren_en_millions(self) -> None:
        assert _classify_excluded_line("(en millions de dollars canadiens)") == "unit"

    def test_real_indicator_not_excluded(self) -> None:
        assert _classify_excluded_line("Risque de credit") is None

    def test_depots_personnels_not_excluded(self) -> None:
        assert _classify_excluded_line("Depots personnels") is None

    def test_actions_ordinaires_not_excluded(self) -> None:
        assert _classify_excluded_line("Actions ordinaires") is None


class TestClassifyExcludedLineDateWithNotes:
    """_classify_excluded_line excludes date lines with trailing footnote markers."""

    def test_date_with_trailing_notes(self) -> None:
        assert _classify_excluded_line("31 janvier 2025 1, 2") == "date"

    def test_date_with_single_trailing_note(self) -> None:
        assert _classify_excluded_line("31 janvier 2025 1") == "date"

    def test_date_without_year(self) -> None:
        assert _classify_excluded_line("31 janvier") == "date"

    def test_date_plain(self) -> None:
        assert _classify_excluded_line("31 janvier 2025") == "date"

    def test_au_date_trailing_noise(self) -> None:
        assert _classify_excluded_line("Au 31 octobre 2024 1)") == "date"


class TestIsDateOnlyLineAggressive:
    """is_date_only_line catches the same aggressive patterns."""

    def test_en_millions(self) -> None:
        assert is_date_only_line("En millions") is True

    def test_en_alone(self) -> None:
        assert is_date_only_line("En") is True

    def test_date_trailing_notes(self) -> None:
        assert is_date_only_line("31 janvier 2025 1, 2") is True

    def test_moyenne_trimestre(self) -> None:
        assert is_date_only_line("moyenne du trimestre clos le 31 janvier 2025") is True

    def test_real_indicator(self) -> None:
        assert is_date_only_line("Actifs totaux") is False


class TestClassifyExcludedLineExactRegression:
    """Regression tests with EXACT strings: unit/date must never appear in added/removed/renamed.

    CIBC/TD: "En millions", "En millions de dollars", "Au 31 janvier 2025", etc.
    """

    def test_en_unit(self) -> None:
        assert _classify_excluded_line("En") == "unit"

    def test_en_millions_unit(self) -> None:
        assert _classify_excluded_line("En millions") == "unit"

    def test_en_millions_de_dollars_sauf_indication_contraire_unit(self) -> None:
        assert _classify_excluded_line(
            "En millions de dollars, sauf indication contraire"
        ) == "unit"

    def test_au_31_janvier_2025_date(self) -> None:
        assert _classify_excluded_line("Au 31 janvier 2025") == "date"

    def test_en_millions_de_dollars_au_31_janvier_2025_unit(self) -> None:
        assert _classify_excluded_line(
            "En millions de dollars, au 31 janvier 2025"
        ) == "unit"

    def test_31_janvier_2025_1_2_date(self) -> None:
        assert _classify_excluded_line("31 janvier 2025 1, 2") == "date"

    def test_en_millions_de_dollars_unit(self) -> None:
        assert _classify_excluded_line("En millions de dollars") == "unit"

    def test_au_30_avril_2025_date(self) -> None:
        assert _classify_excluded_line("Au 30 avril 2025") == "date"
