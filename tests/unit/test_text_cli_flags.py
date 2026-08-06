from __future__ import annotations

import pytest

from vigie.cli import run_text_compare
from vigie.pipelines import complet


def test_text_cli_accepts_short_quarter_flags() -> None:
    parser = run_text_compare.build_parser()
    args = parser.parse_args(["--banque", "BMO", "--annee", "2025", "--T2"])

    assert args.banque == "BMO"
    assert args.annee == 2025
    assert args.trimestre == "T2"


def test_text_cli_rejects_legacy_quarter_option() -> None:
    parser = run_text_compare.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--banque", "BMO", "--annee", "2025", "--quarter", "T2"])


def test_run_text_compare_accepts_strict_sections() -> None:
    parser = run_text_compare.build_parser()
    args = parser.parse_args(["--banque", "BNS", "--annee", "2025", "--T2", "--strict-sections"])
    assert args.strict_sections is True


def test_complet_parser_rejects_english_bank_flag() -> None:
    parser = complet.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--bank", "BNC", "--annee", "2025", "--T2"])


def test_complet_parser_accepts_sans_extraction() -> None:
    parser = complet.build_parser()
    args = parser.parse_args(["--banque", "BNC", "--annee", "2025", "--T2", "--sans-extraction"])
    assert args.banque == "BNC"
    assert args.sans_extraction is True
    assert args.forcer_extraction is False
