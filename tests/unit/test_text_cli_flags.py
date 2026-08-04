from __future__ import annotations

import pytest

from vigie.cli import run_text_compare


def test_text_cli_accepts_short_quarter_flags() -> None:
    parser = run_text_compare.build_parser()
    args = parser.parse_args(["--bank", "BMO", "--year", "2025", "--T2"])

    assert args.bank == "BMO"
    assert args.year == 2025
    assert args.quarter_flag == "T2"


def test_text_cli_rejects_legacy_quarter_option() -> None:
    parser = run_text_compare.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--bank", "BMO", "--year", "2025", "--quarter", "T2"])


def test_run_text_compare_accepts_strict_sections() -> None:
    parser = run_text_compare.build_parser()
    args = parser.parse_args(["--bank", "BNS", "--year", "2025", "--T2", "--strict-sections"])
    assert args.strict_sections is True
