from __future__ import annotations

import pytest

from vigilance.cli import run_text_compare, run_text_extract


@pytest.mark.parametrize("module", [run_text_extract, run_text_compare])
def test_text_clis_accept_short_quarter_flags(module) -> None:
    parser = module.build_parser()
    base_args = ["--bank", "BMO", "--year", "2025", "--T2"]

    if module is run_text_extract:
        base_args += ["--pdf", "dummy.pdf"]

    args = parser.parse_args(base_args)

    assert args.bank == "BMO"
    assert args.year == 2025
    assert args.quarter_flag == "T2"


@pytest.mark.parametrize("module", [run_text_extract, run_text_compare])
def test_text_clis_reject_legacy_quarter_option(module) -> None:
    parser = module.build_parser()
    legacy_args = ["--bank", "BMO", "--year", "2025", "--quarter", "T2"]

    if module is run_text_extract:
        legacy_args += ["--pdf", "dummy.pdf"]

    with pytest.raises(SystemExit):
        parser.parse_args(legacy_args)
