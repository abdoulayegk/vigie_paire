from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "run_text_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_text_pipeline", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
run_text_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_text_pipeline)


def test_build_parser_accepts_short_quarter_flag() -> None:
    parser = run_text_pipeline.build_parser()

    args = parser.parse_args(["--bank", "BMO", "--year", "2025", "--T2"])

    assert args.bank == "BMO"
    assert args.year == 2025
    assert args.quarter_flag == "T2"


def test_build_parser_rejects_legacy_quarter_option() -> None:
    parser = run_text_pipeline.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--bank", "BMO", "--year", "2025", "--quarter", "T2"])
