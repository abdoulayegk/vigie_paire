from __future__ import annotations

import importlib.util
import sys
import types
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


def test_build_parser_accepts_extraction_only_flags() -> None:
    parser = run_text_pipeline.build_parser()

    args = parser.parse_args(
        [
            "--bank",
            "BMO",
            "--year",
            "2025",
            "--T4",
            "--extract-only",
            "--force-extraction",
        ]
    )

    assert args.extract_only is True
    assert args.force_extraction is True


def test_build_parser_rejects_legacy_quarter_option() -> None:
    parser = run_text_pipeline.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--bank", "BMO", "--year", "2025", "--quarter", "T2"])


def test_skip_comparison_runs_extraction_only_with_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous_pdf = tmp_path / "BMO_2024_T4.pdf"
    current_pdf = tmp_path / "BMO_2025_T4.pdf"
    captured: dict[str, object] = {}

    def fake_find_pdf_pair(**_kwargs: object) -> tuple[Path, Path]:
        return previous_pdf, current_pdf

    def fake_run_text_extraction_pipeline(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "extraction_artifact_t1": str(tmp_path / "text_extraction_2024_t4.md"),
            "extraction_artifact_t2": str(tmp_path / "text_extraction_2025_t4.md"),
        }

    fake_pipeline = types.ModuleType("vigilance.text_analysis")
    fake_pipeline.run_text_extraction_pipeline = fake_run_text_extraction_pipeline

    monkeypatch.setattr(run_text_pipeline, "find_pdf_pair", fake_find_pdf_pair)
    monkeypatch.setitem(sys.modules, "vigilance.text_analysis", fake_pipeline)

    rc = run_text_pipeline.main(
        [
            "--bank",
            "BMO",
            "--year",
            "2025",
            "--T4",
            "--skip-comparison",
            "--force-extraction",
            "--out-root",
            str(tmp_path / "outputs" / "resultats"),
        ]
    )

    assert rc == 0
    assert captured["bank_code"] == "bmo"
    assert captured["quarter_current"] == "t4"
    assert captured["force_extraction"] is True
