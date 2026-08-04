from __future__ import annotations

from pathlib import Path

import pytest

from vigie.pipelines import texte as run_text_pipeline


def test_build_parser_accepts_short_quarter_flag() -> None:
    parser = run_text_pipeline.build_parser()

    args = parser.parse_args(["--banque", "BMO", "--annee", "2025", "--T2"])

    assert args.banque == "BMO"
    assert args.annee == 2025
    assert args.trimestre == "T2"


def test_build_parser_accepts_extraction_only_flags() -> None:
    parser = run_text_pipeline.build_parser()

    args = parser.parse_args(
        [
            "--banque",
            "BMO",
            "--annee",
            "2025",
            "--T4",
            "--extraction-seule",
            "--forcer-extraction",
        ]
    )

    assert args.extraction_seule is True
    assert args.forcer_extraction is True


def test_build_parser_rejects_legacy_quarter_option() -> None:
    parser = run_text_pipeline.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--banque", "BMO", "--annee", "2025", "--quarter", "T2"])


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

    monkeypatch.setattr(run_text_pipeline, "find_pdf_pair", fake_find_pdf_pair)
    monkeypatch.setattr(
        "vigie.analyse_texte.pipeline.run_text_extraction_pipeline",
        fake_run_text_extraction_pipeline,
    )

    rc = run_text_pipeline.main(
        [
            "--banque",
            "BMO",
            "--annee",
            "2025",
            "--T4",
            "--sans-comparaison",
            "--forcer-extraction",
            "--sortie",
            str(tmp_path / "outputs" / "resultats"),
        ]
    )

    assert rc == 0
    assert captured["bank_code"] == "bmo"
    assert captured["force_extraction"] is True
    assert captured["pdf_previous"] == previous_pdf
    assert captured["pdf_current"] == current_pdf
