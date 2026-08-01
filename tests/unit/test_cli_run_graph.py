"""Tests unitaires pour la commande CLI vigie-graph-run et le runner asynchrone."""

from __future__ import annotations

import asyncio
from pathlib import Path
from vigilance.cli.run_graph import main
from vigilance.graph.runner import arun_langgraph_comparison


def test_cli_run_graph_main_help(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "vigie-graph-run" in captured.out or "Exécute la comparaison bancaire" in captured.out


def test_cli_run_graph_main_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "resultats" / "rbc"
    exit_code = main(
        [
            "--bank",
            "RBC",
            "--year",
            "2025",
            "--quarter",
            "T4",
            "--year-prev",
            "2024",
            "--quarter-prev",
            "T4",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "comparison.json").exists()


def test_arun_langgraph_comparison_async(tmp_path: Path) -> None:
    out_dir = tmp_path / "async_resultats" / "bmo"
    json_path = asyncio.run(
        arun_langgraph_comparison(
            bank="BMO",
            year_current=2025,
            quarter_current="T4",
            year_previous=2024,
            quarter_previous="T4",
            output_dir=out_dir,
        )
    )
    assert json_path.exists()
