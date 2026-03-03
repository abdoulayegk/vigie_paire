"""Regression tests for eval_matching_harness with gold fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
HARNESS_SCRIPT = REPO_ROOT / "scripts" / "eval_matching_harness.py"


def test_eval_harness_runs_with_fixtures() -> None:
    """Eval harness runs without error on sample comparison JSON."""
    result_path = FIXTURES_DIR / "eval_comparison_sample.json"
    if not result_path.exists():
        pytest.skip("Fixture eval_comparison_sample.json not found")

    proc = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), "--result", str(result_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Summary counts" in proc.stdout


def test_eval_harness_with_gold_computes_metrics() -> None:
    """Eval harness with gold CSVs computes table and rename metrics."""
    result_path = FIXTURES_DIR / "eval_comparison_sample.json"
    gold_tables = FIXTURES_DIR / "gold_tables.csv"
    gold_renames = FIXTURES_DIR / "gold_renames.csv"
    if not result_path.exists() or not gold_tables.exists() or not gold_renames.exists():
        pytest.skip("Gold fixtures not found")

    proc = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPT),
            "--result",
            str(result_path),
            "--gold-tables",
            str(gold_tables),
            "--gold-renames",
            str(gold_renames),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Table match metrics" in proc.stdout
    assert "Rename metrics" in proc.stdout
    assert "precision" in proc.stdout.lower() or "tp" in proc.stdout.lower()


def test_extract_renames_from_sample() -> None:
    """extract_renames returns expected pairs from sample fixture."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_harness", HARNESS_SCRIPT
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    payload = mod.load_result(FIXTURES_DIR / "eval_comparison_sample.json")
    renames = mod.extract_renames(payload)
    assert len(renames) == 2
    assert ("Ratio CET1 (phase-in)", "Ratio CET1") in renames
    assert (
        "Actifs ponderes risques",
        "Actifs ponderes en fonction des risques",
    ) in renames


def test_rename_metrics_with_gold() -> None:
    """rename_metrics computes precision when gold is provided."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_harness", HARNESS_SCRIPT
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    payload = mod.load_result(FIXTURES_DIR / "eval_comparison_sample.json")
    renames = mod.extract_renames(payload)
    gold = mod.load_gold_renames(FIXTURES_DIR / "gold_renames.csv")
    metrics = mod.rename_metrics(renames, gold)
    assert metrics["rename_precision"] is not None
    assert metrics["tp"] == 2
    assert metrics["gold_renames"] == 2
    assert metrics["pred_renames"] == 2
