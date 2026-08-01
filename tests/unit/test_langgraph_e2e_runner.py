"""Tests de bout en bout (E2E) pour le runner LangGraph et l'exportation des fichiers."""

from __future__ import annotations

import json
from pathlib import Path
from vigilance.graph.runner import run_langgraph_comparison


def test_run_langgraph_comparison_e2e(tmp_path: Path) -> None:
    # Set up mock extraction files in tmp_path
    ext_dir = tmp_path / "extractions" / "rbc"
    (ext_dir / "2025" / "t4").mkdir(parents=True, exist_ok=True)
    (ext_dir / "2024" / "t4").mkdir(parents=True, exist_ok=True)

    t24_content = {"tables": [{"table_id": "tbl_p082_i01", "title": "Notations Tableau 58"}]}
    t25_content = {"tables": [{"table_id": "tbl_p085_i01", "title": "Notations Tableau 56"}]}

    with open(ext_dir / "2024" / "t4" / "tables.json", "w", encoding="utf-8") as f:
        json.dump(t24_content, f)

    with open(ext_dir / "2025" / "t4" / "tables.json", "w", encoding="utf-8") as f:
        json.dump(t25_content, f)

    out_dir = tmp_path / "resultats" / "rbc" / "2025_t4_vs_2024_t4"

    json_path = run_langgraph_comparison(
        bank="RBC",
        year_current=2025,
        quarter_current="T4",
        year_previous=2024,
        quarter_previous="T4",
        output_dir=out_dir,
    )

    assert json_path.exists()
    assert (out_dir / "comparison.xlsx").exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["bank_code"].upper() == "RBC"
    assert "global_summary" in data
    assert "matching" in data
