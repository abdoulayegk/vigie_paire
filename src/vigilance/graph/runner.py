"""Connecteur d'exécution de bout-en-bout du pipeline Multi-Agents LangGraph."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vigilance.comparison_excel import generate_comparison_excel
from vigilance.comparison_io import _atomic_write_json
from vigilance.graph.builder import build_comparison_graph
from vigilance.graph.state import ComparisonState
from vigilance.ui_config import OUTPUT_DIR, RESULTATS_DIR

logger = logging.getLogger(__name__)


def run_langgraph_comparison(
    bank: str,
    year_current: int,
    quarter_current: str,
    year_previous: int,
    quarter_previous: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Exécute la comparaison complète de deux rapports bancaires via le graphe LangGraph.

    Args:
        bank: Code de la banque (ex: RBC, BMO, TD).
        year_current: Année du trimestre courant.
        quarter_current: Trimestre courant (ex: T4).
        year_previous: Année du trimestre précédent.
        quarter_previous: Trimestre précédent.
        output_dir: Dossier optionnel pour sauvegarder comparison.json et comparison.xlsx.

    Returns:
        Chemin Path du fichier comparison.json généré.
    """
    bank_clean = bank.strip().lower()
    q_curr = quarter_current.strip().lower()
    q_prev = quarter_previous.strip().lower()

    # Emplacements d'extraction des tables
    extraction_root = OUTPUT_DIR / "extractions"
    t2_path = extraction_root / bank_clean / str(year_current) / q_curr / "tables.json"
    t1_path = extraction_root / bank_clean / str(year_previous) / q_prev / "tables.json"

    previous_cards: list[dict[str, Any]] = []
    current_cards: list[dict[str, Any]] = []

    if t1_path.exists():
        with open(t1_path, "r", encoding="utf-8") as f:
            previous_cards = json.load(f).get("tables", [])

    if t2_path.exists():
        with open(t2_path, "r", encoding="utf-8") as f:
            current_cards = json.load(f).get("tables", [])

    initial_state = ComparisonState(
        bank_code=bank,
        year_current=year_current,
        year_previous=year_previous,
        quarter_current=quarter_current,
        quarter_previous=quarter_previous,
        previous_cards=previous_cards,
        current_cards=current_cards,
    )

    graph = build_comparison_graph()
    final_state = graph.invoke(initial_state)

    # Formatage de la sortie d'analyse unifiée
    payload = {
        "bank_code": bank,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "global_summary": final_state.get("global_summary", {}),
        "matching": {
            "matched_pairs": final_state.get("matched_pairs", []),
            "tables_removed": final_state.get("unmatched_previous", []),
            "tables_added": final_state.get("unmatched_current", []),
        },
        "pair_comparisons": final_state.get("pair_comparisons", []),
    }

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = RESULTATS_DIR / bank_clean / f"{year_current}_{q_curr}_vs_{year_previous}_{q_prev}"

    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "comparison.json"
    excel_path = out_path / "comparison.xlsx"

    _atomic_write_json(json_path, payload)
    logger.info("[LangGraph Runner] Sauvegardé JSON : %s", json_path)

    generate_comparison_excel(payload, excel_path)
    logger.info("[LangGraph Runner] Généré Excel : %s", excel_path)

    return json_path
