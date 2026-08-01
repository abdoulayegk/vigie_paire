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

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = RESULTATS_DIR / bank_clean / f"{year_current}_{q_curr}_vs_{year_previous}_{q_prev}"

    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "comparison.json"
    excel_path = out_path / "comparison.xlsx"

    # Emplacement du JSON de comparaison préexistant en production (le cas échéant)
    prod_json = RESULTATS_DIR / bank_clean / f"{year_current}_{q_curr}_vs_{year_previous}_{q_prev}" / "comparison.json"
    source_json = json_path if json_path.exists() else (prod_json if prod_json.exists() else None)

    # Si des données de diff préexistantes existent, on les enrichit avec l'état LangGraph
    if source_json and source_json.exists():
        try:
            with open(source_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    existing_data["artifact_type"] = "report_comparison"
                    existing_data["global_summary"] = final_state.get("global_summary", existing_data.get("global_summary", {}))
                    _atomic_write_json(json_path, existing_data)
                    logger.info("[LangGraph Runner] Mis à jour JSON conforme pour Dash : %s", json_path)
                    generate_comparison_excel(existing_data, excel_path)
                    return json_path
        except Exception as e:
            logger.warning("[LangGraph Runner] Erreur lors du chargement de %s: %s", source_json, e)

    matched_pairs = final_state.get("matched_pairs", [])
    tables_removed = final_state.get("unmatched_previous", [])
    tables_added = final_state.get("unmatched_current", [])
    pair_comparisons = final_state.get("pair_comparisons", [])

    summary = {
        "matched_pairs_total": len(matched_pairs),
        "tables_added_total": len(tables_added),
        "tables_removed_total": len(tables_removed),
        "indicator_changes_total": sum(len(p.get("added_indicators", []) or []) + len(p.get("removed_indicators", []) or []) for p in pair_comparisons),
        "footnote_changes_total": sum(sum(p.get("footnotes_counts", {}).values()) for p in pair_comparisons if isinstance(p.get("footnotes_counts"), dict)),
        "high_priority_items_total": sum(1 for p in pair_comparisons if p.get("priority") in ("critique", "prioritaire")),
    }

    # Formatage de la sortie d'analyse unifiée avec le schéma canonique "report_comparison" requis par Dash
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "report_comparison",
        "bank_code": bank,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "global_summary": final_state.get("global_summary", {}),
        "matching": {
            "matched_pairs": matched_pairs,
            "tables_removed": tables_removed,
            "tables_added": tables_added,
        },
        "pair_comparisons": pair_comparisons,
        "summary": summary,
    }

    _atomic_write_json(json_path, payload)
    logger.info("[LangGraph Runner] Sauvegardé JSON conforme Dash : %s", json_path)

    generate_comparison_excel(payload, excel_path)
    logger.info("[LangGraph Runner] Généré Excel : %s", excel_path)

    return json_path
