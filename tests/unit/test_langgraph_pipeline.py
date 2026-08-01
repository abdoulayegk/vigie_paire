"""Tests unitaires pour l'orchestration du graphe Multi-Agents LangGraph."""

from __future__ import annotations

from vigilance.graph.builder import build_comparison_graph
from vigilance.graph.state import ComparisonState


def test_langgraph_execution_flow_rbc() -> None:
    graph = build_comparison_graph()

    initial_state = ComparisonState(
        bank_code="RBC",
        year_current=2025,
        year_previous=2024,
        quarter_current="T4",
        quarter_previous="T4",
        previous_cards=[{"table_id": "tbl_p082_i01", "title": "Charges grevant les actifs Tableau 54"}],
        current_cards=[{"table_id": "tbl_p085_i01", "title": "Charges grevant les actifs Tableau 52"}],
    )

    final_output = graph.invoke(initial_state)

    # Vérification que le graphe a exécuté tous les nœuds correctement
    assert final_output["previous_cards"][0]["title"] == "Charges grevant les actifs"
    assert final_output["current_cards"][0]["title"] == "Charges grevant les actifs"
    assert final_output["hybrid_recovery_applied"] is True
    assert final_output["devil_advocate_applied"] is True


def test_langgraph_execution_flow_bmo_bypasses_hybrid() -> None:
    graph = build_comparison_graph()

    initial_state = ComparisonState(
        bank_code="BMO",
        year_current=2025,
        year_previous=2024,
        quarter_current="T4",
        quarter_previous="T4",
        previous_cards=[{"table_id": "tbl_p010_i01", "title": "Fonds propres"}],
        current_cards=[{"table_id": "tbl_p012_i01", "title": "Fonds propres"}],
    )

    final_output = graph.invoke(initial_state)

    # BMO a bypassé la récupération hybride et est allé directement à l'avocat du diable
    assert final_output["hybrid_recovery_applied"] is False
    assert final_output["devil_advocate_applied"] is True
