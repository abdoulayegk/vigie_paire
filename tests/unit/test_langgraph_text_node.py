"""Tests unitaires pour le nœud d'analyse textuelle text_triage_node dans LangGraph."""

from __future__ import annotations

from vigilance.graph.builder import build_comparison_graph
from vigilance.graph.nodes import text_triage_node
from vigilance.graph.state import ComparisonState


def test_text_triage_node_execution() -> None:
    state = ComparisonState(bank_code="RBC", year_current=2025)
    res = text_triage_node(state)

    assert "warnings" in res
    assert len(res["warnings"]) >= 1
    assert "Analyse textuelle validée pour RBC" in res["warnings"][0]


def test_langgraph_pipeline_with_text_triage() -> None:
    graph = build_comparison_graph()
    initial_state = ComparisonState(
        bank_code="CIBC",
        year_current=2025,
        quarter_current="T4",
    )

    final_output = graph.invoke(initial_state)

    assert final_output["global_summary"] != {}
    assert len(final_output["warnings"]) >= 1
    assert "Analyse textuelle validée pour CIBC" in final_output["warnings"][0]
