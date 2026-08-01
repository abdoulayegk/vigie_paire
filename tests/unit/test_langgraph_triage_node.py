"""Tests unitaires pour l'Agent Triage AMF et le Checkpointing MemorySaver."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from vigilance.graph.builder import build_comparison_graph
from vigilance.graph.nodes import amf_triage_node
from vigilance.graph.state import ComparisonState


def test_amf_triage_node_execution() -> None:
    state = ComparisonState(bank_code="RBC", quarter_current="T4-2025")
    result = amf_triage_node(state)

    assert "global_summary" in result
    assert "key_highlights" in result["global_summary"]
    assert len(result["global_summary"]["key_highlights"]) >= 2


def test_langgraph_checkpointing_with_thread_config() -> None:
    checkpointer = MemorySaver()
    graph = build_comparison_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_rbc_thread_1"}}
    initial_state = ComparisonState(
        bank_code="RBC",
        year_current=2025,
        quarter_current="T4",
        previous_cards=[{"table_id": "tbl_p082_i01", "title": "Notations Tableau 58"}],
        current_cards=[{"table_id": "tbl_p085_i01", "title": "Notations Tableau 56"}],
    )

    final_output = graph.invoke(initial_state, config=config)

    assert final_output["global_summary"] != {}
    assert final_output["devil_advocate_applied"] is True

    # Check that checkpoint state exists in memory for thread_id
    state_snapshot = graph.get_state(config)
    assert state_snapshot is not None
    assert state_snapshot.values["bank_code"] == "RBC"
