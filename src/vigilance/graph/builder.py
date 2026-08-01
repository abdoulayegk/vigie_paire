"""Assemblage du graphe d'états LangGraph (StateGraph)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from vigilance.graph.nodes import (
    bank_normalizer_node,
    devil_advocate_node,
    hybrid_recovery_node,
    primary_matcher_node,
)
from vigilance.graph.state import ComparisonState


def _should_run_hybrid_recovery(state: ComparisonState) -> str:
    """Aiguillage conditionnel : vérifie si la récupération hybride doit être exécutée."""
    if state.bank_code.strip().lower() == "rbc" and state.unmatched_previous:
        return "hybrid_recovery"
    return "devil_advocate"


def build_comparison_graph() -> StateGraph:
    """Construit et retourne le graphe d'états Multi-Agents LangGraph."""
    workflow = StateGraph(ComparisonState)

    # Ajout des nœuds agents
    workflow.add_node("normalizer", bank_normalizer_node)
    workflow.add_node("primary_matcher", primary_matcher_node)
    workflow.add_node("hybrid_recovery", hybrid_recovery_node)
    workflow.add_node("devil_advocate", devil_advocate_node)

    # Connexion du flux (edges)
    workflow.add_edge(START, "normalizer")
    workflow.add_edge("normalizer", "primary_matcher")

    # Branchement conditionnel
    workflow.add_conditional_edges(
        "primary_matcher",
        _should_run_hybrid_recovery,
        {
            "hybrid_recovery": "hybrid_recovery",
            "devil_advocate": "devil_advocate",
        },
    )

    workflow.add_edge("hybrid_recovery", "devil_advocate")
    workflow.add_edge("devil_advocate", END)

    return workflow.compile()
