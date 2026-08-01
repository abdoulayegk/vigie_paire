"""Tests unitaires pour l'intégration de LangChain ChatOpenAI et Structured Outputs dans les nœuds LangGraph."""

from __future__ import annotations

from unittest.mock import MagicMock
from vigilance.graph.llm import get_llm
from vigilance.graph.nodes import devil_advocate_node
from vigilance.graph.state import ComparisonState
from vigilance.models.comparison_models import DevilAdvocateResponse, DevilAdvocateMatch


def test_get_llm_factory() -> None:
    llm = get_llm(model_name="gpt-4o", temperature=0.0, api_key="sk-fake-key-for-testing")
    assert llm.model_name == "gpt-4o"
    assert llm.temperature == 0.0
    assert llm.max_retries == 3


def test_devil_advocate_node_with_structured_output_mock() -> None:
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Simulated Pydantic response
    mock_response = DevilAdvocateResponse(
        new_matches=[
            DevilAdvocateMatch(
                previous_table_id="tbl_p082_i01",
                current_table_id="tbl_p085_i01",
                match_confidence=0.98,
                reason="Aligned business indicators",
            )
        ]
    )
    mock_structured.invoke.return_value = mock_response

    state = ComparisonState(bank_code="RBC", year_current=2025)
    res = devil_advocate_node(state, llm=mock_llm)

    assert res["devil_advocate_applied"] is True
    mock_llm.with_structured_output.assert_called_once()
