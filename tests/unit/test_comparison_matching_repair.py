"""Tests du mecanisme de reparation structurelle de l'appariement."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vigilance.comparison_matching import (
    _build_matching_repair_response_model,
    _run_matching_stage,
)


def _card(table_id: str, summary: str = "") -> dict[str, object]:
    return {
        "table_id": table_id,
        "table_summary": summary or table_id,
        "indicators": [summary or table_id],
        "headers": ["2025", "2024"],
        "row_count": 1,
    }


def test_valid_primary_response_keeps_existing_single_call_path() -> None:
    calls: list[dict[str, object]] = []

    def fake_call_openai_json(**kwargs):
        calls.append(kwargs)
        return {
            "current_table_decisions": [
                {
                    "current_table_id": "curr_a",
                    "decision": "matched",
                    "previous_table_id": "prev_a",
                    "match_confidence": 0.97,
                    "reason": "Same table.",
                }
            ]
        }

    result = _run_matching_stage(
        [_card("prev_a")],
        [_card("curr_a")],
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model="gpt-test",
        call_openai_json=fake_call_openai_json,
    )

    assert len(calls) == 1
    assert "response_model" not in calls[0]
    assert result["current_table_decisions"][0]["previous_table_id"] == "prev_a"
    assert result["validation_retries_total"] == 0


def test_wrong_current_namespace_in_previous_id_gets_targeted_repair() -> None:
    calls: list[dict[str, object]] = []
    responses = [
        {
            "current_table_decisions": [
                {
                    "current_table_id": "tbl_p110_i02",
                    "decision": "matched",
                    "previous_table_id": "tbl_p110_i02",
                    "match_confidence": 0.95,
                    "reason": "Same-looking current identifier used by mistake.",
                }
            ]
        },
        {
            "current_table_decisions": [
                {
                    "current_table_id": "CQ::tbl_p110_i02",
                    "decision": "matched",
                    "previous_table_id": "PQ::tbl_p106_i02",
                    "match_confidence": 0.96,
                    "reason": "Same summary, indicators, and headers.",
                }
            ],
            "warnings": [],
        },
    ]

    def fake_call_openai_json(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    result = _run_matching_stage(
        [_card("tbl_p106_i02", "RBC capital table")],
        [_card("tbl_p110_i02", "RBC capital table")],
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model="gpt-test",
        call_openai_json=fake_call_openai_json,
    )

    repair_prompt = json.loads(calls[1]["messages"][-1]["content"])
    assert repair_prompt["required_repair_current_table_ids"] == ["CQ::tbl_p110_i02"]
    assert repair_prompt["allowed_previous_table_ids"] == ["PQ::tbl_p106_i02"]
    assert repair_prompt["validation_diagnostics"]["wrong_namespace_previous_ids"] == ["tbl_p110_i02"]
    assert calls[1]["response_model"] is not None
    assert result["current_table_decisions"] == [
        {
            "current_table_id": "tbl_p110_i02",
            "decision": "matched",
            "reason": "Same summary, indicators, and headers.",
            "previous_table_id": "tbl_p106_i02",
            "match_confidence": 0.96,
        }
    ]


def test_duplicate_previous_assignment_repairs_only_conflicting_tables() -> None:
    calls: list[dict[str, object]] = []
    responses = [
        {
            "current_table_decisions": [
                {
                    "current_table_id": "curr_a",
                    "decision": "matched",
                    "previous_table_id": "prev_a",
                    "match_confidence": 0.91,
                    "reason": "Candidate A.",
                },
                {
                    "current_table_id": "curr_b",
                    "decision": "matched",
                    "previous_table_id": "prev_a",
                    "match_confidence": 0.90,
                    "reason": "Conflicting candidate B.",
                },
                {
                    "current_table_id": "curr_c",
                    "decision": "matched",
                    "previous_table_id": "prev_c",
                    "match_confidence": 0.99,
                    "reason": "Locked valid match.",
                },
            ]
        },
        {
            "current_table_decisions": [
                {
                    "current_table_id": "CQ::curr_a",
                    "decision": "matched",
                    "previous_table_id": "PQ::prev_a",
                    "match_confidence": 0.95,
                    "reason": "Resolved A.",
                },
                {
                    "current_table_id": "CQ::curr_b",
                    "decision": "matched",
                    "previous_table_id": "PQ::prev_b",
                    "match_confidence": 0.95,
                    "reason": "Resolved B.",
                },
            ]
        },
    ]

    def fake_call_openai_json(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    result = _run_matching_stage(
        [_card("prev_a"), _card("prev_b"), _card("prev_c")],
        [_card("curr_a"), _card("curr_b"), _card("curr_c")],
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model="gpt-test",
        call_openai_json=fake_call_openai_json,
    )

    prompt = json.loads(calls[1]["messages"][-1]["content"])
    assert prompt["required_repair_current_table_ids"] == [
        "CQ::curr_a",
        "CQ::curr_b",
    ]
    assert [item["current_table_id"] for item in prompt["locked_decisions"]] == ["curr_c"]
    assert {item["current_table_id"]: item.get("previous_table_id") for item in result["current_table_decisions"]} == {
        "curr_a": "prev_a",
        "curr_b": "prev_b",
        "curr_c": "prev_c",
    }


def test_persistent_invalid_repair_degrades_to_non_blocking_review() -> None:
    responses = [
        {
            "current_table_decisions": [
                {
                    "current_table_id": "curr_a",
                    "decision": "matched",
                    "previous_table_id": "curr_a",
                    "match_confidence": 0.9,
                    "reason": "Wrong namespace.",
                }
            ]
        },
        {
            "current_table_decisions": [
                {
                    "current_table_id": "CQ::curr_a",
                    "decision": "matched",
                    "previous_table_id": "",
                    "match_confidence": 0.9,
                    "reason": "Still invalid.",
                }
            ]
        },
        {
            "current_table_decisions": [
                {
                    "current_table_id": "CQ::curr_a",
                    "decision": "matched",
                    "previous_table_id": "",
                    "match_confidence": 0.9,
                    "reason": "Still invalid after adjudication.",
                }
            ]
        },
    ]

    def fake_call_openai_json(**_kwargs):
        return responses.pop(0)

    result = _run_matching_stage(
        [_card("prev_a")],
        [_card("curr_a")],
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model="gpt-test",
        call_openai_json=fake_call_openai_json,
    )

    assert result["current_table_decisions"][0]["decision"] == "unresolved"
    assert result["validation_retries_total"] == 3
    assert result["matching_validation_failures_total"] == 3
    assert "matching_structure_repair_exhausted:curr_a" in result["warnings"]


def test_repair_schema_rejects_cross_namespace_identifiers() -> None:
    response_model = _build_matching_repair_response_model(
        current_aliases=["CQ::curr_a"],
        previous_aliases=["PQ::prev_a"],
        allowed_decisions={"matched", "unresolved"},
    )

    response_model.model_validate(
        {
            "current_table_decisions": [
                {
                    "current_table_id": "CQ::curr_a",
                    "decision": "matched",
                    "previous_table_id": "PQ::prev_a",
                    "match_confidence": 0.9,
                    "reason": "Valid aliases.",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "CQ::curr_a",
                        "decision": "matched",
                        "previous_table_id": "CQ::curr_a",
                        "match_confidence": 0.9,
                        "reason": "Crossed namespace.",
                    }
                ]
            }
        )
