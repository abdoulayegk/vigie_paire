"""Tests des contrats Pydantic du rapprochement indicateurs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vigie.comparaison.io import TableCard, _table_card
from vigie.comparaison.pipeline.resultat_models import (
    ComparisonRunResult,
    ComparisonSummary,
    MatchingBlock,
    ReferenceResolution,
)
from vigie.comparaison.rapprochement.etat import (
    MatchedPair,
    MatchingResult,
    MatchingState,
    TableRef,
)
from vigie.comparaison.rapprochement.normalisation_reponses import (
    _normalize_matching_response,
)
from vigie.comparaison.rapprochement.contrats import _MatchingValidationError
from vigie.support.models.comparison_models import (
    MatchingDecision,
    PrimaryMatchResponse,
    RecoveryMatchResponse,
)


def test_matching_decision_and_primary_response_roundtrip() -> None:
    decision = MatchingDecision(
        current_table_id="curr_a",
        decision="matched",
        previous_table_id="prev_a",
        match_confidence=0.91,
        reason="same indicators",
    )
    response = PrimaryMatchResponse(current_table_decisions=[decision], warnings=[])
    dumped = response.model_dump(mode="json")
    assert dumped["current_table_decisions"][0]["match_confidence"] == 0.91
    RecoveryMatchResponse.model_validate(
        {
            "current_table_decisions": [
                {
                    "current_table_id": "curr_b",
                    "decision": "added",
                    "reason": "no partner",
                    "previous_table_id": "",
                    "match_confidence": None,
                }
            ]
        }
    )


def test_normalize_accepts_primary_match_response_model() -> None:
    payload = PrimaryMatchResponse(
        current_table_decisions=[
            MatchingDecision(
                current_table_id="curr_a",
                decision="matched",
                previous_table_id="prev_a",
                match_confidence=0.88,
                reason="ok",
            )
        ]
    )
    normalized = _normalize_matching_response(
        payload,
        previous_ids={"prev_a"},
        current_ids={"curr_a"},
        allowed_decisions={"matched", "unresolved"},
    )
    assert normalized["current_table_decisions"][0]["previous_table_id"] == "prev_a"


def test_normalize_rejects_duplicate_previous_assignment() -> None:
    with pytest.raises(_MatchingValidationError):
        _normalize_matching_response(
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "c1",
                        "decision": "matched",
                        "previous_table_id": "p1",
                        "match_confidence": 0.9,
                        "reason": "a",
                    },
                    {
                        "current_table_id": "c2",
                        "decision": "matched",
                        "previous_table_id": "p1",
                        "match_confidence": 0.9,
                        "reason": "b",
                    },
                ]
            },
            previous_ids={"p1"},
            current_ids={"c1", "c2"},
            allowed_decisions={"matched", "unresolved"},
        )


def test_table_card_is_pydantic_and_dict_compatible() -> None:
    card = _table_card(
        {
            "table_id": "tbl_1",
            "section": "capital_management",
            "title": "Ratios",
            "indicators": ["CET1", "Tier 1"],
            "headers": ["2025"],
            "footnotes": [],
        }
    )
    assert isinstance(card, TableCard)
    assert card["table_id"] == "tbl_1"
    assert card.get("first_indicator") == "CET1"
    assert card.model_dump(mode="json")["row_count"] == 2


def test_matching_state_and_result_models() -> None:
    state = MatchingState(
        confirmed_pairs=[
            MatchedPair(
                previous_table_id="p1",
                current_table_id="c1",
                match_confidence=0.95,
                reason="match",
            )
        ],
        tables_added=[TableRef(table_id="c2", reason="new")],
    )
    assert state.confirmed_pairs[0].previous_table_id == "p1"
    result = MatchingResult(
        executed=True,
        matched_pairs=state.confirmed_pairs,
        tables_added=state.tables_added,
        matching_passes_total=2,
    )
    legacy = result.to_legacy_dict()
    assert legacy["matched_pairs"][0]["current_table_id"] == "c1"
    assert legacy["tables_added"][0]["table_id"] == "c2"


def test_comparison_run_result_dump_has_stable_keys() -> None:
    result = ComparisonRunResult(
        schema_version=3,
        artifact_type="report_comparison",
        run_id="run_test",
        bank_code="bnc",
        year_previous=2025,
        quarter_previous="t1",
        year_current=2025,
        quarter_current="t2",
        created_at="2026-01-01T00:00:00",
        model_version="gpt-test",
        prompt_version_match="table_match_v8",
        prompt_version_diff="table_diff_v4",
        reference_resolution=ReferenceResolution(
            mode="automatique",
            year_previous=2025,
            quarter_previous="t1",
            rule="t2->t1",
        ),
        matching=MatchingBlock(matched_pairs=[], tables_added=[], tables_removed=[]),
        pair_comparisons=[],
        run_metrics={"comparison_runtime_sec": 1.0},
        summary=ComparisonSummary(matched_pairs_total=0),
    )
    dumped = result.to_json_dict()
    for key in (
        "schema_version",
        "artifact_type",
        "matching",
        "pair_comparisons",
        "run_metrics",
        "summary",
        "reference_resolution",
    ):
        assert key in dumped
    assert dumped["schema_version"] == 3
    # Round-trip validation used by Dash load path
    again = ComparisonRunResult.model_validate(dumped).to_json_dict()
    assert again["bank_code"] == "bnc"


def test_match_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        MatchingDecision(
            current_table_id="c",
            decision="matched",
            previous_table_id="p",
            match_confidence=1.5,
            reason="bad",
        )
