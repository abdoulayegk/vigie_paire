"""Tests de la recuperation hybride opt-in des tableaux RBC."""

from __future__ import annotations

import json
from pathlib import Path

from vigie.comparaison.rapprochement.moteur_rapprochement import _run_table_matching
from vigie.comparaison.rbc_hybrid_matching import (
    partition_trusted_rbc_primary_pairs,
    run_rbc_hybrid_recovery,
)
from vigie.support.config import get_matching_thresholds


def _card(
    table_id: str,
    title: str,
    indicators: list[str],
    *,
    summary: str | None = None,
    headers: list[str] | None = None,
    footnotes: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "table_id": table_id,
        "section": "risk_management",
        "title": title,
        "table_summary": summary or title,
        "page": int("".join(character for character in table_id if character.isdigit()) or 1),
        "row_count": len(indicators),
        "first_indicator": indicators[0] if indicators else "",
        "headers": headers or ["Indicateur", "2025", "2024"],
        "indicators": indicators,
        "footnotes": footnotes or [],
        "footnote_count": len(footnotes or []),
    }


def _constant_embeddings(**kwargs) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _item in kwargs["inputs"]]


def _semantic_judge_and_inspector(**kwargs) -> dict[str, object]:
    call_kind = kwargs.get("call_kind")
    payload = json.loads(kwargs["messages"][-1]["content"])
    if call_kind == "rbc_hybrid_judge":
        candidates = payload["candidates"]
        best = max(
            candidates,
            key=lambda item: (
                item["objective_facts"]["indicator_smaller_coverage"],
                item["objective_facts"]["indicator_common_count"],
                item["objective_facts"]["summary_exact"],
            ),
        )
        return {
            "assessments": [
                {
                    "previous_table_id": item["previous_table"]["table_id"],
                    "verdict": "same_table" if item is best else "different_table",
                    "confidence": 0.97 if item is best else 0.99,
                    "reason": "Full indicator signature and business purpose align."
                    if item is best
                    else "Different complete table.",
                }
                for item in candidates
            ]
        }
    if call_kind == "rbc_hybrid_final_inspector":
        return {
            "verdict": "confirmed",
            "confidence": 0.98,
            "reason": "Independent full-table review confirms the pair.",
        }
    raise AssertionError(f"Unexpected call kind: {call_kind}")


def test_rbc_hybrid_recovers_known_shifted_block_without_title_only_matching() -> None:
    previous = [
        _card("tbl_p104_i01", "Sources de financement Tableau 55", ["Non garanti", "Garanties", "Total"]),
        _card("tbl_p105_i01", "Composition du financement", ["Banques", "Entreprises", "Total"]),
        _card("tbl_p106_i01", "Notations Tableau 58", ["Moody's", "S&P", "DBRS", "Fitch"]),
        _card(
            "tbl_p106_i02",
            "Obligations découlant de révisions des notations Tableau 59",
            ["Financement des dérivés", "Sûretés à fournir"],
            footnotes=[{"id": "1", "text": "Montants contractuels supplémentaires"}],
        ),
    ]
    current = [
        _card("tbl_p107_i01", "Sources de financement Tableau 53", ["Non garanti", "Garanties", "Total"]),
        _card("tbl_p109_i01", "Composition du financement", ["Banques", "Entreprises", "Total"]),
        _card("tbl_p110_i01", "Notations Tableau 56", ["Moody's", "S&P", "DBRS", "Fitch"]),
        _card(
            "tbl_p110_i02",
            "Obligations découlant de révisions des notations Tableau 57",
            ["Financement des dérivés", "Sûretés à fournir"],
            footnotes=[{"id": "1", "text": "Montants contractuels supplémentaires"}],
        ),
    ]

    result = run_rbc_hybrid_recovery(
        previous,
        current,
        model="gpt-test",
        embedding_model="text-embedding-test",
        top_k=2,
        min_confidence=0.75,
        call_openai_json=_semantic_judge_and_inspector,
        call_openai_embeddings=_constant_embeddings,
    )

    assert {item["current_table_id"]: item.get("previous_table_id") for item in result["current_table_decisions"]} == {
        "tbl_p107_i01": "tbl_p104_i01",
        "tbl_p109_i01": "tbl_p105_i01",
        "tbl_p110_i01": "tbl_p106_i01",
        "tbl_p110_i02": "tbl_p106_i02",
    }
    assert result["hybrid_embedding_calls_total"] == 1
    assert result["hybrid_final_inspector_calls_total"] == 4


def test_embedding_and_llm_cannot_promote_a_title_only_pair() -> None:
    previous = [_card("prev", "Échéances", ["Actifs financiers", "Prêts hypothécaires"])]
    current = [_card("curr", "Échéances", ["Dépôts", "Débentures subordonnées"])]

    def always_same(**kwargs) -> dict[str, object]:
        if kwargs.get("call_kind") == "rbc_hybrid_final_inspector":
            raise AssertionError("A title-only candidate must never reach final inspection")
        payload = json.loads(kwargs["messages"][-1]["content"])
        return {
            "assessments": [
                {
                    "previous_table_id": payload["candidates"][0]["previous_table"]["table_id"],
                    "verdict": "same_table",
                    "confidence": 1.0,
                    "reason": "Same generic title.",
                }
            ]
        }

    result = run_rbc_hybrid_recovery(
        previous,
        current,
        model="gpt-test",
        embedding_model="embedding-test",
        top_k=1,
        min_confidence=0.75,
        call_openai_json=always_same,
        call_openai_embeddings=_constant_embeddings,
    )

    assert result["current_table_decisions"][0]["decision"] == "added"
    assert result["hybrid_final_inspector_calls_total"] == 0


def test_rbc_primary_audit_releases_cascade_but_keeps_supported_pair() -> None:
    previous = [
        _card("prev_sources", "Sources", ["Non garanti", "Garanties", "Total"]),
        _card("prev_notes", "Notations", ["Moody's", "S&P"]),
    ]
    current = [
        _card("curr_sources", "Sources", ["Non garanti", "Garanties", "Total"]),
        _card("curr_notes", "Notations", ["Moody's", "S&P"]),
    ]
    trusted, released = partition_trusted_rbc_primary_pairs(
        [
            {"previous_table_id": "prev_sources", "current_table_id": "curr_sources"},
            {"previous_table_id": "prev_notes", "current_table_id": "curr_sources"},
        ],
        previous,
        current,
    )

    assert [(item["previous_table_id"], item["current_table_id"]) for item in trusted] == [
        ("prev_sources", "curr_sources")
    ]
    assert [(item["previous_table_id"], item["current_table_id"]) for item in released] == [
        ("prev_notes", "curr_sources")
    ]


def test_embedding_failure_is_fail_closed() -> None:
    def broken_embeddings(**_kwargs):
        raise RuntimeError("temporary embedding outage")

    result = run_rbc_hybrid_recovery(
        [_card("prev", "Capital", ["CET1", "Levier"])],
        [_card("curr", "Capital", ["CET1", "Levier"])],
        model="gpt-test",
        embedding_model="embedding-test",
        top_k=2,
        min_confidence=0.75,
        call_openai_json=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")),
        call_openai_embeddings=broken_embeddings,
    )

    assert result["current_table_decisions"][0]["decision"] == "added"
    assert result["warnings"] == ["rbc_hybrid_embeddings_failed:RuntimeError"]


def test_legacy_path_never_calls_embeddings_when_hybrid_is_disabled() -> None:
    responses = iter(
        [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr",
                        "decision": "matched",
                        "previous_table_id": "prev",
                        "match_confidence": 0.99,
                        "reason": "Same complete table.",
                    }
                ]
            },
            {
                "verdict": "confirmed",
                "shared_indicators": ["CET1", "Levier"],
                "confidence": 0.99,
                "reason": "Confirmed.",
            },
        ]
    )

    result = _run_table_matching(
        [_card("prev", "Capital", ["CET1", "Levier"])],
        [_card("curr", "Capital", ["CET1", "Levier"])],
        model="gpt-test",
        call_openai_json=lambda **_kwargs: next(responses),
        hybrid_recovery_enabled=False,
        call_openai_embeddings=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Embeddings are RBC-only")),
    )

    assert result["matched_pairs"][0]["previous_table_id"] == "prev"
    assert result["hybrid_recovery_executed"] == 0


def test_configuration_enables_hybrid_only_for_rbc() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "bank_profiles.yaml"

    assert get_matching_thresholds(config_path, "rbc")["hybrid_embedding_recovery_enabled"] is True
    assert get_matching_thresholds(config_path, "td")["hybrid_embedding_recovery_enabled"] is False
    assert get_matching_thresholds(config_path, "bnc")["hybrid_embedding_recovery_enabled"] is False


def test_clean_title_for_bank_rbc_only() -> None:
    from vigie.comparaison.io import _clean_title_for_bank

    # For RBC: strips "Tableau XX" suffixes
    assert (
        _clean_title_for_bank("Charges grevant les actifs Tableau 54", bank_code="RBC") == "Charges grevant les actifs"
    )
    assert _clean_title_for_bank("Notations Tableau 58", bank_code="rbc") == "Notations"
    assert _clean_title_for_bank("Échéances contractuelles Tableau 62", bank_code="RBC") == "Échéances contractuelles"

    # For other banks (BMO, TD, BNC, BNS, CIBC): preserves exact title
    assert (
        _clean_title_for_bank("Charges grevant les actifs Tableau 54", bank_code="BMO")
        == "Charges grevant les actifs Tableau 54"
    )
    assert _clean_title_for_bank("Notations Tableau 58", bank_code="TD") == "Notations Tableau 58"
    assert (
        _clean_title_for_bank("Échéances contractuelles Tableau 62", bank_code="BNC")
        == "Échéances contractuelles Tableau 62"
    )
