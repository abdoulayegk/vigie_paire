from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vigilance.compare_gpt import (
    DIFF_PROMPT_VERSION,
    MATCH_PROMPT_VERSION,
    OPENAI_COMPARISON_TIMEOUT_SECONDS,
    _call_openai_embeddings,
    _call_openai_json,
    compare_reports_gpt4o,
    normalize_quarter,
    resolve_reference_period,
)


def _table(
    *,
    table_id: str,
    page: int,
    section: str,
    title: str,
    table_summary: str,
    headers: list[str],
    indicators: list[str],
    footnotes: list[dict[str, str]] | None = None,
    bbox: list[float] | None = None,
    extraction_status: str = "ok",
) -> dict:
    return {
        "table_id": table_id,
        "page": page,
        "section": section,
        "title": title,
        "table_summary": table_summary,
        "bbox": bbox,
        "row_count": len(indicators),
        "headers": headers,
        "indicators": indicators,
        "footnotes": footnotes or [],
        "extraction_status": extraction_status,
    }


def _write_tables_json(
    path: Path,
    *,
    bank: str,
    year: int,
    quarter: str,
    tables: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 7,
                "bank_code": bank,
                "year": year,
                "quarter": quarter,
                "created_at": "2026-03-24T10:00:00",
                "tables": tables,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_normalize_quarter_and_reference_period() -> None:
    assert normalize_quarter("Q2") == "t2"
    assert resolve_reference_period(2026, "t2") == (2026, "t1")
    assert resolve_reference_period(2026, "t1") == (2025, "t3")


def test_comparison_openai_clients_use_direct_120_second_timeout(monkeypatch) -> None:
    client_kwargs: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            client_kwargs.append(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="{}"),
                            )
                        ]
                    )
                )
            )
            self.embeddings = SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.25, 0.75])]
                )
            )

    monkeypatch.setattr("vigilance.compare_gpt.get_openai_api_key", lambda: "test-key")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    assert _call_openai_json(model="gpt-test", messages=[]) == {}
    assert _call_openai_embeddings(model="embedding-test", inputs=["table"]) == [
        [0.25, 0.75]
    ]
    assert [kwargs["timeout"] for kwargs in client_kwargs] == [
        OPENAI_COMPARISON_TIMEOUT_SECONDS,
        OPENAI_COMPARISON_TIMEOUT_SECONDS,
    ]
    assert client_kwargs[0]["max_retries"] == 0
    assert "max_retries" not in client_kwargs[1]


def test_compare_reports_gpt4o_uses_canonical_prompt_cards_and_gpt_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_capital",
                page=10,
                section="capital_management",
                title="Capital",
                table_summary="Ratios de capital réglementaire",
                headers=["Indicateur", "Valeur"] + [f"H{i}" for i in range(1, 26)],
                indicators=["Ratio CET1"],
                footnotes=[{"id": "1", "text": "Note A"}],
            ),
            _table(
                table_id="prev_removed",
                page=12,
                section="risk_management",
                title="Risque",
                table_summary="Tableau de risque hérité",
                headers=["Indicateur", "Valeur"],
                indicators=["RWA"],
            ),
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_capital",
                page=11,
                section="capital_management",
                title="Capital",
                table_summary="Ratios de capital réglementaire",
                headers=["Indicateur", "Valeur"],
                indicators=["Ratio CET1", "Ratio de levier"],
                footnotes=[{"id": "1", "text": "Note A mise à jour"}],
            ),
            _table(
                table_id="curr_added",
                page=14,
                section="liquidite",
                title="Liquidité",
                table_summary="Tableau de liquidité à court terme",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
            ),
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_capital",
                        "decision": "matched",
                        "previous_table_id": "prev_capital",
                        "match_confidence": 0.98,
                        "reason": "Même sujet et mêmes indicateurs principaux.",
                    },
                    {
                        "current_table_id": "curr_added",
                        "decision": "unresolved",
                        "reason": "Aucune contrepartie précédente assez solide au pass strict.",
                    },
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_added",
                        "decision": "added",
                        "reason": "Aucun équivalent clair dans le rapport précédent.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Ratio CET1"],
                "confidence": 0.95,
                "reason": "Shared indicator Ratio CET1.",
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
        "diff_indicators": [
            {
                "indicators_added": [
                    {
                        "value": "Ratio de levier",
                        "reason": "Nouvel indicateur courant.",
                        "analyst_assessment": {
                            "relevance_level": 2,
                            "justification": "Nouvel indicateur",
                        },
                    }
                ],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Le tableau gagne un indicateur.",
            },
        ],
        "diff_footnotes": [
            {
                "footnotes_added": [],
                "footnotes_removed": [],
                "footnotes_renamed": [
                    {
                        "previous_id": "1",
                        "current_id": "1",
                        "previous_text": "Note A",
                        "current_text": "Note A mise à jour",
                        "reason": "Même note avec reformulation matérielle.",
                        "analyst_assessment": {
                            "relevance_level": 3,
                            "justification": "Reformulation",
                        },
                    }
                ],
                "reason": "La note est reformulée matériellement.",
            },
        ],
        "inspect_artifacts": [
            {
                "added_verdicts": [
                    {
                        "value": "Ratio de levier",
                        "verdict": "real",
                        "reason": "Nouvel indicateur réel.",
                    }
                ],
                "removed_verdicts": [],
                "artifact_pairs": [],
            },
        ],
    }
    seen_prompts: list[tuple[str, dict]] = []

    def fake_call_openai_json(**kwargs):
        prompt = json.loads(kwargs["messages"][-1]["content"])
        kind = kwargs["call_kind"]
        seen_prompts.append((kind, prompt))
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["matched_pairs"] == [
        {
            "previous_table_id": "prev_capital",
            "current_table_id": "curr_capital",
            "match_confidence": 0.98,
            "reason": "Même sujet et mêmes indicateurs principaux.",
        }
    ]
    assert payload["matching"]["tables_added"][0]["table_id"] == "curr_added"
    assert payload["matching"]["tables_removed"][0]["table_id"] == "prev_removed"
    assert payload["prompt_version_match"] == MATCH_PROMPT_VERSION
    assert payload["prompt_version_diff"] == DIFF_PROMPT_VERSION
    assert payload["pair_comparisons"][0]["diff_mode"] == "gpt"
    assert payload["pair_comparisons"][0]["technical_diff"]["indicators_added"] == [
        {
            "value": "Ratio de levier",
            "reason": "Nouvel indicateur courant.",
            "analyst_assessment": {
                "relevance_level": 2,
                "justification": "Nouvel indicateur",
            },
        }
    ]
    assert payload["pair_comparisons"][0]["technical_diff"]["footnotes_renamed"] == [
        {
            "previous_id": "1",
            "current_id": "1",
            "previous_text": "Note A",
            "current_text": "Note A mise à jour",
            "reason": "Même note avec reformulation matérielle.",
            "analyst_assessment": {
                "relevance_level": 3,
                "justification": "Reformulation",
            },
        }
    ]
    assert payload["run_metrics"]["matching_passes_total"] == 2
    assert payload["run_metrics"]["audit_passes_total"] == 0

    match_kind, match_prompt = seen_prompts[0]
    assert match_kind == "matching"
    prev0 = match_prompt["previous_tables"][0]
    assert set(prev0) == {
        "table_id",
        "section",
        "title",
        "table_summary",
        "page",
        "row_count",
        "first_indicator",
        "footnote_count",
        "headers",
        "indicators",
        "footnotes",
    }
    assert "extraction_index" not in prev0
    assert "header_columns_total" not in prev0
    assert len(prev0["headers"]) == 27


def test_compare_reports_gpt4o_allows_cross_section_matching_in_single_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "rbc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "rbc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="rbc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_lcr",
                page=20,
                section="risk_management",
                title="Ratio de liquidité à court terme",
                table_summary="Mesures de liquidité réglementaires",
                headers=["Indicateur", "Valeur"],
                indicators=["Total des actifs liquides", "LCR"],
            )
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="rbc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_lcr",
                page=24,
                section="liquidite",
                title="Ratio de liquidité à court terme",
                table_summary="Mesures de liquidité réglementaires",
                headers=["Indicateur", "Valeur"],
                indicators=["Total des actifs liquides", "LCR"],
            )
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_lcr",
                        "decision": "matched",
                        "previous_table_id": "prev_lcr",
                        "match_confidence": 0.91,
                        "reason": "Même tableau malgré la section différente.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Total des actifs liquides", "LCR"],
                "confidence": 0.92,
                "reason": "Both indicators match.",
            },
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement sémantique détecté.",
            },
        ],
    }
    seen_modes: list[str] = []

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        if kind == "matching":
            prompt = json.loads(kwargs["messages"][-1]["content"])
            seen_modes.append(prompt["task"])
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["matched_pairs"][0]["current_table_id"] == "curr_lcr"
    assert payload["run_metrics"]["matching_passes_total"] == 1
    assert payload["run_metrics"]["unmatched_after_primary_total"] == 0
    assert payload["run_metrics"]["unmatched_after_rescue_total"] == 0
    assert len(seen_modes) == 1


def test_compare_reports_gpt4o_retries_invalid_matching_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "td" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "td" / "2025" / "t2"
    table_prev = _table(
        table_id="prev_1",
        page=8,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
    )
    table_curr = _table(
        table_id="curr_1",
        page=9,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
    )
    _write_tables_json(
        previous_dir / "tables.json",
        bank="td",
        year=2025,
        quarter="t1",
        tables=[table_prev],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="td",
        year=2025,
        quarter="t2",
        tables=[table_curr],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_unknown",
                        "decision": "matched",
                        "previous_table_id": "prev_1",
                        "match_confidence": 0.90,
                        "reason": "Réponse invalide.",
                    }
                ]
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_1",
                        "decision": "matched",
                        "previous_table_id": "prev_1",
                        "match_confidence": 0.96,
                        "reason": "Même tableau.",
                    }
                ],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Ratio CET1"],
                "confidence": 0.96,
                "reason": "Same indicator.",
            },
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement.",
            },
        ],
    }
    feedbacks: list[str] = []

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        if kind == "matching":
            prompt = json.loads(kwargs["messages"][-1]["content"])
            if "validation_feedback" in prompt:
                feedbacks.append(prompt["validation_feedback"])
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["matched_pairs"][0]["current_table_id"] == "curr_1"
    assert payload["run_metrics"]["matching_output_retries_total"] >= 1
    assert feedbacks


def test_compare_reports_gpt4o_rejects_duplicate_pairs_without_local_scoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "cibc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "cibc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="cibc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_a",
                page=5,
                section="risk_management",
                title="Risque A",
                table_summary="Tableau A",
                headers=["Indicateur", "Valeur"],
                indicators=["Alpha"],
            ),
            _table(
                table_id="prev_b",
                page=6,
                section="risk_management",
                title="Risque B",
                table_summary="Tableau B",
                headers=["Indicateur", "Valeur"],
                indicators=["Bêta"],
            ),
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="cibc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_a",
                page=5,
                section="risk_management",
                title="Risque A",
                table_summary="Tableau A",
                headers=["Indicateur", "Valeur"],
                indicators=["Alpha"],
            )
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_a",
                        "decision": "matched",
                        "previous_table_id": "prev_a",
                        "match_confidence": 0.95,
                        "reason": "Première paire valide.",
                    },
                    {
                        "current_table_id": "curr_a",
                        "decision": "matched",
                        "previous_table_id": "prev_b",
                        "match_confidence": 0.94,
                        "reason": "Doublon à rejeter.",
                    },
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_a",
                        "decision": "matched",
                        "previous_table_id": "prev_a",
                        "match_confidence": 0.95,
                        "reason": "Première paire valide.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Alpha"],
                "confidence": 0.95,
                "reason": "Shared indicator Alpha.",
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement.",
            },
        ],
    }

    monkeypatch.setattr(
        "vigilance.compare_gpt._call_openai_json",
        lambda **kwargs: responses_by_kind[kwargs["call_kind"]].pop(0),
    )

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["matched_pairs"] == [
        {
            "previous_table_id": "prev_a",
            "current_table_id": "curr_a",
            "match_confidence": 0.95,
            "reason": "Première paire valide.",
        }
    ]
    assert payload["run_metrics"]["matching_pairs_llm_duplicates_total"] == 1
    assert payload["run_metrics"]["matching_pairs_llm_deduped_total"] == 0
    assert payload["run_metrics"]["matching_output_retries_total"] >= 1
    assert payload["run_metrics"]["matching_validation_failures_total"] >= 1


def test_compare_reports_gpt4o_recovers_unresolved_pairs_in_second_matching_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "rbc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "rbc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="rbc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_x",
                page=10,
                section="risk_management",
                title="Liquidité",
                table_summary="Mesures de liquidité",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
            ),
            _table(
                table_id="prev_y",
                page=11,
                section="risk_management",
                title="Liquidité secondaire",
                table_summary="Mesures de liquidité secondaires",
                headers=["Indicateur", "Valeur"],
                indicators=["NSFR"],
            ),
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="rbc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_x",
                page=12,
                section="liquidite",
                title="Liquidité",
                table_summary="Mesures de liquidité",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
            )
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_x",
                        "decision": "unresolved",
                        "reason": "Deux candidats précédents restent plausibles au pass strict.",
                    }
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_x",
                        "decision": "matched",
                        "previous_table_id": "prev_x",
                        "match_confidence": 0.93,
                        "reason": "Réconcilié dans le pass de récupération.",
                    }
                ],
                "warnings": [],
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement.",
            },
        ],
    }
    stages: list[str] = []

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        if kind == "matching":
            prompt = json.loads(kwargs["messages"][-1]["content"])
            stages.append(str(prompt.get("stage", "")))
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["matched_pairs"][0]["current_table_id"] == "curr_x"
    assert payload["matching"]["tables_removed"][0]["table_id"] == "prev_y"
    assert payload["run_metrics"]["matching_passes_total"] == 2
    assert payload["run_metrics"]["unmatched_after_primary_total"] == 3
    assert stages == ["primary", "recovery"]


def test_compare_reports_gpt4o_retries_incomplete_matching_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_unmatched",
                page=20,
                section="risk_management",
                title="Ancien tableau",
                table_summary="Ancien tableau de risque",
                headers=["Indicateur", "Valeur"],
                indicators=["RWA"],
            )
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_unmatched_a",
                page=21,
                section="risk_management",
                title="Nouveau tableau A",
                table_summary="Nouveau tableau de risque A",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
            ),
            _table(
                table_id="curr_unmatched_b",
                page=22,
                section="risk_management",
                title="Nouveau tableau B",
                table_summary="Nouveau tableau de risque B",
                headers=["Indicateur", "Valeur"],
                indicators=["NSFR"],
            ),
            _table(
                table_id="curr_unmatched_c",
                page=23,
                section="risk_management",
                title="Nouveau tableau C",
                table_summary="Nouveau tableau de risque C",
                headers=["Indicateur", "Valeur"],
                indicators=["HQLA"],
            ),
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_unmatched_a",
                        "decision": "unresolved",
                        "reason": "Nouveau tableau réel A.",
                    }
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_unmatched_a",
                        "decision": "unresolved",
                        "reason": "Nouveau tableau réel A.",
                    },
                    {
                        "current_table_id": "curr_unmatched_b",
                        "decision": "unresolved",
                        "reason": "Nouveau tableau réel B.",
                    },
                    {
                        "current_table_id": "curr_unmatched_c",
                        "decision": "unresolved",
                        "reason": "Nouveau tableau réel C.",
                    },
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_unmatched_a",
                        "decision": "added",
                        "reason": "Nouveau tableau réel A.",
                    },
                    {
                        "current_table_id": "curr_unmatched_b",
                        "decision": "added",
                        "reason": "Nouveau tableau réel B.",
                    },
                    {
                        "current_table_id": "curr_unmatched_c",
                        "decision": "added",
                        "reason": "Nouveau tableau réel C.",
                    },
                ],
                "warnings": [],
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
    }
    feedbacks: list[str] = []
    required_ids: list[list[str]] = []

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        if kind == "matching":
            prompt = json.loads(kwargs["messages"][-1]["content"])
            required_ids.append(list(prompt.get("required_current_table_ids", [])))
            if "validation_feedback" in prompt:
                feedbacks.append(prompt["validation_feedback"])
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert [item["table_id"] for item in payload["matching"]["tables_added"]] == [
        "curr_unmatched_a",
        "curr_unmatched_b",
        "curr_unmatched_c",
    ]
    assert payload["matching"]["tables_added"][0]["reason"] == "Nouveau tableau réel A."
    assert payload["matching"]["tables_added"][0]["title"] == "Nouveau tableau A"
    assert payload["matching"]["tables_removed"][0]["table_id"] == "prev_unmatched"
    assert payload["run_metrics"]["matching_output_retries_total"] >= 1
    assert payload["run_metrics"]["matching_validation_failures_total"] >= 1
    assert feedbacks
    assert required_ids
    assert required_ids[0] == [
        "curr_unmatched_a",
        "curr_unmatched_b",
        "curr_unmatched_c",
    ]
    assert "curr_unmatched_b" in feedbacks[0]
    assert "curr_unmatched_c" in feedbacks[0]


def test_compare_reports_gpt4o_separates_artifacts_and_extraction_suspects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_artifact",
                page=40,
                section="risk_management",
                title="Rapport de gestion",
                table_summary="",
                headers=[],
                indicators=[],
                bbox=[0.1, 0.1, 0.9, 0.8],
                extraction_status="confirmed_no_table",
            ),
            _table(
                table_id="prev_biz",
                page=39,
                section="risk_management",
                title="Ancre suspect",
                table_summary="",
                headers=["Indicateur", "Valeur"],
                indicators=["Ligne"],
            ),
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_real_add",
                page=41,
                section="risk_management",
                title="Nouveau vrai tableau",
                table_summary="Nouveau sujet métier",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
            ),
            _table(
                table_id="curr_suspect",
                page=42,
                section="risk_management",
                title="Rapport de gestion",
                table_summary="",
                headers=[],
                indicators=[],
                bbox=[0.1, 0.1, 0.9, 0.8],
                extraction_status="suspect_unresolved",
            ),
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            # Stage 1: primary — curr_real_add unresolved
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_real_add",
                        "decision": "unresolved",
                        "reason": "Vrai nouveau tableau, aucune correspondance solide.",
                    },
                ],
                "warnings": [],
            },
            # Stage 2: recovery — curr_real_add added
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_real_add",
                        "decision": "added",
                        "reason": "Vrai nouveau tableau.",
                    },
                ],
                "warnings": [],
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
    }

    monkeypatch.setattr(
        "vigilance.compare_gpt._call_openai_json",
        lambda **kwargs: responses_by_kind[kwargs["call_kind"]].pop(0),
    )

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert [item["table_id"] for item in payload["matching"]["tables_added"]] == ["curr_real_add"]
    assert [item["table_id"] for item in payload["matching"]["tables_removed"]] == ["prev_biz"]
    assert [item["table_id"] for item in payload["matching"]["artifacts_confirmed_previous"]] == ["prev_artifact"]
    assert [item["table_id"] for item in payload["matching"]["extraction_suspects_current"]] == ["curr_suspect"]
    assert "extraction_status=suspect_unresolved" in (
        payload["matching"]["extraction_suspects_current"][0].get("reason") or ""
    )
    assert payload["summary"]["tables_added_total"] == 1
    assert payload["summary"]["tables_removed_total"] == 1
    assert payload["summary"]["artifacts_confirmed_previous_total"] == 1
    assert payload["summary"]["extraction_suspects_current_total"] == 1


def test_compare_reports_gpt4o_preclassifies_artifacts_and_suspects_before_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_artifact",
                page=40,
                section="risk_management",
                title="Rapport de gestion",
                table_summary="",
                headers=[],
                indicators=[],
                bbox=[0.1, 0.1, 0.9, 0.8],
                extraction_status="confirmed_no_table",
            )
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_suspect",
                page=41,
                section="risk_management",
                title="Rapport de gestion",
                table_summary="",
                headers=[],
                indicators=[],
                bbox=[0.1, 0.1, 0.9, 0.8],
                extraction_status="suspect_unresolved",
            )
        ],
    )

    call_kinds: list[str] = []

    def fake_call_openai_json(**kwargs):
        call_kinds.append(kwargs["call_kind"])
        return {}

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    # Both tables are ghost (0 indicators, 0 headers) → no matching call
    assert call_kinds == []
    assert payload["matching"]["tables_added"] == []
    assert payload["matching"]["tables_removed"] == []
    assert [item["table_id"] for item in payload["matching"]["artifacts_confirmed_previous"]] == ["prev_artifact"]
    assert [item["table_id"] for item in payload["matching"]["extraction_suspects_current"]] == ["curr_suspect"]


def test_compare_reports_gpt4o_sends_trivial_ok_tables_to_business_matching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_trivial_ok",
                page=40,
                section="risk_management",
                title="Rapport de gestion",
                table_summary="",
                headers=[],
                indicators=[],
                bbox=[0.1, 0.1, 0.9, 0.8],
                extraction_status="ok",
            )
        ],
    )

    call_kinds: list[str] = []

    def fake_call_openai_json(**kwargs):
        call_kinds.append(kwargs["call_kind"])
        return {}

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    # Ghost filter: 0 indicators + 0 headers → no business tables → no matching call
    assert call_kinds == []
    assert payload["matching"]["tables_added"] == []
    assert payload["matching"]["artifacts_confirmed_current"] == []


def test_compare_reports_gpt4o_always_uses_gpt_for_unchanged_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    table_prev = _table(
        table_id="prev_same",
        page=3,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "1", "text": "Note stable"}],
    )
    table_curr = _table(
        table_id="curr_same",
        page=4,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "1", "text": "Note stable"}],
    )
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[table_prev],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[table_curr],
    )

    call_kinds: list[str] = []
    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_same",
                        "decision": "matched",
                        "previous_table_id": "prev_same",
                        "match_confidence": 0.99,
                        "reason": "Même tableau.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Ratio CET1"],
                "confidence": 0.99,
                "reason": "Identical indicator.",
            },
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement sémantique sur les indicateurs.",
            },
        ],
        "diff_footnotes": [
            {
                "footnotes_added": [],
                "footnotes_removed": [],
                "footnotes_renamed": [],
                "reason": "Aucun changement sémantique sur les notes.",
            },
        ],
    }

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        call_kinds.append(kind)
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert call_kinds == [
        "matching",
        "match_inspector",
        "diff_indicators",
        "diff_footnotes",
    ]
    assert payload["pair_comparisons"][0]["diff_mode"] == "gpt"
    assert payload["pair_comparisons"][0]["technical_diff"]["table_level_change"] == "inchange"
    assert payload["run_metrics"]["comparison_calls_total"] == 4


def test_compare_reports_gpt4o_runs_visual_sanity_for_footnote_only_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    table_prev = _table(
        table_id="prev_same",
        page=3,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "1", "text": "Note A"}],
        bbox=[0.1, 0.2, 0.9, 0.7],
    )
    table_curr = _table(
        table_id="curr_same",
        page=4,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "2", "text": "Note A mise à jour"}],
        bbox=[0.1, 0.2, 0.9, 0.7],
    )
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[table_prev],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[table_curr],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_same",
                        "decision": "matched",
                        "previous_table_id": "prev_same",
                        "match_confidence": 0.99,
                        "reason": "Même tableau.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Ratio CET1"],
                "confidence": 0.99,
                "reason": "Identical indicator.",
            },
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement indicateur.",
            },
        ],
        "diff_footnotes": [
            {
                "footnotes_added": [],
                "footnotes_removed": [],
                "footnotes_renamed": [
                    {
                        "previous_id": "1",
                        "current_id": "2",
                        "previous_text": "Note A",
                        "current_text": "Note A mise à jour",
                        "reason": "Même note, wording mis à jour.",
                        "analyst_assessment": {
                            "relevance_level": 3,
                            "justification": "Reformulation",
                        },
                    }
                ],
                "reason": "Une note change.",
            },
        ],
    }

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        return responses_by_kind[kind].pop(0)

    sanity_calls: list[dict] = []

    def fake_render_visual_sanity_proof(pdf_path, *, page, bbox, **kwargs):
        return b"proof-bytes", "ok"

    def fake_visual_sanity_check(
        previous_render_bytes,
        current_render_bytes,
        diff_result,
        *,
        model,
        call_openai_json,
        usage_recorder=None,
    ):
        sanity_calls.append(diff_result)
        return {
            **diff_result,
            "visual_sanity_applied": True,
            "visual_sanity_rejected_count": 0,
            "visual_sanity_scope": ["indicators", "footnotes", "tables"],
            "visual_sanity_render_mode": "full",
            "visual_sanity_render_status": "ok",
        }

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)
    monkeypatch.setattr(
        "vigilance.compare_gpt.render_visual_sanity_proof",
        fake_render_visual_sanity_proof,
    )
    monkeypatch.setattr("vigilance.compare_gpt.visual_sanity_check", fake_visual_sanity_check)

    source_pdf_previous = tmp_path / "prev.pdf"
    source_pdf_current = tmp_path / "curr.pdf"
    source_pdf_previous.write_bytes(b"%PDF-1.4\n")
    source_pdf_current.write_bytes(b"%PDF-1.4\n")

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
        source_pdf_previous=str(source_pdf_previous),
        source_pdf_current=str(source_pdf_current),
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert len(sanity_calls) == 1
    assert payload["pair_comparisons"][0]["visual_sanity_applied"] is True
    assert payload["pair_comparisons"][0]["visual_sanity_render_mode"] == "full"
    assert payload["pair_comparisons"][0]["technical_diff"]["footnotes_renamed"] != []


def test_compare_reports_gpt4o_filters_table_added_removed_with_visual_sanity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_removed",
                page=12,
                section="liquidite",
                title="Liquidité",
                table_summary="Tableau de liquidité antérieur",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
                bbox=[0.1, 0.2, 0.9, 0.7],
            )
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_added",
                page=14,
                section="liquidite",
                title="Liquidité",
                table_summary="Tableau de liquidité à court terme",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
                bbox=[0.1, 0.2, 0.9, 0.7],
            )
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_added",
                        "decision": "unresolved",
                        "reason": "Aucune contrepartie.",
                    }
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_added",
                        "decision": "added",
                        "reason": "Nouveau tableau.",
                    }
                ],
                "warnings": [],
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
    }

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        return responses_by_kind[kind].pop(0)

    render_calls: list[tuple[str, int]] = []

    def fake_render_visual_sanity_proof(pdf_path, *, page, bbox, **kwargs):
        render_calls.append((str(pdf_path), int(page)))
        return b"proof-bytes", "ok"

    def fake_visual_sanity_check_table_event(
        previous_render_bytes,
        current_render_bytes,
        *,
        event_type,
        table_id,
        table_title,
        model,
        call_openai_json,
        usage_recorder=None,
    ):
        return {
            "confirmed": False,
            "visual_sanity_applied": True,
            "visual_sanity_rejected_count": 1,
            "visual_sanity_scope": ["indicators", "footnotes", "tables"],
            "visual_sanity_render_mode": "full",
            "visual_sanity_render_status": "ok",
        }

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)
    monkeypatch.setattr(
        "vigilance.compare_gpt.render_visual_sanity_proof",
        fake_render_visual_sanity_proof,
    )
    monkeypatch.setattr(
        "vigilance.compare_gpt.visual_sanity_check_table_event",
        fake_visual_sanity_check_table_event,
    )

    source_pdf_previous = tmp_path / "prev.pdf"
    source_pdf_current = tmp_path / "curr.pdf"
    source_pdf_previous.write_bytes(b"%PDF-1.4\n")
    source_pdf_current.write_bytes(b"%PDF-1.4\n")

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
        source_pdf_previous=str(source_pdf_previous),
        source_pdf_current=str(source_pdf_current),
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["tables_added"] == []
    assert payload["matching"]["tables_removed"] == []
    assert payload["summary"]["tables_added_total"] == 0
    assert payload["summary"]["tables_removed_total"] == 0
    assert render_calls == [
        (str(source_pdf_previous), 12),
        (str(source_pdf_current), 14),
        (str(source_pdf_previous), 12),
        (str(source_pdf_current), 14),
    ]


def test_compare_reports_gpt4o_skips_table_visual_sanity_without_anchor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[
            _table(
                table_id="prev_removed",
                page=12,
                section="risk_management",
                title="Risque",
                table_summary="Tableau de risque hérité",
                headers=["Indicateur", "Valeur"],
                indicators=["RWA"],
                bbox=[0.1, 0.2, 0.9, 0.7],
            )
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            _table(
                table_id="curr_added",
                page=14,
                section="liquidite",
                title="Liquidité",
                table_summary="Tableau de liquidité à court terme",
                headers=["Indicateur", "Valeur"],
                indicators=["LCR"],
                bbox=[0.1, 0.2, 0.9, 0.7],
            )
        ],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_added",
                        "decision": "unresolved",
                        "reason": "Aucune contrepartie.",
                    }
                ],
                "warnings": [],
            },
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_added",
                        "decision": "added",
                        "reason": "Nouveau tableau.",
                    }
                ],
                "warnings": [],
            },
        ],
        "devil_advocate": [
            {"new_matches": [], "confirmed_low_confidence": [], "contested_pairs": []},
        ],
    }

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)
    monkeypatch.setattr(
        "vigilance.compare_gpt.render_visual_sanity_proof",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected render")),
    )
    monkeypatch.setattr(
        "vigilance.compare_gpt.visual_sanity_check_table_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sanity")),
    )

    source_pdf_previous = tmp_path / "prev.pdf"
    source_pdf_current = tmp_path / "curr.pdf"
    source_pdf_previous.write_bytes(b"%PDF-1.4\n")
    source_pdf_current.write_bytes(b"%PDF-1.4\n")

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
        source_pdf_previous=str(source_pdf_previous),
        source_pdf_current=str(source_pdf_current),
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert len(payload["matching"]["tables_added"]) == 1
    assert len(payload["matching"]["tables_removed"]) == 1
    assert payload["matching"]["tables_added"][0]["visual_sanity_render_status"] == "skipped_missing_anchor"
    assert payload["matching"]["tables_removed"][0]["visual_sanity_render_status"] == "skipped_missing_anchor"
    assert "confirmed" not in payload["matching"]["tables_added"][0]
    assert "confirmed" not in payload["matching"]["tables_removed"][0]


def test_compare_reports_gpt4o_recomputes_table_level_change_after_noise_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    table_prev = _table(
        table_id="prev_same",
        page=3,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "1", "text": "Voir pages 10 à 11"}],
        bbox=[0.1, 0.2, 0.9, 0.7],
    )
    table_curr = _table(
        table_id="curr_same",
        page=4,
        section="capital_management",
        title="Capital",
        table_summary="Ratios de capital",
        headers=["Indicateur", "Valeur"],
        indicators=["Ratio CET1"],
        footnotes=[{"id": "1", "text": "Voir pages 12 à 13"}],
        bbox=[0.1, 0.2, 0.9, 0.7],
    )
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=[table_prev],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[table_curr],
    )

    responses_by_kind: dict[str, list[dict]] = {
        "matching": [
            {
                "current_table_decisions": [
                    {
                        "current_table_id": "curr_same",
                        "decision": "matched",
                        "previous_table_id": "prev_same",
                        "match_confidence": 0.99,
                        "reason": "Même tableau.",
                    }
                ],
                "warnings": [],
            },
        ],
        "match_inspector": [
            {
                "verdict": "confirmed",
                "shared_indicators": ["Ratio CET1"],
                "confidence": 0.99,
                "reason": "Identical indicator.",
            },
        ],
        "diff_indicators": [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement indicateur.",
            },
        ],
        "diff_footnotes": [
            {
                "footnotes_added": [],
                "footnotes_removed": [],
                "footnotes_renamed": [
                    {
                        "previous_id": "1",
                        "current_id": "1",
                        "previous_text": "Voir pages 10 à 11",
                        "current_text": "Voir pages 12 à 13",
                        "reason": "Référence de page modifiée.",
                        "analyst_assessment": {
                            "relevance_level": 3,
                            "justification": "Page ref",
                        },
                    }
                ],
                "reason": "Une note change.",
            },
        ],
    }

    def fake_call_openai_json(**kwargs):
        kind = kwargs["call_kind"]
        return responses_by_kind[kind].pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["pair_comparisons"][0]["technical_diff"]["footnotes_renamed"] == []
    assert payload["pair_comparisons"][0]["technical_diff"]["table_level_change"] == "inchange"


def test_compare_reports_gpt4o_rejects_non_schema_7_tables_json(tmp_path: Path) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    previous_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    (previous_dir / "tables.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "bank_code": "bnc",
                "year": 2025,
                "quarter": "t1",
                "created_at": "2026-03-24T10:00:00",
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[],
    )

    with pytest.raises(ValueError, match="schema_version=6"):
        compare_reports_gpt4o(
            previous_dir=previous_dir,
            current_dir=current_dir,
            out_root=tmp_path / "comparisons",
            model="gpt-4o-test",
        )
