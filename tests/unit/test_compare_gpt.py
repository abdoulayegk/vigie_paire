from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigilance.compare_gpt import (
    compare_reports_gpt4o,
    normalize_quarter,
    resolve_reference_period,
)
from vigilance.cli.run_compare_gpt4o import main


def _write_tables_json(path: Path, *, bank: str, year: int, quarter: str, tables: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "bank_code": bank,
                "year": year,
                "quarter": quarter,
                "created_at": "2026-03-22T10:00:00",
                "model_version": "gpt-4o",
                "prompt_version": "extract_v1",
                "tables": tables,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_compare_reports_gpt4o_writes_comparison_json(
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
            {
                "table_id": "prev_1",
                "page": 10,
                "section": "capital_management",
                "title": "Capital",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["Ratio CET1", "13.1"]],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [{"id": "1", "text": "Note A"}],
            },
            {
                "table_id": "prev_2",
                "page": 12,
                "section": "risk_management",
                "title": "Risque",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["RWA", "100"]],
                "indicators_raw": ["RWA"],
                "indicators_normalized": ["rwa"],
                "footnotes": [],
            },
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            {
                "table_id": "curr_1",
                "page": 11,
                "section": "capital_management",
                "title": "Capital",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["Ratio CET1", "13.4"], ["Ratio de levier", "4.1"]],
                "indicators_raw": ["Ratio CET1", "Ratio de levier"],
                "indicators_normalized": ["ratio cet1", "ratio de levier"],
                "footnotes": [{"id": "1", "text": "Note A mise a jour"}],
            },
            {
                "table_id": "curr_2",
                "page": 15,
                "section": "liquidite",
                "title": "Liquidite",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["LCR", "120"]],
                "indicators_raw": ["LCR"],
                "indicators_normalized": ["lcr"],
                "footnotes": [],
            },
        ],
    )

    responses = [
        {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 0.97,
                    "reason": "Meme sujet et meme indicateur principal.",
                }
            ],
            "tables_added": [{"table_id": "curr_2", "reason": "Nouveau tableau"}],
            "tables_removed": [{"table_id": "prev_2", "reason": "Tableau absent au trimestre courant"}],
        },
        {
            "indicators_added": [
                {"value": "ratio de levier", "reason": "Ajoute dans le tableau courant"}
            ],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [
                {
                    "previous_id": "1",
                    "current_id": "1",
                    "previous_text": "Note A",
                    "current_text": "Note A mise a jour",
                    "reason": "Meme note reformulee",
                }
            ],
            "reason": "Le tableau conserve sa structure avec un indicateur supplementaire.",
        },
    ]

    seen_prompts: list[dict] = []

    def fake_call_openai_json(**kwargs):
        assert kwargs["model"] == "gpt-4o-test"
        assert isinstance(kwargs["messages"], list)
        prompt = json.loads(kwargs["messages"][-1]["content"])
        seen_prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    previous_pdf = tmp_path / "previous.pdf"
    current_pdf = tmp_path / "current.pdf"
    previous_pdf.write_bytes(b"%PDF-1.4 previous")
    current_pdf.write_bytes(b"%PDF-1.4 current")

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
        source_pdf_previous=str(previous_pdf),
        source_pdf_current=str(current_pdf),
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "report_comparison"
    assert payload["run_id"]
    assert payload["bank_code"] == "bnc"
    assert payload["quarter_previous"] == "t1"
    assert payload["quarter_current"] == "t2"
    assert payload["source_pdf_previous"] == str(previous_pdf)
    assert payload["source_pdf_current"] == str(current_pdf)
    assert payload["archived_pdf_previous"] == str(comparison_path.parent / "previous_report.pdf")
    assert payload["archived_pdf_current"] == str(comparison_path.parent / "current_report.pdf")
    assert (comparison_path.parent / "previous_report.pdf").read_bytes() == previous_pdf.read_bytes()
    assert (comparison_path.parent / "current_report.pdf").read_bytes() == current_pdf.read_bytes()
    assert payload["reference_resolution"]["mode"] == "automatique"
    assert payload["reference_resolution"]["year_previous"] == 2025
    assert payload["reference_resolution"]["quarter_previous"] == "t1"
    assert comparison_path.parent.name == payload["run_id"]
    assert payload["matching"]["matched_pairs"][0]["previous_table_id"] == "prev_1"
    assert payload["matching"]["tables_added"][0]["table_id"] == "curr_2"
    assert payload["matching"]["tables_added"][0]["title"] == "Liquidite"
    assert payload["matching"]["tables_added"][0]["page"] == 15
    assert payload["matching"]["tables_added"][0]["analyst_assessment"]["theme"] == "liquidite"
    assert payload["matching"]["tables_removed"][0]["analyst_assessment"]["theme"] == "autre"
    assert payload["matching"]["tables_removed"][0]["title"] == "Risque"
    assert payload["summary"]["matched_pairs_total"] == 1
    assert payload["summary"]["tables_added_total"] == 1
    assert payload["summary"]["tables_removed_total"] == 1
    assert payload["summary"]["indicator_changes_total"] == 1
    assert payload["summary"]["footnote_changes_total"] == 1
    assert payload["summary"]["high_priority_items_total"] >= 1
    assert payload["pair_comparisons"][0]["technical_diff"]["table_level_change"] == "modifie"
    assert payload["pair_comparisons"][0]["technical_diff"]["indicators_added"][0]["value"] == "ratio de levier"
    assert payload["pair_comparisons"][0]["previous_table"]["title"] == "Capital"
    assert payload["pair_comparisons"][0]["current_table"]["page"] == 11
    assert payload["pair_comparisons"][0]["analyst_assessment"]["theme"] == "capital"
    assert payload["pair_comparisons"][0]["analyst_assessment"]["review_priority"] in {"prioritaire", "critique"}

    match_prompt = seen_prompts[0]
    assert "previous_tables" in match_prompt
    assert match_prompt["response_schema"]["tables_added"][0]["table_id"] == "string"
    assert match_prompt["response_schema"]["tables_removed"][0]["table_id"] == "string"
    prev0 = match_prompt["previous_tables"][0]
    assert prev0["extraction_index"] == 0
    assert prev0["page"] == 10
    assert prev0["row_count"] == 1
    assert prev0["headers"] == ["Indicateur", "Valeur"]
    assert prev0["header_columns_total"] == 2
    assert "rows" not in prev0

    diff_prompt = seen_prompts[1]
    assert "headers" not in diff_prompt["previous_table"]
    assert "rows" not in diff_prompt["previous_table"]
    assert "headers" not in diff_prompt["current_table"]
    assert "rows" not in diff_prompt["current_table"]


def test_compare_reports_gpt4o_accepts_legacy_unmatched_id_keys(
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
            {
                "table_id": "prev_1",
                "page": 10,
                "section": "capital_management",
                "title": "Capital",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["Ratio CET1", "13.1"]],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [],
            },
            {
                "table_id": "prev_2",
                "page": 12,
                "section": "risk_management",
                "title": "Risque",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["RWA", "100"]],
                "indicators_raw": ["RWA"],
                "indicators_normalized": ["rwa"],
                "footnotes": [],
            },
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            {
                "table_id": "curr_1",
                "page": 11,
                "section": "capital_management",
                "title": "Capital",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["Ratio CET1", "13.4"]],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [],
            },
            {
                "table_id": "curr_2",
                "page": 15,
                "section": "liquidite",
                "title": "Liquidite",
                "headers": ["Indicateur", "Valeur"],
                "rows": [["LCR", "120"]],
                "indicators_raw": ["LCR"],
                "indicators_normalized": ["lcr"],
                "footnotes": [],
            },
        ],
    )

    responses = [
        {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 0.97,
                    "reason": "Meme sujet et meme indicateur principal.",
                }
            ],
            "tables_added": [
                {
                    "current_table_id": "curr_2",
                    "reason": "Nouveau tableau",
                }
            ],
            "tables_removed": [
                {
                    "previous_table_id": "prev_2",
                    "reason": "Tableau absent au trimestre courant",
                }
            ],
        },
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Aucun changement semantique.",
        },
    ]

    monkeypatch.setattr(
        "vigilance.compare_gpt._call_openai_json",
        lambda **kwargs: responses.pop(0),
    )

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["matching"]["tables_added"] == [
        {
            "table_id": "curr_2",
            "reason": "Nouveau tableau",
            "title": "Liquidite",
            "page": 15,
            "section": "liquidite",
            "bbox": None,
            "indicators_raw": ["LCR"],
            "indicators_normalized": ["lcr"],
            "footnotes": [],
            "analyst_assessment": {
                "theme": "liquidite",
                "change_significance": "eleve",
                "review_priority": "critique",
                "analyst_summary": "Nouveau tableau sur le theme liquidite a revoir par l'analyste.",
            },
        }
    ]
    assert payload["matching"]["tables_removed"] == [
        {
            "table_id": "prev_2",
            "reason": "Tableau absent au trimestre courant",
            "title": "Risque",
            "page": 12,
            "section": "risk_management",
            "bbox": None,
            "indicators_raw": ["RWA"],
            "indicators_normalized": ["rwa"],
            "footnotes": [],
            "analyst_assessment": {
                "theme": "autre",
                "change_significance": "moyen",
                "review_priority": "normale",
                "analyst_summary": "Tableau supprime sur le theme autre a confirmer par l'analyste.",
            },
        }
    ]


def test_compare_reports_gpt4o_rejects_missing_pathlike_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Chemin requis manquant: previous_dir"):
        compare_reports_gpt4o(
            previous_dir=None,
            current_dir=tmp_path / "current",
            out_root=tmp_path / "comparisons",
            model="gpt-4o-test",
        )


def test_compare_reports_gpt4o_prompts_ignore_numeric_and_date_changes(
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
            {
                "table_id": "prev_num",
                "page": 5,
                "section": "capital_management",
                "title": "Capital au 31 janvier 2025",
                "headers": ["Indicateur", "T1 2025"],
                "rows": [["Ratio CET1", "13.1"]],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [{"id": "1", "text": "Au 31 janvier 2025."}],
            }
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=[
            {
                "table_id": "curr_num",
                "page": 6,
                "section": "capital_management",
                "title": "Capital au 30 avril 2025",
                "headers": ["Indicateur", "T2 2025"],
                "rows": [["Ratio CET1", "13.9"]],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [{"id": "1", "text": "Au 30 avril 2025."}],
            }
        ],
    )

    responses = [
        {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_num",
                    "current_table_id": "curr_num",
                    "match_confidence": 0.99,
                    "reason": "Meme indicateur et meme theme.",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        },
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Les differences numeriques et de date sont ignorees.",
        },
    ]
    seen_rules: list[list[str]] = []

    def fake_call_openai_json(**kwargs):
        prompt = json.loads(kwargs["messages"][-1]["content"])
        seen_rules.append(prompt["rules"])
        return responses.pop(0)

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    technical_diff = payload["pair_comparisons"][0]["technical_diff"]
    assert technical_diff["table_level_change"] == "inchange"
    assert payload["summary"]["indicator_changes_total"] == 0
    assert payload["summary"]["footnote_changes_total"] == 0
    assert any("Ignore all numeric changes" in " ".join(rules) for rules in seen_rules)


def test_compare_reports_gpt4o_classifies_credit_risk_deterministically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bns" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bns" / "2025" / "t2"
    tables = [
        {
            "table_id": "risk_1",
            "page": 8,
            "section": "funding",
            "title": "Expositions au risque de credit de la Banque par regions",
            "headers": ["Indicateur", "Valeur"],
            "rows": [["Amerique latine", "100"]],
            "indicators_raw": ["Amerique latine", "Europe"],
            "indicators_normalized": ["amerique latine", "europe"],
            "footnotes": [],
        }
    ]
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bns",
        year=2025,
        quarter="t1",
        tables=tables,
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bns",
        year=2025,
        quarter="t2",
        tables=tables,
    )

    responses = [
        {
            "matched_pairs": [
                {
                    "previous_table_id": "risk_1",
                    "current_table_id": "risk_1",
                    "match_confidence": 0.99,
                    "reason": "Meme tableau",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        }
    ]

    monkeypatch.setattr(
        "vigilance.compare_gpt._call_openai_json",
        lambda **_: responses.pop(0),
    )

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assessment = payload["pair_comparisons"][0]["analyst_assessment"]
    assert assessment["theme"] == "risque_credit"


def test_compare_reports_gpt4o_short_circuits_trivial_diff_and_records_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t1"
    current_dir = tmp_path / "extractions" / "bnc" / "2025" / "t2"
    tables = [
        {
            "table_id": "same_1",
            "page": 3,
            "section": "capital_management",
            "title": "Capital",
            "headers": ["Indicateur", "T1 2025"],
            "rows": [["Ratio CET1", "13.1"]],
            "indicators_raw": ["Ratio CET1"],
            "indicators_normalized": ["ratio cet1"],
            "footnotes": [{"id": "1", "text": "Note stable"}],
        }
    ]
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t1",
        tables=tables,
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t2",
        tables=tables,
    )

    def fake_call_openai_json(**kwargs):
        if kwargs.get("usage_recorder") is not None:
            kwargs["usage_recorder"].append(
                {
                    "model": kwargs["model"],
                    "call_kind": kwargs.get("call_kind", ""),
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                }
            )
        return {
            "matched_pairs": [
                {
                    "previous_table_id": "same_1",
                    "current_table_id": "same_1",
                    "match_confidence": 1.0,
                    "reason": "Meme tableau",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        }

    monkeypatch.setattr("vigilance.compare_gpt._call_openai_json", fake_call_openai_json)

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o",
        runtime_extraction_sec=4.25,
        extraction_run_metrics={
            "previous": {
                "vision_calls_total": 3,
                "vision_rescue_total": 1,
                "prompt_tokens_total": 1000,
                "completion_tokens_total": 200,
                "total_tokens_total": 1200,
                "estimated_cost_usd": 0.018,
                "cache_hit": False,
            },
            "current": {
                "vision_calls_total": 4,
                "vision_rescue_total": 0,
                "prompt_tokens_total": 900,
                "completion_tokens_total": 180,
                "total_tokens_total": 1080,
                "estimated_cost_usd": 0.016,
                "cache_hit": False,
            },
        },
    )

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    pair = payload["pair_comparisons"][0]
    assert pair["technical_diff"]["table_level_change"] == "inchange"
    assert pair["diff_mode"] == "local_exact_match"
    metrics = payload["run_metrics"]
    assert metrics["runtime_extraction_sec"] == 4.25
    assert metrics["vision_calls_total"] == 7
    assert metrics["vision_rescue_total"] == 1
    assert metrics["comparison_calls_total"] == 1
    assert metrics["comparison_local_diff_skips"] == 1
    assert metrics["prompt_tokens_total"] == 2020
    assert metrics["completion_tokens_total"] == 410
    assert metrics["estimated_cost_usd"] > 0.0


def test_compare_reports_gpt4o_isolates_runs_with_unique_output_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_dir = tmp_path / "extractions" / "bnc" / "2025" / "t3"
    current_dir = tmp_path / "extractions" / "bnc" / "2026" / "t1"
    _write_tables_json(
        previous_dir / "tables.json",
        bank="bnc",
        year=2025,
        quarter="t3",
        tables=[
            {
                "table_id": "prev_1",
                "page": 1,
                "section": "capital",
                "title": "Capital",
                "headers": [],
                "rows": [],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [],
            }
        ],
    )
    _write_tables_json(
        current_dir / "tables.json",
        bank="bnc",
        year=2026,
        quarter="t1",
        tables=[
            {
                "table_id": "curr_1",
                "page": 2,
                "section": "capital",
                "title": "Capital",
                "headers": [],
                "rows": [],
                "indicators_raw": ["Ratio CET1"],
                "indicators_normalized": ["ratio cet1"],
                "footnotes": [],
            }
        ],
    )

    responses = [
        {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 1.0,
                    "reason": "Meme tableau",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        },
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Aucun changement.",
        },
        {
            "matched_pairs": [
                {
                    "previous_table_id": "prev_1",
                    "current_table_id": "curr_1",
                    "match_confidence": 1.0,
                    "reason": "Meme tableau",
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        },
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Aucun changement.",
        },
    ]

    monkeypatch.setattr(
        "vigilance.compare_gpt._call_openai_json",
        lambda **_: responses.pop(0),
    )
    monkeypatch.setattr("vigilance.compare_gpt._make_run_id", lambda: "20260323_143015")

    path_a = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )
    path_b = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=tmp_path / "comparisons",
        model="gpt-4o-test",
    )

    assert path_a != path_b
    assert path_a.parent.name == "20260323_143015"
    assert path_b.parent.name == "20260323_143015_02"


def test_run_compare_gpt4o_cli_resolves_standard_paths(tmp_path: Path, monkeypatch) -> None:
    expected = tmp_path / "comparisons" / "bnc" / "2025_t2_vs_2025_t1" / "comparison.json"

    def fake_compare_reports_gpt4o(**kwargs):
        assert kwargs["previous_dir"] == tmp_path / "extractions" / "bnc" / "2025" / "t1"
        assert kwargs["current_dir"] == tmp_path / "extractions" / "bnc" / "2025" / "t2"
        assert kwargs["out_root"] == tmp_path / "comparisons"
        assert kwargs["reference_resolution"]["mode"] == "automatique"
        assert kwargs["reference_resolution"]["year_previous"] == 2025
        assert kwargs["reference_resolution"]["quarter_previous"] == "t1"
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text("{}", encoding="utf-8")
        return expected

    monkeypatch.setattr(
        "vigilance.cli.run_compare_gpt4o.compare_reports_gpt4o",
        fake_compare_reports_gpt4o,
    )

    rc = main(
        [
            "--bank",
            "bnc",
            "--year-current",
            "2025",
            "--quarter-current",
            "T2",
            "--extraction-root",
            str(tmp_path / "extractions"),
            "--out-root",
            str(tmp_path / "comparisons"),
        ]
    )

    assert rc == 0


def test_resolve_reference_period_follows_business_rule() -> None:
    assert resolve_reference_period(2025, "t2") == (2025, "t1")
    assert resolve_reference_period(2025, "t3") == (2025, "t2")
    assert resolve_reference_period(2026, "t1") == (2025, "t3")
    assert resolve_reference_period(2026, "t4") == (2025, "t4")


def test_normalize_quarter_accepts_t_and_q_variants() -> None:
    assert normalize_quarter("T1") == "t1"
    assert normalize_quarter("Q2") == "t2"
    assert normalize_quarter(" q3 ") == "t3"


def test_normalize_quarter_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        normalize_quarter("t5")
