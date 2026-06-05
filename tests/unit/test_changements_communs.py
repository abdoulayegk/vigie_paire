from __future__ import annotations

import json
from pathlib import Path

from vigilance.changements_communs import (
    build_changements_communs_judge_messages,
    build_changements_communs_source_stats,
    changements_communs_output_path,
    collect_changements_communs_records,
    generate_changements_communs_report,
    load_changements_communs_report,
    write_changements_communs_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_changements_communs_records_reads_v3_and_v1_shapes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "bnc" / "2025_t2_vs_2025_t1" / "text_comparison.json",
        {
            "schema_version": 3,
            "bank_code": "bnc",
            "section_comparisons": [
                {
                    "section_title": "Gestion des risques",
                    "block_comparisons": [
                        {
                            "change_id": "chg_001",
                            "diff_type": "added",
                            "subsection_heading": "Risque climatique",
                            "source_text_t2": "Nouveau paragraphe climatique.",
                            "pages_t2": [12],
                            "change_summary": "Ajout climat",
                            "genai_triage": {
                                "themes_amf": ["ESG_CLIMATIQUE"],
                                "impact_level": "MAJEUR",
                            },
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "cibc" / "2025_t2_vs_2025_t1" / "text_comparison.json",
        {
            "schema_version": 1,
            "bank_code": "cibc",
            "section_comparisons": [
                {
                    "section_title": "Gestion des risques",
                    "block_comparisons": [
                        {
                            "change_id": "ges_chg_137",
                            "diff_type": "added",
                            "change_summary": "Ajout tarifs",
                            "block_t1": None,
                            "block_t2": {
                                "text": "Les tarifs douaniers augmentent le risque.",
                                "page": 37,
                            },
                            "genai_triage": {"impact_level": "MODERE"},
                        }
                    ],
                }
            ],
        },
    )

    records = collect_changements_communs_records(root_dir=tmp_path)

    assert [record.bank_code for record in records] == ["bnc", "cibc"]
    assert records[0].record_id == "bnc:2025_t2_vs_2025_t1:chg_001"
    assert records[0].themes == ("ESG_CLIMATIQUE",)
    assert records[0].pages_after == (12,)
    assert records[1].text_after == "Les tarifs douaniers augmentent le risque."
    assert records[1].pages_after == (37,)


def test_changements_communs_source_stats_counts_banks_and_periods(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "td" / "2026_t1_vs_2025_t3" / "text_comparison.json",
        {
            "bank_code": "td",
            "section_comparisons": [
                {
                    "section_title": "Gestion du capital",
                    "block_comparisons": [
                        {
                            "change_id": "chg",
                            "diff_type": "modified",
                            "source_text_t1": "Avant",
                            "source_text_t2": "Apres",
                            "genai_triage": {"impact_level": "MAJEUR"},
                        }
                    ],
                }
            ],
        },
    )

    stats = build_changements_communs_source_stats(collect_changements_communs_records(root_dir=tmp_path))

    assert stats["total_changes"] == 1
    assert stats["bank_count"] == 1
    assert stats["period_count"] == 1
    assert stats["impact_counts"] == {"MAJEUR": 1}


def test_collect_changements_communs_records_can_filter_one_period(tmp_path: Path) -> None:
    for period, text in [
        ("2025_t2_vs_2025_t1", "Changement T2."),
        ("2025_t3_vs_2025_t2", "Changement T3."),
    ]:
        _write_json(
            tmp_path / "td" / period / "text_comparison.json",
            {
                "bank_code": "td",
                "section_comparisons": [
                    {
                        "section_title": "Gestion des risques",
                        "block_comparisons": [
                            {
                                "change_id": period,
                                "diff_type": "added",
                                "source_text_t2": text,
                                "change_summary": text,
                            }
                        ],
                    }
                ],
            },
        )

    records = collect_changements_communs_records(
        root_dir=tmp_path,
        period="2025_t2_vs_2025_t1",
    )

    assert len(records) == 1
    assert records[0].period == "2025_t2_vs_2025_t1"
    assert records[0].text_after == "Changement T2."


def test_changements_communs_judge_prompt_uses_candidate_record_ids(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "rbc" / "2025_t2_vs_2025_t1" / "text_comparison.json",
        {
            "bank_code": "rbc",
            "section_comparisons": [
                {
                    "section_title": "Gestion des risques",
                    "block_comparisons": [
                        {
                            "change_id": "risk_001",
                            "diff_type": "modified",
                            "source_text_t1": "Ancien texte",
                            "source_text_t2": "Nouveau texte",
                            "change_summary": "Modification risque",
                        }
                    ],
                }
            ],
        },
    )
    record = collect_changements_communs_records(root_dir=tmp_path)[0]

    messages = build_changements_communs_judge_messages(
        topic="risques climatiques",
        candidates=[record],
        min_banks=3,
    )

    assert messages[0]["role"] == "system"
    assert record.record_id in messages[1]["content"]
    assert "min_banks" in messages[1]["content"]
    assert "signal_mineur_2_banques" in messages[1]["content"]


def test_generate_report_classifies_two_bank_signal_as_minor(tmp_path: Path) -> None:
    for bank in ("bmo", "bnc"):
        _write_json(
            tmp_path / bank / "2025_t2_vs_2025_t1" / "text_comparison.json",
            {
                "bank_code": bank,
                "section_comparisons": [
                    {
                        "section_title": "Gestion des risques",
                        "block_comparisons": [
                            {
                                "change_id": "risk_001",
                                "diff_type": "added",
                                "source_text_t2": "Un changement sur les tarifs douaniers.",
                                "change_summary": "Ajout tarifs",
                            }
                        ],
                    }
                ],
            },
        )

    class _EmbeddingItem:
        def __init__(self, index: int) -> None:
            self.index = index
            self.embedding = [1.0, 0.0]

    class _Embeddings:
        def create(self, *, model: str, input: list[str]):
            class _Response:
                pass

            response = _Response()
            response.data = [_EmbeddingItem(index) for index, _ in enumerate(input)]
            return response

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Message(content)

    class _ChatCompletions:
        def create(self, **kwargs):
            user_payload = json.loads(kwargs["messages"][1]["content"])
            record_ids = [item["record_id"] for item in user_payload["candidates"][:2]]
            payload = {
                "signals": [
                    {
                        "theme": "Tarifs douaniers",
                        "summary": "Le LLM surestime le consensus.",
                        "status": "consensus_3_plus",
                        "banks": ["bmo", "bnc", "td"],
                        "evidence": [
                            {
                                "record_id": record_ids[0],
                                "quote": "tarifs douaniers",
                                "why_relevant": "preuve bmo",
                            },
                            {
                                "record_id": record_ids[1],
                                "quote": "tarifs douaniers",
                                "why_relevant": "preuve bnc",
                            }
                        ],
                    }
                ]
            }

            class _Response:
                pass

            response = _Response()
            response.choices = [_Choice(json.dumps(payload))]
            return response

    class _Chat:
        def __init__(self) -> None:
            self.completions = _ChatCompletions()

    class _Client:
        def __init__(self) -> None:
            self.embeddings = _Embeddings()
            self.chat = _Chat()

    report = generate_changements_communs_report(
        "tarifs douaniers",
        period="2025_t2_vs_2025_t1",
        root_dir=tmp_path,
        client=_Client(),
        model="fake-model",
    )

    signal = report["signals"][0]
    assert report["artifact_type"] == "changements_communs_banques"
    assert report["period"] == "2025_t2_vs_2025_t1"
    assert report["analysis_scope"] == "single_period"
    assert report["source_stats"]["periods"] == ["2025_t2_vs_2025_t1"]
    assert signal["banks"] == ["bmo", "bnc"]
    assert signal["bank_count"] == 2
    assert signal["min_banks_met"] is False
    assert signal["status"] == "signal_mineur_2_banques"
    assert report["signal_counts"] == {"total": 1, "consensus": 0, "minor": 1}


def test_period_artifact_lives_under_resultats_changements_communs_period() -> None:
    path = changements_communs_output_path("2025_t2_vs_2025_t1")

    assert path.parts[-4:] == (
        "resultats",
        "changements_communs_banques",
        "2025_t2_vs_2025_t1",
        "changements_communs_banques.json",
    )


def test_write_and_load_changements_communs_report(tmp_path: Path) -> None:
    period = "2025_t2_vs_2025_t1"
    path = changements_communs_output_path(
        period,
        base_dir=tmp_path / "resultats" / "changements_communs_banques",
    )
    payload = {
        "artifact_type": "changements_communs_banques",
        "schema_version": 1,
        "period": period,
        "signals": [],
    }

    written = write_changements_communs_report(payload, path=path)
    loaded = load_changements_communs_report(path=written)

    assert written.parts[-2:] == (
        "2025_t2_vs_2025_t1",
        "changements_communs_banques.json",
    )
    assert loaded == payload
