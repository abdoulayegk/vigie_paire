from __future__ import annotations

import json
from pathlib import Path

from vigilance.quality.quality_gate import evaluate_quality, run_quality_gate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_quality_gate_pass_writes_reports(tmp_path: Path) -> None:
    tables_payload = {
        "bank_code": "bnc",
        "year": 2025,
        "quarter": "t1",
        "created_at": "2026-03-24T10:00:00",
        "schema_version": 7,
        "tables": [
            {
                "table_id": "t1",
                "title": "Capital reglementaire",
                "page": 12,
                "section": "capital",
                "bbox": None,
                "row_count": 2,
                "headers": [],
                "table_summary": "Capital reglementaire",
                "indicators": ["CET1 ratio", "Tier 1 ratio"],
                "extraction_status": "ok",
                "footnotes": [
                    {"id": "1", "text": "Includes transitional arrangements."}
                ],
            }
        ],
    }
    tables_path = tmp_path / "tables.json"
    _write_json(tables_path, tables_payload)

    result = run_quality_gate(
        tables_path=tables_path,
        out_dir=tmp_path,
        bank_code="bnc",
        run_id="run_1",
    )

    assert result["status"] == "PASS"
    assert result["eligible_for_review"] is True
    assert (tmp_path / "quality_report.json").exists()
    assert (tmp_path / "quality_report.md").exists()
    assert (tmp_path / "quality_gate_status.json").exists()


def test_quality_gate_fails_on_suspect_unresolved_tables(tmp_path: Path) -> None:
    tables_payload = {
        "bank_code": "bnc",
        "year": 2025,
        "quarter": "t1",
        "created_at": "2026-03-24T10:00:00",
        "schema_version": 7,
        "tables": [
            {
                "table_id": "suspect_1",
                "title": "Rapport de gestion",
                "page": 40,
                "section": "risk_management",
                "bbox": [0.1, 0.1, 0.9, 0.8],
                "row_count": 0,
                "headers": [],
                "table_summary": "",
                "indicators": [],
                "extraction_status": "suspect_unresolved",
                "footnotes": [],
            }
        ],
    }
    tables_path = tmp_path / "tables.json"
    _write_json(tables_path, tables_payload)

    result = run_quality_gate(
        tables_path=tables_path,
        out_dir=tmp_path,
        bank_code="bnc",
        run_id="run_suspect",
    )

    report = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    assert result["eligible_for_review"] is False
    assert report["summary"]["tables_suspect_unresolved"] == 1
    assert any(
        "extraction_suspect_unresolved_tables=1" in reason
        for reason in report["fail_reasons"]
    )


def test_quality_gate_fails_on_repr_like_footnotes(tmp_path: Path) -> None:
    indicators = {
        "bank_code": "bnc",
        "run_id": "run_2",
        "tables": [
            {
                "table_id": "t2",
                "title": "Gestion des risques",
                "page": 20,
                "section": "risk",
                "indicators_row": ["RWA", "LCR"],
            }
        ],
    }
    footnotes = {
        "bank_code": "bnc",
        "run_id": "run_2",
        "tables": [
            {
                "table_id": "t2",
                "title": "Gestion des risques",
                "page": 20,
                "section": "risk",
                "footnotes": [{"id": "1", "text": "{'id': '1', 'text': 'legacy repr'}"}],
            }
        ],
    }
    report = evaluate_quality(indicators, footnotes)

    assert report["status"] == "FAIL"
    assert report["eligible_for_review"] is False
    assert report["summary"]["tables_failed_footnote_integrity"] == 1
    assert any("footnote_integrity_failed_tables" in r for r in report["fail_reasons"])


def test_quality_gate_fails_policy_on_duplicates_and_titles() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "a",
                "title": "Expositions brutes 79 772 76 163",
                "page": 1,
                "section": "s1",
                "indicators_row": ["x", "x", "x"],
            },
            {
                "table_id": "b",
                "title": "Revenue 100 200 300",
                "page": 2,
                "section": "s2",
                "indicators_row": ["a", "a", "b"],
            },
        ]
    }
    footnotes = {"tables": []}
    report = evaluate_quality(
        indicators,
        footnotes,
        config={
            "duplicate_ratio_threshold": 0.15,
            "max_tables_duplicate_excess": 0,
            "max_contaminated_titles": 0,
        },
    )

    assert report["status"] == "FAIL"
    assert report["summary"]["tables_duplicate_ratio_excess"] >= 1
    assert any("duplicate_ratio_excess_tables" in r for r in report["fail_reasons"])


def test_quality_gate_detects_line_split_suspicion() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "x1",
                "title": "Risk table",
                "page": 9,
                "section": "risk",
                "indicators_row": ["supplementaires1", "a", "-", "Core capital"],
            }
        ]
    }
    footnotes = {"tables": []}
    report = evaluate_quality(indicators, footnotes)
    table = report["tables"][0]
    assert int(table["suspicious_line_splits"]) >= 2


def test_quality_gate_exempts_date_header_titles_by_default() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "d1",
                "title": "Au 31 janvier 2025",
                "page": 1,
                "section": "s",
                "indicators_row": ["CET1"],
            },
            {
                "table_id": "d2",
                "title": "Pour le trimestre clos le 31 janvier 2025 31 octobre 2024",
                "page": 2,
                "section": "s",
                "indicators_row": ["RWA"],
            },
        ]
    }
    report = evaluate_quality(
        indicators,
        {"tables": []},
        config={"max_contaminated_titles": 0},
    )
    assert report["status"] == "PASS"
    assert report["summary"]["tables_title_contaminated"] == 0
    assert report["summary"]["tables_date_header_titles"] == 2
    assert all(not t["title_contaminated"] for t in report["tables"])


def test_quality_gate_still_flags_true_numeric_title_contamination() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "n1",
                "title": "239 323 240 796",
                "page": 3,
                "section": "s",
                "indicators_row": ["Actifs"],
            }
        ]
    }
    report = evaluate_quality(
        indicators,
        {"tables": []},
        config={"max_contaminated_titles": 0},
    )
    assert report["status"] == "FAIL"
    assert report["summary"]["tables_title_contaminated"] == 1
    assert report["summary"]["tables_date_header_titles"] == 0
    assert report["tables"][0]["title_contaminated"] is True


def test_quality_gate_auto_cleaned_title_is_non_blocking() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "n2",
                "title": "BMO (societe mere) 239 323 240 796",
                "page": 4,
                "section": "s",
                "indicators_row": ["Actifs"],
            }
        ]
    }
    report = evaluate_quality(
        indicators,
        {"tables": []},
        config={"max_contaminated_titles": 0},
    )
    assert report["status"] == "PASS"
    assert report["summary"]["tables_title_contaminated"] == 0
    assert report["summary"]["tables_title_auto_cleaned"] == 1
    assert report["tables"][0]["title_auto_cleaned"] is True
    assert report["tables"][0]["title_effective"] == "BMO (societe mere)"


def test_quality_gate_accepts_legacy_indicator_fields_as_fallback() -> None:
    indicators = {
        "tables": [
            {
                "table_id": "legacy_1",
                "title": "Capital reglementaire",
                "page": 8,
                "section": "capital",
                "indicators_raw": ["CET1", "Tier 1"],
                "indicators_normalized": ["cet1", "tier 1"],
            }
        ]
    }
    report = evaluate_quality(indicators, {"tables": []})
    assert report["status"] == "PASS"
    assert report["summary"]["tables_total"] == 1
    assert report["tables"][0]["duplicate_ratio"] == 0.0
