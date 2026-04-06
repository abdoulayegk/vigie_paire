from __future__ import annotations

import json
from pathlib import Path

from vigilance.comparison_canonical import to_canonical_payload
from vigilance.dash_app.services.text_comparison_store import (
    resolve_text_comparison_from_payload,
)


def _write_text_comparison(root_dir: Path) -> None:
    path = root_dir / "bnc" / "2025_t2_vs_2025_t1" / "text_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bank_code": "bnc",
                "quarter_current": "2025_t2",
                "quarter_previous": "2025_t1",
                "global_summary": {},
                "section_comparisons": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _raw_report_comparison_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_type": "report_comparison",
        "bank_code": "bnc",
        "year_previous": 2025,
        "quarter_previous": "t1",
        "year_current": 2025,
        "quarter_current": "t2",
        "matching": {
            "matched_pairs": [],
            "tables_added": [],
            "tables_removed": [],
        },
        "summary": {
            "matched_pairs_total": 0,
            "tables_added_total": 0,
            "tables_removed_total": 0,
            "indicator_changes_total": 0,
            "footnote_changes_total": 0,
            "high_priority_items_total": 0,
        },
    }


def test_resolve_text_comparison_from_raw_payload(tmp_path: Path) -> None:
    _write_text_comparison(tmp_path)

    resolved = resolve_text_comparison_from_payload(
        _raw_report_comparison_payload(),
        root_dir=tmp_path,
    )

    assert resolved is not None
    assert resolved["bank_code"] == "bnc"


def test_resolve_text_comparison_from_canonical_payload(tmp_path: Path) -> None:
    _write_text_comparison(tmp_path)
    canonical = to_canonical_payload(_raw_report_comparison_payload())

    resolved = resolve_text_comparison_from_payload(
        canonical,
        root_dir=tmp_path,
    )

    assert resolved is not None
    assert resolved["quarter_current"] == "2025_t2"
