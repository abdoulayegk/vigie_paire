"""Smoke test for section-aware table export in run_tables CLI."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from vigilance.cli.run_tables import main


def test_run_tables_keeps_section_context(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "banks": {
                    "rbc": {"name": "RBC"},
                },
            }
        ),
        encoding="utf-8",
    )

    ranges_path = tmp_path / "section_ranges.json"
    ranges_path.write_text(
        json.dumps(
            {
                "section_ranges": [
                    {"section": "capital_management", "start": 10, "end": 12},
                    {"section": "Gestion des risques", "start": 20, "end": 23},
                ]
            }
        ),
        encoding="utf-8",
    )

    fake_module = types.ModuleType("vigilance.extraction.docling_processor")

    def fake_extract_tables_docling_by_sections(
        pdf_path: str,
        bank_code: str,
        quarter: str,
        year: int,
        section_ranges: list[dict],
        use_vision_primary: object = None,
    ) -> list[SimpleNamespace]:
        out: list[SimpleNamespace] = []
        for i, sr in enumerate(section_ranges, start=1):
            out.append(
                SimpleNamespace(
                    section=sr["section"],
                    page_number=sr["start"],
                    table_id=f"table_{i}",
                    title=f"Table {i}",
                    headers=["Indicator", "Value"],
                    rows=[["A", "1"]],
                    extraction_method="docling",
                )
            )
        return out

    fake_module.extract_tables_docling_by_sections = fake_extract_tables_docling_by_sections
    monkeypatch.setitem(sys.modules, "vigilance.extraction.docling_processor", fake_module)

    out_root = tmp_path / "outputs"
    main(
        [
            "--bank",
            "rbc",
            "--pdf",
            "dummy.pdf",
            "--quarter",
            "t1-2025",
            "--config",
            str(cfg_path),
            "--ranges_json",
            str(ranges_path),
            "--out_root",
            str(out_root),
        ]
    )

    output_path = out_root / "t1-2025" / "rbc" / "tables_docling.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "tables" in payload
    assert len(payload["tables"]) == 2
    assert all("section" in table for table in payload["tables"])
    assert all("first_column_indicators" in table for table in payload["tables"])
    assert all(table["first_column_indicators"] == ["A"] for table in payload["tables"])
    assert {table["section"] for table in payload["tables"]} == {
        "capital_management",
        "risk_management",
    }
