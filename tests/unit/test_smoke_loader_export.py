"""Smoke tests for config loading and JSON export."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from vigie.support.config.loader import get_bank_cfg, load_config
from vigie.support.models.section_models import SectionRange, SectionRangesResult
from vigie.support.models.table_models import TableArtifact
from vigie.support.report.export_json import write_section_ranges, write_tables_docling


def test_loader_and_export_smoke(tmp_path: Path) -> None:
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
    cfg = load_config(cfg_path)
    bank_cfg = get_bank_cfg(cfg, "rbc")
    assert bank_cfg["name"] == "RBC"

    run_dir = tmp_path / "outputs" / "runs" / "t1-2025" / "rbc"

    ranges = SectionRangesResult(
        bank_code="rbc",
        quarter="t1-2025",
        pdf_path="sample.pdf",
        ranges=[
            SectionRange(
                section="gestion_capital",
                start_page_pdf=10,
                end_page_pdf=20,
                method="toc",
                confidence=0.9,
                evidence={},
            )
        ],
    )
    section_path = write_section_ranges(run_dir, ranges)
    assert section_path.exists()
    section_payload = json.loads(section_path.read_text(encoding="utf-8"))
    assert section_payload["metadata"]["bank_code"] == "rbc"
    assert section_payload["ranges"][0]["start_page_pdf"] == 10

    tables = [
        TableArtifact(
            bank_code="rbc",
            section="gestion_capital",
            page_pdf=12,
            table_id="table_1",
            title="Capital adequacy",
            headers=["Metric", "Value"],
            rows=[["CET1", "13.2%"]],
            first_column_indicators=["CET1"],
            extraction_method="docling",
            quarter="t1-2025",
            pdf_path="sample.pdf",
        )
    ]
    tables_path = write_tables_docling(run_dir, tables)
    assert tables_path.exists()
    tables_payload = json.loads(tables_path.read_text(encoding="utf-8"))
    assert tables_payload["metadata"]["bank_code"] == "rbc"
    assert tables_payload["tables"][0]["table_id"] == "table_1"
    assert tables_payload["tables"][0]["first_column_indicators"] == ["CET1"]
