"""Ensure run_tables exports first-column indicators explicitly."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from vigie.cli.run_tables import main


def test_run_tables_exports_first_column_indicators_region_table(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"version": "1.0", "banks": {"rbc": {"name": "RBC"}}}),
        encoding="utf-8",
    )

    ranges_path = tmp_path / "section_ranges.json"
    ranges_path.write_text(
        json.dumps({"section_ranges": [{"section": "capital_management", "start": 10, "end": 10}]}),
        encoding="utf-8",
    )

    fake_module = types.ModuleType("vigie.extraction.docling.processor")

    def fake_extract_tables_docling_by_sections(
        pdf_path: str,
        bank_code: str,
        quarter: str,
        year: int,
        section_ranges: list[dict],
        use_vision_extraction: object = None,
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                section="capital_management",
                page_number=10,
                table_id="table_24",
                title="TABLEAU 24 - Marges de crédit sur valeur domiciliaire",
                headers=["Région", "Encours"],
                rows=[
                    ["Atlantique", "1072"],
                    ["Québec", "9145"],
                    ["Ontario", "25288"],
                    ["Alberta", "3171"],
                    ["Colombie-Britannique", "10513"],
                    ["Ailleurs au Canada", "719"],
                    ["Total au Canada", "49908"],
                    ["États-Unis", "6806"],
                    ["Total", "56714"],
                ],
                first_column_indicators_raw=[
                    "Atlantique",
                    "Québec",
                    "Ontario",
                    "Alberta",
                    "Colombie-Britannique",
                    "Ailleurs au Canada",
                    "Total au Canada",
                    "États-Unis",
                    "Total",
                ],
                extraction_method="vision_full_gpt4o",
                extraction_status="rescued",
                bbox=[0.08, 0.18, 0.92, 0.43],
                title_clean="Marges de crédit sur valeur domiciliaire",
                table_summary="Encours des marges par région",
                title_raw="TABLEAU 24 - Marges de crédit sur valeur domiciliaire",
                debug_metrics={
                    "bbox_original": [0.10, 0.20, 0.90, 0.40],
                    "bbox_final": [0.08, 0.18, 0.92, 0.43],
                    "bbox_source": "page_context_locator",
                    "bbox_confidence": 0.96,
                    "bbox_verified": True,
                },
                footnotes=[],
                content_source="vision_gpt4o",
            )
        ]

    fake_module.extract_tables_docling_by_sections = fake_extract_tables_docling_by_sections
    monkeypatch.setitem(sys.modules, "vigie.extraction.docling.processor", fake_module)

    out_root = tmp_path / "outputs"
    main(
        [
            "--banque",
            "rbc",
            "--pdf",
            "dummy.pdf",
            "--trimestre",
            "t1-2025",
            "--config",
            str(cfg_path),
            "--ranges_json",
            str(ranges_path),
            "--sortie",
            str(out_root),
        ]
    )

    payload = json.loads((out_root / "t1-2025" / "rbc" / "tables_docling.json").read_text(encoding="utf-8"))
    table = payload["tables"][0]
    assert table["first_column_indicators_raw"] == [
        "Atlantique",
        "Québec",
        "Ontario",
        "Alberta",
        "Colombie-Britannique",
        "Ailleurs au Canada",
        "Total au Canada",
        "États-Unis",
        "Total",
    ]
    assert table["extraction_status"] == "rescued"
    assert table["bbox"] == [0.08, 0.18, 0.92, 0.43]
    assert table["debug_metrics"]["bbox_source"] == "page_context_locator"
