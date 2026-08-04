from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from vigie.cli.run_extract_report import main


def test_run_extract_report_writes_compact_artifacts(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"version": "1.0", "banks": {"bnc": {"name": "BNC"}}}),
        encoding="utf-8",
    )

    locator_module = types.ModuleType("vigie.extraction.localisation_sections.section_locator")
    processor_module = types.ModuleType("vigie.extraction.docling.processor")

    def fake_locate_sections_in_pdf(
        pdf_path: str,
        bank_code: str | None = None,
        quarter: str | None = None,
        year: int = 2025,
    ) -> SimpleNamespace:
        assert pdf_path == "dummy.pdf"
        assert bank_code == "bnc"
        assert quarter == "t1"
        assert year == 2025
        return SimpleNamespace(
            sections=[
                SimpleNamespace(
                    section_type="Gestion des risques",
                    start_page=12,
                    end_page=14,
                )
            ]
        )

    def fake_extract_tables_docling_by_sections(
        pdf_path: str,
        bank_code: str,
        quarter: str,
        year: int,
        section_ranges: list[dict],
        use_vision_extraction: object = None,
    ) -> list[SimpleNamespace]:
        assert pdf_path == "dummy.pdf"
        assert bank_code == "bnc"
        assert quarter == "t1"
        assert year == 2025
        assert section_ranges == [
            {"section": "risk_management", "start": 12, "end": 14}
        ]
        return [
            SimpleNamespace(
                table_id="tbl_p012_i01",
                page_number=12,
                section="risk_management",
                title="Table risque",
                table_summary="Résumé métier du risque",
                title_clean=None,
                headers=["Indicator", "Value"],
                rows=[["Pertes attendues", "100"]],
                first_column_indicators_raw=["Pertes attendues"],
                first_column_indicators=["pertes attendues"],
                footnotes=[{"id": "1", "text": "Footnote de test"}],
                table_index_on_page=1,
            )
        ]

    locator_module.locate_sections_in_pdf = fake_locate_sections_in_pdf
    processor_module.extract_tables_docling_by_sections = (
        fake_extract_tables_docling_by_sections
    )
    monkeypatch.setitem(sys.modules, "vigie.extraction.localisation_sections.section_locator", locator_module)
    monkeypatch.setitem(sys.modules, "vigie.extraction.docling.processor", processor_module)

    out_root = tmp_path / "outputs"
    main(
        [
            "--banque",
            "bnc",
            "--pdf",
            "dummy.pdf",
            "--annee",
            "2025",
            "--trimestre",
            "T1",
            "--config",
            str(cfg_path),
            "--sortie",
            str(out_root),
        ]
    )

    out_dir = out_root / "bnc" / "2025" / "t1"
    assert (out_dir / "tables.json").exists()
    assert (out_dir / "indicators.json").exists()
    assert (out_dir / "footnotes.json").exists()

    tables_payload = json.loads((out_dir / "tables.json").read_text(encoding="utf-8"))
    indicators_payload = json.loads((out_dir / "indicators.json").read_text(encoding="utf-8"))
    footnotes_payload = json.loads((out_dir / "footnotes.json").read_text(encoding="utf-8"))

    assert tables_payload["tables"][0]["table_id"] == "tbl_p012_i01"
    assert indicators_payload["tables"][0]["table_id"] == "tbl_p012_i01"
    assert footnotes_payload["tables"][0]["table_id"] == "tbl_p012_i01"
    assert tables_payload["tables"][0]["section"] == "risk_management"
    assert tables_payload["tables"][0]["row_count"] == 1
    assert tables_payload["tables"][0]["table_summary"] == "Résumé métier du risque"
    assert tables_payload["tables"][0]["indicators"] == ["Pertes attendues"]
    assert indicators_payload["schema_version"] == 7
    assert indicators_payload["tables"][0]["section"] == "risk_management"
    assert indicators_payload["tables"][0]["title"] == "Table risque"
    assert indicators_payload["tables"][0]["indicators"] == ["Pertes attendues"]
    assert footnotes_payload["tables"][0]["footnotes"] == [
        {"id": "1", "text": "Footnote de test"}
    ]
