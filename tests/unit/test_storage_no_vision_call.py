"""Step 4 contract test: when stored extraction exists, _extract_tables returns it
without calling extraction (Vision is never re-invoked for stored data).

This validates the extraction/comparison decoupling requirement.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _make_tables_json(bank_code: str, year: int, quarter: str, tables_json: list) -> Path:
    """Create a tables.json in a temp dir matching the extraction_storage directory layout."""
    base = Path(tempfile.mkdtemp())
    target = base / bank_code / str(year) / quarter
    target.mkdir(parents=True)
    tables_path = target / "tables.json"
    tables_path.write_text(
        json.dumps(
            {
                "tables": tables_json,
                "bank_code": bank_code,
                "year": year,
                "quarter": quarter,
                "created_at": "2026-03-24T10:00:00",
                "schema_version": 7,
            }
        ),
        encoding="utf-8",
    )
    return base


def test_load_extraction_returns_stored_tables() -> None:
    """load_extraction must return previously saved tables without calling Vision."""
    from vigie.extraction.extraction_storage import load_extraction

    stored_table = {
        "bank_code": "bnc",
        "section": "capital",
        "page": 3,
        "table_id": "tableau_0",
        "title": "Tableau 1",
        "table_summary": "Capital réglementaire",
        "bbox": None,
        "headers": ["Période", "T1 2025"],
        "row_count": 1,
        "indicators": ["Ratio CET1"],
        "footnotes": [{"id": "1", "text": "Note provisoire"}],
        "extraction_status": "rescued",
    }

    base_dir = _make_tables_json("bnc", 2025, "t1", [stored_table])
    try:
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is not None
        tables, meta = result
        assert len(tables) == 1
        t = tables[0]
        assert t.table_id == "tableau_0"
        assert t.title == "Tableau 1"
        assert t.extraction_status == "rescued"
        assert meta.get("schema_version") == 7
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_extraction_returns_none_when_missing() -> None:
    """load_extraction must return None when no stored extraction exists."""
    from vigie.extraction.extraction_storage import load_extraction

    base_dir = Path(tempfile.mkdtemp())
    try:
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is None
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_extraction_returns_none_when_tables_empty() -> None:
    """load_extraction must return None when tables.json exists but tables list is empty.

    Empty stored extraction is treated as no valid cache so the caller runs fresh
    extraction (avoids Q1=0 when a previous run persisted 0 tables for t1).
    """
    from vigie.extraction.extraction_storage import load_extraction

    base_dir = _make_tables_json("bnc", 2025, "t1", [])
    try:
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is None
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_table_artifact_from_dict_normalizes_indicators() -> None:
    """Raw indicators with footnote markers are normalized on load (matches fresh Vision path)."""
    from vigie.extraction.extraction_storage import table_artifact_from_dict
    from vigie.support.models.table_models import get_comparison_indicators

    d = {
        "bank_code": "bnc",
        "section": "capital",
        "page": 1,
        "table_id": "t1",
        "title": "Fonds propres",
        "table_summary": "Fonds propres",
        "headers": ["Indicateur", "Valeur"],
        "row_count": 2,
        "indicators": ["Total des fonds propres *", "CET1 ratio (1)"],
        "extraction_method": "vision_minimal",
        "footnotes": [],
    }
    artifact = table_artifact_from_dict(d)
    indicators = get_comparison_indicators(artifact)
    assert "fonds propre" in indicators
    assert "cet1 ratio" in indicators
    assert "*" not in str(indicators)
    assert "(1)" not in str(indicators)


def test_table_artifact_from_dict_uses_raw_pipeline_when_raw_present() -> None:
    """When first_column_indicators_raw is present and non-empty, build indicators from raw (normalize + post_normalize)."""
    from vigie.extraction.extraction_storage import table_artifact_from_dict
    from vigie.support.utils.indicator_cleaner import (
        normalize_indicator_for_comparison,
        post_normalize_indicator,
    )

    raw_list = ["Ratio CET1", "Tier 1", "Total des fonds propres (1)"]
    expected = []
    for ind in raw_list:
        fixed, _, _ = post_normalize_indicator(normalize_indicator_for_comparison(ind))
        if fixed and normalize_indicator_for_comparison(fixed):
            expected.append(fixed)
    d = {
        "bank_code": "bmo",
        "section": "capital",
        "page": 1,
        "table_id": "t1",
        "title": "Capital",
        "table_summary": "Capital",
        "headers": ["Indicateur", "Valeur"],
        "row_count": len(raw_list),
        "indicators": raw_list,
        "extraction_method": "vision_full_gpt4o",
        "footnotes": [],
    }
    artifact = table_artifact_from_dict(d)
    assert artifact.first_column_indicators == expected


def test_save_then_load_roundtrip() -> None:
    """save_extraction then load_extraction must return identical data."""
    from vigie.extraction.extraction_storage import load_extraction, save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="bnc",
            section="capital",
            page_pdf=3,
            table_id="tableau_0",
            title="Tableau 1",
            headers=["Période"],
            rows=[["Ratio CET1", "13.1%"]],
            first_column_indicators=["ratio cet1"],
            first_column_indicators_raw=["Ratio CET1"],
            extraction_method="vision_full_gpt4o",
            footnotes=[{"marker": "1", "text": "Note"}],
            debug_metrics={"vision_status": "ok"},
            table_summary="Capital réglementaire",
            quarter="t1",
            content_source="vision_gpt4o",
            extraction_status="rescued",
        )
        save_extraction(
            bank_code="bnc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={"schema_version": 2},
            base_dir=base_dir,
        )
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is not None
        tables, meta = result
        assert len(tables) == 1
        assert tables[0].table_id == "tableau_0"
        assert tables[0].title == "Tableau 1"
        assert tables[0].table_summary == "Capital réglementaire"
        assert tables[0].extraction_status == "rescued"
        assert meta.get("schema_version") == 7
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_then_load_roundtrip_preserves_rbc_matching_fields() -> None:
    from vigie.extraction.extraction_storage import load_extraction, save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="rbc",
            section="risk_management",
            page_pdf=31,
            table_id="tableau_31",
            title="Lien entre le risque de marche et les principales donnees figurant au bilan",
            headers=["Montant figurant au bilan"],
            rows=[["Prets de detail", "1"]],
            first_column_indicators=["pret de detail"],
            first_column_indicators_raw=["Prets de detail"],
            first_column_groups=["Prets"],
            hierarchical_indicator_signature=["Prets > Prets de detail"],
            title_reliability="reliable",
            table_summary="Risque de marché au bilan",
            extraction_method="vision_full_gpt4o",
            footnotes=[],
            content_source="vision_gpt4o",
        )
        save_extraction(
            bank_code="rbc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={"schema_version": 2},
            base_dir=base_dir,
        )
        result = load_extraction("rbc", 2025, "t1", base_dir)
        assert result is not None
        tables, _ = result
        loaded = tables[0]
        assert loaded.first_column_groups is None
        assert loaded.hierarchical_indicator_signature is None
        assert loaded.title_reliability is None
        assert loaded.table_summary == "Risque de marché au bilan"
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_writes_schema_version_in_tables_json_root() -> None:
    """After save_extraction, tables.json root must contain schema_version and tables."""
    from vigie.extraction.extraction_storage import save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="bnc",
            section="capital",
            page_pdf=1,
            table_id="tableau_0",
            title="T1",
            headers=[],
            rows=[],
            first_column_indicators=[],
            first_column_indicators_raw=[],
            extraction_method="vision_full_gpt4o",
            footnotes=[],
            content_source="vision_gpt4o",
        )
        save_extraction(
            bank_code="bnc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={},
            base_dir=base_dir,
        )
        tables_path = base_dir / "bnc" / "2025" / "t1" / "tables.json"
        raw = json.loads(tables_path.read_text(encoding="utf-8"))
        assert raw.get("schema_version") == 7
        assert "tables" in raw
        assert len(raw["tables"]) == 1
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_extraction_writes_unified_json_and_txt_artifacts() -> None:
    """save_extraction writes the official JSON contract only."""
    from vigie.extraction.extraction_storage import save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="bnc",
            section="capital",
            page_pdf=1,
            table_id="tableau_0",
            title="T1",
            headers=["Indicateur", "Valeur"],
            rows=[["Ratio CET1", "13.1"]],
            first_column_indicators=["ratio cet1"],
            first_column_indicators_raw=["Ratio CET1"],
            extraction_method="vision_full_gpt4o",
            footnotes=[{"marker": "1", "text": "Note provisoire"}],
            table_summary="Capital réglementaire",
            content_source="vision_gpt4o",
        )
        save_extraction(
            bank_code="bnc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={"schema_version": 2},
            base_dir=base_dir,
        )
        target = base_dir / "bnc" / "2025" / "t1"
        indicators = json.loads((target / "indicators.json").read_text(encoding="utf-8"))
        footnotes = json.loads((target / "footnotes.json").read_text(encoding="utf-8"))
        tables = json.loads((target / "tables.json").read_text(encoding="utf-8"))
        assert indicators["quarter"] == "t1"
        assert indicators["schema_version"] == 7
        assert indicators["tables"][0]["section"] == "capital"
        assert indicators["tables"][0]["title"] == "T1"
        assert indicators["tables"][0]["indicators"] == ["Ratio CET1"]
        assert footnotes["tables"][0]["footnotes"] == [{"id": "1", "text": "Note provisoire"}]
        assert "title" not in footnotes["tables"][0]
        assert tables["tables"][0]["table_summary"] == "Capital réglementaire"
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_normalizes_footnotes_to_id_text_in_file() -> None:
    """Saved tables.json must store footnotes with canonical keys id/text, not marker."""
    from vigie.extraction.extraction_storage import save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="bnc",
            section="capital",
            page_pdf=1,
            table_id="tableau_0",
            title="T1",
            headers=[],
            rows=[],
            first_column_indicators=[],
            first_column_indicators_raw=[],
            extraction_method="vision_full_gpt4o",
            footnotes=[{"marker": "1", "text": "Note provisoire"}],
            table_summary="Capital réglementaire",
            content_source="vision_gpt4o",
        )
        save_extraction(
            bank_code="bnc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={},
            base_dir=base_dir,
        )
        tables_path = base_dir / "bnc" / "2025" / "t1" / "tables.json"
        raw = json.loads(tables_path.read_text(encoding="utf-8"))
        footnotes = raw["tables"][0].get("footnotes") or []
        assert len(footnotes) == 1
        fn = footnotes[0]
        assert "id" in fn
        assert fn["id"] == "1"
        assert "text" in fn
        assert fn["text"] == "Note provisoire"
        assert "marker" not in fn
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_extraction_reads_minimal_tables_json() -> None:
    """load_extraction must rebuild TableArtifact from the official minimal tables.json."""
    from vigie.extraction.extraction_storage import load_extraction

    base = Path(tempfile.mkdtemp())
    target = base / "bnc" / "2025" / "t1"
    target.mkdir(parents=True)
    (target / "tables.json").write_text(
        json.dumps(
            {
                "schema_version": 7,
                "created_at": "2026-03-22T10:00:00",
                "bank_code": "bnc",
                "year": 2025,
                "quarter": "t1",
                "tables": [
                    {
                        "table_id": "tableau_1",
                        "page": 7,
                        "section": "capital_management",
                        "title": "Capital",
                        "table_summary": "Capital",
                        "bbox": None,
                        "row_count": 1,
                        "headers": ["Indicateur", "Valeur"],
                        "indicators": ["Ratio CET1"],
                        "footnotes": [{"id": "1", "text": "Note A"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        result = load_extraction("bnc", 2025, "t1", base)
        assert result is not None
        tables, _meta = result
        assert len(tables) == 1
        assert tables[0].table_id == "tableau_1"
        assert tables[0].page_pdf == 7
        assert tables[0].section == "capital_management"
        assert tables[0].first_column_indicators == ["ratio cet1"]
        assert tables[0].first_column_indicators_raw == ["Ratio CET1"]
        assert tables[0].table_summary == "Capital"
        assert tables[0].footnotes == [{"id": "1", "text": "Note A"}]
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


def test_load_extraction_recreates_missing_derived_json_artifacts() -> None:
    """load_extraction regenerates missing derived JSON artifacts locally."""
    from vigie.extraction.extraction_storage import load_extraction, save_extraction
    from vigie.support.models.table_models import TableArtifact

    base_dir = Path(tempfile.mkdtemp())
    try:
        artifact = TableArtifact(
            bank_code="bnc",
            section="capital",
            page_pdf=1,
            table_id="tableau_0",
            title="T1",
            headers=["Indicateur", "Valeur"],
            rows=[["Ratio CET1", "13.1"]],
            first_column_indicators=["ratio cet1"],
            first_column_indicators_raw=["Ratio CET1"],
            extraction_method="vision_full_gpt4o",
            footnotes=[{"marker": "1", "text": "Note provisoire"}],
            table_summary="Capital réglementaire",
            content_source="vision_gpt4o",
        )
        save_extraction(
            bank_code="bnc",
            year=2025,
            quarter="t1",
            tables=[artifact],
            meta={"schema_version": 2},
            base_dir=base_dir,
        )
        target = base_dir / "bnc" / "2025" / "t1"
        (target / "indicators.json").unlink()
        (target / "footnotes.json").unlink()

        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is not None
        assert (target / "indicators.json").exists()
        assert (target / "footnotes.json").exists()
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_extraction_rejects_non_current_schema_version() -> None:
    from vigie.extraction.extraction_storage import load_extraction

    base_dir = Path(tempfile.mkdtemp())
    target = base_dir / "bnc" / "2025" / "t1"
    target.mkdir(parents=True)
    (target / "tables.json").write_text(
        json.dumps(
            {
                "bank_code": "bnc",
                "year": 2025,
                "quarter": "t1",
                "created_at": "2026-03-24T10:00:00",
                "schema_version": 6,
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        assert load_extraction("bnc", 2025, "t1", base_dir) is None
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)
