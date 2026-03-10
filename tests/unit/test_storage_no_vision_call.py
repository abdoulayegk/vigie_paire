"""Step 4 contract test: when stored extraction exists, _extract_tables returns it
without calling extraction (Vision is never re-invoked for stored data).

This validates the extraction/comparison decoupling requirement.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _make_tables_json(
    bank_code: str, year: int, quarter: str, tables_json: list
) -> Path:
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
            }
        ),
        encoding="utf-8",
    )
    meta_path = target / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "bank_code": bank_code,
                "year": year,
                "quarter": quarter,
                "schema_version": 2,
                "extraction_method": "vision_full_gpt4o",
            }
        ),
        encoding="utf-8",
    )
    return base


def test_load_extraction_returns_stored_tables() -> None:
    """load_extraction must return previously saved tables without calling Vision."""
    from app.extraction_storage import load_extraction

    stored_table = {
        "bank_code": "bnc",
        "section": "capital",
        "page_pdf": 3,
        "table_id": "tableau_0",
        "title": "Tableau 1",
        "headers": ["Période", "T1 2025"],
        "rows": [["Ratio CET1", "13.1%"]],
        "first_column_indicators": ["ratio cet1"],
        "first_column_indicators_raw": ["Ratio CET1"],
        "extraction_method": "vision_full_gpt4o",
        "footnotes": [{"marker": "1", "text": "Note provisoire"}],
        "debug_metrics": {"vision_status": "ok"},
        "quarter": "t1",
        "content_source": "vision_gpt4o",
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
        assert meta.get("schema_version") == 2
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_extraction_returns_none_when_missing() -> None:
    """load_extraction must return None when no stored extraction exists."""
    from app.extraction_storage import load_extraction

    base_dir = Path(tempfile.mkdtemp())
    try:
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is None
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_table_artifact_from_dict_normalizes_indicators() -> None:
    """Raw indicators with footnote markers are normalized on load (matches fresh Vision path)."""
    from app.extraction_storage import table_artifact_from_dict
    from vigilance.models.table_models import get_comparison_indicators

    d = {
        "bank_code": "bnc",
        "section": "capital",
        "page_pdf": 1,
        "table_id": "t1",
        "title": "Fonds propres",
        "headers": ["Indicateur", "Valeur"],
        "rows": [["Total des fonds propres *", "100"], ["CET1 ratio (1)", "13"]],
        "first_column_indicators": ["Total des fonds propres *", "CET1 ratio (1)"],
        "first_column_indicators_raw": ["Total des fonds propres *", "CET1 ratio (1)"],
        "extraction_method": "docling",
        "footnotes": [],
    }
    artifact = table_artifact_from_dict(d)
    indicators = get_comparison_indicators(artifact)
    assert "total des fonds propre" in indicators
    assert "cet1 ratio" in indicators
    assert "*" not in str(indicators)
    assert "(1)" not in str(indicators)


def test_save_then_load_roundtrip() -> None:
    """save_extraction then load_extraction must return identical data."""
    from app.extraction_storage import load_extraction, save_extraction
    from vigilance.models.table_models import TableArtifact

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
            quarter="t1",
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
        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is not None
        tables, meta = result
        assert len(tables) == 1
        assert tables[0].table_id == "tableau_0"
        assert tables[0].title == "Tableau 1"
        assert meta.get("schema_version") == 2
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_then_load_roundtrip_preserves_rbc_matching_fields() -> None:
    from app.extraction_storage import load_extraction, save_extraction
    from vigilance.models.table_models import TableArtifact

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
        assert loaded.first_column_groups == ["Prets"]
        assert loaded.hierarchical_indicator_signature == ["Prets > Prets de detail"]
        assert loaded.title_reliability == "reliable"
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_writes_schema_version_in_tables_json_root() -> None:
    """After save_extraction, tables.json root must contain schema_version and tables."""
    from app.extraction_storage import save_extraction
    from vigilance.models.table_models import TableArtifact

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
        assert raw.get("schema_version") == 3
        assert "tables" in raw
        assert len(raw["tables"]) == 1
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_normalizes_footnotes_to_id_text_in_file() -> None:
    """Saved tables.json must store footnotes with canonical keys id/text, not marker."""
    from app.extraction_storage import save_extraction
    from vigilance.models.table_models import TableArtifact

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


def test_load_extraction_accepts_legacy_tables_json_without_schema_version() -> None:
    """load_extraction must load tables.json that has no schema_version at root (legacy)."""
    from app.extraction_storage import load_extraction

    base = Path(tempfile.mkdtemp())
    target = base / "bnc" / "2025" / "t1"
    target.mkdir(parents=True)
    tables_path = target / "tables.json"
    # Legacy: no schema_version at root; footnotes as dict
    tables_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "bank_code": "bnc",
                        "section": "capital",
                        "page_pdf": 2,
                        "table_id": "tableau_0",
                        "title": "Legacy table",
                        "headers": ["A"],
                        "rows": [["x"]],
                        "first_column_indicators": [],
                        "extraction_method": "docling",
                        "footnotes": {"1": "Note one", "2": "Note two"},
                    }
                ],
                "bank_code": "bnc",
                "year": 2025,
                "quarter": "t1",
            }
        ),
        encoding="utf-8",
    )
    meta_path = target / "meta.json"
    meta_path.write_text(
        json.dumps({"bank_code": "bnc", "year": 2025, "quarter": "t1"}),
        encoding="utf-8",
    )
    try:
        result = load_extraction("bnc", 2025, "t1", base)
        assert result is not None
        tables, meta = result
        assert len(tables) == 1
        assert tables[0].title == "Legacy table"
        assert tables[0].table_id == "tableau_0"
        assert tables[0].footnotes is not None
        assert len(tables[0].footnotes) == 2
        # Legacy dict is normalized to list with id/text
        assert tables[0].footnotes[0]["id"] == "1"
        assert tables[0].footnotes[0]["text"] == "Note one"
        assert tables[0].footnotes[1]["id"] == "2"
        assert tables[0].footnotes[1]["text"] == "Note two"
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)
