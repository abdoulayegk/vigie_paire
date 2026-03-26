"""Tests for vigie_extract_v1 schema helpers and builder."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from vigilance.report.vigie_extract_schema import (
    SLUG_TO_CANONICAL,
    SCHEMA_VERSION,
    build_vigie_extract,
    canonical_to_slug,
    compute_features,
    load_artifacts_from_vigie_extract,
    make_table_uid,
    normalize_text,
    parse_first_column,
    parse_footnotes,
    section_title_for_slug,
    write_vigie_extract,
)


# ------------------------------------------------------------------
# normalize_text
# ------------------------------------------------------------------


def test_normalize_text_accents_and_case() -> None:
    assert normalize_text("Prêts hypothécaires résidentiels") == "prets hypothecaires residentiels"


def test_normalize_text_special_chars() -> None:
    assert normalize_text("CET-1 (ratio)") == "cet 1 ratio"


def test_normalize_text_empty() -> None:
    assert normalize_text("") == ""


# ------------------------------------------------------------------
# parse_first_column
# ------------------------------------------------------------------


def test_parse_first_column_basic() -> None:
    indicators = [
        "Ratio de capital CET1",
        "Ratio de capital Tier 1 (1)",
        "Ratio de capital total",
    ]
    result = parse_first_column(indicators)
    assert len(result) == 3

    assert result[0]["row_idx"] == 0
    assert result[0]["text"] == "Ratio de capital CET1"
    assert result[0]["text_norm"] == "ratio de capital cet1"
    assert result[0]["note_refs"] == []

    assert result[1]["row_idx"] == 1
    assert result[1]["text"] == "Ratio de capital Tier 1"
    assert result[1]["note_refs"] == ["1"]


def test_parse_first_column_empty() -> None:
    assert parse_first_column([]) == []


# ------------------------------------------------------------------
# parse_footnotes
# ------------------------------------------------------------------


def test_parse_footnotes_standard() -> None:
    raw = ["(1) Calculé selon Bâle III (révisé).", "(2) Inclut les expositions."]
    result = parse_footnotes(raw)
    assert len(result) == 2
    assert result[0]["marker"] == "1"
    assert result[0]["raw_text"] == "(1) Calculé selon Bâle III (révisé)."
    assert result[0]["text"] == "Calculé selon Bâle III (révisé)."
    assert result[0]["scope"] == "table"
    assert result[1]["marker"] == "2"


def test_parse_footnotes_no_marker() -> None:
    result = parse_footnotes(["Some note without a number."])
    assert len(result) == 1
    assert result[0]["marker"] == ""
    assert result[0]["text"] == "Some note without a number."


def test_parse_footnotes_empty_strings_skipped() -> None:
    assert parse_footnotes(["", "  ", "(1) Real note"]) == [parse_footnotes(["(1) Real note"])[0]]


# ------------------------------------------------------------------
# make_table_uid
# ------------------------------------------------------------------


def test_make_table_uid_with_number() -> None:
    uid = make_table_uid("cibc", 2025, "t2-2025", "gestion_capital_fonds_propres", "28", 30, 0)
    assert uid == "cibc_2025_t22025_gestion_capital_fonds_propres_tbl28_p30"


def test_make_table_uid_no_number() -> None:
    uid = make_table_uid("rbc", 2025, "t1-2025", "gestion_risques", None, 40, 3)
    assert uid == "rbc_2025_t12025_gestion_risques_idx3_p40"


# ------------------------------------------------------------------
# compute_features
# ------------------------------------------------------------------


def test_compute_features() -> None:
    first_col = parse_first_column(["Alpha (1)", "Beta", "Gamma"])
    feat = compute_features(first_col)
    assert feat["n_indicators"] == 3
    assert feat["indicator_set_hash"].startswith("sha1:")
    assert "alpha" in feat["anchors"]
    assert "beta" in feat["anchors"]
    assert "gamma" in feat["anchors"]


# ------------------------------------------------------------------
# canonical_to_slug / section_title_for_slug
# ------------------------------------------------------------------


def test_canonical_to_slug_known() -> None:
    assert canonical_to_slug("capital_management") == "gestion_capital_fonds_propres"
    assert canonical_to_slug("risk_management") == "gestion_risques"
    assert canonical_to_slug("regulatory_updates") == "reglementation"


def test_canonical_to_slug_passthrough() -> None:
    assert canonical_to_slug("custom_section") == "custom_section"


def test_section_title_for_slug_with_evidence() -> None:
    assert section_title_for_slug("gestion_risques", "Gestion du risque de crédit") == "Gestion du risque de crédit"


def test_section_title_for_slug_fallback() -> None:
    title = section_title_for_slug("gestion_risques")
    assert title == "Gestion du risque"


# ------------------------------------------------------------------
# build_vigie_extract (integration)
# ------------------------------------------------------------------


@dataclass
class _FakeTable:
    table_id: str = "tableau_0"
    page_number: int = 30
    title: str | None = "Tableau 28 – Ratios de capital"
    table_number: str | None = "28"
    unit_context: str | None = "En pourcentage"
    headers: list[str] = field(default_factory=lambda: ["", "T2 2025", "T1 2025"])
    rows: list[list[str]] = field(default_factory=list)
    first_column_indicators: list[str] = field(
        default_factory=lambda: ["ratio cet1", "ratio tier 1", "ratio total"]
    )
    first_column_indicators_raw: list[str] = field(
        default_factory=lambda: ["Ratio CET1", "Ratio Tier 1 (1)", "Ratio total"]
    )
    footnotes: list[str] = field(
        default_factory=lambda: ["(1) Calculé selon Bâle III."]
    )
    section: str = "capital_management"
    bbox: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.9, 0.8])
    content_source: str = "vision_gpt4o"
    comparison_eligible: bool = True
    comparison_blockers: list[str] = field(default_factory=list)


def test_build_vigie_extract_structure() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake content")
        pdf_path = f.name

    try:
        section_ranges = [
            {
                "section": "capital_management",
                "start": 25,
                "end": 30,
                "evidence": {"title_found": "Gestion des fonds propres"},
            }
        ]
        tables = [_FakeTable()]

        payload = build_vigie_extract(
            pdf_path=pdf_path,
            bank_code="cibc",
            quarter="t2-2025",
            year=2025,
            section_ranges=section_ranges,
            tables=tables,
        )

        assert payload["schema_version"] == SCHEMA_VERSION
        meta = payload["extraction_meta"]
        assert meta["bank_code"] == "cibc"
        assert meta["year"] == 2025
        assert meta["pdf_hash"].startswith("sha256:")

        sections = payload["sections"]
        assert "gestion_capital_fonds_propres" in sections

        sec = sections["gestion_capital_fonds_propres"]
        assert sec["section_title_pdf"] == "Gestion des fonds propres"
        assert sec["start_page"] == 25
        assert sec["end_page"] == 30
        assert len(sec["tables"]) == 1

        tbl = sec["tables"][0]
        assert tbl["table_uid"].startswith("cibc_2025_")
        assert tbl["table_number"] == "28"
        assert len(tbl["first_column"]) == 3
        assert tbl["first_column"][1]["note_refs"] == ["1"]
        assert tbl["content_source"] == "vision_gpt4o"
        assert tbl["comparison_eligible"] is True
        assert len(tbl["footnotes"]) == 1
        assert tbl["footnotes"][0]["marker"] == "1"
        assert tbl["features"]["n_indicators"] == 3
        assert tbl["bbox"] == [0.1, 0.2, 0.9, 0.8]
    finally:
        Path(pdf_path).unlink(missing_ok=True)


# ------------------------------------------------------------------
# write_vigie_extract
# ------------------------------------------------------------------


def test_write_vigie_extract_creates_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        pdf_path = f.name

    try:
        payload = build_vigie_extract(
            pdf_path=pdf_path,
            bank_code="rbc",
            quarter="t1-2025",
            year=2025,
            section_ranges=[{"section": "risk_management", "start": 10, "end": 20}],
            tables=[],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = write_vigie_extract(tmp_dir, payload)
            assert out_path.exists()
            assert out_path.name == "rbc_t1_2025_2025_extract.json"

            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert data["schema_version"] == SCHEMA_VERSION
            assert "gestion_risques" in data["sections"]
    finally:
        Path(pdf_path).unlink(missing_ok=True)


# ------------------------------------------------------------------
# load_artifacts_from_vigie_extract
# ------------------------------------------------------------------


def test_load_roundtrip() -> None:
    """build -> write -> load should produce TableArtifacts with correct fields."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake")
        pdf_path = f.name

    try:
        table = _FakeTable()
        section_ranges = [
            {"section": "capital_management", "start": 25, "end": 30,
             "evidence": {"title_found": "Gestion des fonds propres"}},
        ]
        payload = build_vigie_extract(
            pdf_path=pdf_path,
            bank_code="cibc",
            quarter="t2-2025",
            year=2025,
            section_ranges=section_ranges,
            tables=[table],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = write_vigie_extract(tmp_dir, payload)
            artifacts = load_artifacts_from_vigie_extract(out_path)

        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.bank_code == "cibc"
        assert art.section == "capital_management"
        assert art.page_pdf == 30
        assert art.title == "Tableau 28 – Ratios de capital"
        assert art.table_number == "28"
        assert art.headers == ["", "T2 2025", "T1 2025"]
        assert art.rows == []
        assert len(art.first_column_indicators) == 3
        assert "Ratio CET1" in art.first_column_indicators
        assert art.first_column_indicators_raw == ["Ratio CET1", "Ratio Tier 1", "Ratio total"]
        assert art.bbox == [0.1, 0.2, 0.9, 0.8]
        assert art.quarter == "t2-2025"
        assert art.extraction_method == "docling"
        assert art.content_source == "vision_gpt4o"
        assert art.comparison_eligible is True
    finally:
        Path(pdf_path).unlink(missing_ok=True)


def test_load_empty_sections() -> None:
    """A payload with no tables produces an empty artifact list."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extraction_meta": {"bank_code": "rbc", "quarter": "t1-2025"},
        "sections": {
            "gestion_risques": {
                "section_title_pdf": "Gestion du risque",
                "start_page": 10,
                "end_page": 20,
                "tables": [],
            }
        },
    }
    artifacts = load_artifacts_from_vigie_extract(payload)
    assert artifacts == []


def test_slug_to_canonical_roundtrip() -> None:
    """Every known slug maps back to its canonical key."""
    for canonical, slug in [
        ("capital_management", "gestion_capital_fonds_propres"),
        ("risk_management", "gestion_risques"),
        ("regulatory_updates", "reglementation"),
    ]:
        assert canonical_to_slug(canonical) == slug
        assert SLUG_TO_CANONICAL[slug] == canonical


def test_loaded_artifacts_default_to_ok_status_and_comparison_eligible() -> None:
    """Artifacts loaded from vigie_extract keep the simplified comparison gate."""
    indicators = ["CET1", "Tier 1", "Total Capital", "Leverage Ratio"]
    first_col = [
        {"row_idx": i, "text": t, "text_norm": t.lower(), "note_refs": []}
        for i, t in enumerate(indicators)
    ]
    base_table = {
        "table_uid": "test_uid",
        "table_id": "tableau_0",
        "page_number": 30,
        "table_number": "28",
        "table_title": "Tableau 28 – Ratios de capital",
        "headers": ["", "T2 2025"],
        "first_column": first_col,
        "footnotes": [],
        "features": {"n_indicators": 4, "indicator_set_hash": "sha1:abc", "anchors": []},
        "quality_flags": [],
        "bbox": [0.1, 0.2, 0.9, 0.8],
    }

    payload_t1: dict = {
        "schema_version": SCHEMA_VERSION,
        "extraction_meta": {"bank_code": "rbc", "quarter": "t1-2025",
                            "source_pdf": "t1.pdf", "extraction_method": "docling"},
        "sections": {
            "gestion_capital_fonds_propres": {
                "section_title_pdf": "Gestion des fonds propres",
                "start_page": 25, "end_page": 30,
                "tables": [base_table],
            }
        },
    }
    payload_t2: dict = {
        "schema_version": SCHEMA_VERSION,
        "extraction_meta": {"bank_code": "rbc", "quarter": "t2-2025",
                            "source_pdf": "t2.pdf", "extraction_method": "docling"},
        "sections": {
            "gestion_capital_fonds_propres": {
                "section_title_pdf": "Gestion des fonds propres",
                "start_page": 25, "end_page": 30,
                "tables": [base_table],
            }
        },
    }

    arts_t1 = load_artifacts_from_vigie_extract(payload_t1)
    arts_t2 = load_artifacts_from_vigie_extract(payload_t2)
    assert len(arts_t1) == 1
    assert len(arts_t2) == 1
    assert arts_t1[0].extraction_status == "ok"
    assert arts_t2[0].extraction_status == "ok"
    assert arts_t1[0].comparison_eligible is True
    assert arts_t2[0].comparison_eligible is True
    assert arts_t1[0].comparison_blockers == []
    assert arts_t2[0].comparison_blockers == []
