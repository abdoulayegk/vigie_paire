"""Regression tests for the stale comparison_blockers bug.

The bug: TableArtifact.__post_init__ used to union old blockers with
inferred blockers, so stale blockers from storage reload or fragment
merge could permanently mark a valid Vision table as ineligible.

The fix: __post_init__ now recomputes blockers purely from current state
(content_source, first_column_indicators_raw, footnotes).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vigilance.models.table_models import TableArtifact


def _vision_table(**overrides) -> TableArtifact:
    """Helper: build a valid Vision table with sane defaults."""
    defaults = dict(
        bank_code="bnc",
        section="capital_management",
        page_pdf=1,
        table_id="t1",
        title="Fonds propres",
        headers=["Indicateur", "Valeur"],
        rows=[["CET1", "13.6%"]],
        first_column_indicators=["cet1"],
        first_column_indicators_raw=["CET1"],
        extraction_method="vision_full_gpt4o",
        footnotes=[{"id": "1", "text": "Note"}],
        content_source="vision_gpt4o",
    )
    defaults.update(overrides)
    return TableArtifact(**defaults)


# ---------- Test 1: Core fix ----------


def test_stale_blockers_cleared_on_valid_vision_table():
    """A Vision table with valid indicators must be eligible even when
    stale blockers are injected (e.g. from storage reload or merge)."""
    art = _vision_table(
        comparison_blockers=["non_vision_content_source", "missing_vision_indicators"],
        comparison_eligible=False,
    )
    assert art.comparison_eligible is True
    assert "non_vision_content_source" not in art.comparison_blockers
    assert "missing_vision_indicators" not in art.comparison_blockers


# ---------- Test 2: Storage roundtrip ----------


def test_storage_roundtrip_does_not_perpetuate_stale_blockers():
    """Loading a stored extraction with stale blockers must yield
    an eligible table if the source fields are valid."""
    from app.extraction_storage import load_extraction

    base_dir = Path(tempfile.mkdtemp())
    try:
        # Manually write a tables.json with stale blockers
        target = base_dir / "bnc" / "2025" / "t1"
        target.mkdir(parents=True)
        stored = {
            "schema_version": 3,
            "tables": [
                {
                    "bank_code": "bnc",
                    "section": "capital_management",
                    "page_pdf": 22,
                    "table_id": "tableau_0",
                    "title": "Ratios des fonds propres",
                    "headers": ["Indicateur", "Valeur"],
                    "rows": [["CET1", "13.6%"]],
                    "first_column_indicators": ["cet1"],
                    "first_column_indicators_raw": ["CET1"],
                    "extraction_method": "vision_full_gpt4o",
                    "footnotes": [{"id": "1", "text": "Note"}],
                    "content_source": "vision_gpt4o",
                    # Stale values from a previous failed run:
                    "comparison_eligible": False,
                    "comparison_blockers": ["missing_vision_indicators"],
                }
            ],
            "bank_code": "bnc",
            "year": 2025,
            "quarter": "t1",
        }
        (target / "tables.json").write_text(
            json.dumps(stored, ensure_ascii=False), encoding="utf-8"
        )
        (target / "meta.json").write_text(
            json.dumps(
                {"schema_version": 3, "bank_code": "bnc", "year": 2025, "quarter": "t1"}
            ),
            encoding="utf-8",
        )

        result = load_extraction("bnc", 2025, "t1", base_dir)
        assert result is not None
        tables, _ = result
        assert len(tables) == 1
        t = tables[0]
        assert t.comparison_eligible is True
        assert "missing_vision_indicators" not in t.comparison_blockers
    finally:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


# ---------- Test 3: Fragment merge ----------


def test_fragment_merge_does_not_propagate_stale_blockers():
    """Merging fragments where one had a stale blocker must not
    infect the merged result if the merged data is valid."""
    from vigilance.compare.table_fragment_merger import merge_table_fragments

    frag_a = _vision_table(
        table_id="frag_a",
        page_pdf=22,
        first_column_indicators=["cet1", "tier 1"],
        first_column_indicators_raw=["CET1", "Tier 1"],
    )
    frag_b = _vision_table(
        table_id="frag_b",
        page_pdf=23,
        first_column_indicators=["ratio tlac"],
        first_column_indicators_raw=["Ratio TLAC"],
        # Simulate stale blocker injected before the fix:
        comparison_blockers=["missing_vision_indicators"],
    )
    # After __post_init__ fix, frag_b should already be clean,
    # but test the merge path end-to-end anyway.
    merged, events = merge_table_fragments([frag_a, frag_b], merge_score_min=0.0)
    # With merge_score_min=0.0 they should merge (same section, consecutive pages)
    for t in merged:
        assert t.comparison_eligible is True
        assert "missing_vision_indicators" not in t.comparison_blockers


# ---------- Test 4: Non-regression — docling stays blocked ----------


def test_non_vision_table_remains_blocked():
    """A docling table without Vision indicators must stay ineligible."""
    art = TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=1,
        table_id="t1",
        title="Legacy",
        headers=["Col"],
        rows=[["X", "1"]],
        first_column_indicators=["x"],
        extraction_method="docling",
        footnotes=[],
    )
    assert art.comparison_eligible is False
    assert "non_vision_content_source" in art.comparison_blockers


# ---------- Test 5: Non-regression — Vision without indicators stays blocked ----------


def test_vision_without_indicators_remains_blocked():
    """A Vision table with empty raw indicators must stay ineligible."""
    art = _vision_table(
        first_column_indicators=[],
        first_column_indicators_raw=[],
    )
    assert art.comparison_eligible is False
    assert "missing_vision_indicators" in art.comparison_blockers


# ---------- Test 6: footnotes_unavailable is non-fatal ----------


def test_footnotes_unavailable_does_not_block_eligibility():
    """A Vision table with indicators but no footnotes must still be eligible.
    footnotes_unavailable is a blocker but NOT a fatal one."""
    art = _vision_table(footnotes=None)
    assert art.comparison_eligible is True
    assert "footnotes_unavailable" in art.comparison_blockers
