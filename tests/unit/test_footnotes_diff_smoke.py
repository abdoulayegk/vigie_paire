"""Smoke tests for footnote comparison wiring in comparison_runner."""

from __future__ import annotations

import pytest

from vigilance.comparison.footnote_comparator import FootnoteComparator, compare_footnotes
from vigilance.utils.footnotes_utils import footnotes_list_to_dict
from app.comparison_runner import _compare_table_footnotes
from vigilance.models.table_models import TableArtifact


def _make_artifact(footnotes: list[str] | None = None, **kwargs) -> TableArtifact:
    defaults = dict(
        bank_code="bnc",
        section="gestion_capital",
        page_pdf=1,
        table_id="T1",
        title="Test Table",
        headers=["Col1", "Col2"],
        rows=[["a", "1"]],
        first_column_indicators=["Indicator A"],
        extraction_method="docling",
        footnotes=footnotes,
    )
    defaults.update(kwargs)
    return TableArtifact(**defaults)


class TestFootnotesListToDict:
    def test_empty(self) -> None:
        assert footnotes_list_to_dict([]) == {}
        assert footnotes_list_to_dict(None) == {}

    def test_plain_strings(self) -> None:
        result = footnotes_list_to_dict(["First note", "Second note"])
        assert result == {"1": "First note", "2": "Second note"}

    def test_dict_items(self) -> None:
        items = [
            {"id": "a", "text": "Note A"},
            {"ref": "2", "value": "Note B"},
        ]
        result = footnotes_list_to_dict(items)
        assert result == {"a": "Note A", "2": "Note B"}


class TestFootnoteComparator:
    def test_no_changes(self) -> None:
        fn = {"1": "This is a footnote about methodology."}
        changes = compare_footnotes(fn, fn)
        assert len(changes) == 0

    def test_new_footnote(self) -> None:
        fn1: dict[str, str] = {}
        fn2 = {"1": "New regulatory requirement."}
        changes = compare_footnotes(fn1, fn2)
        assert len(changes) == 1
        assert changes[0].change_type == "new_footnote"

    def test_removed_footnote(self) -> None:
        fn1 = {"1": "Old footnote about Basel requirements."}
        fn2: dict[str, str] = {}
        changes = compare_footnotes(fn1, fn2)
        assert len(changes) == 1
        assert changes[0].change_type == "removed_footnote"

    def test_modified_footnote(self) -> None:
        fn1 = {"1": "Calculated using old methodology."}
        fn2 = {"1": "Calculated using new BSIF guidelines."}
        changes = compare_footnotes(fn1, fn2)
        assert len(changes) == 1
        assert changes[0].change_type == "modified_footnote"

    def test_classification_regulatory(self) -> None:
        fn1: dict[str, str] = {}
        fn2 = {"1": "Conforme aux normes du BSIF et exigences reglementaires."}
        changes = compare_footnotes(fn1, fn2)
        assert changes[0].category == "REGULATORY"
        assert changes[0].significance in ("MAJOR", "MODERATE")


class TestCompareTableFootnotes:
    def test_no_footnotes(self) -> None:
        t1 = _make_artifact(footnotes=None)
        t2 = _make_artifact(footnotes=None)
        result = _compare_table_footnotes(t1, t2)
        assert result["added"] == []
        assert result["removed"] == []
        assert result["modified"] == []
        assert result["counts"] == {"added": 0, "removed": 0, "modified": 0}

    def test_empty_footnotes(self) -> None:
        t1 = _make_artifact(footnotes=[])
        t2 = _make_artifact(footnotes=[])
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"]["added"] == 0

    def test_added_footnote(self) -> None:
        t1 = _make_artifact(footnotes=[])
        t2 = _make_artifact(footnotes=["New disclosure requirement."])
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"]["added"] == 1
        assert len(result["added"]) == 1
        assert result["added"][0]["change_type"] == "new_footnote"

    def test_removed_footnote(self) -> None:
        t1 = _make_artifact(footnotes=["Old note about risk."])
        t2 = _make_artifact(footnotes=[])
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"]["removed"] == 1

    def test_modified_footnote_same_ref(self) -> None:
        t1 = _make_artifact(footnotes=["Original methodology description."])
        t2 = _make_artifact(footnotes=["Completely revised new approach."])
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"]["modified"] == 1

    def test_identical_footnotes_no_diff(self) -> None:
        fn = ["This is a regulatory footnote."]
        t1 = _make_artifact(footnotes=fn)
        t2 = _make_artifact(footnotes=fn)
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"] == {"added": 0, "removed": 0, "modified": 0}


class TestTableArtifactFootnotes:
    def test_footnotes_field_exists(self) -> None:
        t = _make_artifact(footnotes=["Note 1"])
        assert t.footnotes == ["Note 1"]

    def test_footnotes_default_none(self) -> None:
        t = _make_artifact()
        assert t.footnotes is None

    def test_to_dict_includes_footnotes(self) -> None:
        t = _make_artifact(footnotes=["Note X"])
        d = t.to_dict()
        assert d["footnotes"] == ["Note X"]


class TestTableToArtifactCopiesFootnotes:
    """Verify _table_to_artifact propagates footnotes from ExtractedTable-like objects."""

    def test_copies_footnotes(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact

        fake_table = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1", "Col2"],
            first_column_indicators=["Ind A"],
            first_column_indicators_raw=None,
            section="gestion_capital",
            page_number=5,
            table_id="T42",
            title="My Table",
            extraction_method="docling",
            table_number=None,
            bbox=None,
            footnotes=["Note 1", "Note 2"],
        )
        art = _table_to_artifact(fake_table, bank_code="bnc", quarter="t1", pdf_path="/tmp/test.pdf")
        assert art.footnotes == ["Note 1", "Note 2"]

    def test_none_when_missing(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact

        fake_table = SimpleNamespace(
            rows=[], headers=[], first_column_indicators=[], first_column_indicators_raw=None,
            section="", page_number=1, table_id="T1", title=None,
            extraction_method="docling", table_number=None, bbox=None,
        )
        art = _table_to_artifact(fake_table, bank_code="bnc", quarter="t1", pdf_path="/tmp/test.pdf")
        assert art.footnotes is None


class TestFootnotesDiffInPayload:
    """Verify the payload shape produced by _compare_table_footnotes and review adapter."""

    def test_enabled_produces_footnotes_diff(self) -> None:
        t1 = _make_artifact(footnotes=["Old note."], table_id="T1")
        t2 = _make_artifact(footnotes=["Brand new note."], table_id="T2")
        result = _compare_table_footnotes(t1, t2)
        assert "added" in result
        assert "removed" in result
        assert "modified" in result
        assert "counts" in result
        total = result["counts"]["added"] + result["counts"]["removed"] + result["counts"]["modified"]
        assert total > 0

    def test_disabled_produces_empty(self) -> None:
        t1 = _make_artifact(footnotes=None, table_id="T1")
        t2 = _make_artifact(footnotes=None, table_id="T2")
        result = _compare_table_footnotes(t1, t2)
        assert result["counts"] == {"added": 0, "removed": 0, "modified": 0}


class TestReviewAdapterFootnotes:
    """Verify build_review_items_from_indicator_result produces footnote items."""

    def test_footnote_review_items(self) -> None:
        from app.review_adapters import build_review_items_from_indicator_result

        payload = {
            "table_comparisons": [
                {
                    "table_id_t1": "T1",
                    "table_id_t2": "T2",
                    "title_t1": "Ratios",
                    "title_t2": "Ratios",
                    "page_t1": 10,
                    "page_t2": 11,
                    "section": "gestion_capital",
                    "match_score": 0.9,
                    "added_indicators": [],
                    "removed_indicators": [],
                    "renamed_indicators": [],
                    "table_status": "stable",
                    "footnotes_diff": {
                        "added": [{"change_type": "new_footnote", "footnote_ref": "3", "new_text": "New", "old_text": None}],
                        "removed": [],
                        "modified": [],
                        "counts": {"added": 1, "removed": 0, "modified": 0},
                    },
                    "footnotes_counts": {"added": 1, "removed": 0, "modified": 0},
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        }
        items = build_review_items_from_indicator_result(
            payload, bank_code="bnc", quarter_from="t1", quarter_to="t2",
            pdf_path_t1="/tmp/t1.pdf", pdf_path_t2="/tmp/t2.pdf",
        )
        fn_items = [i for i in items if i.item_type == "footnote"]
        assert len(fn_items) == 1
        assert fn_items[0].change_type == "footnote"
        assert len(fn_items[0].indicators) == 1
        assert fn_items[0].footnote_changes[0]["change_type"] == "new_footnote"

    def test_no_footnote_items_when_no_diff(self) -> None:
        from app.review_adapters import build_review_items_from_indicator_result

        payload = {
            "table_comparisons": [
                {
                    "table_id_t1": "T1", "table_id_t2": "T2",
                    "added_indicators": ["X"], "removed_indicators": [], "renamed_indicators": [],
                    "table_status": "modifie", "section": "gestion_capital",
                    "match_score": 0.8,
                }
            ],
            "tables_added": [],
            "tables_removed": [],
        }
        items = build_review_items_from_indicator_result(
            payload, bank_code="bnc", quarter_from="t1", quarter_to="t2",
            pdf_path_t1="/tmp/t1.pdf", pdf_path_t2="/tmp/t2.pdf",
        )
        fn_items = [i for i in items if i.item_type == "footnote"]
        assert len(fn_items) == 0
