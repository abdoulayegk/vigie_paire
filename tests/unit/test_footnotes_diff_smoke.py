"""Smoke tests for footnote comparison wiring in comparison_runner."""

from __future__ import annotations

import pytest

from vigilance.compare.footnote_comparator import FootnoteComparator, compare_footnotes
from vigilance.utils.footnotes_utils import footnotes_list_to_dict
from app.comparison_runner import _compare_table_footnotes
from vigilance.models.table_models import TableArtifact


def _make_artifact(
    footnotes: list[str] | list[dict[str, str]] | None = None, **kwargs
) -> TableArtifact:
    defaults = dict(
        bank_code="bnc",
        section="gestion_capital",
        page_pdf=1,
        table_id="T1",
        title="Test Table",
        headers=["Col1", "Col2"],
        rows=[["a", "1"]],
        first_column_indicators=["Indicator A"],
        first_column_indicators_raw=["Indicator A"],
        extraction_method="vision_full_gpt4o",
        footnotes=footnotes,
        content_source="vision_gpt4o",
    )
    defaults.update(kwargs)
    return TableArtifact(**defaults)


class TestNormalizeFootnotesToCanonical:
    def test_empty_or_none(self) -> None:
        from vigilance.utils.footnotes_utils import normalize_footnotes_to_canonical

        assert normalize_footnotes_to_canonical(None) == []
        assert normalize_footnotes_to_canonical([]) == []

    def test_mixed_types(self) -> None:
        from vigilance.utils.footnotes_utils import normalize_footnotes_to_canonical

        raw = ["Plain string.", {"id": "2", "text": "Dict note."}]
        out = normalize_footnotes_to_canonical(raw)
        assert out == [
            {"id": "1", "text": "Plain string."},
            {"id": "2", "text": "Dict note."},
        ]


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

    def test_marker_text_items(self) -> None:
        items = [
            {"marker": "(1)", "text": "Texte parenthetique"},
            {"marker": "*", "text": "Texte etoile"},
        ]
        result = footnotes_list_to_dict(items)
        assert result == {"(1)": "Texte parenthetique", "*": "Texte etoile"}

    def test_stringified_dict_recovery(self) -> None:
        items = ["{'id': '1', 'text': 'Recovered text'}"]
        result = footnotes_list_to_dict(items)
        assert result == {"1": "Recovered text"}

    def test_stringified_dict_in_value_recovery(self) -> None:
        items = [
            {"id": "1", "value": "{'id': '1', 'text': 'Nested recovered text'}"},
            {"marker": "2", "value": "Direct value text"},
        ]
        result = footnotes_list_to_dict(items)
        assert result == {
            "1": "Nested recovered text",
            "2": "Direct value text",
        }


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
            first_column_indicators_raw=["Ind A"],
            section="gestion_capital",
            page_number=5,
            table_id="T42",
            title="My Table",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=["Note 1", "Note 2"],
            content_source="vision_gpt4o",
        )
        art = _table_to_artifact(fake_table, bank_code="bnc", quarter="t1", pdf_path="/tmp/test.pdf")
        assert art.footnotes == [
            {"id": "1", "text": "Note 1"},
            {"id": "2", "text": "Note 2"},
        ]

    def test_none_when_missing(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact

        fake_table = SimpleNamespace(
            rows=[], headers=[], first_column_indicators=[], first_column_indicators_raw=[],
            section="", page_number=1, table_id="T1", title=None,
            extraction_method="vision_full_gpt4o", table_number=None, bbox=None,
            content_source="vision_gpt4o",
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


class TestFootnoteNormalizationDoclingLegacy:
    """Regression: Docling-only list[str] footnotes work through pipeline."""

    def test_docling_list_str_normalized_to_canonical(self) -> None:
        from vigilance.utils.footnotes_utils import normalize_footnotes_to_canonical

        raw = ["Methodology note.", "Basel III compliance."]
        out = normalize_footnotes_to_canonical(raw)
        assert out == [
            {"id": "1", "text": "Methodology note."},
            {"id": "2", "text": "Basel III compliance."},
        ]

    def test_docling_through_table_to_artifact_and_comparator(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _compare_table_footnotes, _table_to_artifact

        fake = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1", "Col2"],
            first_column_indicators=["Ind"],
            first_column_indicators_raw=["Ind"],
            section="",
            page_number=1,
            table_id="T1",
            title="Table",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=["Docling footnote text here."],
            content_source="vision_gpt4o",
        )
        art = _table_to_artifact(fake, bank_code="bnc", quarter="t1", pdf_path="/tmp/x.pdf")
        fn_dict = footnotes_list_to_dict(art.footnotes or [])
        assert fn_dict == {"1": "Docling footnote text here."}
        assert "Docling footnote text here." in fn_dict.values()
        assert not any("dict" in str(v) or "'" in str(v) for v in fn_dict.values())

    def test_docling_through_writer_produces_real_text(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact
        from vigilance.extraction.vision_extraction_writer import write_footnotes_json

        fake = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1"],
            first_column_indicators=["Ind"],
            first_column_indicators_raw=["Ind"],
            section="",
            page_number=42,
            table_id="TABLEAU 39",
            title="RATIO DE LIQUIDITE",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=["LCR calcule conformement aux normes BSIF."],
            content_source="vision_gpt4o",
        )
        art = _table_to_artifact(fake, bank_code="rbc", quarter="t1", pdf_path="/tmp/t1.pdf")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_footnotes_json([art], [], out_dir, "rbc", "test_run")
            data = __import__("json").loads((out_dir / "footnotes.json").read_text())
        entry = data["tables"][0]
        assert entry["footnotes_content"]["1"] == "LCR calcule conformement aux normes BSIF."
        assert "dict" not in entry["footnotes_content"]["1"]
        assert "{" not in entry["footnotes_content"]["1"]


class TestFootnoteNormalizationVisionPrimary:
    """Regression: Vision-primary list[dict] footnotes preserved through pipeline."""

    def test_vision_list_dict_preserved_not_stringified(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact

        fake = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1"],
            first_column_indicators=["Ind"],
            first_column_indicators_raw=["Ind"],
            section="",
            page_number=5,
            table_id="T1",
            title="Table",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=[
                {"id": "1", "text": "Le LCR represente la moyenne des 62 donnees quotidiennes."},
                {"id": "2", "text": "Valeurs non ponderees des entrees et sorties."},
            ],
            content_source="vision_gpt4o",
        )
        art = _table_to_artifact(fake, bank_code="rbc", quarter="t1", pdf_path="/tmp/x.pdf")
        assert art.footnotes == [
            {"id": "1", "text": "Le LCR represente la moyenne des 62 donnees quotidiennes."},
            {"id": "2", "text": "Valeurs non ponderees des entrees et sorties."},
        ]
        fn_dict = footnotes_list_to_dict(art.footnotes or [])
        assert fn_dict["1"] == "Le LCR represente la moyenne des 62 donnees quotidiennes."
        assert fn_dict["2"] == "Valeurs non ponderees des entrees et sorties."
        assert not any("{" in v or "'" in v for v in fn_dict.values())

    def test_vision_through_footnote_comparator_and_writer(self) -> None:
        from types import SimpleNamespace
        from app.comparison_runner import _compare_table_footnotes, _table_to_artifact

        fn_vision = [
            {"id": "1", "text": "Original methodology text."},
            {"id": "2", "text": "Regulatory reference."},
        ]
        fake = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1"],
            first_column_indicators=["Ind"],
            first_column_indicators_raw=["Ind"],
            section="",
            page_number=1,
            table_id="T1",
            title="Table",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=fn_vision,
            content_source="vision_gpt4o",
        )
        t1 = _table_to_artifact(fake, bank_code="bnc", quarter="t1", pdf_path="/tmp/t1.pdf")
        t2_art = _table_to_artifact(
            SimpleNamespace(
                rows=[["a", "1"]],
                headers=["Col1"],
                first_column_indicators=["Ind"],
                first_column_indicators_raw=["Ind"],
                section="",
                page_number=1,
                table_id="T2",
                title="Table",
                extraction_method="vision_full_gpt4o",
                table_number=None,
                bbox=None,
                footnotes=[
                    {"id": "1", "text": "Revised methodology text."},
                    {"id": "2", "text": "Regulatory reference."},
                ],
                content_source="vision_gpt4o",
            ),
            bank_code="bnc",
            quarter="t2",
            pdf_path="/tmp/t2.pdf",
        )
        result = _compare_table_footnotes(t1, t2_art)
        assert result["counts"]["modified"] == 1
        mod = result["modified"][0]
        assert mod["old_text"] == "Original methodology text."
        assert mod["new_text"] == "Revised methodology text."
        assert "dict" not in str(mod["old_text"])
        assert "{" not in str(mod["old_text"])

    def test_vision_writer_output_has_real_text(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace
        from app.comparison_runner import _table_to_artifact
        from vigilance.extraction.vision_extraction_writer import write_footnotes_json

        fake = SimpleNamespace(
            rows=[["a", "1"]],
            headers=["Col1"],
            first_column_indicators=["Ind"],
            first_column_indicators_raw=["Ind"],
            section="",
            page_number=42,
            table_id="TABLEAU 39",
            title="RATIO DE LIQUIDITE",
            extraction_method="vision_full_gpt4o",
            table_number=None,
            bbox=None,
            footnotes=[
                {"id": "1", "text": "Le LCR pour le trimestre clos le 31 janvier 2025."},
                {"id": "2", "text": "Valeurs ponderees selon ligne directrice BSIF."},
            ],
            content_source="vision_gpt4o",
        )
        art = _table_to_artifact(fake, bank_code="rbc", quarter="t1", pdf_path="/tmp/t1.pdf")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_footnotes_json([art], [], out_dir, "rbc", "vision_test")
            data = __import__("json").loads((out_dir / "footnotes.json").read_text())
        entry = data["tables"][0]
        assert entry["footnotes_content"]["1"] == "Le LCR pour le trimestre clos le 31 janvier 2025."
        assert entry["footnotes_content"]["2"] == "Valeurs ponderees selon ligne directrice BSIF."
        for v in entry["footnotes_content"].values():
            assert "dict" not in v
            assert "{" not in v

    def test_writer_meta_reports_repr_suspect_count(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace
        from vigilance.extraction.vision_extraction_writer import write_footnotes_json

        table_like = SimpleNamespace(
            table_id="T1",
            title="Table",
            page_number=1,
            footnotes=["{'id': '1', 'text': 'Recovered text'}"],
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_footnotes_json([table_like], [], out_dir, "bnc", "repr_test")
            data = __import__("json").loads((out_dir / "footnotes.json").read_text())
        assert data["meta"]["tables_total"] == 1
        assert data["meta"]["footnote_entries_total"] == 1
        assert data["meta"]["repr_suspect_count"] == 1
        assert data["warnings"][0]["code"] == "repr_suspect_detected"


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
