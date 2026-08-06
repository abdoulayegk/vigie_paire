"""Tests for compute_changed_tables_t1 / compute_changed_tables_t2.

Scenario:
  - 1 matched-changed table (indicator diff)
  - 1 matched-unchanged table (stable)
  - 1 tables_removed entry
  - 2 tables_added entries
  - 1 fusion/split matched pair that shares a T1 id with tables_removed
    (to verify de-duplication)
"""

from __future__ import annotations

import pytest

from vigie.comparaison.canonical import (
    _is_comparison_changed,
    compute_changed_tables_t1,
    compute_changed_tables_t2,
    is_canonical_comparison,
    is_ui_comparison_payload,
    new_empty_ui_comparison_payload,
    to_canonical_payload,
    to_ui_comparison_payload,
)


def _make_comparison(
    tid_t1: str,
    tid_t2: str,
    *,
    status: str = "stable",
    added: list[str] | None = None,
    removed: list[str] | None = None,
    renamed: list[dict] | None = None,
    footnotes_counts: dict | None = None,
    structure_change: bool = False,
) -> dict:
    return {
        "table_id_t1": tid_t1,
        "table_id_t2": tid_t2,
        "table_status": status,
        "added_indicators": added or [],
        "removed_indicators": removed or [],
        "renamed_indicators": renamed or [],
        "structure_change_detected": structure_change,
        "footnotes_counts": footnotes_counts,
    }


def _make_result(
    comparisons: list[dict],
    tables_added: list[dict] | None = None,
    tables_removed: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "comparison_canonical_v1",
        "table_comparisons": comparisons,
        "tables_added": tables_added or [],
        "tables_removed": tables_removed or [],
    }


class TestIsComparisonChanged:
    def test_stable_is_not_changed(self):
        c = _make_comparison("a", "b", status="stable")
        assert not _is_comparison_changed(c)

    def test_modifie_is_changed(self):
        c = _make_comparison("a", "b", status="modifie", added=["ind1"])
        assert _is_comparison_changed(c)

    def test_structure_change_is_changed(self):
        c = _make_comparison("a", "b", status="structure_change", structure_change=True)
        assert _is_comparison_changed(c)

    def test_footnote_only_change_is_changed(self):
        c = _make_comparison("a", "b", status="stable", footnotes_counts={"added": 1, "removed": 0, "modified": 0})
        assert _is_comparison_changed(c)

    def test_zero_footnotes_stable_is_not_changed(self):
        c = _make_comparison("a", "b", status="stable", footnotes_counts={"added": 0, "removed": 0, "modified": 0})
        assert not _is_comparison_changed(c)


class TestComputeChangedTablesBasic:
    """Simple scenario with no overlapping IDs."""

    def test_counts_with_no_changes(self):
        result = _make_result(
            [_make_comparison("t1_a", "t2_a", status="stable")],
        )
        assert compute_changed_tables_t1(result) == 0
        assert compute_changed_tables_t2(result) == 0

    def test_counts_with_all_changed(self):
        result = _make_result(
            [_make_comparison("t1_a", "t2_a", status="modifie", added=["x"])],
            tables_added=[{"table_id": "t2_new"}],
            tables_removed=[{"table_id": "t1_old"}],
        )
        assert compute_changed_tables_t1(result) == 2  # t1_a + t1_old
        assert compute_changed_tables_t2(result) == 2  # t2_a + t2_new

    def test_empty_result(self):
        result = _make_result([])
        assert compute_changed_tables_t1(result) == 0
        assert compute_changed_tables_t2(result) == 0


class TestComputeChangedTablesDeduplication:
    """Scenario from the spec with overlapping IDs to verify de-duplication."""

    @pytest.fixture
    def complex_result(self) -> dict:
        comparisons = [
            _make_comparison("t1_1", "t2_1", status="modifie", added=["ind_new"]),
            _make_comparison("t1_2", "t2_2", status="stable"),
            _make_comparison(
                "t1_dup",
                "t2_split_a",
                status="structure_change",
                structure_change=True,
            ),
        ]
        tables_added = [
            {"table_id": "t2_added_1"},
            {"table_id": "t2_added_2"},
        ]
        tables_removed = [
            {"table_id": "t1_dup"},
        ]
        return _make_result(comparisons, tables_added, tables_removed)

    def test_t1_deduplicates_overlap(self, complex_result):
        # t1_1 (modifie) + t1_dup (structure_change AND in tables_removed) = 2
        assert compute_changed_tables_t1(complex_result) == 2

    def test_t2_no_overlap(self, complex_result):
        # t2_1 (modifie) + t2_split_a (structure_change) + t2_added_1 + t2_added_2 = 4
        assert compute_changed_tables_t2(complex_result) == 4

    def test_stable_excluded(self, complex_result):
        t1_changed = set()
        for c in complex_result["table_comparisons"]:
            if _is_comparison_changed(c):
                t1_changed.add(c["table_id_t1"])
        assert "t1_2" not in t1_changed


class TestComputeChangedTablesFootnotesOnly:
    """Footnote-only changes should count as changed."""

    def test_footnote_change_counted(self):
        result = _make_result(
            [
                _make_comparison(
                    "t1_fn",
                    "t2_fn",
                    status="stable",
                    footnotes_counts={"added": 0, "removed": 2, "modified": 0},
                ),
            ]
        )
        assert compute_changed_tables_t1(result) == 1
        assert compute_changed_tables_t2(result) == 1


class TestComputeChangedTablesMissingFields:
    """Graceful handling of missing or None fields."""

    def test_missing_table_comparisons(self):
        result = {"tables_added": [], "tables_removed": []}
        assert compute_changed_tables_t1(result) == 0
        assert compute_changed_tables_t2(result) == 0

    def test_none_table_id_skipped(self):
        result = _make_result(
            [_make_comparison(None, None, status="modifie", added=["x"])],  # type: ignore[arg-type]
        )
        assert compute_changed_tables_t1(result) == 0
        assert compute_changed_tables_t2(result) == 0


class TestUiPayloadAliases:
    def test_new_ui_payload_aliases_match_legacy_behavior(self):
        payload = {"bank_code": "bnc"}

        legacy = to_canonical_payload(payload)
        explicit = to_ui_comparison_payload(payload)
        empty_payload = new_empty_ui_comparison_payload()

        assert legacy == explicit
        assert is_canonical_comparison(explicit)
        assert is_ui_comparison_payload(explicit)
        assert empty_payload["schema_version"] == "comparison_canonical_v1"
