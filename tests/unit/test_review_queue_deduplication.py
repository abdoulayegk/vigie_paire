"""Unit tests for review queue deduplication and grouping.

Tests cover:
- compute_table_key: stable key generation
- normalize_review_queue: deduplication and grouping
- ChangeItem: serialization and validation status
- ReviewTableItem: summary computation and status
"""

from vigilance.review_models_v2 import (
    ChangeItem,
    ChangeType,
    ReviewTableItem,
    compute_table_key,
    legacy_change_type_to_new,
)
from vigilance.review_queue_normalizer import (
    _change_exists,
    normalize_review_queue,
    sort_review_tables_by_priority,
)


class TestComputeTableKey:
    """Tests for compute_table_key function."""

    def test_key_with_both_table_ids(self):
        """Key should use pair_id when both T1 and T2 IDs present."""
        key = compute_table_key(
            bank_code="rbc",
            section="Credit Risk",
            table_id_t1="tbl_001",
            table_id_t2="tbl_002",
            table_title="Test Table",
        )
        assert key == "rbc::credit risk::tbl_001|tbl_002"

    def test_key_with_only_t2_id(self):
        """Key should work with only T2 ID (for added tables)."""
        key = compute_table_key(
            bank_code="td",
            section="Market Risk",
            table_id_t1="",
            table_id_t2="new_tbl",
            table_title="New Table",
        )
        assert key == "td::market risk::|new_tbl"

    def test_key_with_only_t1_id(self):
        """Key should work with only T1 ID (for removed tables)."""
        key = compute_table_key(
            bank_code="bmo",
            section="Operational Risk",
            table_id_t1="old_tbl",
            table_id_t2="",
            table_title="Old Table",
        )
        assert key == "bmo::operational risk::old_tbl|"

    def test_key_fallback_to_title_hash(self):
        """Key should use title hash when no IDs available."""
        key = compute_table_key(
            bank_code="bnc",
            section="Liquidity",
            table_id_t1="",
            table_id_t2="",
            table_title="Liquidity Coverage Ratio",
        )
        assert key.startswith("bnc::liquidity::title:")
        assert len(key.split("::")[-1]) > 10  # Hash should be present

    def test_key_fallback_includes_page_signature(self):
        """Fallback key should differ for same title/section on different pages."""
        key_p10 = compute_table_key(
            bank_code="bnc",
            section="Liquidity",
            table_id_t1="",
            table_id_t2="",
            table_title="Liquidity Coverage Ratio",
            page_t1=10,
            page_t2=11,
        )
        key_p30 = compute_table_key(
            bank_code="bnc",
            section="Liquidity",
            table_id_t1="",
            table_id_t2="",
            table_title="Liquidity Coverage Ratio",
            page_t1=30,
            page_t2=31,
        )
        assert key_p10 != key_p30

    def test_key_normalizes_section(self):
        """Section should be normalized (lowercase, stripped)."""
        key1 = compute_table_key("rbc", "  Credit Risk  ", "t1", "t2", "Table")
        key2 = compute_table_key("rbc", "credit risk", "t1", "t2", "Table")
        assert key1 == key2

    def test_key_same_for_duplicate_inputs(self):
        """Same inputs should produce same key (deterministic)."""
        key1 = compute_table_key("rbc", "Credit", "t1", "t2", "Table A")
        key2 = compute_table_key("rbc", "Credit", "t1", "t2", "Table A")
        assert key1 == key2


class TestChangeItem:
    """Tests for ChangeItem dataclass."""

    def test_to_dict_and_from_dict(self):
        """ChangeItem should serialize and deserialize correctly."""
        original = ChangeItem(
            change_id="chg_001",
            change_type=ChangeType.INDICATOR_ADDED.value,
            payload={"indicator_name": "Test Indicator"},
            validation_status="pending",
            is_required=True,
        )

        d = original.to_dict()
        restored = ChangeItem.from_dict(d)

        assert restored.change_id == original.change_id
        assert restored.change_type == original.change_type
        assert restored.payload == original.payload
        assert restored.is_required == original.is_required

    def test_is_validated(self):
        """is_validated should return True only for final analyst decisions."""
        pending = ChangeItem("1", "indicator_added", {}, "pending")
        approved = ChangeItem("2", "indicator_added", {}, "approved")
        rejected = ChangeItem("3", "indicator_added", {}, "rejected")
        skipped = ChangeItem("4", "indicator_added", {}, "skipped")

        assert not pending.is_validated()
        assert approved.is_validated()
        assert rejected.is_validated()
        assert not skipped.is_validated()


class TestReviewTableItem:
    """Tests for ReviewTableItem dataclass."""

    def test_compute_summary(self):
        """Summary should count changes by type and status."""
        table = ReviewTableItem(
            table_key="test::section::t1|t2",
            section="Test",
            table_name="Test Table",
            table_number="1",
            table_id_t1="t1",
            table_id_t2="t2",
            page_t1=1,
            page_t2=2,
            changes=[
                ChangeItem("1", "indicator_added", {}, "pending"),
                ChangeItem("2", "indicator_added", {}, "approved"),
                ChangeItem("3", "indicator_removed", {}, "rejected"),
                ChangeItem("4", "footnote_modified", {}, "pending"),
            ],
        )

        summary = table.compute_summary()

        assert summary["total_changes"] == 4
        assert summary["indicators_added"] == 2
        assert summary["indicators_removed"] == 1
        assert summary["footnotes_changed"] == 1
        assert summary["validated"] == 2
        assert summary["pending"] == 2

    def test_is_complete_all_validated(self):
        """is_complete should return True when all required changes are validated."""
        table = ReviewTableItem(
            table_key="test",
            section="Test",
            table_name="Test",
            table_number="1",
            table_id_t1="",
            table_id_t2="",
            page_t1=1,
            page_t2=1,
            changes=[
                ChangeItem("1", "indicator_added", {}, "approved", is_required=True),
                ChangeItem("2", "indicator_removed", {}, "rejected", is_required=True),
                ChangeItem("3", "footnote_added", {}, "pending", is_required=False),
            ],
        )

        # All required changes validated, optional footnote still pending
        assert table.is_complete()

    def test_is_complete_required_pending(self):
        """is_complete should return False when required changes are pending."""
        table = ReviewTableItem(
            table_key="test",
            section="Test",
            table_name="Test",
            table_number="1",
            table_id_t1="",
            table_id_t2="",
            page_t1=1,
            page_t2=1,
            changes=[
                ChangeItem("1", "indicator_added", {}, "pending", is_required=True),
            ],
        )

        assert not table.is_complete()

    def test_update_status(self):
        """update_status should set correct table_status based on changes."""
        # All pending
        table = ReviewTableItem(
            table_key="test",
            section="Test",
            table_name="Test",
            table_number="1",
            table_id_t1="",
            table_id_t2="",
            page_t1=1,
            page_t2=1,
            changes=[
                ChangeItem("1", "indicator_added", {}, "pending"),
            ],
        )
        table.update_status()
        assert table.table_status == "pending"

        # Partial
        table.changes[0].validation_status = "approved"
        table.changes.append(ChangeItem("2", "indicator_removed", {}, "pending"))
        table.update_status()
        assert table.table_status == "partial"

        # Completed
        table.changes[1].validation_status = "approved"
        table.update_status()
        assert table.table_status == "completed"

    def test_to_dict_has_review_id_and_view_mode(self):
        """Serialized table should expose stable review_id and resolved view_mode."""
        table = ReviewTableItem(
            table_key="rbc::risk::t1|t2",
            section="Risk",
            table_name="Table A",
            table_number="1",
            table_id_t1="t1",
            table_id_t2="t2",
            page_t1=1,
            page_t2=2,
            changes=[ChangeItem("c1", "table_added", {"description": "whole table"})],
        )
        d = table.to_dict()
        assert d["review_id"] == "rbc::risk::t1|t2"
        assert d["table_title"] == "Table A"
        assert d["view_mode"] == "table_only"


class TestNormalizeReviewQueue:
    """Tests for normalize_review_queue function."""

    def test_deduplicates_same_table_key(self):
        """Items with same table_key should be merged into one."""
        raw_items = [
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "indicator",
                "indicators": [
                    {"name": "Ind1", "type": "added", "review_status": "pending"},
                ],
            },
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "footnote",
                "indicators": [
                    {
                        "name": "FN1",
                        "type": "added",
                        "footnote_ref": "1",
                        "review_status": "pending",
                    },
                ],
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")

        # Should produce ONE table with TWO changes
        assert len(result) == 1
        assert len(result[0].changes) == 2

    def test_different_tables_not_merged(self):
        """Items with different table_keys should remain separate."""
        raw_items = [
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "indicator",
                "indicators": [
                    {"name": "Ind1", "type": "added"},
                ],
            },
            {
                "section": "Market Risk",
                "table_name": "Table B",
                "table_id_t1": "t3",
                "table_id_t2": "t4",
                "page_t1": 5,
                "page_t2": 6,
                "item_type": "indicator",
                "indicators": [
                    {"name": "Ind2", "type": "removed"},
                ],
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")

        assert len(result) == 2

    def test_filters_out_empty_tables(self):
        """Tables with no changes should be filtered out."""
        raw_items = [
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "indicator",
                "indicators": [],  # No changes
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")

        assert len(result) == 0

    def test_deduplicates_identical_changes(self):
        """Identical changes within same table should be deduplicated."""
        raw_items = [
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "indicator",
                "indicators": [
                    {"name": "Ind1", "name_clean": "ind1", "type": "added"},
                ],
            },
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "page_t1": 1,
                "page_t2": 2,
                "item_type": "indicator",
                "indicators": [
                    {
                        "name": "Ind1",
                        "name_clean": "ind1",
                        "type": "added",
                    },  # Duplicate
                ],
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")

        assert len(result) == 1
        assert len(result[0].changes) == 1  # Only one change, not two

    def test_table_level_precedence_drops_indicator_changes(self):
        """When table_added/table_removed exists, table review is table-only."""
        raw_items = [
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "",
                "table_id_t2": "t2",
                "item_type": "indicator",
                "indicators": [{"name": "Ind1", "name_clean": "ind1", "type": "added"}],
            },
            {
                "section": "Credit Risk",
                "table_name": "Table A",
                "table_id_t1": "",
                "table_id_t2": "t2",
                "change_type": "table_added",
                "item_type": "indicator",
                "indicators": [],
                "indicator": "table entière",
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")
        assert len(result) == 1
        assert len(result[0].changes) == 1
        assert result[0].changes[0].change_type == "table_added"

    def test_sorts_by_section_page_name(self):
        """Result should be sorted by section, page, then name."""
        raw_items = [
            {
                "section": "Market Risk",
                "table_name": "Zebra Table",
                "table_id_t2": "t3",
                "page_t2": 10,
                "indicators": [{"name": "x", "type": "added"}],
            },
            {
                "section": "Credit Risk",
                "table_name": "Alpha Table",
                "table_id_t2": "t1",
                "page_t2": 1,
                "indicators": [{"name": "y", "type": "added"}],
            },
            {
                "section": "Credit Risk",
                "table_name": "Beta Table",
                "table_id_t2": "t2",
                "page_t2": 5,
                "indicators": [{"name": "z", "type": "added"}],
            },
        ]

        result = normalize_review_queue(raw_items, "rbc", "Q3", "Q4")

        # Credit Risk should come before Market Risk
        assert result[0].section == "Credit Risk"
        assert result[1].section == "Credit Risk"
        assert result[2].section == "Market Risk"
        # Within Credit Risk, sorted by page
        assert result[0].page_t2 == 1
        assert result[1].page_t2 == 5


class TestChangeExists:
    """Tests for _change_exists helper."""

    def test_indicator_added_match(self):
        """Should detect duplicate indicator_added by name_clean."""
        existing = [
            ChangeItem("1", "indicator_added", {"indicator_name_clean": "test ind"}),
        ]
        new = ChangeItem("2", "indicator_added", {"indicator_name_clean": "test ind"})

        assert _change_exists(existing, new)

    def test_indicator_added_no_match(self):
        """Should not match different indicators."""
        existing = [
            ChangeItem("1", "indicator_added", {"indicator_name_clean": "ind a"}),
        ]
        new = ChangeItem("2", "indicator_added", {"indicator_name_clean": "ind b"})

        assert not _change_exists(existing, new)

    def test_renamed_match(self):
        """Should detect duplicate renamed by from_clean and to_clean."""
        existing = [
            ChangeItem(
                "1", "indicator_renamed", {"from_clean": "old", "to_clean": "new"}
            ),
        ]
        new = ChangeItem(
            "2", "indicator_renamed", {"from_clean": "old", "to_clean": "new"}
        )

        assert _change_exists(existing, new)

    def test_footnote_match(self):
        """Should detect duplicate footnotes by ref."""
        existing = [
            ChangeItem("1", "footnote_modified", {"footnote_ref": "1"}),
        ]
        new = ChangeItem("2", "footnote_modified", {"footnote_ref": "1"})

        assert _change_exists(existing, new)

    def test_different_types_no_match(self):
        """Different change types should not match."""
        existing = [
            ChangeItem("1", "indicator_added", {"indicator_name_clean": "test"}),
        ]
        new = ChangeItem("2", "indicator_removed", {"indicator_name_clean": "test"})

        assert not _change_exists(existing, new)


class TestSortReviewTablesByPriority:
    """Tests for sort_review_tables_by_priority function."""

    def test_regulatory_first(self):
        """REGLEMENTAIRE relevance should come first."""
        tables = [
            ReviewTableItem(
                table_key="1",
                section="A",
                table_name="Low",
                table_number="1",
                table_id_t1="",
                table_id_t2="",
                page_t1=1,
                page_t2=1,
                relevance="NON_SIGNIFICATIF",
                risk_level="FAIBLE",
            ),
            ReviewTableItem(
                table_key="2",
                section="B",
                table_name="High",
                table_number="2",
                table_id_t1="",
                table_id_t2="",
                page_t1=1,
                page_t2=1,
                relevance="REGLEMENTAIRE",
                risk_level="ELEVE",
            ),
        ]

        sorted_tables = sort_review_tables_by_priority(tables)

        assert sorted_tables[0].table_name == "High"
        assert sorted_tables[1].table_name == "Low"

    def test_more_changes_higher_priority(self):
        """Tables with more changes should rank higher (same relevance)."""
        table_few = ReviewTableItem(
            table_key="1",
            section="A",
            table_name="Few",
            table_number="1",
            table_id_t1="",
            table_id_t2="",
            page_t1=1,
            page_t2=1,
            changes=[ChangeItem("1", "indicator_added", {})],
        )
        table_many = ReviewTableItem(
            table_key="2",
            section="A",
            table_name="Many",
            table_number="2",
            table_id_t1="",
            table_id_t2="",
            page_t1=1,
            page_t2=1,
            changes=[
                ChangeItem("1", "indicator_added", {}),
                ChangeItem("2", "indicator_removed", {}),
                ChangeItem("3", "indicator_renamed", {}),
            ],
        )

        sorted_tables = sort_review_tables_by_priority([table_few, table_many])

        assert sorted_tables[0].table_name == "Many"


class TestLegacyChangeTypeMapping:
    """Tests for legacy_change_type_to_new function."""

    def test_indicator_types(self):
        """Should map indicator types correctly."""
        assert legacy_change_type_to_new("indicator", "added") == "indicator_added"
        assert legacy_change_type_to_new("indicator", "removed") == "indicator_removed"
        assert legacy_change_type_to_new("indicator", "renamed") == "indicator_renamed"

    def test_footnote_types(self):
        """Should map footnote types correctly."""
        assert legacy_change_type_to_new("footnote", "added") == "footnote_added"
        assert legacy_change_type_to_new("footnote", "removed") == "footnote_removed"
        assert legacy_change_type_to_new("footnote", "modified") == "footnote_modified"
