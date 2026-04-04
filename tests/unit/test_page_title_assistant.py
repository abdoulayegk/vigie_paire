"""Tests for page_title_assistant module and DoclingProcessor integration."""

from __future__ import annotations

import pytest

from vigilance.extraction.page_title_assistant import (
    PageTitleResponse,
    PageTitleResult,
    _parse_page_title_response,
)

# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestPageTitleResponseSchema:
    """Pydantic schema rejects unexpected fields and validates structure."""

    def test_valid_response(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": "1",
                    "title_full": "Tableau 1 - Bilan consolidé",
                    "title_semantic": "Bilan consolidé",
                    "bbox_title": [0.05, 0.10, 0.90, 0.13],
                    "confidence": 0.95,
                }
            ]
        }
        resp = PageTitleResponse.model_validate(data)
        assert len(resp.page_table_titles) == 1
        assert resp.page_table_titles[0].table_number == "1"
        assert resp.page_table_titles[0].confidence == 0.95

    def test_rejects_extra_fields(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": "1",
                    "title_full": "Tableau 1",
                    "title_semantic": "T1",
                    "confidence": 0.9,
                    "unexpected_field": "bad",
                }
            ]
        }
        with pytest.raises(Exception):
            PageTitleResponse.model_validate(data)

    def test_empty_candidates(self):
        data = {"page_table_titles": []}
        resp = PageTitleResponse.model_validate(data)
        assert resp.page_table_titles == []

    def test_malformed_json_string(self):
        result = _parse_page_title_response("{invalid json")
        assert result is None

    def test_coerces_none_to_empty_string(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": None,
                    "title_full": "Test",
                    "title_semantic": "Test",
                    "confidence": 0.8,
                }
            ]
        }
        resp = PageTitleResponse.model_validate(data)
        assert resp.page_table_titles[0].table_number == ""

    def test_invalid_bbox_ignored(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": "2",
                    "title_full": "Tableau 2",
                    "title_semantic": "T2",
                    "bbox_title": [0.1, 0.2],  # Only 2 elements
                    "confidence": 0.85,
                }
            ]
        }
        resp = PageTitleResponse.model_validate(data)
        assert resp.page_table_titles[0].bbox_title is None


# ---------------------------------------------------------------------------
# 2. Candidate-to-table assignment
# ---------------------------------------------------------------------------


class TestPageTitleResult:
    def _make_result(self, candidates: list[dict]) -> PageTitleResult:
        return PageTitleResult(page_number=1, candidates=candidates)

    def test_get_candidate_by_number_found(self):
        result = self._make_result(
            [
                {"table_number": "1", "title_semantic": "First", "confidence": 0.9},
                {"table_number": "2", "title_semantic": "Second", "confidence": 0.85},
            ]
        )
        c = result.get_candidate_by_number("2")
        assert c is not None
        assert c["title_semantic"] == "Second"

    def test_get_candidate_by_number_not_found(self):
        result = self._make_result(
            [
                {"table_number": "1", "title_semantic": "First", "confidence": 0.9},
            ]
        )
        c = result.get_candidate_by_number("99")
        assert c is None

    def test_get_candidate_by_number_empty(self):
        result = self._make_result([])
        assert result.get_candidate_by_number("1") is None

    def test_get_candidate_by_number_empty_string(self):
        result = self._make_result(
            [
                {"table_number": "1", "title_semantic": "X", "confidence": 0.9},
            ]
        )
        assert result.get_candidate_by_number("") is None

    def test_bbox_proximity_picks_closest_above(self):
        result = self._make_result(
            [
                {
                    "table_number": "1",
                    "title_semantic": "Far title",
                    "bbox_title": [0.05, 0.05, 0.90, 0.08],
                    "confidence": 0.9,
                },
                {
                    "table_number": "2",
                    "title_semantic": "Close title",
                    "bbox_title": [0.05, 0.18, 0.90, 0.20],
                    "confidence": 0.9,
                },
            ]
        )
        # Table starts at y=0.22
        table_bbox = [0.05, 0.22, 0.95, 0.60]
        c = result.get_candidate_by_bbox_proximity(table_bbox)
        assert c is not None
        assert c["title_semantic"] == "Close title"

    def test_bbox_proximity_ignores_below_table(self):
        result = self._make_result(
            [
                {
                    "table_number": "3",
                    "title_semantic": "Below",
                    "bbox_title": [0.05, 0.65, 0.90, 0.68],
                    "confidence": 0.9,
                },
            ]
        )
        table_bbox = [0.05, 0.22, 0.95, 0.60]
        c = result.get_candidate_by_bbox_proximity(table_bbox)
        assert c is None

    def test_bbox_proximity_no_bbox(self):
        result = self._make_result(
            [
                {"table_number": "1", "title_semantic": "No bbox", "confidence": 0.9},
            ]
        )
        assert result.get_candidate_by_bbox_proximity([0.1, 0.2, 0.9, 0.8]) is None

    def test_bbox_proximity_empty_table_bbox(self):
        result = self._make_result(
            [
                {
                    "table_number": "1",
                    "title_semantic": "T",
                    "bbox_title": [0.1, 0.1, 0.9, 0.15],
                    "confidence": 0.9,
                },
            ]
        )
        assert result.get_candidate_by_bbox_proximity([]) is None


# ---------------------------------------------------------------------------
# 3. Parse response
# ---------------------------------------------------------------------------


class TestParsePageTitleResponse:
    def test_parse_valid_json_string(self):
        raw = '{"page_table_titles": [{"table_number": "5", "title_full": "Tableau 5 - Actifs", "title_semantic": "Actifs", "confidence": 0.92}]}'
        result = _parse_page_title_response(raw)
        assert result is not None
        assert len(result.candidates) == 1
        assert result.candidates[0]["table_number"] == "5"

    def test_parse_dict(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": "3",
                    "title_full": "Table 3",
                    "title_semantic": "T3",
                    "confidence": 0.88,
                }
            ]
        }
        result = _parse_page_title_response(data)
        assert result is not None
        assert len(result.candidates) == 1

    def test_parse_filters_empty_titles(self):
        data = {
            "page_table_titles": [
                {
                    "table_number": "1",
                    "title_full": "",
                    "title_semantic": "",
                    "confidence": 0.5,
                },
                {
                    "table_number": "2",
                    "title_full": "Real",
                    "title_semantic": "Real",
                    "confidence": 0.9,
                },
            ]
        }
        result = _parse_page_title_response(data)
        assert result is not None
        assert len(result.candidates) == 1
        assert result.candidates[0]["table_number"] == "2"

    def test_parse_markdown_fenced(self):
        raw = '```json\n{"page_table_titles": [{"table_number":"1","title_full":"T1","title_semantic":"T1","confidence":0.9}]}\n```'
        result = _parse_page_title_response(raw)
        assert result is not None
        assert len(result.candidates) == 1

    def test_parse_non_dict(self):
        assert _parse_page_title_response("[1,2,3]") is None

    def test_parse_none_string(self):
        assert _parse_page_title_response("null") is None


# ---------------------------------------------------------------------------
# 4. Existing valid Vision title is NOT overwritten
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="DoclingProcessor._apply_page_title_candidates absent; logique de titre refondue."
)
class TestFallbackPolicy:
    """Verify that good existing titles are preserved (simulated via _apply_page_title_candidates)."""

    def test_good_title_not_overwritten(self):
        """A table with a high-quality title should not be changed."""
        from unittest.mock import MagicMock

        from vigilance.extraction.docling_processor import DoclingProcessor

        proc = DoclingProcessor.__new__(DoclingProcessor)
        proc.bank_code_for_patterns = "td"
        proc.extraction_patterns = None

        table = MagicMock()
        table.title = "Tableau 28 - Bilan consolidé condensé intermédiaire"
        table.title_clean = "Bilan consolidé condensé intermédiaire"
        table.title_raw = "Tableau 28 - Bilan consolidé condensé intermédiaire"
        table.table_number = "28"
        table.table_id = "tableau_0"
        table.page_number = 5
        table.bbox = [0.05, 0.20, 0.95, 0.80]
        table.title_resolution_method = "inline_numbered"

        result = PageTitleResult(
            page_number=5,
            candidates=[
                {
                    "table_number": "28",
                    "title_full": "Tableau 28 - Different title",
                    "title_semantic": "Different title",
                    "bbox_title": [0.05, 0.15, 0.90, 0.18],
                    "confidence": 0.95,
                }
            ],
        )

        # With a high threshold (3), the good title (score > 3) won't be replaced
        proc._apply_page_title_candidates([table], result, weak_title_threshold=3)
        assert table.title == "Tableau 28 - Bilan consolidé condensé intermédiaire"

    def test_empty_title_gets_filled(self):
        """A table with an empty title should get filled from candidate."""
        from unittest.mock import MagicMock

        from vigilance.extraction.docling_processor import DoclingProcessor

        proc = DoclingProcessor.__new__(DoclingProcessor)
        proc.bank_code_for_patterns = "td"
        proc.extraction_patterns = None

        table = MagicMock()
        table.title = ""
        table.title_clean = None
        table.title_raw = None
        table.table_number = "5"
        table.table_id = "tableau_1"
        table.page_number = 3
        table.bbox = [0.05, 0.30, 0.95, 0.80]
        table.title_resolution_method = None

        result = PageTitleResult(
            page_number=3,
            candidates=[
                {
                    "table_number": "5",
                    "title_full": "Tableau 5 - Actifs pondérés en fonction des risques",
                    "title_semantic": "Actifs pondérés en fonction des risques",
                    "bbox_title": [0.05, 0.25, 0.90, 0.28],
                    "confidence": 0.92,
                }
            ],
        )

        proc._apply_page_title_candidates([table], result, weak_title_threshold=3)
        assert table.title == "Actifs pondérés en fonction des risques"
        assert "page_level_assist" in table.title_resolution_method
