"""Tests for the deterministic safety net in comparison_diff_gpt."""

from __future__ import annotations

from vigilance.comparison_diff_gpt import (
    _deterministic_footnote_diff,
    _deterministic_indicator_diff,
    _normalize_indicator_text,
    diff_table_pair_gpt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(
    *,
    table_id: str,
    indicators: list[str],
    footnotes: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "table_id": table_id,
        "section": "capital_management",
        "title": "Capital",
        "table_summary": "Ratios de capital",
        "page": 1,
        "row_count": len(indicators),
        "headers": ["Indicateur", "Valeur"],
        "indicators": indicators,
        "footnotes": footnotes or [],
    }


# ---------------------------------------------------------------------------
# _normalize_indicator_text
# ---------------------------------------------------------------------------


class TestNormalizeIndicatorText:
    def test_strips_footnote_markers(self):
        assert _normalize_indicator_text("catégorie 1 (4)") == "catégorie 1"

    def test_strips_superscript_digits(self):
        assert _normalize_indicator_text("CET1¹") == "cet1"

    def test_collapses_whitespace(self):
        assert _normalize_indicator_text("  Ratio   CET1  ") == "ratio cet1"


# ---------------------------------------------------------------------------
# _deterministic_indicator_diff
# ---------------------------------------------------------------------------


class TestDeterministicIndicatorDiff:
    def test_exact_removal_detected(self):
        """Série 9 is absent from current → must appear in det_removed."""
        prev = ["Série 1", "Série 5", "Série 9", "Total"]
        curr = ["Série 1", "Série 5", "Total"]
        result = _deterministic_indicator_diff(prev, curr)
        assert result["det_removed"] == ["Série 9"]
        assert result["det_added"] == []
        assert result["det_renamed"] == []

    def test_exact_addition_detected(self):
        prev = ["Ratio CET1"]
        curr = ["Ratio CET1", "Ratio AT1"]
        result = _deterministic_indicator_diff(prev, curr)
        assert result["det_added"] == ["Ratio AT1"]
        assert result["det_removed"] == []

    def test_footnote_marker_ignored(self):
        """'catégorie 1 (4)' vs 'catégorie 1' should NOT be flagged."""
        prev = ["catégorie 1 (4)"]
        curr = ["catégorie 1"]
        result = _deterministic_indicator_diff(prev, curr)
        assert result["det_removed"] == []
        assert result["det_added"] == []

    def test_no_changes_when_identical(self):
        indicators = ["A", "B", "C"]
        result = _deterministic_indicator_diff(indicators, indicators)
        assert result["det_removed"] == []
        assert result["det_added"] == []
        assert result["det_renamed"] == []

    def test_fuzzy_rename_detected(self):
        prev = ["Ratio de fonds propres CET1"]
        curr = ["Ratio de fonds propres CET1 ajusté"]
        result = _deterministic_indicator_diff(prev, curr, fuzzy_threshold=0.70)
        assert len(result["det_renamed"]) == 1
        assert result["det_renamed"][0]["previous"] == prev[0]
        assert result["det_renamed"][0]["current"] == curr[0]

    def test_ocr_whitespace_tolerance(self):
        """'Ratio CET1' vs 'Ratio  CET1' should be treated as same."""
        prev = ["Ratio CET1"]
        curr = ["Ratio  CET1"]
        result = _deterministic_indicator_diff(prev, curr)
        assert result["det_removed"] == []
        assert result["det_added"] == []


# ---------------------------------------------------------------------------
# _deterministic_footnote_diff
# ---------------------------------------------------------------------------


class TestDeterministicFootnoteDiff:
    def test_added_footnote_detected(self):
        prev = [{"id": "1", "text": "Note A"}]
        curr = [{"id": "1", "text": "Note A"}, {"id": "2", "text": "Note B"}]
        result = _deterministic_footnote_diff(prev, curr)
        assert len(result["det_added"]) == 1
        assert result["det_added"][0]["id"] == "2"

    def test_removed_footnote_detected(self):
        prev = [{"id": "1", "text": "Note A"}, {"id": "2", "text": "Note B"}]
        curr = [{"id": "1", "text": "Note A"}]
        result = _deterministic_footnote_diff(prev, curr)
        assert len(result["det_removed"]) == 1
        assert result["det_removed"][0]["id"] == "2"

    def test_text_change_detected(self):
        prev = [{"id": "1", "text": "Comprennent les engagements de la Banque."}]
        curr = [{"id": "1", "text": "Comprennent aussi les engagements de la Banque."}]
        result = _deterministic_footnote_diff(prev, curr)
        assert len(result["det_modified"]) == 1
        assert result["det_modified"][0]["previous_id"] == "1"

    def test_date_only_change_ignored(self):
        prev = [{"id": "1", "text": "En vigueur depuis le 31 janvier 2025."}]
        curr = [{"id": "1", "text": "En vigueur depuis le 30 avril 2025."}]
        result = _deterministic_footnote_diff(prev, curr)
        assert result["det_modified"] == []

    def test_pure_renumbering_ignored(self):
        """Same text, different ID → pure renumbering, not a real change."""
        prev = [{"id": "7", "text": "Comprennent les engagements."}]
        curr = [{"id": "8", "text": "Comprennent les engagements."}]
        result = _deterministic_footnote_diff(prev, curr)
        assert result["det_added"] == []
        assert result["det_removed"] == []
        assert result["det_modified"] == []

    def test_no_changes(self):
        fns = [{"id": "1", "text": "Note A"}]
        result = _deterministic_footnote_diff(fns, fns)
        assert result["det_added"] == []
        assert result["det_removed"] == []
        assert result["det_modified"] == []


# ---------------------------------------------------------------------------
# Post-diff GPT Inspector integration via diff_table_pair_gpt
# (Replaces the old deterministic guard tests)
# ---------------------------------------------------------------------------


class TestPostGPTGuard:
    def test_inspector_catches_missed_removal_as_real(self):
        """When GPT diff misses Série 9, Inspector confirms it as real → stays."""
        call_kinds: list[str] = []
        responses = [
            # GPT indicator diff: misses Série 9
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement.",
            },
            # Inspector is NOT called because GPT returned 0 adds/removes
        ]

        def fake_call(*, model, messages, usage_recorder=None, call_kind="", **kwargs):
            call_kinds.append(call_kind)
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(
                table_id="prev", indicators=["Série 1", "Série 5", "Série 9", "Total"]
            ),
            _table(table_id="curr", indicators=["Série 1", "Série 5", "Total"]),
            model="gpt-4o-test",
            call_openai_json=fake_call,
        )

        # With the new architecture, the Inspector is only called if GPT found adds/removes.
        # If GPT missed Série 9 entirely (returned no adds/removes), inspector won't be called.
        # This is by design — the hints in the prompt should guide GPT to catch it.
        assert "diff_indicators" in call_kinds
        assert "inspect_artifacts" not in call_kinds

    def test_inspector_filters_artifact_but_keeps_real(self):
        """GPT returns both a real removal and an artifact — Inspector filters the artifact."""
        responses = [
            # indicator diff
            {
                "indicators_added": [{"value": "Goodwill³", "reason": "New."}],
                "indicators_removed": [
                    {"value": "Série 9", "reason": "Removed."},
                    {"value": "Goodwill", "reason": "Removed."},
                ],
                "indicators_renamed": [],
                "reason": "Changes.",
            },
            # inspector
            {
                "added_verdicts": [
                    {
                        "value": "Goodwill³",
                        "verdict": "artifact",
                        "reason": "footnote marker noise",
                    },
                ],
                "removed_verdicts": [
                    {
                        "value": "Série 9",
                        "verdict": "real",
                        "reason": "genuinely removed",
                    },
                    {
                        "value": "Goodwill",
                        "verdict": "artifact",
                        "reason": "footnote marker noise",
                    },
                ],
                "artifact_pairs": [
                    {
                        "removed": "Goodwill",
                        "added": "Goodwill³",
                        "reason": "same indicator",
                    },
                ],
            },
        ]

        def fake_call(*, model, messages, usage_recorder=None, call_kind="", **kwargs):
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(table_id="prev", indicators=["Série 1", "Série 9", "Goodwill"]),
            _table(table_id="curr", indicators=["Série 1", "Goodwill³"]),
            model="gpt-4o-test",
            call_openai_json=fake_call,
        )

        td = result["technical_diff"]
        removed_values = [r["value"] for r in td["indicators_removed"]]
        assert "Série 9" in removed_values
        assert "Goodwill" not in removed_values
        assert td["indicators_added"] == []
        assert td["table_level_change"] == "modifie"

    def test_inspector_not_called_for_pure_renames(self):
        """When GPT only returns renames (no adds/removes), inspector is skipped."""
        call_kinds: list[str] = []
        responses = [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [
                    {"previous": "Old", "current": "New", "reason": "Renamed."}
                ],
                "reason": "Renamed.",
            },
        ]

        def fake_call(*, model, messages, usage_recorder=None, call_kind="", **kwargs):
            call_kinds.append(call_kind)
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(table_id="prev", indicators=["Old"]),
            _table(table_id="curr", indicators=["New"]),
            model="gpt-4o-test",
            call_openai_json=fake_call,
        )

        assert call_kinds == ["diff_indicators"]
        assert result["diff_calls_total"] == 1

    def test_all_artifacts_filtered_flips_to_inchange(self):
        """When inspector filters all artifacts, table flips from modifie to inchange."""
        responses = [
            {
                "indicators_added": [{"value": "C²", "reason": "New."}],
                "indicators_removed": [{"value": "C", "reason": "Gone."}],
                "indicators_renamed": [],
                "reason": "Changes.",
            },
            {
                "added_verdicts": [
                    {"value": "C²", "verdict": "artifact", "reason": "superscript"}
                ],
                "removed_verdicts": [
                    {"value": "C", "verdict": "artifact", "reason": "superscript"}
                ],
                "artifact_pairs": [{"removed": "C", "added": "C²", "reason": "same"}],
            },
        ]

        def fake_call(*, model, messages, usage_recorder=None, call_kind="", **kwargs):
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(table_id="prev", indicators=["A", "B", "C"]),
            _table(table_id="curr", indicators=["A", "B", "C²"]),
            model="gpt-4o-test",
            call_openai_json=fake_call,
        )

        assert result["technical_diff"]["table_level_change"] == "inchange"
        assert result["technical_diff"]["indicators_added"] == []
        assert result["technical_diff"]["indicators_removed"] == []
