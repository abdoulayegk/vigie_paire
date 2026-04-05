"""Tests for the GPT Inspector artifact filter in comparison_diff_gpt."""

from __future__ import annotations

from vigilance.comparison_diff_gpt import (
    _inspect_diff_artifacts_gpt,
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
# _inspect_diff_artifacts_gpt — unit tests
# ---------------------------------------------------------------------------


class TestInspectDiffArtifactsGpt:
    def test_skips_call_when_no_adds_removes_or_renames(self):
        """If nothing to inspect, inspector must not be called and diff is returned as-is."""
        calls: list[str] = []

        def fake_call_openai_json(**kwargs):
            calls.append(kwargs["call_kind"])
            return {}

        diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "No changes.",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["A"]),
            _table(table_id="curr", indicators=["A"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        assert calls == []  # No GPT call
        assert result is diff  # Same object returned

    def test_inspects_renames_and_filters_artifacts(self):
        """Inspector marks a rename as artifact → rename is removed from output."""

        def fake_call_openai_json(**kwargs):
            return {
                "added_verdicts": [],
                "removed_verdicts": [],
                "renamed_verdicts": [
                    {
                        "previous": "A",
                        "current": "B",
                        "verdict": "artifact",
                        "reason": "hierarchical prefix noise",
                    },
                ],
                "artifact_pairs": [],
            }

        diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [{"previous": "A", "current": "B", "reason": "ok"}],
            "reason": "Renamed.",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["A"]),
            _table(table_id="curr", indicators=["B"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        assert result["indicators_renamed"] == []

    def test_filters_artifact_added_and_removed(self):
        """Inspector marks both an added and a removed indicator as artifacts → both removed."""

        def fake_call_openai_json(**kwargs):
            return {
                "added_verdicts": [
                    {
                        "value": "Total (bloc 2)",
                        "verdict": "artifact",
                        "reason": "disambiguation noise",
                    },
                ],
                "removed_verdicts": [
                    {
                        "value": "31 octobre 2025 – Total",
                        "verdict": "artifact",
                        "reason": "disambiguation noise",
                    },
                ],
                "artifact_pairs": [
                    {
                        "removed": "31 octobre 2025 – Total",
                        "added": "Total (bloc 2)",
                        "reason": "same row",
                    },
                ],
            }

        diff = {
            "indicators_added": [{"value": "Total (bloc 2)", "reason": "New."}],
            "indicators_removed": [{"value": "31 octobre 2025 – Total", "reason": "Gone."}],
            "indicators_renamed": [],
            "reason": "Changes detected.",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["31 octobre 2025 – Total"]),
            _table(table_id="curr", indicators=["Total (bloc 2)"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        assert result["indicators_added"] == []
        assert result["indicators_removed"] == []
        # Reason and renames are preserved
        assert result["reason"] == "Changes detected."
        assert result["indicators_renamed"] == []

    def test_keeps_real_changes(self):
        """Inspector marks a real added indicator → it stays in the diff."""

        def fake_call_openai_json(**kwargs):
            return {
                "added_verdicts": [
                    {
                        "value": "Ratio de levier",
                        "verdict": "real",
                        "reason": "Genuinely new row.",
                    },
                ],
                "removed_verdicts": [],
                "artifact_pairs": [],
            }

        diff = {
            "indicators_added": [
                {
                    "value": "Ratio de levier",
                    "reason": "New.",
                    "analyst_assessment": {
                        "relevance_level": 1,
                        "justification": "New regulatory ratio.",
                    },
                },
            ],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "New indicator.",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["Ratio CET1"]),
            _table(table_id="curr", indicators=["Ratio CET1", "Ratio de levier"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        assert len(result["indicators_added"]) == 1
        assert result["indicators_added"][0]["value"] == "Ratio de levier"

    def test_mixed_real_and_artifact(self):
        """Inspector filters one artifact and keeps one real indicator."""

        def fake_call_openai_json(**kwargs):
            return {
                "added_verdicts": [
                    {
                        "value": "Goodwill³",
                        "verdict": "artifact",
                        "reason": "footnote marker variance",
                    },
                    {
                        "value": "Ratio AT1",
                        "verdict": "real",
                        "reason": "genuinely new",
                    },
                ],
                "removed_verdicts": [
                    {
                        "value": "Goodwill",
                        "verdict": "artifact",
                        "reason": "footnote marker variance",
                    },
                ],
                "artifact_pairs": [
                    {
                        "removed": "Goodwill",
                        "added": "Goodwill³",
                        "reason": "same indicator",
                    },
                ],
            }

        diff = {
            "indicators_added": [
                {"value": "Goodwill³", "reason": "New."},
                {"value": "Ratio AT1", "reason": "New regulatory."},
            ],
            "indicators_removed": [
                {"value": "Goodwill", "reason": "Removed."},
            ],
            "indicators_renamed": [],
            "reason": "Changes.",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["Goodwill", "Ratio CET1"]),
            _table(table_id="curr", indicators=["Goodwill³", "Ratio CET1", "Ratio AT1"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        assert len(result["indicators_added"]) == 1
        assert result["indicators_added"][0]["value"] == "Ratio AT1"
        assert result["indicators_removed"] == []

    def test_handles_malformed_inspector_response_gracefully(self):
        """If inspector returns garbage, original diff passes through unchanged."""

        def fake_call_openai_json(**kwargs):
            return {"unexpected_field": "oops"}

        diff = {
            "indicators_added": [{"value": "X", "reason": "New."}],
            "indicators_removed": [{"value": "Y", "reason": "Gone."}],
            "indicators_renamed": [],
            "reason": "",
        }
        result = _inspect_diff_artifacts_gpt(
            diff,
            _table(table_id="prev", indicators=["Y"]),
            _table(table_id="curr", indicators=["X"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )
        # No filtering happened — original items preserved
        assert len(result["indicators_added"]) == 1
        assert len(result["indicators_removed"]) == 1


# ---------------------------------------------------------------------------
# Integration: diff_table_pair_gpt with Inspector
# ---------------------------------------------------------------------------


class TestDiffTablePairGptWithInspector:
    def test_inspector_called_when_diff_has_adds(self):
        """Inspector is called as 3rd call when diff_indicators returns adds."""
        call_kinds: list[str] = []
        responses = [
            # 1st call: diff_indicators
            {
                "indicators_added": [{"value": "Total (bloc 2)", "reason": "New row."}],
                "indicators_removed": [{"value": "31 oct – Total", "reason": "Gone."}],
                "indicators_renamed": [],
                "reason": "Disambiguation variance.",
            },
            # 2nd call: diff_footnotes
            {
                "footnotes_added": [],
                "footnotes_removed": [],
                "footnotes_renamed": [],
                "reason": "",
            },
            # 3rd call: inspector
            {
                "added_verdicts": [
                    {
                        "value": "Total (bloc 2)",
                        "verdict": "artifact",
                        "reason": "same row",
                    },
                ],
                "removed_verdicts": [
                    {
                        "value": "31 oct – Total",
                        "verdict": "artifact",
                        "reason": "same row",
                    },
                ],
                "artifact_pairs": [
                    {
                        "removed": "31 oct – Total",
                        "added": "Total (bloc 2)",
                        "reason": "extraction noise",
                    },
                ],
            },
        ]

        def fake_call_openai_json(**kwargs):
            call_kinds.append(kwargs["call_kind"])
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(
                table_id="prev",
                indicators=["Ratio CET1", "31 oct – Total"],
                footnotes=[{"id": "1", "text": "Note A"}],
            ),
            _table(
                table_id="curr",
                indicators=["Ratio CET1", "Total (bloc 2)"],
                footnotes=[{"id": "1", "text": "Note A"}],
            ),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )

        assert call_kinds == ["diff_indicators", "diff_footnotes", "inspect_artifacts"]
        assert result["diff_calls_total"] == 3
        assert result["technical_diff"]["indicators_added"] == []
        assert result["technical_diff"]["indicators_removed"] == []
        assert result["technical_diff"]["table_level_change"] == "inchange"

    def test_inspector_not_called_when_no_adds_or_removes(self):
        """Inspector is NOT called when diff_indicators has only renames or nothing."""
        call_kinds: list[str] = []
        responses = [
            {
                "indicators_added": [],
                "indicators_removed": [],
                "indicators_renamed": [],
                "reason": "Aucun changement.",
            },
        ]

        def fake_call_openai_json(**kwargs):
            call_kinds.append(kwargs["call_kind"])
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(table_id="prev", indicators=["Ratio CET1"]),
            _table(table_id="curr", indicators=["Ratio CET1"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )

        assert call_kinds == ["diff_indicators"]
        assert result["diff_calls_total"] == 1
        assert result["technical_diff"]["table_level_change"] == "inchange"

    def test_unicode_apostrophe_artifact_filtered(self):
        """Unicode apostrophe artifact (d\u2019imp\u00f4t vs d'imp\u00f4t) is caught by inspector."""
        call_kinds: list[str] = []
        responses = [
            {
                "indicators_added": [{"value": "Cr\u00e9dit d\u2019imp\u00f4t", "reason": "New."}],
                "indicators_removed": [{"value": "Cr\u00e9dit d'imp\u00f4t", "reason": "Gone."}],
                "indicators_renamed": [],
                "reason": "Change detected.",
            },
            {
                "added_verdicts": [
                    {
                        "value": "Cr\u00e9dit d\u2019imp\u00f4t",
                        "verdict": "artifact",
                        "reason": "unicode apostrophe",
                    },
                ],
                "removed_verdicts": [
                    {
                        "value": "Cr\u00e9dit d'imp\u00f4t",
                        "verdict": "artifact",
                        "reason": "unicode apostrophe",
                    },
                ],
                "artifact_pairs": [
                    {
                        "removed": "Cr\u00e9dit d'imp\u00f4t",
                        "added": "Cr\u00e9dit d\u2019imp\u00f4t",
                        "reason": "same",
                    },
                ],
            },
        ]

        def fake_call_openai_json(**kwargs):
            call_kinds.append(kwargs["call_kind"])
            return responses.pop(0)

        result = diff_table_pair_gpt(
            _table(table_id="prev", indicators=["Cr\u00e9dit d'imp\u00f4t"]),
            _table(table_id="curr", indicators=["Cr\u00e9dit d\u2019imp\u00f4t"]),
            model="gpt-4o-test",
            call_openai_json=fake_call_openai_json,
        )

        assert call_kinds == ["diff_indicators", "inspect_artifacts"]
        assert result["diff_calls_total"] == 2
        assert result["technical_diff"]["indicators_added"] == []
        assert result["technical_diff"]["indicators_removed"] == []
        assert result["technical_diff"]["table_level_change"] == "inchange"
