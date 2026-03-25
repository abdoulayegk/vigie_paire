from __future__ import annotations

from vigilance.comparison_diff_gpt import diff_table_pair_gpt


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


def test_diff_table_pair_gpt_skips_footnote_call_when_both_tables_have_no_footnotes() -> None:
    call_kinds: list[str] = []
    responses = [
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "Aucun changement.",
        }
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
    assert result["diff_mode"] == "gpt"
    assert result["diff_calls_total"] == 1
    assert result["technical_diff"]["footnotes_added"] == []
    assert result["technical_diff"]["footnotes_removed"] == []
    assert result["technical_diff"]["footnotes_renamed"] == []
    assert result["technical_diff"]["table_level_change"] == "inchange"


def test_diff_table_pair_gpt_structurally_classifies_one_sided_footnotes_without_gpt_call() -> None:
    call_kinds: list[str] = []
    responses = [
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "Aucun changement sur les indicateurs.",
        }
    ]

    def fake_call_openai_json(**kwargs):
        call_kinds.append(kwargs["call_kind"])
        return responses.pop(0)

    result = diff_table_pair_gpt(
        _table(table_id="prev", indicators=["Ratio CET1"], footnotes=[]),
        _table(
            table_id="curr",
            indicators=["Ratio CET1"],
            footnotes=[{"id": "1", "text": "Nouvelle note"}],
        ),
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert call_kinds == ["diff_indicators"]
    assert result["diff_mode"] == "gpt"
    assert result["diff_calls_total"] == 1
    assert result["technical_diff"]["footnotes_added"] == [
        {
            "id": "1",
            "text": "Nouvelle note",
            "reason": "Footnote present only in current table.",
        }
    ]
    assert result["technical_diff"]["footnotes_removed"] == []
    assert result["technical_diff"]["footnotes_renamed"] == []
    assert result["technical_diff"]["table_level_change"] == "modifie"


def test_diff_table_pair_gpt_calls_both_gpt_specialists_when_footnotes_exist_on_both_sides() -> None:
    call_kinds: list[str] = []
    responses = [
        {
            "indicators_added": [{"value": "Ratio de levier", "reason": "Nouveau."}],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "Nouveau indicateur.",
        },
        {
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [
                {
                    "previous_id": "1",
                    "current_id": "2",
                    "previous_text": "Note A",
                    "current_text": "Note A mise à jour",
                    "reason": "Même note, wording mis à jour.",
                }
            ],
            "reason": "Note modifiée matériellement.",
        },
    ]

    def fake_call_openai_json(**kwargs):
        call_kinds.append(kwargs["call_kind"])
        return responses.pop(0)

    result = diff_table_pair_gpt(
        _table(
            table_id="prev",
            indicators=["Ratio CET1"],
            footnotes=[{"id": "1", "text": "Note A"}],
        ),
        _table(
            table_id="curr",
            indicators=["Ratio CET1", "Ratio de levier"],
            footnotes=[{"id": "2", "text": "Note A mise à jour"}],
        ),
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert call_kinds == ["diff_indicators", "diff_footnotes"]
    assert result["technical_diff"]["indicators_added"] == [
        {"value": "Ratio de levier", "reason": "Nouveau."}
    ]
    assert result["technical_diff"]["footnotes_renamed"] == [
        {
            "previous_id": "1",
            "current_id": "2",
            "previous_text": "Note A",
            "current_text": "Note A mise à jour",
            "reason": "Même note, wording mis à jour.",
        }
    ]


def test_diff_table_pair_gpt_retries_malformed_indicator_response() -> None:
    responses = [
        {"indicators_added": "bad", "reason": "malformed"},
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "Aucun changement.",
        },
    ]

    def fake_call_openai_json(**kwargs):
        return responses.pop(0)

    result = diff_table_pair_gpt(
        _table(table_id="prev", indicators=["Ratio CET1"]),
        _table(table_id="curr", indicators=["Ratio CET1"]),
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert result["technical_diff"]["table_level_change"] == "inchange"


def test_diff_table_pair_gpt_retries_malformed_footnote_response() -> None:
    responses = [
        {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "reason": "Aucun changement sur les indicateurs.",
        },
        {"footnotes_added": "bad", "reason": "malformed"},
        {
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Aucun changement sur les notes.",
        },
    ]

    def fake_call_openai_json(**kwargs):
        return responses.pop(0)

    result = diff_table_pair_gpt(
        _table(
            table_id="prev",
            indicators=["Ratio CET1"],
            footnotes=[{"id": "1", "text": "Note A"}],
        ),
        _table(
            table_id="curr",
            indicators=["Ratio CET1"],
            footnotes=[{"id": "1", "text": "Note A"}],
        ),
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert result["technical_diff"]["footnotes_added"] == []
    assert result["technical_diff"]["footnotes_removed"] == []
    assert result["technical_diff"]["footnotes_renamed"] == []
