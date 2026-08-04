from __future__ import annotations

from vigie.comparaison.differences.comparaison_paire import diff_table_pair_gpt
from vigie.comparaison.differences.normalisation_elements import _table_context


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
    fn_added = result["technical_diff"]["footnotes_added"]
    assert len(fn_added) == 1
    assert fn_added[0]["id"] == "1"
    assert fn_added[0]["text"] == "Nouvelle note"
    assert fn_added[0]["reason"] == "Footnote present only in current table."
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
        # Inspector: confirms the added indicator is real
        {
            "added_verdicts": [
                {
                    "value": "Ratio de levier",
                    "verdict": "real",
                    "reason": "genuinely new",
                },
            ],
            "removed_verdicts": [],
            "artifact_pairs": [],
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

    assert call_kinds == ["diff_indicators", "diff_footnotes", "inspect_artifacts"]
    ind_added = result["technical_diff"]["indicators_added"]
    assert len(ind_added) == 1
    assert ind_added[0]["value"] == "Ratio de levier"
    assert ind_added[0]["reason"] == "Nouveau."
    fn_renamed = result["technical_diff"]["footnotes_renamed"]
    assert len(fn_renamed) == 1
    assert fn_renamed[0]["previous_id"] == "1"
    assert fn_renamed[0]["current_id"] == "2"
    assert fn_renamed[0]["previous_text"] == "Note A"
    assert fn_renamed[0]["current_text"] == "Note A mise à jour"


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


def test_table_context_filters_standalone_dates() -> None:
    """Pure date strings like '31 octobre 2024' should be stripped from indicators."""
    entry = _table(
        table_id="t1",
        indicators=[
            "Dépôts provenant d'autres banques",
            "Certificats de dépôt",
            "31 octobre 2024",
            "Au 30 avril 2025",
        ],
    )
    ctx = _table_context(entry)
    names = [ind["name"] for ind in ctx["indicators"]]
    assert "31 octobre 2024" not in names
    assert "Au 30 avril 2025" not in names
    assert "Dépôts provenant d'autres banques" in names
    assert "Certificats de dépôt" in names
    assert len(names) == 2
