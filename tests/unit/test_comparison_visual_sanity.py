from __future__ import annotations

from vigilance.comparison_visual_sanity import (
    render_visual_sanity_proof,
    visual_sanity_check,
    visual_sanity_check_table_event,
)


def test_visual_sanity_filters_indicator_and_footnote_rejections() -> None:
    diff_result = {
        "technical_diff": {
            "indicators_added": [{"value": "Ratio de levier", "reason": "new"}],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [{"id": "7", "text": "Nouvelle note", "reason": "new"}],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "modifie",
        },
        "reason": "test",
        "diff_mode": "gpt",
    }

    def fake_call_openai_json(**kwargs):
        assert kwargs["call_kind"] == "visual_sanity_check"
        return {
            "verdicts": [
                {
                    "item_id": "indicator_added::Ratio de levier",
                    "item_type": "indicator_added",
                    "verdict": "rejected",
                    "reason": "visible on both sides",
                }
            ],
            "overall_assessment": "one false positive",
        }

    result = visual_sanity_check(
        b"pq-proof",
        b"cq-proof",
        diff_result,
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert result["technical_diff"]["indicators_added"] == []
    assert result["technical_diff"]["footnotes_added"] == [
        {"id": "7", "text": "Nouvelle note", "reason": "new"}
    ]
    assert result["technical_diff"]["table_level_change"] == "modifie"
    assert result["visual_sanity_applied"] is True
    assert result["visual_sanity_rejected_count"] == 1
    assert result["visual_sanity_scope"] == ["indicators", "footnotes", "tables"]
    assert result["visual_sanity_render_mode"] == "full"
    assert result["visual_sanity_render_status"] == "ok"


def test_visual_sanity_table_event_rejects_false_positive_table() -> None:
    def fake_call_openai_json(**kwargs):
        assert kwargs["call_kind"] == "visual_sanity_check"
        return {
            "verdicts": [
                {
                    "item_id": "table_added::tbl_1 | Tableau de liquidité",
                    "item_type": "table_added",
                    "verdict": "rejected",
                    "reason": "table visible on both sides",
                }
            ],
            "overall_assessment": "reject table event",
        }

    verdict = visual_sanity_check_table_event(
        b"pq-proof",
        b"cq-proof",
        event_type="table_added",
        table_id="tbl_1",
        table_title="Tableau de liquidité",
        model="gpt-4o-test",
        call_openai_json=fake_call_openai_json,
    )

    assert verdict["confirmed"] is False
    assert verdict["visual_sanity_applied"] is True
    assert verdict["visual_sanity_rejected_count"] == 1
    assert verdict["visual_sanity_render_mode"] == "full"
    assert verdict["visual_sanity_render_status"] == "ok"


def test_render_visual_sanity_proof_reports_missing_pdf() -> None:
    raw, status = render_visual_sanity_proof(
        "",
        page=4,
        bbox=[0.1, 0.2, 0.9, 0.7],
    )

    assert raw is None
    assert status == "skipped_missing_pdf"


def test_render_visual_sanity_proof_reports_missing_bbox_without_fallback() -> None:
    raw, status = render_visual_sanity_proof(
        "/tmp/report.pdf",
        page=4,
        bbox=None,
    )

    assert raw is None
    assert status == "skipped_missing_bbox"


def test_visual_sanity_check_skips_when_no_visual_items() -> None:
    diff_result = {
        "technical_diff": {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "inchange",
        },
        "reason": "test",
        "diff_mode": "gpt",
    }

    result = visual_sanity_check(
        b"pq-proof",
        b"cq-proof",
        diff_result,
        model="gpt-4o-test",
        call_openai_json=lambda **kwargs: {},
    )

    assert result["visual_sanity_applied"] is False
    assert result["visual_sanity_rejected_count"] == 0
    assert result["visual_sanity_render_status"] == "ok"
