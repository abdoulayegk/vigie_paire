from __future__ import annotations

import json

from vigilance.dash_app.services.text_review import (
    apply_text_review_decision,
    write_text_review_to_disk,
)


def test_apply_text_review_decision_updates_all_change_buckets() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [{"change_id": "chg-1"}],
                "block_comparisons": [{"change_id": "chg-1"}],
            }
        ]
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="chg-1",
        status="rejected",
        comment="Pas une nouvelle idée.",
    )

    assert found is True
    for bucket in ("all_block_comparisons", "block_comparisons"):
        review = updated["section_comparisons"][0][bucket][0]["_analyst_review"]
        assert review["status"] == "rejected"
        assert review["comment"] == "Pas une nouvelle idée."
        assert review["nouvelle_idee_override"] is False


def test_apply_text_review_decision_updates_observation_and_source_changes() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_observation_comparisons": [
                    {"change_id": "obs-1", "source_change_ids": ["chg-1", "chg-2"]}
                ],
                "observation_comparisons": [
                    {"change_id": "obs-1", "source_change_ids": ["chg-1", "chg-2"]}
                ],
                "all_block_comparisons": [{"change_id": "chg-1"}, {"change_id": "chg-2"}],
                "block_comparisons": [{"change_id": "chg-1"}],
            }
        ]
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="obs-1",
        status="approved",
        comment="Observation valide.",
    )

    assert found is True
    section = updated["section_comparisons"][0]
    for bucket in (
        "all_observation_comparisons",
        "observation_comparisons",
        "all_block_comparisons",
        "block_comparisons",
    ):
        for change in section[bucket]:
            review = change["_analyst_review"]
            assert review["status"] == "approved"
            assert review["comment"] == "Observation valide."


def test_write_text_review_to_disk_can_skip_excel_regeneration(tmp_path, monkeypatch) -> None:
    payload = {
        "bank_code": "bnc",
        "quarter_current": "2025_t3",
        "quarter_previous": "2025_t2",
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "_analyst_review": {"status": "skipped", "comment": "À revoir plus tard."},
                    }
                ]
            }
        ],
    }
    target_dir = tmp_path / "bnc" / "2025_t3_vs_2025_t2"
    target_dir.mkdir(parents=True)
    target_json = target_dir / "text_comparison.json"
    target_json.write_text("{}", encoding="utf-8")
    called = {"excel": False}

    def _fake_generate_text_comparison_excel(*args, **kwargs):
        called["excel"] = True

    monkeypatch.setattr("vigilance.dash_app.services.text_review.TEXT_COMPARISON_DIR", tmp_path)
    monkeypatch.setattr(
        "vigilance.dash_app.services.text_review.generate_text_comparison_excel",
        _fake_generate_text_comparison_excel,
    )

    assert write_text_review_to_disk(payload, regenerate_excel=False) is True

    saved = json.loads(target_json.read_text(encoding="utf-8"))
    review = saved["section_comparisons"][0]["all_block_comparisons"][0]["_analyst_review"]
    assert review["status"] == "skipped"
    assert called["excel"] is False
