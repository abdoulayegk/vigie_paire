from __future__ import annotations

import json

from vigilance.dash_app.services.review_writeback import write_back_to_disk


def test_write_back_to_disk_treats_skipped_as_pending(tmp_path, monkeypatch) -> None:
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text(
        json.dumps(
            {
                "pair_comparisons": [
                    {
                        "previous_table_id": "t1",
                        "current_table_id": "t2",
                        "technical_diff": {
                            "indicators_added": [{"value": "ratio cet1"}],
                            "indicators_removed": [],
                            "indicators_renamed": [],
                            "footnotes_added": [],
                            "footnotes_removed": [],
                            "footnotes_renamed": [],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    called = {"xlsx": False}

    def _fake_generate_comparison_excel(*args, **kwargs):
        called["xlsx"] = True

    monkeypatch.setattr(
        "vigilance.dash_app.services.review_writeback.generate_comparison_excel",
        _fake_generate_comparison_excel,
    )
    queue = [
        {
            "table_id_t1": "t1",
            "table_id_t2": "t2",
            "changes": [
                {
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Ratio CET1"},
                    "validation_status": "skipped",
                    "validation_notes": "À revoir",
                }
            ],
        }
    ]

    assert write_back_to_disk(compare_path, queue) is True

    saved = json.loads(compare_path.read_text(encoding="utf-8"))
    added = saved["pair_comparisons"][0]["technical_diff"]["indicators_added"][0]
    assert "_analyst_review" not in added
    assert saved["review_decisions_summary"]["pending"] == 1
    assert saved["review_decisions_summary"]["matched"] == 0
    assert called["xlsx"] is True
