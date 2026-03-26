from __future__ import annotations

from app.review_storage import (
    REVIEW_STATE_SCHEMA_VERSION,
    get_review_state_path,
    load_review_state,
    save_review_state,
)


def test_review_state_sidecar_roundtrip(tmp_path) -> None:
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text("{}", encoding="utf-8")

    saved_path = save_review_state(
        compare_path,
        review_items=[{"change_id": "chg-1", "review_status": "approved"}],
        review_queue=[{"table_key": "tbl-1", "changes": []}],
        review_selection={"review_id": "tbl-1", "change_id": None},
        review_current_idx=2,
        current_change_idx=1,
        current_indicator_idx=3,
        preferred_store="review_queue",
        source="unit-test",
    )

    assert saved_path == get_review_state_path(compare_path)
    assert saved_path is not None and saved_path.exists()

    payload = load_review_state(compare_path)
    assert payload is not None
    assert payload["schema_version"] == REVIEW_STATE_SCHEMA_VERSION
    assert payload["compare_path"] == str(compare_path)
    assert payload["preferred_store"] == "review_queue"
    assert payload["review_current_idx"] == 2
    assert payload["current_change_idx"] == 1
    assert payload["current_indicator_idx"] == 3
    assert payload["source"] == "unit-test"
