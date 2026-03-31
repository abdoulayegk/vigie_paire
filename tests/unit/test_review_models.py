from __future__ import annotations

from vigilance.review_models import ReviewItem


def test_review_item_roundtrip_preserves_future_persistence_fields() -> None:
    original = ReviewItem(
        change_id="chg-1",
        change_type="added",
        indicator="Ratio CET1",
        review_status="approved",
        comment="ok",
        review_user="analyste",
        review_timestamp="2026-03-18T09:30:00Z",
        edited_value="Ratio CET1 ajusté",
    )

    restored = ReviewItem.from_dict(original.to_dict())

    assert restored.review_user == "analyste"
    assert restored.review_timestamp == "2026-03-18T09:30:00Z"
    assert restored.edited_value == "Ratio CET1 ajusté"
