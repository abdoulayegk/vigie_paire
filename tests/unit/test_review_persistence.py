from __future__ import annotations

from types import SimpleNamespace

from app.review_models import ReviewItem
from app.review_storage import load_review_state, save_review_state
from vigilance.dash_app import app as dash_app


def test_review_items_from_v2_queue_preserves_validation_metadata() -> None:
    queue = [
        {
            "table_key": "bnc::capital::pair",
            "section": "capital_management",
            "table_name": "Capital",
            "table_number": "1",
            "table_id_t1": "t1",
            "table_id_t2": "t2",
            "page_t1": 10,
            "page_t2": 11,
            "table_status": "partial",
            "changes": [
                {
                    "change_id": "chg_1",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Ratio CET1"},
                    "validation_status": "approved",
                    "validation_notes": "Valide par analyste",
                    "validated_at": "2026-03-18T10:00:00Z",
                    "validated_by": "analyst",
                    "is_required": True,
                }
            ],
        }
    ]

    items = dash_app._review_items_from_v2_queue(queue)

    assert len(items) == 1
    assert items[0].comment == "Valide par analyste"
    assert items[0].review_user == "analyst"
    assert items[0].review_timestamp == "2026-03-18T10:00:00Z"


def test_init_review_items_restores_persisted_state(monkeypatch, tmp_path) -> None:
    compare_path = tmp_path / "bnc_q2_vs_q1_2025.json"
    compare_path.write_text("{}", encoding="utf-8")
    persisted_queue = [
        {
            "table_key": "bnc::capital::pair",
            "section": "capital_management",
            "table_name": "Capital",
            "table_number": "1",
            "table_id_t1": "t1",
            "table_id_t2": "t2",
            "page_t1": 10,
            "page_t2": 11,
            "table_status": "partial",
            "changes": [
                {
                    "change_id": "chg_1",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Ratio CET1"},
                    "validation_status": "approved",
                    "is_required": True,
                }
            ],
        }
    ]
    save_review_state(
        compare_path,
        review_queue=persisted_queue,
        review_selection={"review_id": "bnc::capital::pair", "change_id": "chg_1"},
        preferred_store="review_queue",
        source="seed",
    )

    monkeypatch.setattr(
        dash_app,
        "build_review_items_from_indicator_result",
        lambda *args, **kwargs: [
            ReviewItem(
                change_id="raw-1",
                change_type="added",
                indicator="Raw",
                section="capital_management",
            )
        ],
    )
    monkeypatch.setattr(
        dash_app,
        "build_normalized_review_queue",
        lambda *args, **kwargs: [],
    )

    serialized, serialized_v2, selection, current_idx, change_idx, dbg = (
        dash_app.init_review_items(
            {"bank_code": "bnc", "quarter_from": "q1", "quarter_to": "q2"},
            {"pdf_previous": "/tmp/t1.pdf", "pdf_current": "/tmp/t2.pdf"},
            {"compare_path": str(compare_path)},
        )
    )

    assert serialized_v2 == persisted_queue
    assert serialized[0]["table_name"] == "Capital"
    assert selection["review_id"] == "bnc::capital::pair"
    assert current_idx == 0
    assert change_idx == 0
    assert dbg["trigger"] == "persisted_init"


def test_on_validate_change_v2_persists_review_state(monkeypatch, tmp_path) -> None:
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        dash_app,
        "ctx",
        SimpleNamespace(triggered_id="btn-approve-change-v2"),
    )

    queue = [
        {
            "table_key": "bnc::capital::pair",
            "section": "capital_management",
            "table_name": "Capital",
            "table_number": "1",
            "table_id_t1": "t1",
            "table_id_t2": "t2",
            "page_t1": 10,
            "page_t2": 11,
            "table_status": "pending",
            "changes": [
                {
                    "change_id": "chg_1",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Ratio CET1"},
                    "validation_status": "pending",
                    "is_required": True,
                }
            ],
        }
    ]

    new_queue, selection, change_idx, table_idx = dash_app.on_validate_change_v2(
        1,
        None,
        None,
        queue,
        {"review_id": "bnc::capital::pair", "change_id": "chg_1"},
        {"section": "all", "status": "all"},
        "Confirme",
        {"compare_path": str(compare_path)},
    )

    assert new_queue[0]["changes"][0]["validation_status"] == "approved"
    assert selection["review_id"] == "bnc::capital::pair"
    assert change_idx == 0
    assert table_idx == 0

    persisted = load_review_state(compare_path)
    assert persisted is not None
    assert persisted["preferred_store"] == "review_queue"
    assert persisted["review_queue"][0]["changes"][0]["validation_notes"] == "Confirme"
    assert persisted["review_selection"]["review_id"] == "bnc::capital::pair"
