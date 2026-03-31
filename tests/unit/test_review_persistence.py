from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.comparison_canonical import to_canonical_payload
from app.ui_io import load_comparison_result
from app.review_models import ReviewItem
from app.review_storage import load_review_state, save_review_state
from vigilance.dash_app import app as dash_app


class _FakeTable:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


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
                indicator="1 ajouté(s)",
                section="capital_management",
                table_name="Capital",
                table_id_t1="t1",
                table_id_t2="t2",
                indicators=[
                    {
                        "name": "Ratio CET1",
                        "type": "added",
                        "review_status": "pending",
                    }
                ],
            )
        ],
    )
    monkeypatch.setattr(
        dash_app,
        "build_normalized_review_queue",
        lambda *args, **kwargs: [_FakeTable(persisted_queue[0])],
    )

    serialized, serialized_v2, selection, _last_positions, change_idx = (
        dash_app.init_review_items(
            {"bank_code": "bnc", "quarter_from": "q1", "quarter_to": "q2"},
            {"pdf_previous": "/tmp/t1.pdf", "pdf_current": "/tmp/t2.pdf"},
            {"compare_path": str(compare_path)},
        )
    )

    assert serialized_v2 == persisted_queue
    assert serialized[0]["table_name"] == "Capital"
    assert selection["review_id"] == "bnc::capital::pair"
    assert change_idx == 0


def test_init_review_items_discards_incompatible_persisted_state(
    monkeypatch, tmp_path
) -> None:
    compare_path = tmp_path / "bnc_q2_vs_q1_2025.json"
    compare_path.write_text("{}", encoding="utf-8")
    save_review_state(
        compare_path,
        review_items=[
            {
                "change_id": "old-1",
                "change_type": "added",
                "indicator": "Old",
                "section": "capital_management",
                "table_name": "Capital",
                "table_id_t1": "t1",
                "table_id_t2": "t2",
                "item_type": "indicator",
                "indicators": [
                    {
                        "name": "Ratio CET1",
                        "type": "added",
                        "review_status": "pending",
                    }
                ],
            }
        ],
        review_selection={"review_id": "old-review", "change_id": "old-1"},
        preferred_store="review_items",
        source="seed",
    )

    monkeypatch.setattr(
        dash_app,
        "build_review_items_from_indicator_result",
        lambda *args, **kwargs: [
            ReviewItem(
                change_id="fresh-1",
                change_type="added",
                indicator="Fresh",
                section="risk_management",
                table_name="Risk",
                table_id_t1="rt1",
                table_id_t2="rt2",
                indicators=[
                    {
                        "name": "Ratio TLAC",
                        "type": "added",
                        "review_status": "pending",
                    }
                ],
            )
        ],
    )
    monkeypatch.setattr(
        dash_app,
        "build_normalized_review_queue",
        lambda *args, **kwargs: [
            _FakeTable(
                {
                    "table_key": "bnc::risk::pair",
                    "review_id": "bnc::risk::pair",
                    "section": "risk_management",
                    "table_name": "Risk",
                    "table_number": "1",
                    "table_id_t1": "rt1",
                    "table_id_t2": "rt2",
                    "page_t1": 20,
                    "page_t2": 21,
                    "table_status": "pending",
                    "changes": [
                        {
                            "change_id": "chg_1",
                            "change_type": "indicator_added",
                            "payload": {"indicator_name": "Ratio TLAC"},
                            "validation_status": "pending",
                            "is_required": True,
                        }
                    ],
                }
            )
        ],
    )

    serialized, serialized_v2, selection, _last_positions, change_idx = (
        dash_app.init_review_items(
            {"bank_code": "bnc", "quarter_from": "q1", "quarter_to": "q2"},
            {"pdf_previous": "/tmp/t1.pdf", "pdf_current": "/tmp/t2.pdf"},
            {"compare_path": str(compare_path)},
        )
    )

    assert serialized[0]["change_id"] == "fresh-1"
    assert serialized[0]["table_name"] == "Risk"
    assert serialized_v2[0]["table_name"] == "Risk"
    assert selection["review_id"] == "bnc::risk::pair"
    assert change_idx == 0


def test_init_review_items_v2_ignores_legacy_cursor_restore(
    monkeypatch, tmp_path
) -> None:
    compare_path = tmp_path / "td_q1_vs_q3_2026.json"
    compare_path.write_text("{}", encoding="utf-8")
    save_review_state(
        compare_path,
        review_items=[
            {
                "change_id": "legacy-1",
                "change_type": "added",
                "indicator": "Legacy",
                "section": "capital_management",
                "table_name": "Legacy Table",
                "table_id_t1": "legacy-t1",
                "table_id_t2": "legacy-t2",
                "item_type": "indicator",
                "indicators": [
                    {
                        "name": "Legacy indicator",
                        "type": "added",
                        "review_status": "approved",
                    }
                ],
            }
        ],
        review_selection={},
        review_current_idx=6,
        current_change_idx=4,
        preferred_store="review_items",
        source="seed",
    )

    monkeypatch.setattr(
        dash_app,
        "build_review_items_from_indicator_result",
        lambda *args, **kwargs: [
            ReviewItem(
                change_id="fresh-1",
                change_type="added",
                indicator="Fresh",
                section="capital_management",
                table_name="Fresh Table",
                table_id_t1="fresh-t1",
                table_id_t2="fresh-t2",
                indicators=[
                    {
                        "name": "Fresh indicator",
                        "type": "added",
                        "review_status": "pending",
                    }
                ],
            )
        ],
    )
    monkeypatch.setattr(
        dash_app,
        "build_normalized_review_queue",
        lambda *args, **kwargs: [
            _FakeTable(
                {
                    "table_key": "td::capital::pair",
                    "review_id": "td::capital::pair",
                    "section": "capital_management",
                    "table_name": "Fresh Table",
                    "table_number": "1",
                    "table_id_t1": "fresh-t1",
                    "table_id_t2": "fresh-t2",
                    "page_t1": 38,
                    "page_t2": 33,
                    "table_status": "pending",
                    "changes": [
                        {
                            "change_id": "chg_1",
                            "change_type": "indicator_added",
                            "payload": {"indicator_name": "Fresh indicator"},
                            "validation_status": "pending",
                            "is_required": True,
                        }
                    ],
                }
            )
        ],
    )

    serialized, serialized_v2, selection, _last_positions, change_idx = (
        dash_app.init_review_items(
            {"bank_code": "td", "quarter_from": "q3", "quarter_to": "q1"},
            {"pdf_previous": "/tmp/t1.pdf", "pdf_current": "/tmp/t2.pdf"},
            {"compare_path": str(compare_path)},
        )
    )

    assert serialized[0]["change_id"] == "fresh-1"
    assert serialized_v2[0]["review_id"] == "td::capital::pair"
    assert selection == {"review_id": "td::capital::pair", "change_id": "chg_1"}
    assert change_idx == 0


def test_td_review_queue_keeps_structure_and_actions_tables_separate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compare_path = (
        repo_root / "outputs" / "comparisons" / "td" / "2026_t1_vs_2025_t3" / "comparison.json"
    )
    raw = load_comparison_result(compare_path)
    payload = to_canonical_payload(raw)

    review_items = dash_app.build_review_items_from_indicator_result(
        payload,
        bank_code=str(payload.get("bank_code") or "td"),
        quarter_from=dash_app.quarter_label_from_payload(payload, "previous"),
        quarter_to=dash_app.quarter_label_from_payload(payload, "current"),
        pdf_path_t1=str(compare_path.parent / "previous_report.pdf"),
        pdf_path_t2=str(compare_path.parent / "current_report.pdf"),
    )
    queue = dash_app.build_normalized_review_queue(
        payload,
        [item.to_dict() for item in review_items],
        pdf_path_t1=str(compare_path.parent / "previous_report.pdf"),
        pdf_path_t2=str(compare_path.parent / "current_report.pdf"),
    )
    queue_by_name = {table.table_name: table.to_dict() for table in queue}

    structure = queue_by_name["STRUCTURE DE FONDS PROPRES ET RATIOS – Bâle III"]
    actions = queue_by_name["ACTIONS ET AUTRES TITRES¹"]

    assert structure["table_id_t1"] == "tbl_p038_i01"
    assert structure["table_id_t2"] == "tbl_p033_i01"
    assert structure["page_t1"] == 38
    assert structure["page_t2"] == 33

    assert actions["table_id_t1"] == "tbl_p039_i01"
    assert actions["table_id_t2"] == "tbl_p034_i01"
    assert actions["page_t1"] == 39
    assert actions["page_t2"] == 34


def test_legacy_review_callbacks_are_removed() -> None:
    assert not hasattr(dash_app, "on_modern_nav")
    assert not hasattr(dash_app, "on_review_navigate")
    assert not hasattr(dash_app, "on_review_status")


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

    new_queue, selection, change_idx = dash_app.on_validate_change_v2(
        1,
        None,
        None,
        queue,
        {"review_id": "bnc::capital::pair", "change_id": "chg_1"},
        {"section": "all", "status": "all"},
        {},
        "Confirme",
        {"compare_path": str(compare_path)},
    )

    assert new_queue[0]["changes"][0]["validation_status"] == "approved"
    assert selection["review_id"] == "bnc::capital::pair"
    assert change_idx == 0

    persisted = load_review_state(compare_path)
    assert persisted is not None
    assert persisted["preferred_store"] == "review_queue"
    assert persisted["review_queue"][0]["changes"][0]["validation_notes"] == "Confirme"
    assert persisted["review_selection"]["review_id"] == "bnc::capital::pair"
