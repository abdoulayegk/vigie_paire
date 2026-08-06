"""Regression : un re-rendu de la file de revue ne doit jamais valider ni naviguer.

Dash relance un callback quand un de ses Input se retrouve dans un bloc de layout
reconstruit par un autre callback. Ces declenchements arrivent avec ``n_clicks``
a 0 alors que ``ctx.triggered_id`` designe un bouton : sans garde, chaque
decision analyste rejouait une validation et un saut de tableau non demandes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from dash.exceptions import PreventUpdate

from vigie.interface.callbacks import review_flow as review_mod


def _queue() -> list[dict]:
    """Deux tableaux de deux changements chacun, tous en attente."""
    return [
        {
            "review_id": "bnc::capital::pair",
            "table_key": "bnc::capital::pair",
            "section": "capital_management",
            "table_name": "Capital",
            "table_status": "pending",
            "changes": [
                {
                    "change_id": "chg_1",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Ratio CET1"},
                    "validation_status": "approved",
                    "is_required": True,
                },
                {
                    "change_id": "chg_2",
                    "change_type": "indicator_removed",
                    "payload": {"indicator_name": "Ratio de levier"},
                    "validation_status": "pending",
                    "is_required": True,
                },
            ],
        },
        {
            "review_id": "bnc::credit::pair",
            "table_key": "bnc::credit::pair",
            "section": "credit_risk",
            "table_name": "Credit",
            "table_status": "pending",
            "changes": [
                {
                    "change_id": "chg_3",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Provisions"},
                    "validation_status": "pending",
                    "is_required": True,
                },
            ],
        },
    ]


_FILTERS = {"section": "all", "status": "all"}
_SELECTION = {"review_id": "bnc::capital::pair", "change_id": "chg_1"}


def test_phantom_approve_does_not_overwrite_a_rejection(monkeypatch, tmp_path) -> None:
    """Le re-rendu du panneau de detail ne doit pas repasser un rejet en validation."""
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id="btn-approve-change-v2"),
    )

    queue = _queue()
    queue[0]["changes"][0]["validation_status"] = "rejected"

    with pytest.raises(PreventUpdate):
        review_mod.on_validate_change_v2(
            0,  # n_clicks remis a zero par le re-rendu, pas un clic analyste
            0,
            0,
            queue,
            _SELECTION,
            _FILTERS,
            {},
            "",
            {"compare_path": str(compare_path)},
        )

    assert queue[0]["changes"][0]["validation_status"] == "rejected"


def test_real_reject_click_still_applies(monkeypatch, tmp_path) -> None:
    """Un vrai clic sur Rejeter reste applique et laisse l'analyste sur place."""
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id="btn-reject-change-v2"),
    )

    new_queue, selection, _ = review_mod.on_validate_change_v2(
        0,
        1,
        0,
        _queue(),
        _SELECTION,
        _FILTERS,
        {},
        "",
        {"compare_path": str(compare_path)},
    )

    assert new_queue[0]["changes"][0]["validation_status"] == "rejected"
    assert selection == _SELECTION


def test_phantom_queue_item_does_not_jump_to_another_table(monkeypatch) -> None:
    """Le re-rendu de la file ne doit pas selectionner un autre tableau."""
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id={"type": "queue-table-item-v2", "review_id": "bnc::credit::pair"}),
    )

    with pytest.raises(PreventUpdate):
        review_mod.on_navigate_table_v2(
            0,
            0,
            [0, 0],
            _queue(),
            _SELECTION,
            _FILTERS,
            {},
        )


def test_real_queue_item_click_still_navigates(monkeypatch) -> None:
    """Un vrai clic sur un tableau de la file change bien de tableau."""
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id={"type": "queue-table-item-v2", "review_id": "bnc::credit::pair"}),
    )

    selection, _ = review_mod.on_navigate_table_v2(
        0,
        0,
        [0, 1],
        _queue(),
        _SELECTION,
        _FILTERS,
        {},
    )

    assert selection["review_id"] == "bnc::credit::pair"


def test_phantom_prev_table_does_not_navigate(monkeypatch) -> None:
    """Le bouton Tableau precedent reconstruit ne doit pas declencher de saut."""
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id="btn-prev-table-v2"),
    )

    with pytest.raises(PreventUpdate):
        review_mod.on_navigate_table_v2(
            0,
            0,
            [0, 0],
            _queue(),
            {"review_id": "bnc::credit::pair", "change_id": "chg_3"},
            _FILTERS,
            {},
        )


def test_phantom_prev_change_does_not_navigate(monkeypatch) -> None:
    """Le bouton Precedent reconstruit ne doit pas changer le changement affiche."""
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id="btn-prev-change-v2"),
    )

    with pytest.raises(PreventUpdate):
        review_mod.on_navigate_change_v2(
            0,
            0,
            _queue(),
            {"review_id": "bnc::capital::pair", "change_id": "chg_2"},
            _FILTERS,
            {},
        )


def test_phantom_filter_button_does_not_reset_section(monkeypatch) -> None:
    """Le re-rendu de la barre de filtres ne doit pas revenir a 'toutes les sections'."""
    monkeypatch.setattr(
        review_mod,
        "ctx",
        SimpleNamespace(triggered_id={"type": "filter-section-v2", "value": "all"}),
    )

    with pytest.raises(PreventUpdate):
        review_mod.on_filter_section([0, 0, 0], {"section": "credit_risk", "status": "all"})
