from __future__ import annotations

import base64

from dash.development.base_component import Component

from vigilance.dash_app.callbacks import proof_flow as proof_mod
from vigilance.dash_app.components.review_detail_v2 import _build_change_full_detail
from vigilance.dash_app.components.review_display_shared import (
    build_proofs_section,
    compute_flag_state,
)
from vigilance.dash_app.layouts import page_results
from vigilance.dash_app.services import pdf_rendering as pdf_mod


def _flatten_text(node: object) -> str:
    parts: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            parts.append(current)
            continue
        if isinstance(current, Component):
            children = getattr(current, "children", None)
            if isinstance(children, list):
                stack.extend(children)
            elif children is not None:
                stack.append(children)
    return " ".join(parts)


def test_build_proofs_section_shows_heading_and_visual_context() -> None:
    section = build_proofs_section(
        item={
            "change_type": "added",
            "page_t1": 10,
            "page_t2": 12,
        },
        img_t1_b64="abc",
        img_t2_b64="def",
        proof_display_mode="full",
    )

    text = _flatten_text(section)
    assert "Preuves visuelles : courant vs précédent" in text
    assert "Page 10" in text
    assert "Page 12" in text
    assert "Mode page complète encadrée" in text


def test_build_proofs_section_shows_full_without_bbox_label() -> None:
    section = build_proofs_section(
        item={
            "change_type": "added",
            "page_t1": 10,
            "page_t2": 12,
        },
        img_t1_b64="abc",
        img_t2_b64="def",
        proof_display_mode="full",
        proof_result_t1={"status": "ok", "mode_effective": "full_without_bbox"},
        proof_result_t2={"status": "ok", "mode_effective": "full_without_bbox"},
    )

    text = _flatten_text(section)
    assert "Mode page complète" in text


def test_build_proofs_section_table_added_shows_previous_placeholder() -> None:
    section = build_proofs_section(
        item={
            "change_type": "table_added",
            "page_t1": None,
            "page_t2": 8,
        },
        img_t1_b64=None,
        img_t2_b64="def",
        proof_display_mode="crop",
    )

    text = _flatten_text(section)
    assert "Aucun tableau dans le trimestre précédent" in text
    assert "Mode zoom tableau" in text


def test_build_proofs_section_shows_crop_bbox_missing_message() -> None:
    section = build_proofs_section(
        item={
            "change_type": "modified",
            "page_t1": 10,
            "page_t2": 12,
        },
        img_t1_b64=None,
        img_t2_b64=None,
        proof_display_mode="crop",
        proof_result_t1={"status": "bbox_missing", "mode_effective": "crop"},
        proof_result_t2={"status": "bbox_missing", "mode_effective": "crop"},
    )

    text = _flatten_text(section)
    assert "Zoom indisponible: zone de tableau non détectée." in text


def test_build_proofs_section_shows_footnote_bbox_missing_message() -> None:
    section = build_proofs_section(
        item={
            "change_type": "modified",
            "page_t1": 10,
            "page_t2": 12,
        },
        img_t1_b64=None,
        img_t2_b64=None,
        proof_display_mode="footnote",
        proof_result_t1={"status": "bbox_missing", "mode_effective": "footnote"},
        proof_result_t2={"status": "bbox_missing", "mode_effective": "footnote"},
    )

    text = _flatten_text(section)
    assert "Zone footnote indisponible: zone de tableau non détectée." in text


def test_update_review_proofs_returns_non_empty_panel_with_fallback_images(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        proof_mod,
        "_get_proof_render_result_for_item",
        lambda item, side, paths, proof_display_mode="crop", **kwargs: {
            "image_b64": None,
            "status": "render_failed",
            "mode_effective": proof_display_mode,
        },
    )

    result = proof_mod.update_review_proofs(
        [
            {
                "review_id": "rid-1",
                "table_key": "rid-1",
                "table_name": "Capital",
                "section": "capital_management",
                "page_t1": 5,
                "page_t2": 6,
                "bbox_t1": None,
                "bbox_t2": None,
                "source_pdf_t1": "/tmp/t1.pdf",
                "source_pdf_t2": "/tmp/t2.pdf",
                "changes": [
                    {
                        "change_id": "chg-1",
                        "change_type": "indicator_added",
                        "payload": {"indicator_name": "Ratio CET1"},
                        "validation_status": "pending",
                    }
                ],
            }
        ],
        {"review_id": "rid-1", "change_id": "chg-1"},
        {},
        True,
        "crop",
    )

    text = _flatten_text(result)
    assert "Preuves visuelles : courant vs précédent" in text
    assert "Rendu impossible pour cette preuve." in text


def test_update_review_proofs_uses_requested_display_mode(monkeypatch) -> None:
    seen_modes: list[str] = []

    def _fake_get(item, side, paths, proof_display_mode="crop", **kwargs):
        seen_modes.append(proof_display_mode)
        return {
            "image_b64": "abc",
            "status": "ok",
            "mode_effective": proof_display_mode,
        }

    monkeypatch.setattr(proof_mod, "_get_proof_render_result_for_item", _fake_get)

    result = proof_mod.update_review_proofs(
        [
            {
                "review_id": "rid-1",
                "table_key": "rid-1",
                "table_name": "Footnote table",
                "section": "risk_management",
                "page_t1": 20,
                "page_t2": 21,
                "bbox_t1": None,
                "bbox_t2": None,
                "source_pdf_t1": "/tmp/t1.pdf",
                "source_pdf_t2": "/tmp/t2.pdf",
                "changes": [
                    {
                        "change_id": "chg-1",
                        "change_type": "footnote_modified",
                        "payload": {"new_text": "Nouvelle note"},
                        "validation_status": "pending",
                    }
                ],
            }
        ],
        {"review_id": "rid-1", "change_id": "chg-1"},
        {},
        True,
        "footnote",
    )

    text = _flatten_text(result)
    assert seen_modes == ["footnote", "footnote"]
    assert "Mode note de bas de tableau" in text


def test_footnote_proof_flags_follow_current_left_previous_right() -> None:
    added_state = compute_flag_state(
        {
            "change_type": "modified",
            "selected_change_type": "footnote_added",
        }
    )
    removed_state = compute_flag_state(
        {
            "change_type": "modified",
            "selected_change_type": "footnote_removed",
        }
    )

    assert added_state["t2_class"] == "proof-card proof-flag-t2"
    assert added_state["t1_class"] == "proof-card"
    assert added_state["badge_t2"] == "Trimestre courant - note ajoutée"
    assert added_state["badge_t1"] == "Trimestre précédent"

    assert removed_state["t2_class"] == "proof-card"
    assert removed_state["t1_class"] == "proof-card proof-flag-t1"
    assert removed_state["badge_t2"] == "Trimestre courant"
    assert removed_state["badge_t1"] == "Trimestre précédent - note supprimée"


def test_footnote_detail_uses_current_then_previous_semantics() -> None:
    added_detail = _build_change_full_detail(
        {
            "change_type": "footnote_added",
            "payload": {"old_text": "", "new_text": "Nouvelle note"},
        }
    )
    removed_detail = _build_change_full_detail(
        {
            "change_type": "footnote_removed",
            "payload": {"old_text": "Ancienne note", "new_text": ""},
        }
    )

    added_text = _flatten_text(added_detail)
    removed_text = _flatten_text(removed_detail)

    assert "Trimestre courant - note ajoutée" in added_text
    assert "Nouvelle note" in added_text
    assert "Trimestre précédent" in added_text
    assert "Élément absent" in added_text

    assert "Trimestre courant" in removed_text
    assert "Élément absent" in removed_text
    assert "Trimestre précédent - note supprimée" in removed_text
    assert "Ancienne note" in removed_text


def test_update_review_proofs_table_removed_without_bbox_falls_back_to_full(
    monkeypatch,
) -> None:
    seen_modes: list[str] = []

    def _fake_get(item, side, paths, proof_display_mode="crop", **kwargs):
        seen_modes.append(proof_display_mode)
        return {
            "image_b64": "abc" if side == "t1" else None,
            "status": "ok" if side == "t1" else "page_missing",
            "mode_effective": proof_display_mode,
        }

    monkeypatch.setattr(proof_mod, "_get_proof_render_result_for_item", _fake_get)

    result = proof_mod.update_review_proofs(
        [
            {
                "review_id": "rid-rem-1",
                "table_key": "rid-rem-1",
                "table_name": "Tableau supprimé",
                "section": "risk_management",
                "page_t1": 18,
                "page_t2": None,
                "bbox_t1": None,
                "bbox_t2": None,
                "source_pdf_t1": "/tmp/t1.pdf",
                "source_pdf_t2": "/tmp/t2.pdf",
                "changes": [
                    {
                        "change_id": "chg-rem-1",
                        "change_type": "table_removed",
                        "payload": {},
                        "validation_status": "pending",
                    }
                ],
            }
        ],
        {"review_id": "rid-rem-1", "change_id": "chg-rem-1"},
        {},
        True,
        "crop",
    )

    text = _flatten_text(result)
    assert seen_modes == ["full", "full"]
    assert "Mode page complète encadrée" in text


def test_update_review_proofs_table_added_without_bbox_falls_back_to_full(
    monkeypatch,
) -> None:
    seen_modes: list[str] = []

    def _fake_get(item, side, paths, proof_display_mode="crop", **kwargs):
        seen_modes.append(proof_display_mode)
        return {
            "image_b64": "abc" if side == "t2" else None,
            "status": "ok" if side == "t2" else "page_missing",
            "mode_effective": proof_display_mode,
        }

    monkeypatch.setattr(proof_mod, "_get_proof_render_result_for_item", _fake_get)

    result = proof_mod.update_review_proofs(
        [
            {
                "review_id": "rid-add-1",
                "table_key": "rid-add-1",
                "table_name": "Tableau ajouté",
                "section": "capital_management",
                "page_t1": None,
                "page_t2": 22,
                "bbox_t1": None,
                "bbox_t2": None,
                "source_pdf_t1": "/tmp/t1.pdf",
                "source_pdf_t2": "/tmp/t2.pdf",
                "changes": [
                    {
                        "change_id": "chg-add-1",
                        "change_type": "table_added",
                        "payload": {},
                        "validation_status": "pending",
                    }
                ],
            }
        ],
        {"review_id": "rid-add-1", "change_id": "chg-add-1"},
        {},
        True,
        "crop",
    )

    text = _flatten_text(result)
    assert seen_modes == ["full", "full"]
    assert "Mode page complète encadrée" in text


def test_v2_meta_and_proofs_resolve_same_review_selection(monkeypatch) -> None:
    queue = [
        {
            "review_id": "rid-structure",
            "table_key": "rid-structure",
            "table_name": "STRUCTURE DE FONDS PROPRES ET RATIOS – Bâle III",
            "section": "capital_management",
            "page_t1": 38,
            "page_t2": 33,
            "bbox_t1": [0.1, 0.1, 0.9, 0.9],
            "bbox_t2": [0.1, 0.1, 0.9, 0.9],
            "source_pdf_t1": "/tmp/prev.pdf",
            "source_pdf_t2": "/tmp/curr.pdf",
            "changes": [
                {
                    "change_id": "chg-structure",
                    "change_type": "indicator_added",
                    "payload": {"indicator_name": "Crypto"},
                    "validation_status": "pending",
                    "is_required": True,
                }
            ],
        },
        {
            "review_id": "rid-actions",
            "table_key": "rid-actions",
            "table_name": "ACTIONS ET AUTRES TITRES¹",
            "section": "capital_management",
            "page_t1": 39,
            "page_t2": 34,
            "bbox_t1": [0.2, 0.2, 0.8, 0.8],
            "bbox_t2": [0.2, 0.2, 0.8, 0.8],
            "source_pdf_t1": "/tmp/prev.pdf",
            "source_pdf_t2": "/tmp/curr.pdf",
            "changes": [
                {
                    "change_id": "chg-actions",
                    "change_type": "indicator_removed",
                    "payload": {"indicator_name": "Serie 32"},
                    "validation_status": "pending",
                    "is_required": True,
                }
            ],
        },
    ]
    selection = {"review_id": "rid-actions", "change_id": "chg-actions"}
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        proof_mod,
        "_get_proof_render_result_for_item",
        lambda item, side, paths, proof_display_mode="crop", **kwargs: {
            "image_b64": "abc",
            "status": "ok",
            "mode_effective": proof_display_mode,
        },
    )

    def _capture_proofs(item, **kwargs):
        seen["proofs"] = str(item.get("table_name") or "")
        return "proofs"

    def _capture_meta(table, **kwargs):
        seen["meta"] = str(table.get("table_name") or "")
        return "meta"

    monkeypatch.setattr(proof_mod, "build_proofs_section", _capture_proofs)
    monkeypatch.setattr(proof_mod, "build_review_detail_v2", _capture_meta)

    assert (
        proof_mod.update_review_proofs(queue, selection, {}, True, "crop") == "proofs"
    )
    assert proof_mod.update_review_meta(queue, selection, True) == "meta"
    assert seen == {
        "proofs": "ACTIONS ET AUTRES TITRES¹",
        "meta": "ACTIONS ET AUTRES TITRES¹",
    }


def test_legacy_nav_buttons_hidden_when_v2_active(monkeypatch) -> None:
    view = page_results.build_page_results()

    ids: set[str] = set()
    stack = [view]
    while stack:
        current = stack.pop()
        if isinstance(current, Component):
            current_id = getattr(current, "id", None)
            if isinstance(current_id, str):
                ids.add(current_id)
            children = getattr(current, "children", None)
            if isinstance(children, list):
                stack.extend(children)
            elif children is not None:
                stack.append(children)

    assert "btn-prev" not in ids
    assert "btn-next" not in ids
    assert "results-review-tab" not in ids
    assert "nav-debug-panel" not in ids
    assert "store-review-data" not in ids
    assert "store-current-review-index" not in ids


def test_results_page_hides_quick_section_overview() -> None:
    view = page_results.build_page_results()

    text = _flatten_text(view)

    assert "Vue rapide par section" not in text
    assert "Repérez rapidement les tableaux touchés" not in text


def test_get_proof_render_result_returns_bbox_missing_for_crop() -> None:
    result = pdf_mod._get_proof_render_result_for_item(
        {
            "page_t1": 4,
            "source_ref_t1": "/tmp/t1.pdf",
            "bbox_t1": None,
        },
        "t1",
        {"pdf_t1": "/tmp/t1.pdf"},
        proof_display_mode="crop",
    )

    assert result == {
        "image_b64": None,
        "status": "bbox_missing",
        "mode_effective": "crop",
    }


def test_get_proof_render_result_returns_bbox_missing_for_footnote() -> None:
    result = pdf_mod._get_proof_render_result_for_item(
        {
            "page_t1": 4,
            "source_ref_t1": "/tmp/t1.pdf",
            "bbox_t1": None,
        },
        "t1",
        {"pdf_t1": "/tmp/t1.pdf"},
        proof_display_mode="footnote",
    )

    assert result == {
        "image_b64": None,
        "status": "bbox_missing",
        "mode_effective": "footnote",
    }


def test_footnote_render_passes_highlights_to_crop(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_crop_footnote(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return b"abc"

    monkeypatch.setattr(
        "vigilance.utils.pdf_crop.crop_footnote_region_to_bytes",
        _fake_crop_footnote,
    )

    result = pdf_mod._get_proof_render_result_for_item(
        {
            "page_t1": 4,
            "source_ref_t1": "/tmp/t1.pdf",
            "bbox_t1": [0.1, 0.2, 0.9, 0.6],
        },
        "t1",
        {"pdf_t1": "/tmp/t1.pdf"},
        proof_display_mode="footnote",
        highlight_rects=[[0.2, 0.7, 0.8, 0.74]],
        secondary_highlight_rects=[[0.2, 0.75, 0.8, 0.79]],
    )

    assert result == {
        "image_b64": base64.b64encode(b"abc").decode("ascii"),
        "status": "ok",
        "mode_effective": "footnote",
    }
    assert seen["kwargs"]["highlight_rects"] == [[0.2, 0.7, 0.8, 0.74]]
    assert seen["kwargs"]["secondary_highlight_rects"] == [[0.2, 0.75, 0.8, 0.79]]


def test_get_proof_render_result_uses_full_without_bbox(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf_mod,
        "render_full_proof_bytes",
        lambda *args, **kwargs: (b"abc", "ok", "full_without_bbox"),
    )

    result = pdf_mod._get_proof_render_result_for_item(
        {
            "page_t1": 4,
            "source_ref_t1": "/tmp/t1.pdf",
            "bbox_t1": None,
        },
        "t1",
        {"pdf_t1": "/tmp/t1.pdf"},
        proof_display_mode="full",
    )

    assert result == {
        "image_b64": base64.b64encode(b"abc").decode("ascii"),
        "status": "ok",
        "mode_effective": "full_without_bbox",
    }


def test_get_proof_image_b64_tolerates_none_pdf_paths(monkeypatch) -> None:
    monkeypatch.setattr(pdf_mod, "get_pdf_preview", lambda *args, **kwargs: None)

    result = pdf_mod._get_proof_image_b64(
        {
            "page_t1": 4,
            "source_ref_t1": None,
            "bbox_t1": None,
        },
        "t1",
        {"pdf_t1": None, "pdf_previous": None},
    )

    assert result is None
