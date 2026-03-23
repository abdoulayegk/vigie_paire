from __future__ import annotations

from types import SimpleNamespace

from dash.development.base_component import Component

from vigilance.dash_app import app as dash_app
from vigilance.dash_app.components.review_detail import build_proofs_section


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
    assert "Preuves visuelles T1/T2" in text
    assert "Page 10" in text
    assert "Page 12" in text
    assert "Mode page complète + bbox" in text


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
    assert "Aucun tableau dans le trimestre precedent" in text
    assert "Mode focus tableau" in text


def test_update_review_proofs_returns_non_empty_panel_with_fallback_images(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dash_app,
        "_get_proof_image_b64_for_item",
        lambda item, side, paths, proof_display_mode="crop": None,
    )

    result = dash_app.update_review_proofs(
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
    assert "Preuves visuelles T1/T2" in text
    assert "Image non disponible" in text


def test_update_review_proofs_uses_requested_display_mode(monkeypatch) -> None:
    seen_modes: list[str] = []

    def _fake_get(item, side, paths, proof_display_mode="crop"):
        seen_modes.append(proof_display_mode)
        return "abc"

    monkeypatch.setattr(dash_app, "_get_proof_image_b64_for_item", _fake_get)

    result = dash_app.update_review_proofs(
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


def test_get_proof_image_b64_tolerates_none_pdf_paths(monkeypatch) -> None:
    monkeypatch.setattr(dash_app, "get_pdf_preview", lambda *args, **kwargs: None)

    result = dash_app._get_proof_image_b64(
        {
            "page_t1": 4,
            "source_ref_t1": None,
            "bbox_t1": None,
        },
        "t1",
        {"pdf_t1": None, "pdf_previous": None},
    )

    assert result is None
