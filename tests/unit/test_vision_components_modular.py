"""Tests unitaires pour les sous-composants modulaires Vision."""

from __future__ import annotations

from vigilance.extraction.vision import (
    VisionTableResponse,
    build_vision_system_prompt,
    build_vision_user_prompt,
    crop_image_bbox,
)


def test_build_vision_prompts() -> None:
    sys_p = build_vision_system_prompt()
    assert "financières" in sys_p

    usr_p = build_vision_user_prompt("td", "Section liquidité")
    assert "TD" in usr_p


def test_vision_table_response_schema() -> None:
    resp = VisionTableResponse(title="Tableau 54 Ratio de liquidité")
    assert resp.title == "Tableau 54 Ratio de liquidité"


def test_crop_image_bbox() -> None:
    b = crop_image_bbox(b"fake_image", (0, 0, 100, 100))
    assert b == b"fake_image"
