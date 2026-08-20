"""Tests de la couche de localisation geometrique au niveau de la page."""

from __future__ import annotations

import pytest

from vigie.extraction.page_table_locator import (
    OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS,
    PAGE_LOCATOR_MAX_COMPLETION_TOKENS,
    PageTableLocator,
    _parse_page_layout,
    build_near_full_page_crop_plan,
    build_page_table_crop_plan,
    should_use_page_context_rescue,
)


def _layout_payload(*, confidence: float = 0.96) -> dict:
    return {
        "tables": [
            {
                "table_bbox": [0.08, 0.18, 0.92, 0.43],
                "title_bbox": [0.08, 0.14, 0.70, 0.17],
                "footnotes_bbox": [0.08, 0.44, 0.90, 0.48],
                "title_text": "Ratios de fonds propres",
                "continuation": False,
                "confidence": confidence,
            },
            {
                "table_bbox": [0.08, 0.55, 0.92, 0.82],
                "title_bbox": [0.08, 0.51, 0.70, 0.54],
                "footnotes_bbox": None,
                "title_text": "Actifs ponderes en fonction des risques",
                "continuation": False,
                "confidence": 0.97,
            },
        ],
        "table_count": 2,
    }


def test_crop_plan_includes_associated_title_and_footnotes_without_neighbor() -> None:
    layout = _parse_page_layout(_layout_payload(), page_number=30)

    assert layout is not None
    plan = build_page_table_crop_plan(
        layout,
        [0.09, 0.19, 0.91, 0.42],
    )

    assert plan is not None
    assert plan.bbox_norm == (0.08, 0.18, 0.92, 0.43)
    assert plan.top_extension == pytest.approx(0.045)
    assert plan.bottom_extension == pytest.approx(0.055)
    assert plan.bbox_norm[3] + plan.bottom_extension < 0.55
    assert plan.continuation is False
    assert plan.table_count == 2


def test_crop_plan_rejects_low_confidence_locator_region() -> None:
    layout = _parse_page_layout(_layout_payload(confidence=0.60), page_number=30)

    assert layout is not None
    assert build_page_table_crop_plan(layout, [0.09, 0.19, 0.91, 0.42]) is None


def test_crop_plan_preserves_docling_left_edge_when_locator_cuts_labels() -> None:
    """Le localisateur serre parfois la grille et coupe la colonne de libelles.

    L'ancre Docling (target_bbox) est plus large a gauche : le plan doit
    restaurer ce bord gauche sans toucher au cadrage vertical du locator.
    """
    payload = {
        "tables": [
            {
                # Locator serre sur la grille (bord gauche trop a droite).
                "table_bbox": [0.24, 0.12, 0.93, 0.56],
                "title_bbox": [0.07, 0.08, 0.70, 0.11],
                "footnotes_bbox": [0.07, 0.57, 0.90, 0.62],
                "title_text": "Tableau 43",
                "continuation": False,
                "confidence": 0.97,
            }
        ],
        "table_count": 1,
    }
    layout = _parse_page_layout(payload, page_number=87)
    assert layout is not None

    # Ancre Docling pleine largeur (colonne des libelles incluse).
    plan = build_page_table_crop_plan(
        layout,
        [0.065, 0.087, 0.936, 0.558],
    )

    assert plan is not None
    assert plan.bbox_norm[0] == pytest.approx(0.065)
    assert plan.bbox_norm[1] == pytest.approx(0.12)
    assert plan.bbox_norm[2] == pytest.approx(0.936)
    assert plan.bbox_norm[3] == pytest.approx(0.56)
    assert plan.top_extension == pytest.approx(0.045)
    assert plan.bottom_extension == pytest.approx(0.065)


def test_near_full_page_plan_accepts_one_reliable_region() -> None:
    payload = _layout_payload()
    payload["tables"] = payload["tables"][:1]
    payload["table_count"] = 1
    layout = _parse_page_layout(payload, page_number=30)

    assert layout is not None
    plan = build_near_full_page_crop_plan(layout)

    assert plan is not None
    assert plan.bbox_norm == (0.08, 0.18, 0.92, 0.43)


def test_near_full_page_plan_refuses_multiple_reliable_regions() -> None:
    layout = _parse_page_layout(_layout_payload(), page_number=30)

    assert layout is not None
    assert build_near_full_page_crop_plan(layout) is None


def test_invalid_title_and_footnote_associations_are_not_used() -> None:
    payload = _layout_payload()
    payload["tables"][0]["title_bbox"] = [0.05, 0.70, 0.40, 0.73]
    payload["tables"][0]["footnotes_bbox"] = [0.05, 0.02, 0.40, 0.05]

    layout = _parse_page_layout(payload, page_number=30)

    assert layout is not None
    assert layout.tables[0].title_bbox is None
    assert layout.tables[0].footnotes_bbox is None


def test_page_locator_cache_reuses_one_call_for_tables_on_same_page(monkeypatch) -> None:
    locator = PageTableLocator(model="gpt-5.4-test")
    calls: list[bytes] = []
    expected = _parse_page_layout(_layout_payload(), page_number=30)

    def fake_locate(page_image_bytes: bytes, page_number: int):
        calls.append(page_image_bytes)
        return expected

    monkeypatch.setattr(locator, "_locate_uncached", fake_locate)

    first = locator.locate_page(b"page-image", 30, pdf_sha="same-pdf")
    second = locator.locate_page(b"rendered-again", 30, pdf_sha="same-pdf")

    assert first is expected
    assert second is expected
    assert calls == [b"page-image"]


def test_page_locator_persistent_cache_stabilizes_separate_runs(
    tmp_path,
    monkeypatch,
) -> None:
    expected = _parse_page_layout(_layout_payload(), page_number=30)
    first_locator = PageTableLocator(
        model="gpt-5.4-test",
        use_cache=True,
        cache_dir=str(tmp_path),
    )
    calls: list[bytes] = []

    def fake_locate(page_image_bytes: bytes, page_number: int):
        calls.append(page_image_bytes)
        return expected

    monkeypatch.setattr(first_locator, "_locate_uncached", fake_locate)
    first = first_locator.locate_page(b"page-image", 30, pdf_sha="same-pdf")

    second_locator = PageTableLocator(
        model="gpt-5.4-test",
        use_cache=True,
        cache_dir=str(tmp_path),
    )

    def fail_if_called(*_args):
        raise AssertionError("persistent cache should avoid a second API call")

    monkeypatch.setattr(second_locator, "_locate_uncached", fail_if_called)
    second = second_locator.locate_page(
        b"different-render",
        30,
        pdf_sha="same-pdf",
    )

    assert first == expected
    assert second == expected
    assert calls == [b"page-image"]


def test_page_locator_client_has_direct_120_second_timeout(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(
        "vigie.extraction.page_table_locator.get_client", lambda **kwargs: captured.append(kwargs) or object()
    )
    locator = PageTableLocator(model="gpt-5.4-test")

    locator._ensure_client()

    assert captured == [
        {
            "timeout": OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS,
            "max_retries": 1,
        }
    ]
    assert OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS == 120.0


def test_page_context_trigger_accepts_missing_targeted_result() -> None:
    assert should_use_page_context_rescue(
        True,
        ["low_density_vertical"],
    )
    assert should_use_page_context_rescue(False, ["missing_result"])
    assert not should_use_page_context_rescue(True, ["missing_table_summary"])


def test_page_locator_retries_with_json_object_on_empty_structured_response(
    monkeypatch,
) -> None:
    import json

    calls: list[dict] = []

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str, finish_reason: str = "stop") -> None:
            self.message = FakeMessage(content)
            self.finish_reason = finish_reason

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    responses = ["", json.dumps(_layout_payload())]

    def fake_chat_completions_create(_client, **kwargs) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(responses[len(calls) - 1])

    monkeypatch.setattr(
        "vigie.extraction.page_table_locator.get_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vigie.extraction.page_table_locator.chat_completions_create",
        fake_chat_completions_create,
    )

    locator = PageTableLocator(model="gpt-5.4-test")
    layout = locator._locate_uncached(b"page-image", 34)

    assert layout is not None
    assert len(layout.tables) == 2
    assert len(calls) == 2
    assert calls[0]["profile"] == "locator"
    assert calls[0]["max_completion_tokens"] == PAGE_LOCATOR_MAX_COMPLETION_TOKENS
    assert PAGE_LOCATOR_MAX_COMPLETION_TOKENS == 128_000
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert calls[1]["profile"] == "locator"
    assert calls[1]["max_completion_tokens"] == PAGE_LOCATOR_MAX_COMPLETION_TOKENS


def test_parse_page_layout_accepts_tables_without_table_count() -> None:
    payload = _layout_payload()
    del payload["table_count"]

    layout = _parse_page_layout(payload, page_number=33)

    assert layout is not None
    assert len(layout.tables) == 2
    assert layout.tables[0].title_text == "Ratios de fonds propres"
