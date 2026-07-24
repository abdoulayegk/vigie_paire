"""Tests de la couche de localisation geometrique au niveau de la page."""

from __future__ import annotations

import pytest

from vigilance.extraction.page_table_locator import (
    OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS,
    PageTableLocator,
    _parse_page_layout,
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


def test_invalid_title_and_footnote_associations_are_not_used() -> None:
    payload = _layout_payload()
    payload["tables"][0]["title_bbox"] = [0.05, 0.70, 0.40, 0.73]
    payload["tables"][0]["footnotes_bbox"] = [0.05, 0.02, 0.40, 0.05]

    layout = _parse_page_layout(payload, page_number=30)

    assert layout is not None
    assert layout.tables[0].title_bbox is None
    assert layout.tables[0].footnotes_bbox is None


def test_page_locator_cache_reuses_one_call_for_tables_on_same_page(monkeypatch) -> None:
    locator = PageTableLocator(api_key="test-key", model="gpt-4o-test")
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
        api_key="test-key",
        model="gpt-4o-test",
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
        api_key="test-key",
        model="gpt-4o-test",
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

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    locator = PageTableLocator(api_key="test-key", model="gpt-4o-test")

    locator._ensure_client()

    assert captured == [
        {
            "api_key": "test-key",
            "timeout": OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS,
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
