from __future__ import annotations

from types import SimpleNamespace

import pytest

from vigilance.extraction.docling_processor import _build_indicator_reference_text
from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionFullResult,
)
from vigilance.extraction.vision_qa_inspector import QAResult, VisionTableInspector
from vigilance.utils.openai_schema import build_strict_openai_response_format


def _result(
    *,
    title: str = "",
    summary: str = "",
    indicators: list[str] | None = None,
    headers: list[str] | None = None,
    footnotes: list[dict[str, str]] | None = None,
) -> VisionFullResult:
    return VisionFullResult(
        table_title=title,
        table_summary=summary,
        headers=list(headers or []),
        indicators=list(indicators or []),
        footnotes_content=list(footnotes or []),
    )


def test_qa_prompt_limits_indicator_audit_to_leftmost_column() -> None:
    inspector = VisionTableInspector(model="gpt-4o-test")

    prompt = inspector._build_qa_prompt('{"indicators":["Programme A"]}')

    assert 'FIRST / LEFTMOST COLUMN' in prompt
    assert 'Text appearing under "États-Unis", "Europe"' in prompt
    assert "must NEVER be reported as a missing indicator" in prompt


def test_indicator_reference_text_is_disabled_for_multi_textual_headers() -> None:
    raw_text = """
    Canada  États-Unis  Europe
    Programme de titres de fiducie de capital (20 milliards de dollars)
    Programme de billets à moyen terme de premier rang liés du Canada (5 milliards de dollars)
    Programme d’obligations sécurisées législatives inscrit à la FCA du Royaume-Uni (100 milliards de dollars)
    """

    assert _build_indicator_reference_text(raw_text, max_chars=4000) is None


def test_indicator_reference_text_keeps_single_column_content() -> None:
    raw_text = """
    Programme de titres de fiducie de capital (20 milliards de dollars)
    Programme de billets à moyen terme de premier rang liés du Canada (5 milliards de dollars)
    Programme de titres adossés à des créances – lignes de crédit domiciliaires (Genesis Trust II)
    """.strip()

    assert _build_indicator_reference_text(raw_text, max_chars=4000) == raw_text


def test_quality_pass_prefers_initial_candidate_when_rescue_adds_right_column_noise(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    clean_indicators = [
        "Titres de fiducie",
        "Billets moyen terme",
        "Titres adossés",
    ]
    contaminated_tail = [
        "Programme de titres de fiducie de capital et de créance inscrit à la SEC des États-Unis (F-3) (75 milliards de dollars américains)",
        "Programme d’obligations sécurisées législatives inscrit à la Financial Conduct Authority (FCA) du Royaume-Uni (100 milliards de dollars)",
        "Programme de billets à moyen terme – marché mondial inscrit à la FCA (40 milliards de dollars américains)",
        "Programme de billets structurés à moyen terme liés – marché mondial non inscrit (20 milliards de dollars américains)",
    ]

    def fake_extract(**kwargs):
        headers = ["Canada", "États-Unis", "Europe"]
        if kwargs.get("rescue_mode"):
            return _result(
                summary="",
                indicators=clean_indicators + contaminated_tail,
                headers=headers,
            )
        return _result(
            summary="",
            indicators=clean_indicators,
            headers=headers,
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="td",
        bbox_norm=[0.05, 0.40, 0.94, 0.62],
        vision_cfg={},
        get_variant_crop_fn=lambda **kwargs: None,
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert result.acceptance_reason == "rescued_without_summary_strong_structure"
    assert result.selected_candidate_name == "initial"
    assert result.indicators == clean_indicators


def test_qa_inspector_uses_openai_compatible_strict_schema(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeParseCompletions:
        def parse(self, **kwargs):
            captured["model"] = kwargs["model"]
            captured["messages"] = kwargs["messages"]
            captured["response_format"] = kwargs["response_format"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=QAResult(
                                is_perfect=True,
                                missing_elements=[],
                                justification="ok",
                            )
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.beta = SimpleNamespace(
                chat=SimpleNamespace(completions=FakeParseCompletions())
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    inspector = VisionTableInspector(model="gpt-4o-test")
    result = inspector.inspect_extraction(b"fake-image", {"indicators": ["Programme A"]})

    assert result == QAResult(
        is_perfect=True,
        missing_elements=[],
        justification="ok",
    )
    assert captured["model"] == "gpt-4o-test"
    assert captured["response_format"] is QAResult
    assert isinstance(captured["messages"], list)


def test_strict_schema_builder_rejects_map_like_objects() -> None:
    class UnsupportedPayload(QAResult):
        metadata: dict[str, str]

    with pytest.raises(RuntimeError, match="map-like object not allowed"):
        build_strict_openai_response_format(
            UnsupportedPayload,
            name="UnsupportedPayload",
        )
