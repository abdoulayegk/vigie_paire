"""Tests du chemin Vision structuré (TDM annuelle / transitions) hors extraction tables."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from vigie.extraction.genai_toc_detector import (
    AnnualTocAnalysisLLM,
    AnnualTocBoundaryLLM,
    AnnualTocEntryLLM,
    GenAITOCDetector,
    PageTransitionLLM,
)


def _parsed_response(parsed, *, finish_reason: str = "stop", refusal: str | None = None):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_annual_toc_schema_rejects_unknown_section_type() -> None:
    with pytest.raises(ValidationError):
        AnnualTocBoundaryLLM(
            section_type="gestion_capital",  # type: ignore[arg-type]
            title_found="Gestion du capital",
            start_page=53,
            successor_title="Gestion des risques",
            successor_page=62,
            confidence=0.9,
        )


def test_call_vision_structured_returns_parsed_model() -> None:
    detector = GenAITOCDetector(api_key="test-key")
    expected = PageTransitionLLM(
        confirmed=True,
        confidence=0.91,
        observed_title="Gestion du capital",
        previous_page_belongs_to_prior_section=True,
        candidate_page_starts_expected_section=True,
        reason="Titre de chapitre en tête de page.",
    )
    parse = MagicMock(return_value=_parsed_response(expected))
    detector._client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    result = detector._call_vision_structured(
        "prompt",
        ["img"],
        response_format=PageTransitionLLM,
        max_completion_tokens=1200,
    )

    assert result == expected
    parse.assert_called_once()
    kwargs = parse.call_args.kwargs
    assert kwargs["response_format"] is PageTransitionLLM
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_completion_tokens"] == 1200


def test_call_vision_structured_soft_fails_on_truncation() -> None:
    detector = GenAITOCDetector(api_key="test-key")
    parse = MagicMock(
        return_value=_parsed_response(
            AnnualTocAnalysisLLM(
                is_master_toc=True,
                confidence=0.9,
                entries=[],
                boundaries=[],
                warnings=[],
            ),
            finish_reason="length",
        )
    )
    detector._client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    result = detector._call_vision_structured(
        "prompt",
        ["img"],
        response_format=AnnualTocAnalysisLLM,
    )
    assert result is None


def test_call_vision_structured_soft_fails_on_invalid_json_exception() -> None:
    detector = GenAITOCDetector(api_key="test-key")
    parse = MagicMock(side_effect=ValueError('Expecting property name enclosed in double quotes'))
    detector._client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    result = detector._call_vision_structured(
        "prompt",
        ["img"],
        response_format=AnnualTocAnalysisLLM,
    )
    assert result is None


def test_analyze_annual_toc_page_maps_structured_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = GenAITOCDetector(api_key="test-key")
    monkeypatch.setattr(detector, "_page_to_base64", lambda *_args, **_kwargs: "img")
    structured = AnnualTocAnalysisLLM(
        is_master_toc=True,
        confidence=0.94,
        entries=[AnnualTocEntryLLM(title="Gestion du capital", page=53, level=0)],
        boundaries=[
            AnnualTocBoundaryLLM(
                section_type="capital_management",
                title_found="Gestion du capital",
                start_page=53,
                successor_title="Gestion des risques",
                successor_page=62,
                confidence=0.95,
            ),
            AnnualTocBoundaryLLM(
                section_type="risk_management",
                title_found="Gestion des risques",
                start_page=70,
                successor_title="Bad successor",
                successor_page=70,
                confidence=0.5,
            ),
        ],
        warnings=["offset ambigu"],
    )
    monkeypatch.setattr(detector, "_call_vision_structured", lambda *a, **k: structured)

    analysis = detector.analyze_annual_toc_page("/tmp/report.pdf", 17)

    assert analysis.is_master_toc is True
    assert analysis.confidence == 0.94
    assert analysis.entries == [{"title": "Gestion du capital", "page": 53, "level": 0}]
    assert len(analysis.boundaries) == 1
    assert analysis.boundaries[0].section_type == "capital_management"
    assert analysis.boundaries[0].successor_page == 62
    assert analysis.warnings == ["offset ambigu"]


def test_validate_section_transition_maps_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = GenAITOCDetector(api_key="test-key")
    monkeypatch.setattr(detector, "_page_to_base64", lambda *_args, **_kwargs: "img")
    structured = PageTransitionLLM(
        confirmed=True,
        confidence=0.88,
        observed_title="Gestion des risques",
        previous_page_belongs_to_prior_section=True,
        candidate_page_starts_expected_section=True,
        reason="Chapitre démarre en haut de page.",
    )
    monkeypatch.setattr(detector, "_call_vision_structured", lambda *a, **k: structured)

    validation = detector.validate_section_transition(
        "/tmp/report.pdf",
        61,
        62,
        section_type="risk_management",
        expected_title="Gestion des risques",
    )

    assert validation.confirmed is True
    assert validation.confidence == 0.88
    assert validation.observed_title == "Gestion des risques"
    assert validation.previous_page_belongs_to_prior_section is True
    assert validation.candidate_page_starts_expected_section is True
