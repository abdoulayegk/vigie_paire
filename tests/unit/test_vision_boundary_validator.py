from __future__ import annotations

from types import SimpleNamespace

from vigilance.text_analysis.vision_boundary_validator import (
    OpenAITextBoundaryValidator,
    VisionBoundaryAssessment,
)


def _segment(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        page=4,
        bbox_norm=[0.1, 0.2, 0.9, 0.3],
    )


def _validator(tmp_path) -> OpenAITextBoundaryValidator:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    return OpenAITextBoundaryValidator(
        pdf_path=pdf_path,
        cache_dir=tmp_path / "cache",
        client=object(),
        model="gpt-4o",
        confidence_threshold=0.90,
        max_calls=2,
    )


def test_high_confidence_narrative_continuation_is_applied_and_cached(
    tmp_path,
    monkeypatch,
) -> None:
    validator = _validator(tmp_path)
    calls = []

    def assessment(*args):
        calls.append(True)
        return VisionBoundaryAssessment(
            same_sentence="yes",
            reading_order="previous_then_next",
            previous_block_type="narrative",
            next_block_type="narrative",
            allow_remove_previous=False,
            allow_remove_next=False,
            confidence=0.97,
            justification="Même phrase dans la même colonne.",
        )

    monkeypatch.setattr(validator, "_request_assessment", assessment)
    previous = _segment("Le cadre demeure")
    current = _segment("La Banque...")

    first = validator.validate(previous, current)
    second = validator.validate(previous, current)

    assert first.apply_merge is True
    assert second.apply_merge is True
    assert second.cached is True
    assert len(calls) == 1


def test_low_confidence_or_error_keeps_boundary_fail_closed(tmp_path, monkeypatch) -> None:
    validator = _validator(tmp_path)
    monkeypatch.setattr(
        validator,
        "_request_assessment",
        lambda *args: VisionBoundaryAssessment(
            same_sentence="yes",
            reading_order="previous_then_next",
            previous_block_type="narrative",
            next_block_type="narrative",
            allow_remove_previous=False,
            allow_remove_next=False,
            confidence=0.72,
            justification="Lecture incertaine.",
        ),
    )
    low = validator.validate(_segment("A"), _segment("B"))
    assert low.apply_merge is False
    assert low.status == "reviewed_fail_closed"

    monkeypatch.setattr(
        validator,
        "_request_assessment",
        lambda *args: (_ for _ in ()).throw(TimeoutError("indisponible")),
    )
    error = validator.validate(_segment("C"), _segment("D"))
    assert error.apply_merge is False
    assert error.status == "error_fail_closed"
