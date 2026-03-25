from __future__ import annotations

from pathlib import Path

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionFullResult,
)


def _result(
    *,
    title: str = "",
    summary: str = "",
    indicators: list[str] | None = None,
    headers: list[str] | None = None,
    footnotes: list[dict[str, str]] | None = None,
    no_table_detected: bool = False,
    retry_reasons: list[str] | None = None,
) -> VisionFullResult:
    return VisionFullResult(
        table_title=title,
        table_summary=summary,
        headers=list(headers or []),
        indicators=list(indicators or []),
        footnotes_content=list(footnotes or []),
        no_table_detected=no_table_detected,
        retry_reasons=list(retry_reasons or []),
    )


def test_quality_pass_accepts_complete_initial_result(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    calls: list[tuple[bytes, bool]] = []

    def fake_extract(**kwargs):
        calls.append((kwargs["crop_bytes"], bool(kwargs.get("rescue_mode"))))
        return _result(
            title="Tableau 1 - Capital",
            summary="Ratios de capital réglementaires",
            indicators=["Ratio CET1", "Ratio Tier 1"],
            headers=["Mesure", "Valeur"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
    )

    assert result is not None
    assert result.extraction_status == "ok"
    assert result.acceptance_reason == "initial_complete"
    assert result.selected_candidate_name == "initial"
    assert calls == [(b"initial", False)]


def test_quality_pass_forces_rescue_when_summary_missing(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        if kwargs.get("rescue_mode"):
            return _result(
                title="Tableau 1 - Capital",
                summary="Ratios de capital réglementaires",
                indicators=["Ratio CET1", "Ratio Tier 1"],
                headers=["Mesure", "Valeur"],
            )
        return _result(
            title="Tableau 1 - Capital",
            summary="",
            indicators=["Ratio CET1", "Ratio Tier 1"],
            headers=["Mesure", "Valeur"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert result.acceptance_reason == "rescued_summary_recovered"
    assert result.selected_candidate_name == "same_crop_rescue"
    assert "missing_table_summary" in result.rejection_reasons


def test_quality_pass_allows_rescue_without_summary_when_structure_is_strong(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        return _result(
            title="Tableau 7 - Risque de crédit",
            summary="",
            indicators=[
                "Pertes de crédit attendues",
                "Exposition brute",
                "Radiations nettes",
            ],
            headers=["Mesure", "Valeur"],
            footnotes=[{"id": "1", "text": "Note de périmètre"}],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert result.acceptance_reason == "rescued_without_summary_strong_structure"
    assert "missing_table_summary" in result.rejection_reasons


def test_quality_pass_marks_confirmed_no_table_after_repeated_no_table_evidence(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        return _result(
            title="Rapport de gestion",
            summary="",
            indicators=[],
            headers=[],
            no_table_detected=True,
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.1, 0.9, 0.75],
        vision_cfg={},
        get_variant_crop_fn=lambda **kwargs: None,
    )

    assert result is not None
    assert result.extraction_status == "confirmed_no_table"
    assert result.no_table_evidence_count >= 2


def test_quality_pass_marks_suspect_when_no_summary_and_structure_remains_weak(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        return _result(
            title="Tableau 4 - Divers",
            summary="",
            indicators=["Total"],
            headers=[],
            footnotes=[],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
        get_variant_crop_fn=lambda **kwargs: None,
    )

    assert result is not None
    assert result.extraction_status == "suspect_unresolved"
    assert result.acceptance_reason == "suspect_unresolved_after_rescue_exhaustion"


def test_quality_pass_rejects_non_empty_summary_with_weak_indicator_only(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        return _result(
            title="Tableau 9 - Divers",
            summary="Sujet générique",
            indicators=["Total"],
            headers=[],
            footnotes=[],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
        get_variant_crop_fn=lambda **kwargs: None,
    )

    assert result is not None
    assert result.extraction_status == "suspect_unresolved"
    assert "weak_indicator_only" in result.rejection_reasons


def test_quality_pass_rejects_generic_title_contamination_even_with_summary(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_extract(**kwargs):
        return _result(
            title="Rapport de gestion",
            summary="Ratios de capital réglementaires",
            indicators=["Ratio CET1", "Ratio Tier 1"],
            headers=[],
            footnotes=[],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.1, 0.9, 0.7],
        vision_cfg={},
        get_variant_crop_fn=lambda **kwargs: None,
    )

    assert result is not None
    assert result.extraction_status == "suspect_unresolved"
    assert "generic_page_title" in result.rejection_reasons
    assert "dominant_contamination" in result.rejection_reasons


def test_quality_pass_prefers_top_trim_candidate_with_summary_over_noisier_candidate(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    def fake_variant_crop(
        *, bbox_override: list[float] | None = None, bottom_extension: float | None = None
    ) -> bytes | None:
        if bbox_override is not None:
            return b"top_trim" if bbox_override[1] > 0.14 else b"tight_body"
        if bottom_extension and bottom_extension > 0.0:
            return b"bottom_extended"
        return None

    def fake_extract(**kwargs):
        crop = kwargs["crop_bytes"]
        rescue_mode = bool(kwargs.get("rescue_mode"))
        if crop == b"initial" and not rescue_mode:
            return _result(
                title="Rapport de gestion",
                summary="",
                indicators=["Ratio CET1", "Ratio Tier 1"],
                headers=[],
            )
        if crop == b"initial" and rescue_mode:
            return _result(
                title="Rapport de gestion",
                summary="",
                indicators=["Ratio CET1", "Ratio Tier 1"],
                headers=[],
            )
        if crop == b"top_trim":
            return _result(
                title="Tableau 1 - Capital",
                summary="Ratios de capital réglementaires",
                indicators=["Ratio CET1", "Ratio Tier 1"],
                headers=["Mesure", "Valeur"],
            )
        if crop == b"bottom_extended":
            return _result(
                title="Rapport de gestion",
                summary="",
                indicators=[
                    "Ratio CET1",
                    "Ratio Tier 1",
                    "Texte narratif de page",
                ],
                headers=[],
            )
        return _result(
            title="Rapport de gestion",
            summary="",
            indicators=["Ratio CET1"],
            headers=[],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="bnc",
        bbox_norm=[0.1, 0.1, 0.9, 0.7],
        vision_cfg={},
        initial_bottom_extension=0.0,
        get_variant_crop_fn=fake_variant_crop,
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert result.selected_candidate_name == "top_trim"
    assert result.acceptance_reason == "rescued_summary_recovered"


def test_extract_cache_hit_preserves_decision_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(
        api_key="test-key",
        model="gpt-4o-test",
        use_cache=True,
    )
    monkeypatch.setattr(
        "vigilance.extraction.vision_full_extractor.get_vision_cache_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "vigilance.extraction.vision_full_extractor.cache_get",
        lambda cache_dir, cache_key: {
            "table_title": "Tableau 1 - Capital",
            "table_summary": "Ratios de capital réglementaires",
            "headers": ["Mesure", "Valeur"],
            "indicators": ["Ratio CET1", "Ratio Tier 1"],
            "footnotes_content": [{"id": "1", "text": "Note"}],
            "no_table_detected": False,
            "vision_status": "ok",
            "warnings": [],
            "retry_reasons": ["missing_table_summary"],
            "requested_max_completion_tokens": 64000,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "rescue_used": True,
            "extraction_status": "rescued",
            "acceptance_reason": "rescued_summary_recovered",
            "rejection_reasons": ["missing_table_summary"],
            "selected_candidate_name": "top_trim",
            "no_table_evidence_count": 1,
            "summary_present": True,
            "indicator_count": 2,
            "candidate_quality_rank": [1, 0, 1, 2, 1, 2, 1],
        },
    )

    result = extractor.extract(
        crop_bytes=b"cached",
        bank_code="bnc",
        pdf_sha="sha",
        page_number=1,
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
    )

    assert result is not None
    assert result.acceptance_reason == "rescued_summary_recovered"
    assert result.selected_candidate_name == "top_trim"
    assert result.rejection_reasons == ["missing_table_summary"]
    assert result.candidate_quality_rank == [1, 0, 1, 2, 1, 2, 1]
