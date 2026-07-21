from __future__ import annotations

from pathlib import Path

import pytest

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionFullResult,
    _normalize_footnote_marker_id,
)
from vigilance.extraction.vision_qa_inspector import QAResult


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


@pytest.fixture(autouse=True)
def _stub_qa_inspector(monkeypatch) -> None:
    monkeypatch.setattr(
        "vigilance.extraction.vision_qa_inspector.VisionTableInspector.inspect_extraction",
        lambda self, image_bytes, extracted_json: QAResult(
            is_perfect=True,
            missing_elements=[],
            justification="test stub",
        ),
    )


def test_quality_pass_accepts_complete_initial_result(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    calls: list[tuple[bytes, bool]] = []

    def fake_extract(**kwargs):
        calls.append((kwargs["crop_bytes"], bool(kwargs.get("rescue_mode"))))
        return _result(
            title="Tableau 1 - Capital",
            summary="Ratios de capital réglementaires",
            indicators=["Ratio CET1", "Ratio Tier 1", "Ratio de levier"],
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


def test_footnote_marker_id_normalizes_parenthetical_and_superscript_forms() -> None:
    assert _normalize_footnote_marker_id("(1)") == "1"
    assert _normalize_footnote_marker_id("¹") == "1"
    assert _normalize_footnote_marker_id("1") == "1"
    assert _normalize_footnote_marker_id("(10)") == "10"


def test_expected_marker_catalog_does_not_force_absent_markers(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    calls: list[tuple[bytes, bool]] = []

    def fake_extract(**kwargs):
        calls.append((kwargs["crop_bytes"], bool(kwargs.get("rescue_mode"))))
        return _result(
            title="Tableau de capital",
            summary="Ratios de capital reglementaires",
            indicators=["Ratio CET1", "Ratio Tier 1", "Ratio de levier"],
            headers=["Mesure", "Valeur"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={"expected_markers": ["(1)", "(2)", "(3)", "(4)"]},
    )

    assert result is not None
    assert result.extraction_status == "ok"
    assert "missing_expected_footnotes" not in result.rejection_reasons
    assert calls == [(b"initial", False)]


def test_observed_markers_accept_mixed_footnote_id_formats(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    calls: list[tuple[bytes, bool]] = []

    def fake_extract(**kwargs):
        calls.append((kwargs["crop_bytes"], bool(kwargs.get("rescue_mode"))))
        return _result(
            title="Ratio de liquidite (1)",
            summary="Ratio de liquidite a court terme",
            indicators=["Actifs liquides", "Sorties de tresorerie (3)", "Ratio"],
            headers=["Valeur non ponderee²", "Valeur ajustee"],
            footnotes=[
                {"id": "1", "text": "Methode de calcul"},
                {"id": "(2)", "text": "Valeurs non ponderees"},
                {"id": "³", "text": "Depots stables"},
            ],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={"expected_markers": ["(1)", "(2)", "(3)"]},
    )

    assert result is not None
    assert result.extraction_status == "ok"
    assert "missing_expected_footnotes" not in result.rejection_reasons
    assert calls == [(b"initial", False)]


def test_observed_missing_marker_still_forces_rescue(monkeypatch) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    calls: list[bool] = []

    def fake_extract(**kwargs):
        rescue_mode = bool(kwargs.get("rescue_mode"))
        calls.append(rescue_mode)
        return _result(
            title="Ratio de liquidite (1)",
            summary="Ratio de liquidite a court terme",
            indicators=["Actifs liquides", "Sorties de tresorerie", "Ratio"],
            headers=["Mesure", "Valeur"],
            footnotes=([{"id": "1", "text": "Methode de calcul"}] if rescue_mode else []),
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={"expected_markers": ["(1)", "(2)", "(3)"]},
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert "missing_expected_footnotes" in result.rejection_reasons
    assert calls == [False, True]


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
    assert result.extraction_status == "rescued"
    assert result.acceptance_reason == "data_richness_override_rescue_exhaustion"


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
    assert result.extraction_status == "rescued"


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
    assert result.extraction_status == "rescued"
    assert (
        "generic_page_title" in result.rejection_reasons or "generic_title_without_support" in result.rejection_reasons
    )
    assert "dominant_contamination" in result.rejection_reasons


# ---------------------------------------------------------------------------
# Guardrail tests: tables with extracted data must NEVER be confirmed_no_table
# ---------------------------------------------------------------------------


class TestDataRichnessOverrideGuardrail:
    """Verrouille le comportement: toute table avec des donnees extraites
    (indicators ou headers non-vides) ne doit JAMAIS etre classee
    confirmed_no_table.  Ces tests sont des garde-fous de non-regression."""

    def test_single_indicator_prevents_confirmed_no_table(self, monkeypatch) -> None:
        """Meme 1 seul indicator suffit a empecher confirmed_no_table."""
        extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

        def fake_extract(**kwargs):
            return _result(
                title="Tableau 99",
                summary="",
                indicators=["Total"],
                headers=[],
                no_table_detected=True,
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
        assert result.extraction_status != "confirmed_no_table", (
            "BUG REGRESSION: une table avec des indicators ne doit JAMAIS etre confirmed_no_table"
        )

    def test_single_header_prevents_confirmed_no_table(self, monkeypatch) -> None:
        """Meme 1 seul header suffit a empecher confirmed_no_table."""
        extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

        def fake_extract(**kwargs):
            return _result(
                title="Tableau X",
                summary="",
                indicators=[],
                headers=["T2-2025"],
                no_table_detected=True,
            )

        monkeypatch.setattr(extractor, "extract", fake_extract)

        result = extractor.extract_with_quality_pass(
            crop_bytes=b"initial",
            bank_code="bmo",
            bbox_norm=[0.1, 0.1, 0.9, 0.7],
            vision_cfg={},
            get_variant_crop_fn=lambda **kwargs: None,
        )

        assert result is not None
        assert result.extraction_status != "confirmed_no_table", (
            "BUG REGRESSION: une table avec des headers ne doit JAMAIS etre confirmed_no_table"
        )

    def test_rich_table_rescued_not_discarded(self, monkeypatch) -> None:
        """Simule un tableau riche (24 indicators, 3 headers) comme le
        TABLEAU 12 de BMO — doit etre rescued, jamais confirmed_no_table."""
        extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

        def fake_extract(**kwargs):
            return _result(
                title="TABLEAU 12",
                summary="Fonds propres réglementaires et TLAC",
                indicators=[f"Indicateur {i}" for i in range(24)],
                headers=["T2-2025", "T1-2025", "T2-2024"],
                no_table_detected=True,
            )

        monkeypatch.setattr(extractor, "extract", fake_extract)

        result = extractor.extract_with_quality_pass(
            crop_bytes=b"initial",
            bank_code="bmo",
            bbox_norm=[0.05, 0.4, 0.94, 0.7],
            vision_cfg={},
            get_variant_crop_fn=lambda **kwargs: None,
        )

        assert result is not None
        assert result.extraction_status != "confirmed_no_table", (
            "BUG REGRESSION: un tableau riche ne doit JAMAIS etre confirmed_no_table"
        )
        assert result.extraction_status in ("ok", "rescued")

    def test_truly_empty_table_remains_confirmed_no_table(self, monkeypatch) -> None:
        """Contre-test: une table VIDE (0 indicators, 0 headers) doit rester
        confirmed_no_table. Le garde-fou ne doit pas tout laisser passer."""
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
        assert result.extraction_status == "confirmed_no_table", (
            "Une table VIDE sans indicators ni headers DOIT rester confirmed_no_table"
        )

    def test_whitespace_only_indicators_count_as_empty(self, monkeypatch) -> None:
        """Des indicators qui ne contiennent que des espaces = vide."""
        extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

        def fake_extract(**kwargs):
            return _result(
                title="Tableau vide",
                summary="",
                indicators=["", "  ", "\n"],
                headers=["", " "],
                no_table_detected=True,
            )

        monkeypatch.setattr(extractor, "extract", fake_extract)

        result = extractor.extract_with_quality_pass(
            crop_bytes=b"initial",
            bank_code="bmo",
            bbox_norm=[0.1, 0.1, 0.9, 0.7],
            vision_cfg={},
            get_variant_crop_fn=lambda **kwargs: None,
        )

        assert result is not None
        assert result.extraction_status == "confirmed_no_table", (
            "Des indicators/headers whitespace-only comptent comme vides"
        )

    @pytest.mark.parametrize(
        "indicators,headers",
        [
            (["Ratio CET1"], []),
            ([], ["T1-2025"]),
            (["Ratio CET1", "Ratio Tier 1"], ["T1-2025", "T2-2025"]),
            ([f"Ind {i}" for i in range(39)], ["Sans échéance", "< 6 mois", "6-12 mois", "> 1 an"]),
        ],
        ids=["one_indicator", "one_header", "small_table", "large_nsfr_table"],
    )
    def test_any_non_empty_data_is_never_confirmed_no_table(self, monkeypatch, indicators, headers) -> None:
        """Parametrise: toute combinaison non-vide doit survivre au garde-fou."""
        extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

        def fake_extract(**kwargs):
            return _result(
                title="Tableau X",
                summary="",
                indicators=indicators,
                headers=headers,
                no_table_detected=True,
            )

        monkeypatch.setattr(extractor, "extract", fake_extract)

        result = extractor.extract_with_quality_pass(
            crop_bytes=b"initial",
            bank_code="cibc",
            bbox_norm=[0.05, 0.2, 0.94, 0.8],
            vision_cfg={},
            get_variant_crop_fn=lambda **kwargs: None,
        )

        assert result is not None
        assert result.extraction_status != "confirmed_no_table", (
            f"BUG REGRESSION: indicators={len(indicators)}, headers={len(headers)} "
            f"ne doit JAMAIS etre confirmed_no_table"
        )


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
