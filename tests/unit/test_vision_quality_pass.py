from __future__ import annotations

from pathlib import Path

import pytest

from vigilance.extraction.vision_full_extractor import (
    OPENAI_VISION_TIMEOUT_SECONDS,
    VisionFullExtractor,
    VisionFullResult,
    _normalize_footnote_marker_id,
    _select_targeted_rescue_variant,
)
from vigilance.extraction.vision_qa_inspector import QAResult
from vigilance.utils.page_layout_context import clamp_variant_crop_to_neighbors


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


def test_vision_client_uses_direct_120_second_timeout(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    extractor._ensure_client()

    assert captured == [
        {
            "api_key": "test-key",
            "timeout": OPENAI_VISION_TIMEOUT_SECONDS,
        }
    ]
    assert OPENAI_VISION_TIMEOUT_SECONDS == 120.0


@pytest.mark.parametrize(
    "reasons,critiques,qa_missing,expected",
    [
        (["missing_expected_footnotes"], [], "", "bottom_extended"),
        ([], ["notes de bas de page manquantes"], "", "bottom_extended"),
        (["top_context_missing_title"], [], "", "top_extended"),
        (["generic_page_title"], [], "", "top_trim"),
        (["dominant_contamination"], [], "", "tight_body"),
        (["low_density_vertical"], [], "", "body_expanded"),
        (["missing_table_summary"], [], "", "same_crop_rescue"),
    ],
)
def test_targeted_rescue_router_selects_one_relevant_variant(
    reasons,
    critiques,
    qa_missing,
    expected,
) -> None:
    assert _select_targeted_rescue_variant(reasons, critiques, qa_missing) == expected


def test_quality_pass_uses_only_bottom_extension_for_missing_footnote(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    variant_calls: list[dict] = []
    inspected_images: list[bytes] = []

    def fake_inspect(self, image_bytes, extracted_json):
        inspected_images.append(image_bytes)
        return QAResult(
            is_perfect=True,
            missing_elements=[],
            justification="complete",
        )

    monkeypatch.setattr(
        "vigilance.extraction.vision_qa_inspector.VisionTableInspector.inspect_extraction",
        fake_inspect,
    )

    def fake_variant_crop(**kwargs) -> bytes:
        variant_calls.append(kwargs)
        return b"bottom_extended"

    def fake_extract(**kwargs):
        if kwargs["crop_bytes"] == b"bottom_extended":
            return _result(
                title="Ratio de liquidite (1)",
                summary="Ratio de liquidite a court terme",
                indicators=["Actifs liquides", "Sorties de tresorerie", "Ratio"],
                headers=["Mesure", "Valeur"],
                footnotes=[{"id": "1", "text": "Methode de calcul"}],
            )
        return _result(
            title="Ratio de liquidite (1)",
            summary="Ratio de liquidite a court terme",
            indicators=["Actifs liquides", "Sorties de tresorerie", "Ratio"],
            headers=["Mesure", "Valeur"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={"expected_markers": ["(1)"]},
        initial_bottom_extension=0.02,
        get_variant_crop_fn=fake_variant_crop,
    )

    assert result is not None
    assert result.selected_candidate_name == "bottom_extended"
    assert variant_calls == [{"bottom_extension": 0.08}]
    assert inspected_images == [b"initial", b"bottom_extended"]


def test_quality_pass_uses_page_context_after_incomplete_targeted_rescue(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    page_context_calls: list[bool] = []

    def fake_variant_crop(**kwargs) -> bytes:
        return b"bottom_extended"

    def fake_page_context_crop() -> dict:
        page_context_calls.append(True)
        return {
            "crop_bytes": b"page_context",
            "bbox_norm": [0.08, 0.18, 0.92, 0.43],
            "bottom_extension": 0.05,
            "confidence": 0.96,
            "title_text": "Ratios de liquidite",
            "continuation": False,
            "table_count": 2,
        }

    def fake_extract(**kwargs):
        crop = kwargs["crop_bytes"]
        if crop == b"page_context":
            return _result(
                title="Ratios de liquidite (1)",
                summary="Ratios reglementaires de liquidite",
                indicators=["Actifs liquides", "Sorties nettes", "Ratio LCR"],
                headers=["Mesure", "T2 2026"],
                footnotes=[{"id": "1", "text": "Methode de calcul complete"}],
            )
        return _result(
            title="Ratios de liquidite (1)",
            summary="Ratios reglementaires de liquidite",
            indicators=["Actifs liquides", "Sorties nettes", "Ratio LCR"],
            headers=["Mesure", "T2 2026"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.42],
        vision_cfg={"expected_markers": ["(1)"]},
        get_variant_crop_fn=fake_variant_crop,
        get_page_context_crop_fn=fake_page_context_crop,
    )

    assert result is not None
    assert result.selected_candidate_name == "page_context_rescue"
    assert result.extraction_status == "rescued"
    assert result.selected_bbox_norm == [0.08, 0.18, 0.92, 0.43]
    assert result.bbox_source == "page_context_locator"
    assert result.bbox_confidence == pytest.approx(0.96)
    assert result.page_context_title == "Ratios de liquidite"
    assert result.page_context_continuation is False
    assert result.page_context_table_count == 2
    assert page_context_calls == [True]


def test_quality_pass_uses_page_context_when_targeted_pass_returns_none(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    page_context_calls: list[bool] = []

    def fake_extract(**kwargs):
        if kwargs["crop_bytes"] == b"page_context":
            return _result(
                title="Prêts douteux bruts",
                summary="Prêts douteux par portefeuille",
                indicators=["Prêts aux particuliers", "Prêts aux entreprises", "Total"],
                headers=["Portefeuille", "T2 2026"],
            )
        return None

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.1, 0.2, 0.9, 0.42],
        get_variant_crop_fn=lambda **_kwargs: b"targeted",
        get_page_context_crop_fn=lambda: (
            page_context_calls.append(True)
            or {
                "crop_bytes": b"page_context",
                "bbox_norm": [0.08, 0.18, 0.92, 0.43],
                "confidence": 0.97,
                "title_text": "Prêts douteux bruts",
                "continuation": False,
                "table_count": 1,
            }
        ),
    )

    assert result is not None
    assert result.selected_candidate_name == "page_context_rescue"
    assert result.selected_bbox_norm == [0.08, 0.18, 0.92, 0.43]
    assert page_context_calls == [True]


def test_quality_pass_accepts_compact_two_row_table_without_self_healing(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")
    extraction_calls: list[bytes] = []

    def fake_extract(**kwargs):
        extraction_calls.append(kwargs["crop_bytes"])
        return _result(
            title="Répartition géographique",
            summary="Répartition par région",
            indicators=["Canada", "Total"],
            headers=["Catégorie", "T1 2026", "T2 2026", "Variation"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.10, 0.20, 0.90, 0.40],
        vision_cfg={},
        get_variant_crop_fn=lambda **_kwargs: b"unexpected_rescue",
    )

    assert result is not None
    assert result.extraction_status == "ok"
    assert result.selected_candidate_name == "initial"
    assert result.recrop_attempted is False
    assert extraction_calls == [b"initial"]


def test_quality_pass_preserves_three_generic_rows_without_summary(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(api_key="test-key", model="gpt-4o-test")

    monkeypatch.setattr(
        extractor,
        "extract",
        lambda **_kwargs: _result(
            title="Répartition géographique",
            summary="",
            indicators=["Canada", "Autres", "Total"],
            headers=["Catégorie", "Valeur"],
        ),
    )

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"initial",
        bank_code="rbc",
        bbox_norm=[0.10, 0.20, 0.90, 0.40],
        vision_cfg={},
        get_variant_crop_fn=lambda **_kwargs: b"same_crop",
    )

    assert result is not None
    assert result.extraction_status == "rescued"
    assert result.indicators == ["Canada", "Autres", "Total"]
    assert result.acceptance_reason == "rescued_without_summary_strong_structure"


def test_variant_crop_is_clamped_before_a_close_following_table() -> None:
    table_bbox = [0.1, 0.1, 0.9, 0.4]
    next_table_top = 0.413
    page_map = {
        30: [
            (1, table_bbox),
            (2, [0.1, next_table_top, 0.9, 0.8]),
        ]
    }

    safe_bbox, safe_bottom, _safe_top = clamp_variant_crop_to_neighbors(
        table_idx=1,
        page_num=30,
        table_bbox=table_bbox,
        page_table_map=page_map,
        bottom_extension=0.06,
    )

    assert safe_bbox[3] + safe_bottom <= next_table_top - 0.005 + 1e-9


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


def test_extract_ignores_cached_result_without_structural_rows(
    monkeypatch,
) -> None:
    extractor = VisionFullExtractor(
        api_key="test-key",
        model="gpt-4o-test",
        use_cache=True,
    )
    monkeypatch.setattr(
        "vigilance.extraction.vision_full_extractor.cache_get",
        lambda *_args: {
            "table_title": "Correction de valeur",
            "table_summary": "",
            "headers": [],
            "indicators": [],
            "footnotes_content": [],
            "no_table_detected": True,
        },
    )
    attempted: list[bool] = []

    def fake_ensure_client() -> None:
        attempted.append(True)
        extractor._client = None

    monkeypatch.setattr(extractor, "_ensure_client", fake_ensure_client)

    result = extractor.extract(
        crop_bytes=b"retry-this-crop",
        bank_code="rbc",
        pdf_sha="same-pdf",
        page_number=33,
        bbox_norm=[0.1, 0.2, 0.9, 0.8],
        vision_cfg={},
    )

    assert result is None
    assert attempted == [True]


def test_quality_pass_cache_preserves_final_self_healing_decision(
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
    calls: list[bytes] = []

    def fake_extract(**kwargs):
        calls.append(kwargs["crop_bytes"])
        return _result(
            title="Petit tableau",
            summary="Répartition géographique",
            indicators=["Canada", "Autres", "Total"],
            headers=["Région", "T2 2026"],
        )

    monkeypatch.setattr(extractor, "extract", fake_extract)
    kwargs = {
        "crop_bytes": b"stable-crop",
        "bank_code": "rbc",
        "pdf_sha": "same-pdf",
        "page_number": 9,
        "bbox_norm": [0.1, 0.2, 0.9, 0.8],
        "vision_cfg": {},
    }

    first = extractor.extract_with_quality_pass(**kwargs)
    second = extractor.extract_with_quality_pass(**kwargs)

    assert first is not None
    assert second is not None
    assert first.indicators == ["Canada", "Autres", "Total"]
    assert second.indicators == first.indicators
    assert second.extraction_status == first.extraction_status
    assert second.qa_inspected is True
    assert calls == [b"stable-crop"]


def test_quality_pass_does_not_cache_unresolved_empty_candidate(
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
    calls: list[bytes] = []

    def fake_extract(**kwargs):
        calls.append(kwargs["crop_bytes"])
        return _result(no_table_detected=True)

    monkeypatch.setattr(extractor, "extract", fake_extract)
    kwargs = {
        "crop_bytes": b"unresolved-crop",
        "bank_code": "rbc",
        "pdf_sha": "same-pdf",
        "page_number": 31,
        "bbox_norm": [0.1, 0.2, 0.9, 0.8],
        "vision_cfg": {},
    }

    first = extractor.extract_with_quality_pass(**kwargs)
    calls_after_first = len(calls)
    second = extractor.extract_with_quality_pass(**kwargs)

    assert first is not None
    assert second is not None
    assert first.indicators == []
    assert second.indicators == []
    assert calls_after_first > 0
    assert len(calls) > calls_after_first
