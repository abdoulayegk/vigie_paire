from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigilance.text_analysis_pipeline import (
    _align_chunks_tfidf,
    _build_comparison_batches,
    _format_alignments_for_prompt,
    ChunkAlignment,
    ChunkCandidate,
    PDFBlock,
    ResolvedSection,
    SectionAudit,
    SemanticUnit,
    _allowed_target_sections,
    _build_section_audit,
    _build_global_summary,
    _build_text_extraction_markdown,
    _call_json_completion,
    _chunk_subsection_text,
    _classify_block_type,
    _compare_section_texts,
    _default_triage,
    _derive_legacy_fields,
    _extract_audits_for_pdf,
    _extract_section_text_from_markdown,
    _FEW_SHOT_TRIAGE_AMF,
    _format_page_marker,
    _format_page_suffix,
    _gpt_match_orphan_headings,
    _is_new_major_or_allowed_moderate,
    _is_non_cosmetic_change,
    _looks_like_footnote,
    _max_output_tokens_for_model,
    _normalize_heading,
    _pair_subsections,
    _parse_page_index_from_markdown,
    _parse_subsections,
    _resolve_orphan_subsections,
    _resolve_sections,
    _rewrite_page_markers_for_display,
    _sanitize_semantic_text,
    _section_window_for_page,
    TextAnalysisQualityError,
    run_text_analysis_pipeline,
)
from vigilance.text_analysis.chunk_alignment import _tfidf_similarity_matrix_from_texts
from vigilance.text_analysis.semantic_chunking import SemanticChunkingError
from vigilance.text_analysis.comparison import (
    ChunkComparisonLLMResponse,
    _attach_alignment_metadata,
    _materialize_semantic_alignment_decisions,
)
from vigilance.text_analysis.global_reconciliation import (
    _ReconciliationResponse,
    _components,
    _one_sided_nodes,
    reconcile_global_change_fragments,
)
from vigilance.text_comparison.change_segments import build_change_segments_from_texts
from vigilance.text_analysis.subsection_matching import OrphanMatchLLMResponse, OrphanSubsection
from vigilance.text_extraction.text_extraction_markdown_writer import (
    get_raw_docling_markdown_path,
    has_current_text_extraction_cache_schema,
    stamp_text_extraction_cache_schema,
)
from vigilance.text_analysis.docling_markdown import (
    DoclingSegment,
    _assign_segments_to_sections,
    _build_text_extraction_markdown_from_docling,
    _matchable_section_segments,
    _parse_docling_markdown,
    _should_keep_docling_segment,
)
from vigilance.text_analysis.extraction import (
    _augment_table_regions_with_composite_grids,
    _docling_page_batches,
)
from vigilance.text_analysis.normalization import _infer_table_footnote_bboxes
from vigilance.text_analysis.markdown import _is_out_of_scope_accounting_heading


class _FakeChoice:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.message = type("FakeMessage", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.choices = [_FakeChoice(content, finish_reason=finish_reason)]


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.max_completion_tokens_seen: list[int | None] = []

    def create(self, **kwargs):
        self.max_completion_tokens_seen.append(kwargs.get("max_completion_tokens"))
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.chat = _FakeChat(responses)


def test_sanitize_semantic_text_removes_numbers_and_regulatory_refs() -> None:
    raw = "En 2026, le ratio CET1 atteint 13,2 % selon OSFI et la banque renforce sa stratégie de capital."

    cleaned = _sanitize_semantic_text(raw)

    assert "2026" not in cleaned
    assert "13,2" not in cleaned
    assert "CET1" not in cleaned
    assert "OSFI" not in cleaned
    assert "stratégie de capital" in cleaned


def test_sanitize_semantic_text_rephrases_regulatory_frameworks() -> None:
    raw = "Le Groupe a mis en œuvre des réformes de III et une ligne directrice sur le levier avec un coussin de ratio de levier de %."

    cleaned = _sanitize_semantic_text(raw)

    assert "III" not in cleaned
    assert "ligne directrice" not in cleaned.lower()
    assert "%" not in cleaned
    assert "La banque a" in cleaned
    assert "exigences de levier" in cleaned


def test_sanitize_semantic_text_expands_residual_acronyms() -> None:
    raw = "L'approche fondée sur des indicateurs répartit les treize indicateurs en cinq catégories pour les BISM et améliore la VaR."

    cleaned = _sanitize_semantic_text(raw)

    assert "BISM" not in cleaned
    assert "VaR" not in cleaned
    assert "banques d'importance systémique" in cleaned
    assert "mesure de risque de marché" in cleaned


def test_keep_change_for_major_relevant() -> None:
    triage = {
        "is_relevant": True,
        "impact_level": "MAJEUR",
        "nouvelle_idee": False,
        "themes_amf": ["MODIFICATION_METHODOLOGIE"],
    }

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_allowed_target_sections_match_bank_matrix() -> None:
    assert _allowed_target_sections("bnc") == {"gestion_capital", "gestion_risques"}
    assert _allowed_target_sections("td") == {"gestion_capital", "gestion_risques"}
    assert _allowed_target_sections("rbc") == {
        "gestion_capital",
        "gestion_risques",
        "gestion_reglementation",
    }


def test_resolve_sections_ignores_regulatory_for_bnc(monkeypatch, tmp_path: Path) -> None:
    class _FakeItem:
        def __init__(self, section_type: str, start_page: int, end_page: int):
            self.section_type = section_type
            self.start_page = start_page
            self.end_page = end_page
            self.anchor_page = start_page
            self.anchor_text = section_type
            self.anchor_bbox_norm = [0.1, 0.2, 0.8, 0.25]

    class _FakeMapping:
        sections = [
            _FakeItem("capital_management", 10, 20),
            _FakeItem("risk_management", 21, 40),
            _FakeItem("regulatory_updates", 41, 43),
        ]

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline.locate_sections_in_pdf",
        lambda pdf_path, bank_code=None, quarter=None, year=2025: _FakeMapping(),
    )

    resolved = _resolve_sections(tmp_path / "dummy.pdf", "bnc")

    assert set(resolved) == {"gestion_capital", "gestion_risques"}


def test_resolve_sections_passes_t4_context_and_filters_regulatory(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class _FakeItem:
        def __init__(self, section_type: str, start_page: int, end_page: int):
            self.section_type = section_type
            self.start_page = start_page
            self.end_page = end_page
            self.anchor_page = start_page
            self.anchor_text = section_type
            self.anchor_bbox_norm = [0.1, 0.2, 0.8, 0.25]

    class _FakeMapping:
        sections = [
            _FakeItem("capital_management", 10, 20),
            _FakeItem("risk_management", 21, 40),
            _FakeItem("regulatory_updates", 41, 43),
        ]

    def fake_locate_sections_in_pdf(
        pdf_path: str,
        bank_code: str | None = None,
        quarter: str | None = None,
        year: int = 2025,
    ) -> _FakeMapping:
        captured.update(
            {
                "pdf_path": pdf_path,
                "bank_code": bank_code,
                "quarter": quarter,
                "year": year,
            }
        )
        return _FakeMapping()

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline.locate_sections_in_pdf",
        fake_locate_sections_in_pdf,
    )

    resolved = _resolve_sections(
        tmp_path / "dummy.pdf",
        "bmo",
        quarter="t4",
        year=2025,
    )

    assert captured["bank_code"] == "bmo"
    assert captured["quarter"] == "t4"
    assert captured["year"] == 2025
    assert set(resolved) == {"gestion_capital", "gestion_risques"}


def test_keep_change_for_new_moderate_with_strong_amf_theme() -> None:
    """Un changement MODERE est retenu s'il porte un thème AMF fort
    (NOUVELLE_MENTION_REGLEMENTAIRE, MODIFICATION_METHODOLOGIE, ...).
    """
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "themes_amf": ["NOUVELLE_MENTION_REGLEMENTAIRE"],
    }

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_keep_change_for_new_moderate_with_risque_emergent_theme() -> None:
    """Un changement MODERE sur RISQUE_EMERGENT (cyber, IA, fraude) doit toujours être retenu."""
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "themes_amf": ["RISQUE_EMERGENT"],
    }

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_drop_non_substantive_moderate_change() -> None:
    """Un MODERE sans nouvelle idée et sans thème fort est rejeté."""
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "themes_amf": ["DIVULGATION_AJOUT"],
    }

    assert _is_new_major_or_allowed_moderate(triage) is False


def test_default_triage_includes_amf_v2_and_legacy_fields() -> None:
    """Le triage par défaut produit le schéma AMF v2 + champs hérités pour rétro-compatibilité."""
    triage = _default_triage()

    assert triage["source"] == "gpt4o_triage_amf_compact_v1"
    assert triage["themes_amf"] == []
    assert triage["exclusion_reason"] == "non_pertinent_autre"
    assert triage["is_relevant"] is False
    assert triage["category"] == "NON_PERTINENT"
    assert triage["confidence"] == 0.0
    assert triage["impact_it"] == "INDETERMINE"
    assert triage["changement_posture"] == "AUCUN"
    assert triage["justification_posture"] == ""
    assert triage["statut_mise_en_oeuvre"] == "INDETERMINE"
    assert triage["confiance_posture"] == "INDETERMINE"
    assert triage["signals"]["methodology_change"] is False
    assert count_complete_sentences(triage["relevance_reason"]) == 2


def test_triage_few_shots_request_compact_relevance_reason() -> None:
    assert "relevance_reason" in _FEW_SHOT_TRIAGE_AMF
    assert "impact_it" not in _FEW_SHOT_TRIAGE_AMF
    assert "justification_posture" not in _FEW_SHOT_TRIAGE_AMF
    assert _FEW_SHOT_TRIAGE_AMF.count("Exemple ") == 9
    assert "transfert de responsabilité de gouvernance" in _FEW_SHOT_TRIAGE_AMF
    assert "comité renommé pertinent" in _FEW_SHOT_TRIAGE_AMF
    outputs = [
        json.loads(line.removeprefix("Output : "))
        for line in _FEW_SHOT_TRIAGE_AMF.splitlines()
        if line.startswith("Output : ")
    ]
    assert len(outputs) == 9
    for output in outputs:
        validated = TriageAMFCompactLLMResultWithIndex(**output)
        assert count_complete_sentences(validated.relevance_reason) == 2
        assert (
            "Ce changement est pertinent pour la vigie AMF"
            not in validated.relevance_reason
        )
        assert "Ce changement n’est pas pertinent" not in validated.relevance_reason


def test_derive_legacy_fields_maps_methodology_theme() -> None:
    """MODIFICATION_METHODOLOGIE doit activer signals.methodology_change."""
    legacy = _derive_legacy_fields(
        {
            "is_relevant": True,
            "themes_amf": ["MODIFICATION_METHODOLOGIE", "EXIGENCES_REGLEMENTAIRES"],
            "impact_level": "MAJEUR",
        }
    )

    assert legacy["signals"]["methodology_change"] is True
    assert legacy["category"] == "REGLEMENTAIRE"
    assert legacy["risk_level"] == "ELEVEE"


def test_derive_legacy_fields_maps_risque_emergent_to_risque_category() -> None:
    """RISQUE_EMERGENT doit mapper sur la catégorie héritée RISQUE."""
    legacy = _derive_legacy_fields(
        {
            "is_relevant": True,
            "themes_amf": ["RISQUE_EMERGENT", "GOUVERNANCE_RISQUES"],
            "impact_level": "MAJEUR",
        }
    )

    assert legacy["category"] == "RISQUE"


def test_derive_legacy_fields_maps_data_and_cloud_to_risque_category() -> None:
    legacy = _derive_legacy_fields(
        {
            "is_relevant": True,
            "themes_amf": ["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"],
            "impact_level": "MODERE",
        }
    )

    assert legacy["category"] == "RISQUE"


def test_derive_legacy_fields_maps_montant_reglementaire_to_quantitative_signal() -> None:
    """MONTANT_REGLEMENTAIRE active signals.quantitative_changed."""
    legacy = _derive_legacy_fields(
        {
            "is_relevant": True,
            "themes_amf": ["RATIOS_REGLEMENTAIRES", "MONTANT_REGLEMENTAIRE"],
            "impact_level": "MAJEUR",
        }
    )

    assert legacy["signals"]["quantitative_changed"] is True
    assert legacy["category"] == "CAPITAL"


def test_is_non_cosmetic_change_rejects_irrelevant_triage() -> None:
    """Un triage non pertinent (themes_amf vide) est rejeté de la rétention."""
    triage = {"is_relevant": False, "themes_amf": []}

    assert _is_non_cosmetic_change(triage) is False


def test_is_non_cosmetic_change_keeps_relevant_with_themes() -> None:
    """Un triage pertinent avec au moins un thème AMF est retenu."""
    triage = {"is_relevant": True, "themes_amf": ["DIVULGATION_AJOUT"]}

    assert _is_non_cosmetic_change(triage) is True


def test_call_json_completion_retries_with_larger_token_budget_after_truncation() -> None:
    client = _FakeClient(
        responses=[
            _FakeResponse('{"changes":[{"diff_type":"added","text_t1":"","text_t2":"texte tronqué', finish_reason="length"),
            _FakeResponse('{"changes":[{"diff_type":"added","text_t1":"","text_t2":"texte complet","change_summary":"Ajout."}]}'),
        ]
    )

    payload = _call_json_completion(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Compare"}],
        max_tokens=100,
    )

    assert payload["changes"][0]["text_t2"] == "texte complet"
    assert client.chat.completions.max_completion_tokens_seen == [100, _max_output_tokens_for_model("gpt-4o")]


def test_call_json_completion_uses_model_max_by_default() -> None:
    client = _FakeClient(
        responses=[
            _FakeResponse('{"changes":[]}')
        ]
    )

    payload = _call_json_completion(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Compare"}],
    )

    assert payload == {"changes": []}
    assert client.chat.completions.max_completion_tokens_seen == [None]


def test_compare_section_texts_surfaces_section_key_on_json_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._call_structured_completion_with_correction",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid structured output from model")),
    )

    with pytest.raises(RuntimeError, match="gestion_risques"):
        _compare_section_texts(
            client=object(),
            model="gpt-4o",
            section_key="gestion_risques",
            text_t1="Texte T1",
            text_t2="Texte T2",
        )


def test_pipeline_retains_non_cosmetic_changes_and_discards_cosmetic(monkeypatch, tmp_path: Path) -> None:
    pdf_previous = tmp_path / "prev.pdf"
    pdf_current = tmp_path / "curr.pdf"
    pdf_previous.write_bytes(b"%PDF-1.4 prev")
    pdf_current.write_bytes(b"%PDF-1.4 curr")

    section = ResolvedSection(
        section_key="gestion_risques",
        title="Gestion des risques",
        start_page=1,
        end_page=2,
        anchor_page=1,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
    )
    audit_prev = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=1,
        end_page=1,
        anchor_page=1,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
        included_blocks=[PDFBlock("p001_b001", 1, [0.1, 0.3, 0.9, 0.4], "Texte exact T1", 1, "narrative", True, "")],
        excluded_blocks=[],
    )
    audit_curr = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=2,
        end_page=2,
        anchor_page=2,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
        included_blocks=[PDFBlock("p002_b001", 2, [0.1, 0.3, 0.9, 0.4], "Texte exact T2", 1, "narrative", True, "")],
        excluded_blocks=[],
    )

    monkeypatch.setattr("vigilance.text_analysis_pipeline._build_openai_client", lambda: object())
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._resolve_sections",
        lambda pdf_path, bank_code, quarter=None, year=None: {"gestion_risques": section},
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._extract_audits_for_pdf",
        lambda **kwargs: ([audit_prev], "") if "prev" in str(kwargs["pdf_path"]) else ([audit_curr], ""),
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_section_texts",
        lambda **kwargs: [
            {"change_id": "c1", "section_key": "gestion_risques", "diff_type": "added",
             "semantic_text_t1": "", "semantic_text_t2": "Nouvelle idee", "source_text_t1": "",
             "source_text_t2": "Nouvelle idee", "source_block_ids_t1": [], "source_block_ids_t2": [],
             "source_refs_t1": [], "source_refs_t2": [], "pages_t1": [], "pages_t2": [],
             "source_resolution_t1": "markdown", "source_resolution_t2": "markdown",
             "evidence_t1": {"pages": [], "snippet": ""}, "evidence_t2": {"pages": [], "snippet": ""},
             "change_summary": "Ajout."},
            {"change_id": "c2", "section_key": "gestion_risques", "diff_type": "modified",
             "semantic_text_t1": "Avant", "semantic_text_t2": "Après", "source_text_t1": "Avant",
             "source_text_t2": "Après", "source_block_ids_t1": [], "source_block_ids_t2": [],
             "source_refs_t1": [], "source_refs_t2": [], "pages_t1": [], "pages_t2": [],
             "source_resolution_t1": "markdown", "source_resolution_t2": "markdown",
             "evidence_t1": {"pages": [], "snippet": ""}, "evidence_t2": {"pages": [], "snippet": ""},
             "change_summary": "Modification."},
            {"change_id": "c3", "section_key": "gestion_risques", "diff_type": "modified",
             "semantic_text_t1": "Non substantif avant", "semantic_text_t2": "Non substantif apres",
             "source_text_t1": "Non substantif avant", "source_text_t2": "Non substantif apres",
             "source_block_ids_t1": [], "source_block_ids_t2": [], "source_refs_t1": [],
             "source_refs_t2": [], "pages_t1": [], "pages_t2": [],
             "source_resolution_t1": "markdown", "source_resolution_t2": "markdown",
             "evidence_t1": {"pages": [], "snippet": ""}, "evidence_t2": {"pages": [], "snippet": ""},
             "change_summary": "Reformulation non substantive."},
        ],
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._triage_section_changes",
        lambda **kwargs: [
            {
                **kwargs["changes"][0],
                "genai_triage": {
                    "is_relevant": True,
                    "themes_amf": ["DIVULGATION_AJOUT", "RISQUE_EMERGENT"],
                    "impact_level": "MODERE",
                    "nouvelle_idee": True,
                    "explanation": "",
                    "action_requise": "information",
                    "exclusion_reason": None,
                    "category": "RISQUE",
                    "signals": {"regulatory_reference_added": False, "methodology_change": False},
                    "source": "gpt4o_triage_amf_v2",
                },
            },
            {
                **kwargs["changes"][1],
                "genai_triage": {
                    "is_relevant": True,
                    "themes_amf": ["MODIFICATION_TEXTE_RISQUE"],
                    "impact_level": "MAJEUR",
                    "nouvelle_idee": False,
                    "explanation": "",
                    "action_requise": "revue_prioritaire",
                    "exclusion_reason": None,
                    "category": "RISQUE",
                    "signals": {"regulatory_reference_added": False, "methodology_change": False},
                    "source": "gpt4o_triage_amf_v2",
                },
            },
            {
                **kwargs["changes"][2],
                "genai_triage": {
                    "is_relevant": False,
                    "themes_amf": [],
                    "impact_level": "MINEUR",
                    "nouvelle_idee": False,
                    "explanation": "",
                    "action_requise": "aucune",
                    "exclusion_reason": "reformulation_mineure",
                    "category": "NON_PERTINENT",
                    "signals": {"regulatory_reference_added": False, "methodology_change": False},
                    "source": "gpt4o_triage_amf_v2",
                },
            },
        ],
    )

    payload, _out_path = run_text_analysis_pipeline(
        bank_code="td",
        year_current=2025,
        quarter_current="t2",
        pdf_previous=pdf_previous,
        pdf_current=pdf_current,
        out_root=tmp_path / "outputs",
        model="gpt-4o",
    )

    section_payload = payload["section_comparisons"][0]
    retained_ids = [c["change_id"] for c in section_payload["block_comparisons"]]
    all_ids = [c["change_id"] for c in section_payload["all_block_comparisons"]]
    assert "c1" in retained_ids
    assert "c2" in retained_ids
    assert "c3" not in retained_ids
    assert set(all_ids) == {"c1", "c2", "c3"}
    assert section_payload["summary"]["retained_changes"] == 2
    assert payload["global_summary"]["counts"]["total_relevant"] == 2
    assert payload["pipeline"] == "gpt4o_markdown_source_of_truth"


def test_build_global_summary_distinguishes_detected_and_relevant_changes() -> None:
    summary = _build_global_summary(
        [
            {
                "block_comparisons": [
                    {
                        "change_summary": "Ajout réglementaire.",
                        "genai_triage": {
                            "is_relevant": True,
                            "impact_level": "MAJEUR",
                            "category": "REGLEMENTAIRE",
                            "action_requise": "revue_prioritaire",
                        },
                    },
                    {
                        "change_summary": "Changement de date.",
                        "genai_triage": {
                            "is_relevant": False,
                            "impact_level": "MINEUR",
                            "category": "NON_PERTINENT",
                            "action_requise": "aucune",
                        },
                    },
                ]
            }
        ]
    )

    assert summary["counts"]["total"] == 2
    assert summary["counts"]["total_detected"] == 2
    assert summary["counts"]["total_relevant"] == 1
    assert "2 changement(s) textuel(s) détecté(s)" in summary["executive_overview"]
    assert "dont 1 substantiel(s)" in summary["executive_overview"]
    assert summary["pertinence_globale"] == "MOYENNE"


def test_section_window_starts_after_anchor_and_stops_before_next_anchor_same_page() -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=10,
        end_page=10,
        anchor_page=10,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.30, 0.8, 0.35],
    )
    next_section = ResolvedSection(
        section_key="gestion_risques",
        title="Gestion des risques",
        start_page=10,
        end_page=12,
        anchor_page=10,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.72, 0.8, 0.77],
    )

    top, bottom = _section_window_for_page(section, 10, next_section)

    assert top == 0.35
    assert bottom == 0.72


def test_section_window_uses_end_anchor_without_next_section() -> None:
    section = ResolvedSection(
        section_key="gestion_risques",
        title="Gestion des risques",
        start_page=84,
        end_page=129,
        anchor_page=84,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
        end_anchor_page=129,
        end_anchor_text="NORMES ET MÉTHODES COMPTABLES",
        end_anchor_bbox_norm=[0.1, 0.65, 0.9, 0.68],
    )

    top, bottom = _section_window_for_page(section, 129, next_section=None)

    assert top == 0.0
    assert bottom == 0.65


def test_assign_segments_stops_at_end_boundary_heading() -> None:
    risk_paragraph = (
        "Les cibles de réduction des émissions de GES sont calculées ou ses cibles "
        "en matière d'émissions de GES sont établies conformément aux normes."
    )
    accounting_paragraph = (
        "Les méthodes comptables significatives utilisées pour préparer les états financiers "
        "consolidés sont décrites ci-dessous."
    )
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=128,
        end_page=129,
        anchor_page=84,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
        included_blocks=[
            PDFBlock(
                "p129_b001",
                129,
                [0.1, 0.20, 0.9, 0.25],
                risk_paragraph,
                1,
                "narrative",
                True,
                "",
            ),
            PDFBlock(
                "p129_b002",
                129,
                [0.1, 0.40, 0.9, 0.45],
                "Faits nouveaux et événements subséquents",
                2,
                "other",
                False,
                "",
                "section_header",
                heading_level=3,
            ),
        ],
        excluded_blocks=[],
        end_anchor_page=129,
        end_anchor_text="NORMES ET MÉTHODES COMPTABLES",
        end_anchor_bbox_norm=[0.1, 0.65, 0.9, 0.68],
    )
    segments = [
        DoclingSegment(kind="paragraph", text=risk_paragraph),
        DoclingSegment(kind="heading", text="Faits nouveaux et événements subséquents", heading_level=3),
        DoclingSegment(kind="heading", text="NORMES ET MÉTHODES COMPTABLES", heading_level=2),
        DoclingSegment(kind="paragraph", text=accounting_paragraph),
    ]

    assigned = _assign_segments_to_sections(segments, [audit])
    risk_segments = assigned["gestion_risques"]

    assert any(segment.text == risk_paragraph for segment in risk_segments)
    assert not any(segment.text == accounting_paragraph for segment in risk_segments)
    assert not any("NORMES ET MÉTHODES COMPTABLES" in segment.text for segment in risk_segments)


def test_assign_segments_keeps_internal_accounting_heading_before_later_risk_content() -> None:
    introductory_paragraph = "La Banque surveille les facteurs susceptibles d'avoir une incidence sur ses activités."
    internal_accounting_heading = "Conventions, méthodes et estimations comptables utilisées par la Banque"
    internal_accounting_paragraph = (
        "Les conventions utilisées par la Banque exigent des estimations portant sur des questions incertaines."
    )
    credit_heading = "Risque de crédit"
    credit_paragraph = "Le risque de crédit représente la possibilité de subir une perte financière."
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=72,
        end_page=118,
        anchor_page=72,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
        included_blocks=[
            PDFBlock(
                "p082_b001",
                82,
                [0.1, 0.20, 0.9, 0.25],
                introductory_paragraph,
                1,
                "narrative",
                True,
                "",
            ),
            PDFBlock(
                "p083_b002",
                83,
                [0.1, 0.30, 0.9, 0.35],
                internal_accounting_paragraph,
                2,
                "narrative",
                True,
                "",
            ),
            PDFBlock(
                "p084_b002",
                84,
                [0.1, 0.30, 0.9, 0.35],
                credit_paragraph,
                2,
                "narrative",
                True,
                "",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p083_b001",
                83,
                [0.1, 0.20, 0.9, 0.25],
                internal_accounting_heading,
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
            PDFBlock(
                "p084_b001",
                84,
                [0.1, 0.20, 0.9, 0.25],
                credit_heading,
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
        ],
    )
    segments = [
        DoclingSegment(kind="paragraph", text=introductory_paragraph),
        DoclingSegment(kind="heading", text=internal_accounting_heading, heading_level=2),
        DoclingSegment(kind="paragraph", text=internal_accounting_paragraph),
        DoclingSegment(kind="heading", text=credit_heading, heading_level=2),
        DoclingSegment(kind="paragraph", text=credit_paragraph),
    ]

    assigned = _assign_segments_to_sections(segments, [audit])
    risk_segments = assigned["gestion_risques"]

    assert any(segment.text == internal_accounting_paragraph for segment in risk_segments)
    assert any(segment.text == credit_heading for segment in risk_segments)
    assert any(segment.text == credit_paragraph for segment in risk_segments)


def test_assign_segments_stops_at_td_accounting_heading_before_declared_end_page() -> None:
    risk_paragraph = "La Banque continue de surveiller les risques environnementaux et sociaux."
    accounting_heading = "Normes et méthodes comptables"
    accounting_paragraph = "La direction doit exercer son jugement pour évaluer les méthodes comptables."
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=84,
        end_page=132,
        anchor_page=84,
        anchor_text="Facteurs de risque et gestion des risques",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
        included_blocks=[
            PDFBlock(
                "p130_b001",
                130,
                [0.1, 0.20, 0.9, 0.25],
                risk_paragraph,
                1,
                "narrative",
                True,
                "",
            ),
            PDFBlock(
                "p132_b001",
                132,
                [0.1, 0.20, 0.9, 0.25],
                accounting_paragraph,
                1,
                "narrative",
                True,
                "",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p131_b001",
                131,
                [0.1, 0.20, 0.9, 0.25],
                accounting_heading,
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
        ],
    )
    segments = [
        DoclingSegment(kind="paragraph", text=risk_paragraph),
        DoclingSegment(kind="heading", text=accounting_heading, heading_level=2),
        DoclingSegment(kind="paragraph", text=accounting_paragraph),
    ]

    assigned = _assign_segments_to_sections(segments, [audit])
    risk_segments = assigned["gestion_risques"]

    assert any(segment.text == risk_paragraph for segment in risk_segments)
    assert not any(segment.text == accounting_paragraph for segment in risk_segments)


def test_is_out_of_scope_accounting_heading_detects_note_titles() -> None:
    titles = [
        "CONSOLIDATION DES ENTITÉS STRUCTURÉES",
        "Transactions entre parties liées",
        "Convention sur les comptes de dépôt assurés",
        "Instruments financiers",
        "Méthodes comptables utilisées par la Banque",
        "NORMES ET MÉTHODES COMPTABLES",
        "Jugements, estimations et hypothèses comptables",
    ]

    assert all(_is_out_of_scope_accounting_heading(title) for title in titles)


def test_is_out_of_scope_accounting_heading_allows_risk_titles() -> None:
    titles = [
        "Risque de crédit",
        "Gouvernance des risques",
        "FACTEURS DE RISQUE ET GESTION DES RISQUES",
        "Risque opérationnel",
        "Appétit pour le risque",
    ]

    assert not any(_is_out_of_scope_accounting_heading(title) for title in titles)


def test_should_keep_docling_segment_rejects_accounting_heading() -> None:
    segment = DoclingSegment(
        kind="heading",
        text="CONSOLIDATION DES ENTITÉS STRUCTURÉES",
        heading_level=3,
    )

    assert _should_keep_docling_segment(segment, audits=None) is False


def test_should_keep_docling_segment_keeps_audited_regulatory_paragraph_despite_ratios() -> None:
    """Les pourcentages et le mot « total » ne doivent pas annuler l'audit PDF."""
    paragraph = (
        "Le Bureau du surintendant des institutions financières Canada exige des institutions "
        "de dépôt qu'elles atteignent des exigences minimales de 7 %, de 8,5 % et de 10,5 % "
        "pour les actions ordinaires et assimilées de T1, les fonds propres de T1 et le total "
        "des fonds propres. Les exigences du premier pilier sont de 8,0 %, de 9,5 % et de 11,5 %."
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=58,
        end_page=58,
        anchor_page=58,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock(
                "p058_d007",
                58,
                [0.1, 0.57, 0.9, 0.67],
                paragraph,
                7,
                "narrative",
                True,
            )
        ],
        excluded_blocks=[],
    )
    segment = DoclingSegment(kind="paragraph", text=paragraph)

    assert _should_keep_docling_segment(segment, audits=None) is True
    assert _should_keep_docling_segment(segment, audits=[audit]) is True


@pytest.mark.parametrize(
    ("bank", "paragraph"),
    [
        (
            "bmo",
            "Le total des capitaux propres a augmenté de 8,2 milliards de dollars depuis le 31 octobre 2023, "
            "et les actions ordinaires ont progressé de 1,0 milliard pendant l'exercice.",
        ),
        (
            "bnc",
            "Le total de l'actif pondéré en fonction des risques ne doit pas être inférieur à 72,5 % du total "
            "calculé selon les approches standardisées, après un coefficient de 67,5 % en 2024.",
        ),
        (
            "bns",
            "Au 31 octobre 2024, le total des actifs s'élevait à 1 412 milliards de dollars, en hausse de "
            "1 milliard par rapport à 2023, malgré une baisse de 26 milliards des dépôts auprès des banques centrales.",
        ),
        (
            "cibc",
            "Au 31 octobre 2024, le total de l'actif avait augmenté de 66,3 G$, ou 7 %, par rapport à 2023, "
            "dont environ 1,4 G$ attribuable à l'appréciation du dollar américain.",
        ),
        (
            "rbc",
            "Au 31 octobre 2024, le risque de perte maximal relativement aux fiducies non consolidées se chiffrait "
            "à 3 milliards de dollars, comparativement à 3 milliards au 31 octobre 2023.",
        ),
        (
            "td",
            "Selon Bâle III, le total des fonds propres comprend trois composantes et les ratios de catégorie 1, "
            "de catégorie 2 et du total sont calculés par rapport aux actifs pondérés en fonction des risques.",
        ),
    ],
)
def test_should_keep_audited_narrative_samples_from_every_bank(bank: str, paragraph: str) -> None:
    """Les paragraphes chiffrés observés dans les six banques restent comparables."""
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=58,
        end_page=58,
        anchor_page=58,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock(
                f"{bank}_p058_d001",
                58,
                [0.1, 0.3, 0.9, 0.4],
                paragraph,
                1,
                "narrative",
                True,
            )
        ],
        excluded_blocks=[],
    )
    segment = DoclingSegment(kind="paragraph", text=paragraph)

    assert _should_keep_docling_segment(segment, audits=None) is True
    assert _should_keep_docling_segment(segment, audits=[audit]) is True


@pytest.mark.parametrize(
    ("bank", "note", "follows_table", "requires_audited_footnote"),
    [
        ("bmo", "1 Les réserves de fonds propres sont calculées conformément à la ligne directrice du BSIF.", False, True),
        ("bnc", "(1) Les ratios sont calculés selon les exigences de Bâle III publiées par le BSIF.", False, False),
        ("bns", "1) Les montants des périodes précédentes ont été retraités afin de refléter une nouvelle norme.", False, False),
        ("cibc", "i) Les expositions sur dérivés sont présentées selon les règles réglementaires applicables.", True, False),
        ("rbc", "¹ Se reporter aux notes afférentes aux états financiers consolidés pour plus de précisions.", False, False),
        ("td", "Note : le ratio de levier est calculé conformément aux exigences de levier du BSIF.", False, False),
    ],
)
def test_note_formats_are_kept_unless_confirmed_as_table_notes(
    bank: str,
    note: str,
    follows_table: bool,
    requires_audited_footnote: bool,
) -> None:
    """Un marqueur de note seul ne doit pas retirer un texte de la comparaison."""
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=58,
        end_page=58,
        anchor_page=58,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=(
            []
            if requires_audited_footnote
            else [PDFBlock(f"{bank}_p058_d010", 58, [0.1, 0.7, 0.9, 0.75], note, 10, "narrative", True)]
        ),
        excluded_blocks=(
            [
                PDFBlock(
                    f"{bank}_p058_d010",
                    58,
                    [0.1, 0.7, 0.9, 0.75],
                    note,
                    10,
                    "table_footnote",
                    False,
                    "table_footnote",
                    "footnote",
                )
            ]
            if requires_audited_footnote
            else []
        ),
    )

    assert _should_keep_docling_segment(
        DoclingSegment(kind="paragraph", text=note, follows_table=follows_table),
        audits=[audit],
    ) is (not requires_audited_footnote)


def test_build_docling_markdown_keeps_narrative_around_bns_d22_figure() -> None:
    """La figure D22 ne doit pas faire disparaître les paragraphes réglementaires voisins."""
    before_figure = (
        "Les banques canadiennes sont assujetties aux exigences de suffisance des fonds propres "
        "publiées par le Comité de Bâle sur le contrôle bancaire. Trois ratios fondés sur le risque "
        "sont utilisés pour évaluer la suffisance des fonds propres réglementaires de la Banque."
    )
    osfi_requirements = (
        "Le Bureau du surintendant des institutions financières Canada exige des institutions de "
        "dépôt qu'elles atteignent des exigences minimales de 7 %, de 8,5 % et de 10,5 % pour les "
        "actions ordinaires et assimilées de T1, les fonds propres de T1 et le total des fonds propres."
    )
    stability_buffer = (
        "En juin 2018, le BSIF a mis en œuvre la réserve pour stabilité intérieure que les banques "
        "d'importance systémique intérieure doivent constituer comme réserve supplémentaire au titre "
        "du deuxième pilier et réexamine la réserve deux fois par an."
    )
    current_requirements = (
        "En juin 2023, le BSIF a annoncé que la réserve pour stabilité intérieure serait portée à "
        "3,5 % des actifs pondérés en fonction des risques. Les exigences minimales s'établissent à "
        "11,5 %, à 13,0 % et à 15,0 % pour les ratios de fonds propres réglementaires."
    )
    leverage = (
        "Outre les exigences de ratio de fonds propres fondées sur le risque, Bâle III a introduit "
        "un ratio de levier simple qui vient compléter les exigences de fonds propres fondées sur le risque."
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=58,
        end_page=58,
        anchor_page=58,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock("p058_d001", 58, [0.1, 0.22, 0.9, 0.3], before_figure, 1, "narrative", True),
            PDFBlock("p058_d002", 58, [0.1, 0.56, 0.9, 0.66], osfi_requirements, 2, "narrative", True),
            PDFBlock("p058_d003", 58, [0.1, 0.67, 0.9, 0.74], stability_buffer, 3, "narrative", True),
            PDFBlock("p058_d004", 58, [0.1, 0.75, 0.9, 0.82], current_requirements, 4, "narrative", True),
            PDFBlock("p058_d005", 58, [0.1, 0.85, 0.9, 0.92], leverage, 5, "narrative", True),
        ],
        excluded_blocks=[
            PDFBlock(
                "p058_d000",
                58,
                [0.1, 0.18, 0.5, 0.2],
                "Fonds propres réglementaires",
                0,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
            PDFBlock(
                "p058_d006",
                58,
                [0.1, 0.83, 0.5, 0.84],
                "Ratio de levier",
                6,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
        ],
    )
    raw_docling = "\n\n".join(
        [
            "## Fonds propres réglementaires",
            before_figure,
            "## D22 Exigences en matière de ratios de fonds propres réglementaires minimaux (au 31 octobre 2024)",
            "<!-- image -->",
            osfi_requirements,
            stability_buffer,
            current_requirements,
            "## Ratio de levier",
            leverage,
        ]
    )

    markdown = _build_text_extraction_markdown_from_docling([audit], raw_docling_markdown=raw_docling)

    for paragraph in (before_figure, osfi_requirements, stability_buffer, current_requirements, leverage):
        assert paragraph in markdown
    assert "### Ratio de levier" in markdown
    assert "D22 Exigences" not in markdown


def test_text_extraction_cache_schema_invalidates_legacy_markdown() -> None:
    legacy = "## Gestion du capital\n\nTexte narratif.\n"
    previous_schema = (
        "<!-- vigilance-text-extraction-schema: 4 -->\n\n"
        "## Gestion du capital\n\nTexte narratif.\n"
    )

    stamped = stamp_text_extraction_cache_schema(legacy)

    assert has_current_text_extraction_cache_schema(legacy) is False
    assert has_current_text_extraction_cache_schema(previous_schema) is False
    assert has_current_text_extraction_cache_schema(stamped) is True
    assert stamped.endswith(legacy)


def test_docling_page_batches_bound_memory_and_preserve_page_order() -> None:
    pages = [10, 11, 12, 20, 21, 25]

    batches = _docling_page_batches(pages)

    assert batches == [
        (10, 11, [10, 11]),
        (12, 12, [12]),
        (20, 21, [20, 21]),
        (25, 25, [25]),
    ]


def test_build_markdown_omits_accounting_headings_in_risk_section() -> None:
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=84,
        end_page=94,
        anchor_page=84,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
        included_blocks=[
            PDFBlock(
                "p094_b001",
                94,
                [0.1, 0.20, 0.9, 0.25],
                "La Banque juge qu'il est d'importance critique d'évaluer à intervalles réguliers le contexte.",
                1,
                "narrative",
                True,
                "",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p129_b001",
                129,
                [0.1, 0.70, 0.9, 0.73],
                "CONSOLIDATION DES ENTITÉS STRUCTURÉES",
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=3,
            ),
            PDFBlock(
                "p094_b002",
                94,
                [0.1, 0.30, 0.9, 0.33],
                "FACTEURS DE RISQUE ET GESTION DES RISQUES",
                2,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=3,
            ),
        ],
    )
    raw_docling = "\n".join(
        [
            "# CONSOLIDATION DES ENTITÉS STRUCTURÉES",
            "La Banque juge qu'il est d'importance critique d'évaluer à intervalles réguliers le contexte.",
            "# FACTEURS DE RISQUE ET GESTION DES RISQUES",
        ]
    )

    markdown = _build_text_extraction_markdown_from_docling([audit], raw_docling_markdown=raw_docling)

    assert "### CONSOLIDATION DES ENTITÉS STRUCTURÉES" not in markdown
    assert "### FACTEURS DE RISQUE ET GESTION DES RISQUES" not in markdown
    assert "La Banque juge qu'il est d'importance critique" in markdown


def test_build_section_audit_excludes_blocks_outside_target_section_and_tables() -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=5,
        end_page=5,
        anchor_page=5,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.25, 0.8, 0.30],
    )
    blocks = [
        PDFBlock("p005_b001", 5, [0.1, 0.10, 0.9, 0.14], "Texte avant section", 1),
        PDFBlock("p005_b002", 5, [0.1, 0.33, 0.9, 0.40], "La banque améliore sa stratégie de capital et renforce sa gestion des risques.", 2),
        PDFBlock("p005_b003", 5, [0.1, 0.55, 0.9, 0.62], "31 45 78 90 120 150", 3),
    ]

    audit = _build_section_audit(
        section=section,
        next_section=None,
        page_blocks={5: blocks},
        repeated_text_counts={},
        table_bboxes_by_page={5: [[0.08, 0.52, 0.92, 0.64]]},
        footnote_bboxes_by_page={5: [[0.0, 0.64, 1.0, 0.72]]},
    )

    assert [block.block_id for block in audit.included_blocks] == ["p005_b002"]
    assert audit.excluded_blocks[0].exclusion_reason == "outside_target_section"
    assert audit.excluded_blocks[1].block_type == "table"
    assert audit.excluded_blocks[1].exclusion_reason == "table_like_block"


def test_build_section_audit_keeps_every_in_scope_non_table_block() -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=5,
        end_page=5,
        anchor_page=5,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
    )
    short_label = "Crédit"
    percentage = "Le ratio de fonds propres atteint 13,8 %."
    standalone_note = "(1) Ce passage décrit une exigence réglementaire autonome."
    blocks = [
        PDFBlock("p005_b001", 5, [0.1, 0.20, 0.9, 0.24], short_label, 1),
        PDFBlock("p005_b002", 5, [0.1, 0.26, 0.9, 0.32], percentage, 2),
        PDFBlock("p005_b003", 5, [0.1, 0.34, 0.9, 0.40], standalone_note, 3, "footnote"),
        PDFBlock("p005_b004", 5, [0.1, 0.42, 0.9, 0.48], "10 20 30 40", 4),
        PDFBlock("p005_b005", 5, [0.1, 0.50, 0.9, 0.54], "(2) Note sous tableau.", 5),
    ]

    audit = _build_section_audit(
        section=section,
        next_section=None,
        page_blocks={5: blocks},
        repeated_text_counts={},
        table_bboxes_by_page={5: [[0.08, 0.40, 0.92, 0.49]]},
        footnote_bboxes_by_page={5: [[0.0, 0.49, 1.0, 0.56]]},
    )

    assert [block.block_id for block in audit.included_blocks] == ["p005_b001", "p005_b002", "p005_b003"]
    assert [(block.block_id, block.block_type) for block in audit.excluded_blocks] == [
        ("p005_b004", "table"),
        ("p005_b005", "table_footnote"),
    ]


@pytest.mark.parametrize("marker", ["s.o.", "S.O.", "- s.o."])
def test_build_section_audit_excludes_standalone_not_applicable_marker(marker: str) -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=5,
        end_page=5,
        anchor_page=5,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.15],
    )
    audit = _build_section_audit(
        section=section,
        next_section=None,
        page_blocks={5: [PDFBlock("p005_b001", 5, [0.1, 0.20, 0.9, 0.24], marker, 1)]},
        repeated_text_counts={},
        table_bboxes_by_page={},
        footnote_bboxes_by_page={},
    )

    assert audit.included_blocks == []
    assert [(block.block_type, block.exclusion_reason) for block in audit.excluded_blocks] == [
        ("not_applicable", "not_applicable")
    ]


def test_build_section_audit_excludes_running_chrome_and_table_unit_label() -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=65,
        end_page=65,
        anchor_page=65,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.01, 0.8, 0.02],
    )
    audit = _build_section_audit(
        section=section,
        next_section=None,
        page_blocks={
            65: [
                PDFBlock(
                    "p065_m001",
                    65,
                    [0.07, 0.04, 0.95, 0.09],
                    "Rapport de gestion Gestion du capital",
                    1,
                ),
                PDFBlock(
                    "p065_d002",
                    65,
                    [0.07, 0.25, 0.35, 0.28],
                    "(en millions de dollars canadiens)",
                    2,
                ),
                PDFBlock(
                    "p065_d003",
                    65,
                    [0.07, 0.30, 0.95, 0.36],
                    "La Banque maintient des fonds propres suffisants pour couvrir les risques inhérents à ses activités.",
                    3,
                ),
                PDFBlock(
                    "p065_m004",
                    65,
                    [0.79, 0.95, 0.96, 0.98],
                    "65 Banque Nationale du Canada Rapport annuel 2025",
                    4,
                ),
            ]
        },
        repeated_text_counts={},
        table_bboxes_by_page={},
        footnote_bboxes_by_page={},
    )

    assert [block.block_id for block in audit.included_blocks] == ["p065_d003"]
    assert [(block.block_type, block.exclusion_reason) for block in audit.excluded_blocks] == [
        ("header_footer", "running_header_footer"),
        ("table", "table_like_block"),
        ("header_footer", "running_header_footer"),
    ]


def test_inferred_table_footnote_zone_covers_bnc_visual_gap() -> None:
    footnotes = _infer_table_footnote_bboxes({96: [[0.04, 0.18, 0.93, 0.573]]})
    note = PDFBlock(
        "p096_d009",
        96,
        [0.04, 0.687, 0.60, 0.695],
        "(7) Pour de plus amples renseignements, se reporter aux notes afférentes aux états financiers consolidés.",
        7,
    )

    assert footnotes[96] == [[0.0, 0.573, 1.0, 0.713]]
    assert _classify_block_type(note, {}, [], footnotes[96]) == "table_footnote"


def test_composite_grid_region_excludes_cells_caption_and_table_note() -> None:
    narrative = PDFBlock(
        "p066_d001",
        66,
        [0.05, 0.12, 0.95, 0.20],
        (
            "Le capital économique permet à la Banque de couvrir ses risques et le ratio réglementaire "
            "atteint 13,8 %, tandis que les fonds propres disponibles totalisent 525 M$."
        ),
        1,
    )
    caption = PDFBlock(
        "p066_d004",
        66,
        [0.05, 0.25, 0.40, 0.27],
        "Répartition des risques par secteur d'exploitation",
        4,
        source_label="section_header",
        heading_level=1,
    )
    blocks = [narrative, caption]
    line_number = 5
    for column_x in (0.17, 0.48, 0.79):
        for label, y, value in (
            ("Crédit", 0.56, "4 290"),
            ("Marché", 0.58, "228"),
            ("Opérationnel", 0.60, "518"),
            ("Total", 0.67, "5 121"),
        ):
            blocks.append(
                PDFBlock(
                    f"p066_d{line_number:03d}",
                    66,
                    [column_x, y, column_x + 0.08, y + 0.008],
                    label,
                    line_number,
                )
            )
            line_number += 1
            blocks.append(
                PDFBlock(
                    f"p066_d{line_number:03d}",
                    66,
                    [column_x + 0.09, y, column_x + 0.14, y + 0.008],
                    value,
                    line_number,
                )
            )
            line_number += 1
    note = PDFBlock(
        "p066_d083",
        66,
        [0.05, 0.72, 0.80, 0.74],
        (
            "Consulter le « Mode de présentation de l'information » aux pages 14 à 20 "
            "pour de plus amples renseignements sur les mesures de gestion du capital."
        ),
        line_number,
    )
    following_narrative = PDFBlock(
        "p067_d001",
        66,
        [0.05, 0.86, 0.95, 0.92],
        "La Banque applique ensuite son cadre de gestion des risques à l'ensemble de ses activités.",
        line_number + 1,
    )
    blocks.extend([note, following_narrative])
    table_regions = {66: [[0.79, 0.63, 0.93, 0.68]]}

    _augment_table_regions_with_composite_grids({66: blocks}, table_regions)
    composite = [bbox for bbox in table_regions[66] if bbox[0] == 0.0 and bbox[2] == 1.0]
    footnote_regions = _infer_table_footnote_bboxes(table_regions)

    assert len(composite) == 1
    assert caption.block_type == "table"
    assert all(block.block_type == "table" for block in blocks[2:-2])
    assert narrative.block_type == "other"
    assert following_narrative.block_type == "other"
    assert _classify_block_type(note, {}, table_regions[66], footnote_regions[66]) == "table_footnote"
    assert _classify_block_type(narrative, {}, table_regions[66], footnote_regions[66]) == "narrative"


def test_raw_docling_markdown_path_uses_role_year_and_quarter(tmp_path: Path) -> None:
    assert get_raw_docling_markdown_path(tmp_path, "TD", 2025, "T4", "current") == (
        tmp_path / "outputs" / "text_extractions" / "td" / "2025" / "t4" / "td_current_2025_t4.md"
    )
    assert get_raw_docling_markdown_path(tmp_path, "td", 2024, "t4", "previous") == (
        tmp_path / "outputs" / "text_extractions" / "td" / "2024" / "t4" / "td_previous_2024_t4.md"
    )


def test_extract_audits_for_pdf_writes_raw_docling_markdown_before_filtering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_markdown = (
        "# Gestion des risques\n\n"
        "Texte brut Docling avec un tableau encore présent.\n\n"
        "| 2025 | 2024 |\n| --- | --- |\n"
    )

    def _fake_extract_docling_page_blocks(pdf_path: Path, page_numbers: list[int]):
        assert pdf_path == tmp_path / "td.pdf"
        assert page_numbers == [7]
        return (
            {
                7: [
                    PDFBlock(
                        "p007_d001",
                        7,
                        [0.10, 0.35, 0.90, 0.42],
                        "La banque maintient une gestion prudente des risques et renforce ses contrôles internes.",
                        1,
                    ),
                    PDFBlock("p007_d002", 7, [0.10, 0.60, 0.90, 0.70], "2025 2024 2023 2022", 2),
                ]
            },
            {7: [[0.08, 0.58, 0.92, 0.72]]},
            {},
            raw_markdown,
        )

    monkeypatch.setattr(
        "vigilance.text_analysis.extraction._extract_docling_page_blocks",
        _fake_extract_docling_page_blocks,
    )
    raw_path = tmp_path / "outputs" / "text_extractions" / "td" / "2025" / "t4" / "td_current_2025_t4.md"
    section = ResolvedSection(
        section_key="gestion_risques",
        title="Gestion des risques",
        start_page=7,
        end_page=7,
        anchor_page=7,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.10, 0.20, 0.90, 0.30],
    )

    audits, written_raw = _extract_audits_for_pdf(
        pdf_path=tmp_path / "td.pdf",
        sections={"gestion_risques": section},
        raw_docling_markdown_path=raw_path,
    )

    assert raw_path.read_text(encoding="utf-8") == raw_markdown
    assert "[p." not in written_raw
    assert "bbox=" not in written_raw
    assert "block_id=" not in written_raw
    assert [block.block_id for block in audits[0].included_blocks] == ["p007_d001"]
    assert audits[0].excluded_blocks[0].block_id == "p007_d002"


def test_extract_section_text_from_markdown_returns_matching_section() -> None:
    md = (
        "## Gestion du capital\n\n"
        "La banque maintient un niveau prudent de fonds propres.\n\n"
        "Elle vise un ratio CET1 supérieur aux exigences réglementaires.\n\n"
        "## Gestion des risques\n\n"
        "La banque surveille les risques géopolitiques.\n"
    )

    capital = _extract_section_text_from_markdown(md, "gestion_capital")
    risques = _extract_section_text_from_markdown(md, "gestion_risques")

    assert "fonds propres" in capital
    assert "CET1" in capital
    assert "Gestion des risques" not in capital
    assert "géopolitiques" in risques
    assert "fonds propres" not in risques


def test_extract_section_text_from_markdown_returns_empty_for_missing_section() -> None:
    md = "## Gestion du capital\n\nQuelques paragraphes.\n"

    result = _extract_section_text_from_markdown(md, "gestion_reglementation")

    assert result == ""


def test_classify_block_type_rejects_rating_table_like_block() -> None:
    block = PDFBlock(
        "p043_b010",
        43,
        [0.1, 0.2, 0.9, 0.4],
        "Au 31 janvier 2026 Moody's S&P Fitch DBRS Dépôts/contrepartie Aa1 A+ AA AA Ancienne dette de premier rang Aa2 A+ AA AA Dette de premier rang A2 A- AA- AA (bas) Actions privilégiées FPUNV Baa2 BBB- BBB+",
        10,
    )

    assert _classify_block_type(block, {}) == "table"


def test_classify_block_type_rejects_block_overlapping_docling_table_bbox() -> None:
    block = PDFBlock(
        "p043_d002",
        43,
        [0.11, 0.22, 0.88, 0.36],
        "Un bloc qui traverserait un tableau détecté par Docling.",
        2,
    )

    assert _classify_block_type(block, {}, [[0.10, 0.20, 0.90, 0.38]]) == "table"


def test_classify_block_type_rejects_block_overlapping_table_footnote_bbox() -> None:
    block = PDFBlock(
        "p043_d003",
        43,
        [0.05, 0.35, 0.95, 0.40],
        "(1) Comprennent les actifs donnés en garantie dans le cadre du financement.",
        3,
    )

    assert _classify_block_type(block, {}, [], [[0.0, 0.33, 1.0, 0.42]]) == "table_footnote"


def test_looks_like_footnote_accepts_bare_numeric_note_marker() -> None:
    text = "1 Le 18 décembre 2025, le BSIF a confirmé que la réserve pour stabilité intérieure était maintenue à 3,5 %."

    assert _looks_like_footnote(text) is True


def test_looks_like_footnote_accepts_superscript_digit_marker() -> None:
    text = "³ Pour de plus amples renseignements, se reporter à la note 23 afférente aux états financiers consolidés."

    assert _looks_like_footnote(text) is True


def test_classify_block_type_rejects_ns_table_note_marker() -> None:
    block = PDFBlock(
        "p050_d011",
        50,
        [0.05, 0.62, 0.95, 0.67],
        "n.s. Le calcul de l'effet de diversification sur le cours le plus haut et sur le cours le plus bas n'est pas significatif puisqu'ils peuvent survenir à des jours différents et pour divers types de risques.",
        11,
    )

    assert _classify_block_type(block, {}, [], [[0.0, 0.60, 1.0, 0.70]]) == "table_footnote"


def test_classify_block_type_rejects_long_explicit_footnote_before_narrative_rule() -> None:
    block = PDFBlock(
        "p033_d007",
        33,
        [0.05, 0.60, 0.95, 0.67],
        (
            "(3) Pour de plus amples renseignements, se reporter au tableau illustrant la distribution "
            "de la VaR des portefeuilles de négociation par catégorie de risque et leur effet de diversification, "
            "et au tableau illustrant la sensibilité aux taux d'intérêt, présentés aux pages suivantes ainsi qu'à "
            "la section « Risque de marché » du Rapport annuel 2025."
        ),
        7,
    )

    assert _classify_block_type(block, {}) == "footnote"


def test_classify_block_type_keeps_narrative_paragraph_after_table_footnotes() -> None:
    block = PDFBlock(
        "p043_d004",
        43,
        [0.05, 0.40, 0.95, 0.50],
        (
            "La Banque veille à ce que ses niveaux de fonds propres excèdent en tout temps les "
            "limites minimales relatives aux capitaux propres établies par le BSIF, y compris la RSI. "
            "Une structure solide de capital permet à la Banque de couvrir les risques inhérents "
            "à ses activités, de soutenir ses secteurs d’exploitation et de protéger sa clientèle."
        ),
        4,
    )

    assert _classify_block_type(block, {}, [], [[0.0, 0.33, 1.0, 0.42]]) == "narrative"


def test_classify_block_type_keeps_regulatory_narrative_with_many_numbers() -> None:
    block = PDFBlock(
        "p044_d010",
        44,
        [0.05, 0.42, 0.95, 0.72],
        (
            "La Banque ainsi que toutes les autres grandes banques canadiennes doivent maintenir des ratios "
            "minimaux de fonds propres établis par le BSIF, soit un ratio des fonds propres CET1 d’au moins 11,5 %, "
            "un ratio des fonds propres de catégorie 1 d’au moins 13,0 % et un ratio du total des fonds propres d’au moins 15,0 %. "
            "Tous ces ratios incluent une réserve de conservation de 2,5 % et une surcharge de 1,0 %, tandis que la RSI est maintenue à 3,5 %."
        ),
        10,
    )

    assert _classify_block_type(block, {}) == "narrative"


def test_build_section_audit_keeps_narrative_between_two_tables() -> None:
    section = ResolvedSection(
        section_key="gestion_capital",
        title="Gestion du capital",
        start_page=6,
        end_page=6,
        anchor_page=6,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.10, 0.8, 0.14],
    )
    blocks = [
        PDFBlock("p006_b001", 6, [0.08, 0.20, 0.92, 0.30], "100 200 300 400 500 600", 1),
        PDFBlock(
            "p006_b002",
            6,
            [0.08, 0.37, 0.92, 0.47],
            "La Banque maintient un niveau de fonds propres prudent afin de couvrir les risques inhérents à ses activités et protéger sa clientèle.",
            2,
        ),
        PDFBlock("p006_b003", 6, [0.08, 0.55, 0.92, 0.66], "700 800 900 1000 1100 1200", 3),
    ]

    audit = _build_section_audit(
        section=section,
        next_section=None,
        page_blocks={6: blocks},
        repeated_text_counts={},
        table_bboxes_by_page={6: [[0.06, 0.18, 0.94, 0.32], [0.06, 0.53, 0.94, 0.68]]},
        footnote_bboxes_by_page={6: [[0.0, 0.32, 1.0, 0.35], [0.0, 0.68, 1.0, 0.72]]},
    )

    assert [block.block_id for block in audit.included_blocks] == ["p006_b002"]


def test_chunk_comparison_llm_response_rejects_invalid_diff_types() -> None:
    with pytest.raises(Exception, match="diff_type"):
        ChunkComparisonLLMResponse.model_validate(
            {
                "changes": [
                    {
                        "alignment_id": "a00",
                        "diff_type": "invalid_type",
                        "text_t1": "Ancienne idée commune.",
                        "text_t2": "Ancienne idée commune.",
                        "change_summary": "",
                    }
                ]
            }
        )


def test_chunk_comparison_llm_response_rejects_missing_alignment_id() -> None:
    with pytest.raises(Exception, match="alignment_id"):
        ChunkComparisonLLMResponse.model_validate(
            {
                "changes": [
                    {
                        "diff_type": "modified",
                        "text_t1": "Ancienne idée commune.",
                        "text_t2": "Nouvelle idée commune.",
                        "change_summary": "Modification.",
                    }
                ]
            }
        )


def test_compare_section_texts_prompt_requests_all_observable_changes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_structured_completion(client, *, model, messages, **kwargs):
        captured["messages"] = messages
        return ChunkComparisonLLMResponse(changes=[])

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._call_structured_completion_with_correction",
        _fake_structured_completion,
    )

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Risque de stratégie\n\nLa banque surveille ce risque au premier trimestre.",
        text_t2="### Risque de stratégie\n\nCe risque est surveillé par la banque au deuxième trimestre.",
    )

    prompt = "\n".join(str(msg.get("content", "")) for msg in captured["messages"])
    assert "tous les changements observables" in prompt
    assert "Ne masque pas les reformulations" in prompt
    assert "retourne quand même diff_type='modified'" in prompt


def test_build_text_extraction_markdown_keeps_headings_and_narrative_only() -> None:
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=10,
        end_page=10,
        anchor_page=10,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[
            PDFBlock("p010_d002", 10, [0.1, 0.33, 0.9, 0.36], "Le 12 décembre 2023, la Banque avait débuté un programme de rachat d'actions ordinaires.", 2, "narrative", True, "", "paragraph"),
        ],
        excluded_blocks=[
            PDFBlock("p010_d001", 10, [0.1, 0.28, 0.8, 0.31], "Rachat d'actions ordinaires", 1, "other", False, "non_narrative_block", "section_header"),
            PDFBlock("p010_d003", 10, [0.1, 0.55, 0.9, 0.60], "TABLEAU 5 100 200 300", 3, "table", False, "table_like_block", "caption"),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit])

    assert "## Gestion du capital" in markdown
    assert "### Rachat d'actions ordinaires" in markdown
    assert "Le 12 décembre 2023" in markdown
    assert "TABLEAU 5" not in markdown


def test_build_text_extraction_markdown_inline_pdf_page_on_headings_only() -> None:
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=60,
        end_page=60,
        anchor_page=60,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[
            PDFBlock(
                "p060_d002",
                60,
                [0.1, 0.33, 0.9, 0.36],
                "La banque maintient un niveau prudent de fonds propres.",
                2,
                "narrative",
                True,
                "",
                "paragraph",
            ),
        ],
        excluded_blocks=[],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit])
    page_index, section_start_pages = _parse_page_index_from_markdown(markdown)

    assert markdown.startswith("## Gestion du capital [pdf.60]")
    assert "[pdf." not in markdown.split("La banque", 1)[1]
    assert page_index["gestion_capital"] == [
        (60, "La banque maintient un niveau prudent de fonds propres.")
    ]
    assert section_start_pages["gestion_capital"] == 60


def test_format_page_suffix_pdf_only() -> None:
    assert _format_page_suffix(84) == " [pdf.84]"
    assert _format_page_marker(62) == "[pdf.62]"


def test_extract_section_text_from_markdown_strips_inline_pdf_page_marker() -> None:
    md = (
        "## Gestion du capital [pdf.60]\n\n"
        "### Sous-section [pdf.61]\n\n"
        "La banque maintient un niveau prudent de fonds propres.\n"
    )

    capital = _extract_section_text_from_markdown(md, "gestion_capital")

    assert capital == "### Sous-section\n\nLa banque maintient un niveau prudent de fonds propres."


def test_extract_section_text_from_markdown_strips_legacy_standalone_markers() -> None:
    md = (
        "[p.58 | pdf.60]\n"
        "## Gestion du capital\n\n"
        "[p.58 | pdf.60]\n"
        "La banque maintient un niveau prudent de fonds propres.\n"
    )

    capital = _extract_section_text_from_markdown(md, "gestion_capital")

    assert capital == "La banque maintient un niveau prudent de fonds propres."


def test_parse_page_index_inherits_page_from_heading() -> None:
    md = (
        "## Gestion du capital [pdf.60]\n\n"
        "### Objectifs [pdf.61]\n\n"
        "Premier paragraphe.\n\n"
        "Deuxieme paragraphe.\n"
    )
    page_index, section_start_pages = _parse_page_index_from_markdown(md)

    assert section_start_pages["gestion_capital"] == 60
    assert page_index["gestion_capital"] == [
        (61, "Premier paragraphe."),
        (61, "Deuxieme paragraphe."),
    ]


def test_rewrite_migrates_legacy_markers_to_pdf_inline() -> None:
    md = "[p.60]\n## Gestion du capital\n\n[p.61]\nUn paragraphe.\n"

    rewritten = _rewrite_page_markers_for_display(md)
    page_index, section_start_pages = _parse_page_index_from_markdown(rewritten)

    assert rewritten.startswith("## Gestion du capital [pdf.60]")
    assert "[pdf." not in "Un paragraphe."
    assert section_start_pages["gestion_capital"] == 60
    assert page_index["gestion_capital"] == [(60, "Un paragraphe.")]


def test_build_text_extraction_markdown_excludes_orphan_heading_from_matching() -> None:
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=10,
        end_page=10,
        anchor_page=10,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[],
        excluded_blocks=[
            PDFBlock("p010_d001", 10, [0.1, 0.28, 0.8, 0.31], "Accord de Bâle", 1, "other", False, "non_narrative_block", "section_header"),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit])

    assert "### Accord de Bâle" not in markdown


def test_docling_parser_marks_first_segment_after_table() -> None:
    segments = _parse_docling_markdown(
        "| 2025 | 2024 |\n"
        "| --- | --- |\n"
        "| 10 | 9 |\n\n"
        "(5) Les montants sont présentés avant déduction des provisions.\n\n"
        "Le paragraphe narratif suivant doit demeurer dans le flux.\n"
    )

    assert segments[0].kind == "table"
    assert segments[1].text.startswith("(5)")
    assert segments[1].follows_table is True
    assert segments[2].follows_table is False


def test_docling_standalone_footnote_is_kept_without_table_context() -> None:
    note = (
        "(5) La juste valeur des titres de participation désignés à la juste valeur "
        "est présentée aux notes afférentes aux états financiers consolidés."
    )
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=30,
        end_page=30,
        anchor_page=30,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock("p030_d001", 30, [0.1, 0.6, 0.9, 0.7], note, 1),
        ],
        excluded_blocks=[],
    )

    assert _should_keep_docling_segment(
        DoclingSegment(kind="paragraph", text=note),
        [audit],
    ) is True


@pytest.mark.parametrize(
    ("text", "follows_table"),
    [
        ("1 Les méthodes de présentation reposent sur la ligne directrice B-20.", False),
        ("4 Comprennent la dette de premier rang et excluent des billets structurés.", False),
        ("n. s. - non significatif", False),
    ],
)
def test_docling_filter_keeps_note_forms_without_table_context(
    text: str,
    follows_table: bool,
) -> None:
    assert _should_keep_docling_segment(
        DoclingSegment(kind="paragraph", text=text, follows_table=follows_table)
    ) is True


@pytest.mark.parametrize("marker", ["s.o.", "S.O."])
def test_docling_filter_excludes_standalone_not_applicable_marker(marker: str) -> None:
    assert _should_keep_docling_segment(DoclingSegment(kind="paragraph", text=marker)) is False


@pytest.mark.parametrize(
    "text",
    [
        "65 Banque Nationale du Canada Rapport annuel 2025",
        "Rapport de gestion",
        "Rapport de gestion Gestion des risques",
        "(en millions de dollars canadiens)",
        "(25) (23) (21) (19) (17) (15) (13) (11) (9) (7) (5) (3) (1) 1 3 5 7 9 11 13",
    ],
)
def test_docling_filter_excludes_running_chrome_and_table_unit_label(text: str) -> None:
    segment = DoclingSegment(kind="heading", text=text, follows_table=False)

    assert _should_keep_docling_segment(segment) is False


def test_docling_filter_excludes_cell_confirmed_by_composite_table_audit() -> None:
    cell = PDFBlock(
        "p066_d007",
        66,
        [0.17, 0.56, 0.20, 0.57],
        "Crédit",
        7,
        "table",
        False,
        "table_like_block",
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=66,
        end_page=66,
        anchor_page=66,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[],
        excluded_blocks=[cell],
    )

    assert _should_keep_docling_segment(
        DoclingSegment(kind="paragraph", text="Crédit"),
        [audit],
    ) is False


def test_docling_filter_keeps_long_dated_numbered_narrative_disclosure() -> None:
    text = (
        "2 Le 31 janvier 2025, la Banque a racheté des actions ordinaires afin "
        "de gérer son capital et de respecter les exigences réglementaires."
    )

    assert _should_keep_docling_segment(
        DoclingSegment(kind="paragraph", text=text, follows_table=True)
    ) is True


def test_docling_parser_keeps_inline_note_clause_with_the_narrative() -> None:
    segments = _parse_docling_markdown(
        "Les prêts sont garantis au Canada et au Yukon. (5) Nous calculons "
        "le ratio prêt-valeur moyen selon les données du tableau.\n"
    )

    assert len(segments) == 1
    assert segments[0].text == (
        "Les prêts sont garantis au Canada et au Yukon. (5) Nous calculons "
        "le ratio prêt-valeur moyen selon les données du tableau."
    )


def test_build_docling_markdown_removes_explicit_table_footnote() -> None:
    narrative = (
        "La Banque maintient un niveau de fonds propres prudent et surveille "
        "régulièrement les risques associés à ses activités."
    )
    note = (
        "(5) La juste valeur des titres de participation désignés à la juste valeur "
        "est présentée aux notes afférentes aux états financiers consolidés."
    )
    raw_docling = (
        "## GESTION DES RISQUES\n\n"
        f"{narrative}\n\n"
        "| 2025 | 2024 |\n"
        "| --- | --- |\n"
        "| 10 | 9 |\n\n"
        f"{note}\n\n"
        f"{narrative}\n"
    )
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=30,
        end_page=30,
        anchor_page=30,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock("p030_d001", 30, [0.1, 0.25, 0.9, 0.35], narrative, 1),
        ],
        excluded_blocks=[
            PDFBlock(
                "p030_d000",
                30,
                [0.1, 0.1, 0.9, 0.2],
                "GESTION DES RISQUES",
                0,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=2,
            ),
            PDFBlock(
                "p030_d002",
                30,
                [0.1, 0.6, 0.9, 0.7],
                note,
                2,
                "table_footnote",
                False,
                "table_footnote",
            ),
        ],
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=raw_docling,
    )

    assert narrative in markdown
    assert note not in markdown
    assert "(5)" not in markdown


def test_build_docling_markdown_keeps_all_non_table_content() -> None:
    short_label = "Crédit"
    financial_paragraph = "Le ratio de fonds propres atteint 13,8 % et le portefeuille vaut 525 M$."
    standalone_note = "(1) Cette exigence s'applique à toutes les filiales réglementées."
    table_note = "(2) Les montants sont exprimés en millions de dollars."
    raw_docling = (
        "## GESTION DU CAPITAL\n\n"
        f"{short_label}\n\n"
        f"{financial_paragraph}\n\n"
        f"{standalone_note}\n\n"
        "| Catégorie | Valeur |\n"
        "| --- | ---: |\n"
        "| Crédit | 395 |\n\n"
        f"{table_note}\n"
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=30,
        end_page=30,
        anchor_page=30,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock("p030_d001", 30, [0.1, 0.25, 0.9, 0.29], short_label, 1, "other", True),
            PDFBlock("p030_d002", 30, [0.1, 0.30, 0.9, 0.36], financial_paragraph, 2, "narrative", True),
            PDFBlock("p030_d003", 30, [0.1, 0.37, 0.9, 0.43], standalone_note, 3, "footnote", True),
        ],
        excluded_blocks=[
            PDFBlock(
                "p030_d004",
                30,
                [0.1, 0.50, 0.9, 0.56],
                table_note,
                4,
                "table_footnote",
                False,
                "table_footnote",
            ),
        ],
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=raw_docling,
    )

    assert short_label in markdown
    assert financial_paragraph in markdown
    assert standalone_note in markdown
    assert "| Crédit | 395 |" not in markdown
    assert table_note not in markdown


def test_build_docling_markdown_falls_back_to_audited_blocks_when_raw_markdown_is_incomplete() -> None:
    visible = "Le ratio de fonds propres atteint 13,8 %."
    missing = "Le portefeuille réglementaire vaut 525 M$."
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=30,
        end_page=30,
        anchor_page=30,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock("p030_d001", 30, [0.1, 0.25, 0.9, 0.31], visible, 1, "narrative", True),
            PDFBlock(
                "p030_d002",
                30,
                [0.1, 0.33, 0.9, 0.39],
                missing,
                2,
                "narrative",
                True,
                source_label="section_header",
            ),
        ],
        excluded_blocks=[],
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=f"## Gestion du capital\n\n{visible}\n",
    )

    assert visible in markdown
    assert missing in markdown


def test_parse_docling_markdown_preserves_one_table_boundary() -> None:
    segments = _parse_docling_markdown(
        "\n".join(
            [
                "## Activités de négociation",
                "| Facteur | VaR |",
                "| --- | ---: |",
                "| Taux | 13,0 |",
                "Le texte narratif reprend après le tableau.",
            ]
        )
    )

    assert [segment.kind for segment in segments] == ["heading", "table", "paragraph"]
    assert segments[-1].follows_table is True


def test_matchable_segments_drop_table_caption_chain_and_resume_parent_heading() -> None:
    segments = [
        DoclingSegment(kind="heading", text="Activités de négociation"),
        DoclingSegment(kind="paragraph", text="Le tableau suivant présente la VaR."),
        DoclingSegment(kind="heading", text="VaR des portefeuilles de négociation"),
        DoclingSegment(kind="heading", text="Exercice terminé le 31 octobre"),
        DoclingSegment(kind="table", text="[table]"),
        DoclingSegment(kind="paragraph", text="La VaR moyenne a augmenté cette année."),
    ]

    selected = _matchable_section_segments(segments)

    assert [segment.text for segment in selected if segment.kind == "heading"] == [
        "Activités de négociation"
    ]
    assert [segment.kind for segment in selected] == [
        "heading",
        "paragraph",
        "table",
        "paragraph",
    ]


def test_docling_markdown_accepts_numbered_parent_heading_without_reinserting_it() -> None:
    parent = "La gestion du capital en 2024"
    child = "Activités de gestion"
    paragraph = "La Banque poursuit ses activités de gestion du capital."
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=62,
        end_page=62,
        anchor_page=57,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.04, 0.05, 0.90, 0.08],
        included_blocks=[
            PDFBlock(
                "p062_d001",
                62,
                [0.04, 0.09, 0.33, 0.11],
                parent,
                1,
                "other",
                True,
                "",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p062_d002",
                62,
                [0.04, 0.12, 0.16, 0.14],
                child,
                2,
                "other",
                True,
                "",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p062_d003",
                62,
                [0.04, 0.15, 0.90, 0.20],
                paragraph,
                3,
                "narrative",
                True,
            ),
        ],
        excluded_blocks=[],
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=f"## {parent}\n\n## {child}\n\n{paragraph}\n",
    )

    assert parent not in markdown
    assert f"### {child} [pdf.62]" in markdown
    assert paragraph in markdown


def test_docling_markdown_keeps_parent_across_table_and_inserts_missing_text_locally() -> None:
    intro = "Le tableau suivant présente la VaR des portefeuilles de négociation."
    missing = "Les montants sont présentés avant impôts selon un niveau de confiance de 99 %."
    after = "La VaR totale de négociation moyenne a augmenté pendant l'exercice."
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=98,
        end_page=98,
        anchor_page=98,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.04, 0.05, 0.90, 0.08],
        included_blocks=[
            PDFBlock(
                "p098_d001",
                98,
                [0.04, 0.09, 0.50, 0.11],
                "Activités de négociation",
                1,
                "other",
                True,
                "",
                "section_header",
                heading_level=1,
            ),
            PDFBlock("p098_d002", 98, [0.04, 0.12, 0.90, 0.15], intro, 2, "narrative", True),
            PDFBlock("p098_d005", 98, [0.04, 0.55, 0.90, 0.58], missing, 5, "narrative", True),
            PDFBlock("p098_d006", 98, [0.04, 0.59, 0.90, 0.62], after, 6, "narrative", True),
        ],
        excluded_blocks=[
            PDFBlock(
                "p098_d003",
                98,
                [0.04, 0.16, 0.40, 0.18],
                "VaR des portefeuilles de négociation (1) (2) *",
                3,
                "table",
                False,
                "table_like_block",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p098_d004",
                98,
                [0.04, 0.19, 0.30, 0.21],
                "Exercice terminé le 31 octobre",
                4,
                "table",
                False,
                "table_like_block",
                "section_header",
                heading_level=1,
            ),
        ],
        table_regions=[
            {
                "table_id": "gestion_risques_p098_tbl_01",
                "page": 98,
                "region_type": "table",
                "bbox": [0.04, 0.16, 0.93, 0.53],
            }
        ],
    )
    raw_docling = "\n\n".join(
        [
            "## Activités de négociation",
            intro,
            "## VaR des portefeuilles de négociation (1) (2) *",
            "## Exercice terminé le 31 octobre",
            "| Facteur | VaR |",
            "| --- | ---: |",
            "| Taux | 13,0 |",
            after,
        ]
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=raw_docling,
    )

    assert "### Activités de négociation [pdf.98]" in markdown
    assert "### VaR des portefeuilles" not in markdown
    assert "### Exercice terminé" not in markdown
    assert intro in markdown
    assert missing in markdown
    assert after in markdown
    assert markdown.index(intro) < markdown.index(missing) < markdown.index(after)


def test_docling_alignment_prefers_exact_heading_and_rejects_page_footer() -> None:
    heading = "Ratio de liquidité à long terme"
    paragraph = (
        "Le CBCB a élaboré le ratio de liquidité à long terme afin de promouvoir "
        "la résilience du secteur bancaire."
    )
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=106,
        end_page=107,
        anchor_page=106,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.04, 0.05, 0.90, 0.08],
        included_blocks=[
            PDFBlock(
                "p106_d001",
                106,
                [0.04, 0.82, 0.23, 0.84],
                heading,
                1,
                "other",
                True,
                "",
                "section_header",
                heading_level=2,
            ),
            PDFBlock(
                "p106_d002",
                106,
                [0.04, 0.85, 0.90, 0.91],
                paragraph,
                2,
                "narrative",
                True,
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p106_d003",
                106,
                [0.90, 0.94, 0.94, 0.96],
                "106",
                3,
                "header_footer",
                False,
                "running_header_footer",
                "section_header",
            ),
            PDFBlock(
                "p107_d001",
                107,
                [0.07, 0.20, 0.95, 0.82],
                f"Passifs et actifs du NSFR {heading} (%) 124 % 123 %",
                4,
                "table",
                False,
                "table_like_block",
                "pymupdf_fallback",
            ),
        ],
    )

    markdown = _build_text_extraction_markdown_from_docling(
        [audit],
        raw_docling_markdown=(
            f"## Gestion des risques\n\n## {heading}\n\n{paragraph}\n\n106\n"
        ),
    )

    assert f"### {heading} [pdf.106]" in markdown
    assert paragraph in markdown
    assert "\n106\n" not in markdown


def test_fallback_markdown_keeps_mislabeled_long_section_header_as_comparable_text() -> None:
    paragraph = (
        "La Banque maintient un ratio CET1 de 13,8 % et renforce ses contrôles "
        "afin de préserver une base de fonds propres suffisante."
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=30,
        end_page=30,
        anchor_page=30,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.9, 0.2],
        included_blocks=[
            PDFBlock(
                "p030_d001",
                30,
                [0.1, 0.25, 0.9, 0.35],
                paragraph,
                1,
                "other",
                True,
                "",
                "section_header",
            )
        ],
        excluded_blocks=[],
    )

    markdown = _build_text_extraction_markdown([audit])

    assert paragraph in markdown
    assert f"### {paragraph}" not in markdown
    section = _extract_section_text_from_markdown(markdown, "gestion_capital")
    chunks = _chunk_subsection_text(dict(_parse_subsections(section))["__intro__"])
    assert [chunk.text for chunk in chunks] == [paragraph]


def test_build_text_extraction_markdown_from_docling_keeps_headings_and_order() -> None:
    raw_docling = (
        "## SITUATION FINANCIÈRE DU GROUPE\n\n"
        "Texte hors périmètre qui ne doit pas être retenu.\n\n"
        "## Situation des fonds propres\n\n"
        "| 2025 | 2024 |\n"
        "| --- | --- |\n"
        "| 10 | 9 |\n\n"
        "## OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES\n\n"
        "Les objectifs de la Banque en matière de gestion des fonds propres sont les suivants :\n\n"
        "- Maintenir des fonds propres adéquats compte tenu du profil de risque de la Banque.\n\n"
        "## SOURCES DES FONDS PROPRES\n\n"
        "Les fonds propres de la Banque proviennent principalement des actionnaires ordinaires.\n"
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=74,
        end_page=74,
        anchor_page=73,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[
            PDFBlock(
                "p074_d002",
                74,
                [0.1, 0.33, 0.9, 0.36],
                "Les objectifs de la Banque en matière de gestion des fonds propres sont les suivants :",
                2,
                "narrative",
                True,
                "",
                "paragraph",
            ),
            PDFBlock(
                "p074_d003",
                74,
                [0.1, 0.40, 0.9, 0.43],
                "Maintenir des fonds propres adéquats compte tenu du profil de risque de la Banque.",
                3,
                "narrative",
                True,
                "",
                "paragraph",
            ),
            PDFBlock(
                "p074_d004",
                74,
                [0.1, 0.50, 0.9, 0.53],
                "Les fonds propres de la Banque proviennent principalement des actionnaires ordinaires.",
                4,
                "narrative",
                True,
                "",
                "paragraph",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p074_d000",
                74,
                [0.1, 0.12, 0.8, 0.15],
                "SITUATION FINANCIÈRE DU GROUPE",
                0,
                "other",
                False,
                "outside_target_section",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p074_d000b",
                74,
                [0.1, 0.18, 0.8, 0.21],
                "Situation des fonds propres",
                0,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p074_d001",
                74,
                [0.1, 0.28, 0.8, 0.31],
                "OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES",
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=1,
            ),
            PDFBlock(
                "p074_d005",
                74,
                [0.1, 0.55, 0.8, 0.58],
                "SOURCES DES FONDS PROPRES",
                5,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=1,
            ),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit], raw_docling_markdown=raw_docling)

    assert "## Gestion du capital" in markdown
    assert "### OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES" in markdown
    assert "### SOURCES DES FONDS PROPRES" in markdown
    assert "Les objectifs de la Banque" in markdown
    assert "Maintenir des fonds propres adéquats" in markdown
    assert "| 2025 |" not in markdown
    assert "Situation des fonds propres" not in markdown
    assert "SITUATION FINANCIÈRE DU GROUPE" not in markdown
    assert "Texte hors périmètre" not in markdown
    assert markdown.index("### OBJECTIFS") < markdown.index("Les objectifs de la Banque")
    assert markdown.index("Les objectifs de la Banque") < markdown.index("### SOURCES DES FONDS PROPRES")


def test_docling_heading_does_not_match_words_inside_capital_paragraph() -> None:
    raw_docling = (
        "## Contrôle des risques\n\n"
        "Ce texte relève de la gestion des risques et ne fait pas partie du capital.\n\n"
        "## OBJECTIFS DE LA BANQUE\n\n"
        "La Banque maintient des fonds propres adéquats.\n"
    )
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=80,
        end_page=80,
        anchor_page=80,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.1, 0.8, 0.15],
        included_blocks=[
            PDFBlock(
                "p080_d007",
                80,
                [0.1, 0.2, 0.9, 0.3],
                "Le processus englobe les fonctions de gestion et de contrôle des risques et des fonds propres.",
                7,
                "narrative",
                True,
                "",
                "paragraph",
            ),
            PDFBlock(
                "p080_d009",
                80,
                [0.1, 0.4, 0.9, 0.45],
                "La Banque maintient des fonds propres adéquats.",
                9,
                "narrative",
                True,
                "",
                "paragraph",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p080_d008",
                80,
                [0.1, 0.35, 0.8, 0.38],
                "OBJECTIFS DE LA BANQUE",
                8,
                "other",
                False,
                "non_narrative_block",
                "section_header",
                heading_level=1,
            ),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown_from_docling([audit], raw_docling_markdown=raw_docling)

    assert "Contrôle des risques" not in markdown
    assert "Ce texte relève de la gestion des risques" not in markdown
    assert "### OBJECTIFS DE LA BANQUE" in markdown
    assert "La Banque maintient des fonds propres adéquats." in markdown


def test_build_text_extraction_markdown_never_reintroduces_table_block_as_heading() -> None:
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=76,
        end_page=76,
        anchor_page=75,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[
            PDFBlock(
                "p076_d002",
                76,
                [0.1, 0.33, 0.9, 0.36],
                "Les objectifs de la Banque en matière de gestion des fonds propres sont les suivants :",
                2,
                "narrative",
                True,
                "",
                "paragraph",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p076_d001",
                76,
                [0.1, 0.28, 0.8, 0.31],
                "OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES",
                1,
                "table",
                False,
                "table_like_block",
                "section_header",
                heading_level=1,
            ),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit])

    assert "### OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES" not in markdown
    assert "Les objectifs de la Banque" in markdown


def test_classify_block_type_preserves_docling_heading_over_table_overlap() -> None:
    block = PDFBlock(
        "p076_d001",
        76,
        [0.1, 0.28, 0.8, 0.31],
        "OBJECTIFS DE LA BANQUE EN MATIÈRE DE GESTION DES FONDS PROPRES",
        1,
        "other",
        False,
        "",
        "section_header",
        heading_level=1,
    )
    table_bboxes = [[0.08, 0.20, 0.92, 0.35]]

    assert _classify_block_type(block, {}, table_bboxes=table_bboxes) == "other"


def test_classify_block_type_rejects_table_column_header_labeled_as_section_header() -> None:
    block = PDFBlock(
        "p077_d010",
        77,
        [0.2, 0.40, 0.5, 0.43],
        "Réserve de conservation des fonds propres",
        10,
        "other",
        False,
        "",
        "section_header",
    )
    table_bboxes = [[0.08, 0.35, 0.95, 0.55]]

    assert _classify_block_type(block, {}, table_bboxes=table_bboxes) == "table"


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0.90, 0.02, 0.96, 0.05], "header_footer"),
        ([0.90, 0.93, 0.96, 0.97], "header_footer"),
        ([0.45, 0.45, 0.52, 0.49], "table"),
    ],
)
def test_classify_block_type_removes_isolated_printed_page_numbers(
    bbox: list[float],
    expected: str,
) -> None:
    block = PDFBlock(
        "p106_m001",
        106,
        bbox,
        "106",
        1,
        "other",
        False,
        "",
        "pymupdf_fallback",
    )

    assert _classify_block_type(block, {}) == expected


def test_build_text_extraction_markdown_excludes_structural_parent_heading() -> None:
    audit = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=10,
        end_page=10,
        anchor_page=10,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[
            PDFBlock(
                "p010_d003",
                10,
                [0.1, 0.36, 0.8, 0.45],
                "La Banque décrit les contrôles et les responsabilités associés au risque.",
                3,
                "narrative",
                True,
                "",
                "text",
            ),
        ],
        excluded_blocks=[
            PDFBlock(
                "p010_d001",
                10,
                [0.1, 0.28, 0.8, 0.31],
                "Risque opérationnel",
                1,
                "other",
                False,
                "non_narrative_block",
                "section_header",
            ),
            PDFBlock(
                "p010_d002",
                10,
                [0.1, 0.32, 0.8, 0.35],
                "Résilience opérationnelle",
                2,
                "other",
                False,
                "non_narrative_block",
                "section_header",
            ),
        ],
        semantic_units=[],
    )

    markdown = _build_text_extraction_markdown([audit])

    assert "### Risque opérationnel" not in markdown
    assert "### Résilience opérationnelle" in markdown


def test_compare_section_texts_skips_empty_orphan_headings(monkeypatch) -> None:
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kw: [],
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Header sans corps\n\n### Header apparié\n\nCorps T1.",
        text_t2="### Header apparié\n\nCorps T2.",
    )

    assert [change for change in changes if change["diff_type"] == "removed"] == []


def test_compare_section_texts_skips_matched_table_only_subsection(monkeypatch) -> None:
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kw: pytest.fail("Un tableau ne doit pas être envoyé à GPT."),
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_capital",
        text_t1="### Répartition\n\n| Catégorie | Valeur |\n| --- | ---: |\n| Crédit | 395 |\n",
        text_t2="### Répartition\n\n| Catégorie | Valeur |\n| --- | ---: |\n| Crédit | 436 |\n",
    )

    assert changes == []


def test_compare_section_texts_sends_financial_paragraphs_to_comparison(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _capture_comparison(**kwargs):
        captured["text_t1"] = kwargs["text_t1"]
        captured["text_t2"] = kwargs["text_t2"]
        raise RuntimeError("comparison reached")

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        _capture_comparison,
    )

    with pytest.raises(RuntimeError, match="comparison reached"):
        _compare_section_texts(
            client=object(),
            model="gpt-4o",
            section_key="gestion_capital",
            text_t1=(
                "### Ratio CET1\n\n"
                "Le ratio CET1 atteint 13,8 % et les fonds propres totalisent 525 M$.\n"
            ),
            text_t2=(
                "### Ratio CET1\n\n"
                "Le ratio CET1 atteint 14,2 % et les fonds propres totalisent 540 M$.\n"
            ),
        )

    assert "13,8 %" in captured["text_t1"]
    assert "525 M$" in captured["text_t1"]
    assert "14,2 %" in captured["text_t2"]
    assert "540 M$" in captured["text_t2"]


def test_run_text_analysis_pipeline_writes_md_as_source_of_truth(monkeypatch, tmp_path: Path) -> None:
    pdf_previous = tmp_path / "prev.pdf"
    pdf_current = tmp_path / "curr.pdf"
    pdf_previous.write_bytes(b"prev-pdf")
    pdf_current.write_bytes(b"curr-pdf")

    section = ResolvedSection(
        section_key="gestion_risques",
        title="Gestion des risques",
        start_page=3,
        end_page=4,
        anchor_page=3,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
    )
    audit_prev = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=3,
        end_page=4,
        anchor_page=3,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
        included_blocks=[PDFBlock("p003_b001", 3, [0.1, 0.3, 0.9, 0.4], "Texte exact T1", 1, "narrative", True, "")],
        excluded_blocks=[],
    )
    audit_curr = SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=3,
        end_page=4,
        anchor_page=3,
        anchor_text="Gestion des risques",
        anchor_bbox_norm=[0.1, 0.2, 0.9, 0.25],
        included_blocks=[PDFBlock("p004_b001", 4, [0.1, 0.3, 0.9, 0.4], "Texte exact T2", 1, "narrative", True, "")],
        excluded_blocks=[],
    )

    compare_texts_kwargs: dict = {}

    monkeypatch.setattr("vigilance.text_analysis_pipeline._build_openai_client", lambda: object())
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._resolve_sections",
        lambda pdf_path, bank_code, quarter=None, year=None: {"gestion_risques": section},
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._extract_audits_for_pdf",
        lambda **kwargs: ([audit_prev], "") if "prev" in str(kwargs["pdf_path"]) else ([audit_curr], ""),
    )

    def _fake_compare_section_texts(**kwargs):
        compare_texts_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_section_texts", _fake_compare_section_texts)
    monkeypatch.setattr("vigilance.text_analysis_pipeline._triage_section_changes", lambda **kwargs: [])

    payload, out_path = run_text_analysis_pipeline(
        bank_code="td",
        year_current=2025,
        quarter_current="t2",
        pdf_previous=pdf_previous,
        pdf_current=pdf_current,
        out_root=tmp_path / "outputs",
        model="gpt-4o",
    )

    # .md files are written and contain the right content
    assert payload["extraction_artifact_t1"] == "text_extraction_2025_t1.md"
    assert payload["extraction_artifact_t2"] == "text_extraction_2025_t2.md"
    assert out_path.exists()

    extraction_prev = out_path.parent / "text_extraction_2025_t1.md"
    extraction_curr = out_path.parent / "text_extraction_2025_t2.md"
    assert extraction_prev.exists()
    assert extraction_curr.exists()

    prev_md = extraction_prev.read_text()
    curr_md = extraction_curr.read_text()
    assert "## Gestion des risques" in prev_md
    assert "Texte exact T1" in prev_md
    assert "## Gestion des risques" in curr_md
    assert "Texte exact T2" in curr_md

    # GPT comparison received the .md section text directly (not SemanticUnits)
    assert "Texte exact T1" in compare_texts_kwargs.get("text_t1", "")
    assert "Texte exact T2" in compare_texts_kwargs.get("text_t2", "")
    assert payload["pipeline"] == "gpt4o_markdown_source_of_truth"


# ---------------------------------------------------------------------------
# Phase 1: subsection splitting and pairing
# ---------------------------------------------------------------------------


def test_parse_subsections_splits_on_triple_hash() -> None:
    md = (
        "Intro avant le premier heading.\n\n"
        "### Risque de marché\n\n"
        "Corps du risque de marché.\n\n"
        "### Risque de liquidité\n\n"
        "Corps du risque de liquidité.\n"
    )

    subs = _parse_subsections(md)

    assert len(subs) == 3
    assert subs[0] == ("__intro__", "Intro avant le premier heading.")
    assert subs[1][0] == "Risque de marché"
    assert "Corps du risque de marché" in subs[1][1]
    assert subs[2][0] == "Risque de liquidité"
    assert "Corps du risque de liquidité" in subs[2][1]


def test_parse_subsections_returns_single_intro_when_no_headings() -> None:
    md = "Texte sans sous-sections.\n\nDeuxième paragraphe.\n"

    subs = _parse_subsections(md)

    assert len(subs) == 1
    assert subs[0][0] == "__intro__"
    assert "Texte sans sous-sections" in subs[0][1]


def test_parse_subsections_returns_empty_for_blank_text() -> None:
    assert _parse_subsections("") == []
    assert _parse_subsections("   \n  ") == []


def test_parse_subsections_excludes_empty_headings_before_orphan_matching() -> None:
    md = (
        "### Situation des fonds propres\n\n"
        "### OBJECTIFS DE LA BANQUE\n\n"
        "La Banque maintient des fonds propres adéquats.\n"
    )

    assert _parse_subsections(md) == [
        ("OBJECTIFS DE LA BANQUE", "La Banque maintient des fonds propres adéquats.")
    ]


def test_chunk_subsection_text_splits_long_paragraphs_into_chunks() -> None:
    paragraphs = [
        (
            "Le risque de stratégie s'entend de la possibilité d'une perte financière ou d'une atteinte "
            "à la réputation attribuable à des stratégies commerciales inefficaces et à des réponses "
            "inadéquates aux changements du contexte commercial."
        ),
        (
            "Le risque de stratégie découle du risque que l'adoption de stratégies d'entreprise ou "
            "d'affaires n'aboutisse pas au résultat attendu en raison d'une mauvaise prise de décision "
            "ou d'une mise en œuvre inefficace."
        ),
        (
            "Le groupe Stratégies de l'organisation supervise le processus de planification stratégique "
            "et travaille avec les secteurs d'activité afin de détecter, de surveiller et d'atténuer les "
            "risques à l'échelle de l'organisation."
        ),
        (
            "Le cadre promeut la cohérence et la conformité aux normes de gestion, y compris l'utilisation "
            "des résultats de simulations de crise pour éclairer les décisions et tester les hypothèses "
            "stratégiques."
        ),
        (
            "Le risque stratégique englobe également le risque d'entreprise découlant des activités propres "
            "à l'entreprise et les répercussions que ces activités pourraient avoir sur les résultats."
        ),
        (
            "Notre performance financière dépend notamment de notre capacité à mettre en œuvre les plans "
            "stratégiques qu'élabore la direction et à repérer les risques émergents d'importance."
        ),
    ]

    chunks = _chunk_subsection_text(
        "\n\n".join(paragraphs),
        subsection_heading="Risque de stratégie",
        section_title="Gestion des risques",
    )

    assert [chunk.chunk_id for chunk in chunks] == ["c00", "c01", "c02", "c03", "c04", "c05"]
    assert [chunk.kind for chunk in chunks] == ["paragraph"] * 6
    assert chunks[0].hierarchy_path == "Gestion des risques > Risque de stratégie"
    assert chunks[5].text.startswith("Notre performance financière")


def test_chunk_subsection_text_keeps_pdf_bullet_list_as_one_chunk() -> None:
    text = (
        "‰ est appropriée, compte tenu des ratios cibles de BMO pour les fonds propres réglementaires;\n\n"
        "‰ soutient les stratégies des groupes d'exploitation de BMO et tient compte du contexte du marché;\n\n"
        "‰ maintient la confiance des déposants, des investisseurs et des organismes de réglementation."
    )

    chunks = _chunk_subsection_text(text, subsection_heading="Objectif", section_title="Gestion du capital")

    assert len(chunks) == 1
    assert chunks[0].kind == "list"
    assert "‰" not in chunks[0].text
    assert chunks[0].text.count("\n") == 2


def test_chunk_subsection_text_keeps_markdown_bullet_list_as_one_chunk() -> None:
    text = (
        "- premier élément de liste qui décrit une exigence de gouvernance;\n\n"
        "- deuxième élément de liste qui décrit une exigence de surveillance;\n\n"
        "- troisième élément de liste qui décrit une exigence de communication."
    )

    chunks = _chunk_subsection_text(text, subsection_heading="Objectif", section_title="Gestion du capital")

    assert len(chunks) == 1
    assert chunks[0].kind == "list"
    assert "- " not in chunks[0].text
    assert chunks[0].text.count("\n") == 2


def test_chunk_subsection_text_groups_checkbox_list_without_brackets() -> None:
    text = (
        "[] La Banque renforce ses contrôles de risque de crédit.\n\n"
        "[] Le cadre prévoit une surveillance accrue des portefeuilles sensibles.\n\n"
        "[] Les résultats sont transmis périodiquement au comité des risques."
    )

    chunks = _chunk_subsection_text(text, subsection_heading="Contrôles", section_title="Gestion des risques")

    assert len(chunks) == 1
    assert chunks[0].kind == "list"
    assert "[" not in chunks[0].text
    assert "]" not in chunks[0].text
    assert "La Banque renforce" in chunks[0].text
    assert "surveillance accrue" in chunks[0].text
    assert "transmis périodiquement" in chunks[0].text


def test_chunk_subsection_text_keeps_short_paragraph_independent() -> None:
    first = (
        "Ce paragraphe est assez long pour former un chunk autonome et décrit les responsabilités de "
        "surveillance, de contrôle, de gouvernance et de reddition de comptes dans la section."
    )
    short = "Bloc court."
    second = (
        "Ce second paragraphe est aussi assez long pour rester séparé et il décrit les mécanismes de "
        "suivi, les rapports périodiques et les indicateurs utilisés par la direction."
    )

    chunks = _chunk_subsection_text("\n\n".join([first, short, second]), subsection_heading="Gouvernance")

    assert [chunk.text for chunk in chunks] == [first, short, second]


def test_chunk_subsection_text_merges_first_short_label_with_its_paragraph() -> None:
    first = "Demande de capital"
    second = (
        "Ce paragraphe est suffisamment long pour absorber le libellé court qui le précède et former "
        "un seul chunk utile pour la comparaison sémantique entre deux rapports trimestriels."
    )

    chunks = _chunk_subsection_text("\n\n".join([first, second]), subsection_heading="Cadre")

    assert [chunk.text for chunk in chunks] == [f"{first}\n\n{second}"]


@pytest.mark.parametrize(
    "text",
    [
        "Crédit",
        "395",
        "Sans objet",
        "s.o. Sans objet",
        "Le tableau ci-dessus présente la variation des actifs.",
        "[pdf.66]",
        "Le ratio atteint 13,8 %.",
        "Le portefeuille vaut 525 M$.",
        "| Crédit | Marché | Opérationnel |",
        "Financement spécialisé aux États-Unis et International",
    ],
)
def test_chunk_subsection_text_keeps_every_non_table_fragment(text: str) -> None:
    chunks = _chunk_subsection_text(text, subsection_heading="Capital")

    assert len(chunks) == 1
    assert chunks[0].text


@pytest.mark.parametrize("marker", ["s.o.", "S.O.", "- s.o."])
def test_chunk_subsection_text_excludes_standalone_not_applicable_marker(marker: str) -> None:
    assert _chunk_subsection_text(marker, subsection_heading="Capital") == []


def test_chunk_subsection_text_excludes_only_a_structural_markdown_table() -> None:
    table = "| Catégorie | Valeur |\n| --- | ---: |\n| Crédit | 395 |"

    assert _chunk_subsection_text(table, subsection_heading="Capital") == []


def test_chunk_subsection_text_keeps_short_complete_narrative_sentence() -> None:
    text = "Un comité de risque est créé."

    chunks = _chunk_subsection_text(text, subsection_heading="Gouvernance")

    assert [chunk.text for chunk in chunks] == [text]


def test_chunk_subsection_text_requires_semantic_services_for_complex_paragraph() -> None:
    first_idea = " ".join(["La Banque surveille les risques de crédit de façon continue."] * 35)
    second_idea = " ".join(["Toutefois, le cadre prévoit des contrôles additionnels pour les portefeuilles sensibles."] * 35)
    third_idea = " ".join(["Par ailleurs, les résultats sont transmis aux comités de surveillance."] * 35)
    paragraph = f"{first_idea} {second_idea} {third_idea}"

    with pytest.raises(SemanticChunkingError, match="aucun fallback"):
        _chunk_subsection_text(paragraph, subsection_heading="Contrôles")


def test_chunk_subsection_text_excludes_markdown_headings() -> None:
    paragraph = (
        "Ce paragraphe narratif est assez long pour être conservé comme chunk autonome et il ne doit "
        "pas inclure le titre markdown qui le précède dans le texte remis au modèle."
    )

    chunks = _chunk_subsection_text(f"### Titre à exclure\n\n{paragraph}", subsection_heading="Titre réel")

    assert len(chunks) == 1
    assert chunks[0].text == paragraph
    assert "###" not in chunks[0].text
    assert "Titre à exclure" not in chunks[0].text


def test_align_chunks_tfidf_matches_shifted_chunks_and_marks_added() -> None:
    previous = [
        (
            "La définition du risque stratégique décrit la possibilité de pertes financières, "
            "de décisions commerciales inefficaces et de réponses inadéquates au contexte."
        ),
        (
            "Le groupe Stratégies de l'organisation supervise la planification stratégique, "
            "les contrôles de gouvernance et la surveillance des risques émergents."
        ),
    ]
    current = [
        previous[0],
        (
            "La banque ajoute un paragraphe distinct sur l'intelligence artificielle, les modèles "
            "analytiques et la surveillance des nouveaux outils numériques."
        ),
        previous[1],
    ]
    chunks_t1 = _chunk_subsection_text("\n\n".join(previous), subsection_heading="Risque de stratégie")
    chunks_t2 = _chunk_subsection_text("\n\n".join(current), subsection_heading="Risque de stratégie")

    alignments = _align_chunks_tfidf(chunks_t1, chunks_t2)
    matched_pairs = {
        (alignment.chunk_t1.chunk_id, alignment.chunk_t2.chunk_id)
        for alignment in alignments
        if alignment.chunk_t1 and alignment.chunk_t2
    }
    added = [alignment for alignment in alignments if alignment.alignment_type == "possible_added"]

    assert ("c00", "c00") in matched_pairs
    assert ("c01", "c02") in matched_pairs
    assert len(added) == 1
    assert added[0].chunk_t2.chunk_id == "c01"
    assert added[0].candidates_t1_for_t2


def test_align_chunks_tfidf_enforces_one_to_one() -> None:
    previous = (
        "Le cadre de gouvernance du risque stratégique prévoit une surveillance indépendante, "
        "des simulations de crise et des rapports réguliers au conseil."
    )
    current = "\n\n".join([previous, previous])
    chunks_t1 = _chunk_subsection_text(previous, subsection_heading="Risque de stratégie")
    chunks_t2 = _chunk_subsection_text(current, subsection_heading="Risque de stratégie")

    alignments = _align_chunks_tfidf(chunks_t1, chunks_t2)
    matched = [alignment for alignment in alignments if alignment.chunk_t1 and alignment.chunk_t2]
    added = [alignment for alignment in alignments if alignment.alignment_type == "possible_added"]

    assert len(matched) == 1
    assert matched[0].chunk_t1.chunk_id == "c00"
    assert len(added) == 1
    assert added[0].chunk_t2.chunk_id == "c01"


def test_align_chunks_tfidf_handles_empty_sklearn_vocabulary() -> None:
    chunks_t1 = _chunk_subsection_text("123 456 789", subsection_heading="Risque de stratégie", min_chars=0)
    chunks_t2 = _chunk_subsection_text("le la de et un une", subsection_heading="Risque de stratégie", min_chars=0)

    alignments = _align_chunks_tfidf(chunks_t1, chunks_t2)
    matched = [alignment for alignment in alignments if alignment.chunk_t1 and alignment.chunk_t2]
    added = [alignment for alignment in alignments if alignment.alignment_type == "possible_added"]
    removed = [alignment for alignment in alignments if alignment.alignment_type == "possible_removed"]

    assert matched == []
    assert [alignment.chunk_t2.text for alignment in added if alignment.chunk_t2] == ["le la de et un une"]
    assert [alignment.chunk_t1.text for alignment in removed if alignment.chunk_t1] == ["123 456 789"]


def test_semantic_alignment_decision_confirms_ambiguous_pair_before_triage() -> None:
    previous = "La Banque applique une limite de risque de crédit pour ses portefeuilles commerciaux."
    current = "La Banque applique une limite de risque de crédit révisée pour ses portefeuilles commerciaux."
    chunk_t1 = _chunk_subsection_text(previous, subsection_heading="Risque de crédit")[0]
    chunk_t2 = _chunk_subsection_text(current, subsection_heading="Risque de crédit")[0]
    alignment = ChunkAlignment("a00", "ambiguous", chunk_t1, chunk_t2, 0.42, [], [], "test")

    scoped = _attach_alignment_metadata(
        [
            {
                "alignment_id": "a00",
                "diff_type": "modified",
                "source_text_t1": previous,
                "source_text_t2": current,
                "alignment_decision": "same_disclosure",
                "alignment_confidence": "high",
                "alignment_rationale": "Même limite de risque, avec une mise à jour explicite.",
            }
        ],
        [alignment],
    )

    assert scoped[0]["alignment_type"] == "ambiguous"
    assert scoped[0]["alignment_decision"] == "same_disclosure"
    assert scoped[0]["alignment_confidence"] == "high"


def test_semantic_distinct_disclosures_are_materialized_as_added_and_removed() -> None:
    changes = [
        {
            "alignment_id": "a02",
            "alignment_type": "ambiguous",
            "alignment_decision": "distinct_disclosures",
            "alignment_confidence": "high",
            "alignment_rationale": "Deux émissions distinctes, à des dates et en devises différentes.",
            "diff_type": "modified",
            "source_text_t1": "Émission américaine de juillet 2024.",
            "source_text_t2": "Émission canadienne de décembre 2024.",
            "semantic_text_t1": "Émission américaine de juillet.",
            "semantic_text_t2": "Émission canadienne de décembre.",
            "evidence_t1": {"pages": [], "snippet": "Émission américaine"},
            "evidence_t2": {"pages": [], "snippet": "Émission canadienne"},
            "change_summary": "Les conditions d'émission diffèrent.",
        }
    ]

    materialized = _materialize_semantic_alignment_decisions(changes)

    assert [change["diff_type"] for change in materialized] == ["removed", "added"]
    assert materialized[0]["source_text_t2"] == ""
    assert materialized[1]["source_text_t1"] == ""
    assert all(change["alignment_type"] == "semantic_distinct" for change in materialized)
    assert all(change["alignment_decision"] == "distinct_disclosures" for change in materialized)
    assert all(change["semantic_alignment_group_id"] == "a02" for change in materialized)


def test_global_reconciliation_removes_bnc_style_resegmented_fragments(monkeypatch) -> None:
    """One old block split into two moved current blocks must not survive as 3 changes."""
    mitigation = (
        "Malgré ces mesures préventives, la Banque compte sur des mécanismes d'atténuation "
        "élaborés avec les propriétaires d'entente et les tiers concernés."
    )
    b10 = (
        "Face à un écosystème de tiers plus vaste, le BSIF a publié la ligne directrice B-10 "
        "sur la gestion du risque lié aux tiers, entrée en vigueur le premier mai 2024."
    )
    old_block = f"{mitigation} {b10}"
    changes = [
        {
            "change_id": "old_c16",
            "section_key": "gestion_risques",
            "subsection_heading": "Description",
            "diff_type": "removed",
            "source_text_t1": old_block,
            "source_text_t2": "",
        },
        {
            "change_id": "new_c00",
            "section_key": "gestion_risques",
            "subsection_heading": "Dépendance envers les tiers et les modèles",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": b10,
        },
        {
            "change_id": "new_c02",
            "section_key": "gestion_risques",
            "subsection_heading": "Dépendance envers les tiers et les modèles",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": mitigation,
        },
    ]

    response = _ReconciliationResponse.model_validate(
        {
            "decision": "moved_unchanged",
            "confidence": "high",
            "rationale": "Le bloc T1 est conservé dans T2 sous deux fragments déplacés et réordonnés.",
            "matches": [
                {"t1_node_id": "n0000", "t2_node_id": "n0001", "text_t1": b10, "text_t2": b10},
                {"t1_node_id": "n0000", "t2_node_id": "n0002", "text_t1": mitigation, "text_t2": mitigation},
            ],
        }
    )
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._call_structured_completion_with_correction",
        lambda *_args, **_kwargs: response,
    )

    reconciled, audit = reconcile_global_change_fragments(
        client=object(),
        model="gpt-4o",
        changes=changes,
    )

    assert reconciled == []
    assert audit[0]["decision"] == "moved_unchanged"
    assert audit[0]["fully_covered"] is True
    assert audit[0]["applied"] is True


def test_global_reconciliation_keeps_a_genuine_unmatched_addition(monkeypatch) -> None:
    addition = (
        "Face aux défis actuels, la Banque adopte une approche proactive et investit dans "
        "la modernisation des infrastructures pour renforcer sa résilience opérationnelle."
    )
    changes = [
        {
            "change_id": "new_security",
            "section_key": "gestion_risques",
            "subsection_heading": "Sécurité de l'information",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": addition,
        }
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._call_structured_completion_with_correction",
        lambda *_args, **_kwargs: pytest.fail("Aucun candidat opposé : pas d'appel GPT de réconciliation."),
    )

    reconciled, audit = reconcile_global_change_fragments(
        client=object(),
        model="gpt-4o",
        changes=changes,
    )

    assert reconciled == changes
    assert audit == []


def test_bnc_t4_global_reconciliation_detects_the_split_third_party_block() -> None:
    """The BNC artifact from the screenshot forms one cross-subsection component."""
    artifact = Path("outputs/resultats/bnc/2025_t4_vs_2024_t4/text_comparison.json")
    if not artifact.exists():
        pytest.skip("Artefact local BNC T4 absent.")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    rows = [
        change
        for section in payload.get("section_comparisons", [])
        for change in section.get("all_block_comparisons", [])
    ]
    unique_rows = list({row.get("change_id"): row for row in rows if row.get("change_id")}.values())
    expected_ids = {
        "gestion_risques_description_change_094",
        "gestion_risques_dépendance_envers_les_tiers_et_les_modèl_change_140",
        "gestion_risques_dépendance_envers_les_tiers_et_les_modèl_change_142",
    }
    present_ids = {str(row.get("change_id") or "") for row in unique_rows}
    if not expected_ids.issubset(present_ids):
        pytest.skip(
            "Les change_id historiques du cas BNC T4 ne sont plus présents dans l'artefact local."
        )

    components, _edges = _components(_one_sided_nodes(unique_rows))
    component_ids = [
        {str(node.change.get("change_id") or "") for node in component}
        for component in components
    ]

    assert expected_ids in component_ids


def test_alignment_prompt_limits_weak_candidate_context() -> None:
    primary_text = (
        "Le risque stratégique est surveillé par un cadre de gouvernance précis avec des contrôles "
        "internes et des rapports réguliers destinés au conseil."
    )
    long_candidate_text = (
        "Ce candidat alternatif contient un texte très long qui ne devrait pas être recopié au complet "
        "dans le prompt lorsque l'alignement est faible mais déjà apparié à une paire principale. "
        "La limitation protège la taille du contexte transmis au modèle et conserve seulement un "
        "extrait utile pour vérifier rapidement qu'il ne s'agit pas d'un meilleur candidat local. "
        "Cette dernière phrase ne doit pas apparaître dans le prompt si l'extrait est bien tronqué."
    )
    primary_t1 = _chunk_subsection_text(primary_text, subsection_heading="Risque de stratégie")[0]
    primary_t2 = _chunk_subsection_text(primary_text, subsection_heading="Risque de stratégie")[0]
    long_candidate = _chunk_subsection_text(long_candidate_text, subsection_heading="Risque de stratégie")[0]
    alignments = [
        ChunkAlignment(
            alignment_id="a00",
            alignment_type="matched_weak",
            chunk_t1=primary_t1,
            chunk_t2=primary_t2,
            similarity_score=0.53,
            candidates_t1_for_t2=[
                ChunkCandidate("c00", long_candidate.chunk_id, 0.22, long_candidate),
            ],
            candidates_t2_for_t1=[],
            reason="tfidf_one_to_one",
        )
    ]

    prompt_t1, prompt_t2 = _format_alignments_for_prompt(alignments)
    prompt = f"{prompt_t1}\n{prompt_t2}"

    assert "extrait 300 caractères" in prompt
    assert "Cette dernière phrase ne doit pas apparaître" not in prompt


def test_build_comparison_batches_uses_type_specific_sizes() -> None:
    base_chunk = _chunk_subsection_text(
        "Le risque stratégique est surveillé par un cadre de gouvernance précis avec des contrôles internes.",
        subsection_heading="Risque de stratégie",
    )[0]
    alignments = [
        ChunkAlignment(f"a{index:02d}", "matched_strong", base_chunk, base_chunk, 0.95, [], [], "test")
        for index in range(6)
    ]
    alignments.extend(
        ChunkAlignment(f"w{index:02d}", "matched_weak", base_chunk, base_chunk, 0.55, [], [], "test")
        for index in range(4)
    )
    alignments.extend(
        [
            ChunkAlignment("x00", "ambiguous", base_chunk, base_chunk, 0.40, [], [], "test"),
            ChunkAlignment("x01", "possible_added", None, base_chunk, 0.0, [], [], "test"),
            ChunkAlignment("x02", "possible_removed", base_chunk, None, 0.0, [], [], "test"),
        ]
    )

    batches = _build_comparison_batches(
        alignments=alignments,
        heading_label="Risque de stratégie",
        heading_slug="risque_de_strategie",
    )

    assert [(batch.alignment_type, len(batch.alignments)) for batch in batches] == [
        ("matched_strong", 5),
        ("matched_strong", 1),
        ("matched_weak", 3),
        ("matched_weak", 1),
        ("ambiguous", 1),
        ("possible_added", 1),
        ("possible_removed", 1),
    ]
    assert [batch.batch_id for batch in batches] == ["b00", "b01", "b02", "b03", "b04", "b05", "b06"]


def test_normalize_heading_strips_table_prefix_and_lowercases() -> None:
    assert _normalize_heading("T22 Mesures du risque de marché") == "mesures du risque de marché"
    assert _normalize_heading("Risque de liquidité") == "risque de liquidité"


def test_normalize_heading_strips_pdf_page_suffixes() -> None:
    assert (
        _normalize_heading("Structure de la gouvernance du risque [pdf.62]")
        == "structure de la gouvernance du risque"
    )
    assert (
        _normalize_heading("Structure de la gouvernance du risque [pdf.56]")
        == "structure de la gouvernance du risque"
    )


def test_normalize_heading_treats_renamed_tariff_headings_as_distinct() -> None:
    # Exact-match normalisation: these two are NOT the same after stripping
    n1 = _normalize_heading("Incidence des tarifs")
    n2 = _normalize_heading("Incidence des tarifs douaniers")
    assert n1 != n2


def test_pair_subsections_matches_identical_headings() -> None:
    subs_t1 = [("Risque de marché", "Corps T1"), ("Risque de liquidité", "Liquide T1")]
    subs_t2 = [("Risque de marché", "Corps T2"), ("Risque de liquidité", "Liquide T2")]

    pairs = _pair_subsections(subs_t1, subs_t2)

    assert len(pairs) == 2
    assert all(h1 is not None and h2 is not None for h1, _, h2, _ in pairs)


def test_pair_subsections_matches_headings_with_different_pdf_page_suffixes() -> None:
    subs_t1 = [("Structure de la gouvernance du risque [pdf.62]", "Corps T1")]
    subs_t2 = [("Structure de la gouvernance du risque [pdf.56]", "Corps T2")]

    pairs = _pair_subsections(subs_t1, subs_t2)

    assert pairs == [
        (
            "Structure de la gouvernance du risque [pdf.62]",
            "Corps T1",
            "Structure de la gouvernance du risque [pdf.56]",
            "Corps T2",
        )
    ]


def test_pair_subsections_marks_t1_only_heading_as_removed() -> None:
    subs_t1 = [("Risque de marché", "Corps T1"), ("Risque opérationnel", "Opérationnel T1")]
    subs_t2 = [("Risque de marché", "Corps T2")]

    pairs = _pair_subsections(subs_t1, subs_t2)

    removed = [(h1, h2) for h1, _, h2, _ in pairs if h2 is None]
    assert len(removed) == 1
    assert removed[0][0] == "Risque opérationnel"


def test_pair_subsections_marks_t2_only_heading_as_added() -> None:
    subs_t1 = [("Risque de marché", "Corps T1")]
    subs_t2 = [("Risque de marché", "Corps T2"), ("Incidence des tarifs", "Nouveau T2")]

    pairs = _pair_subsections(subs_t1, subs_t2)

    added = [(h1, h2) for h1, _, h2, _ in pairs if h1 is None]
    assert len(added) == 1
    assert added[0][1] == "Incidence des tarifs"


def test_compare_section_texts_rejects_non_empty_sections_without_subsections() -> None:
    """Sections non vides sans ### doivent échouer explicitement."""
    with pytest.raises(TextAnalysisQualityError, match="sans sous-sections ###"):
        _compare_section_texts(
            client=object(),
            model="gpt-4o",
            section_key="gestion_risques",
            text_t1="Texte T1 sans sous-sections.",
            text_t2="Texte T2 sans sous-sections.",
        )


def test_compare_section_texts_marks_empty_matched_subsection_side_as_removed() -> None:
    """Un heading apparié vide côté courant devient un retrait synthétique."""
    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Responsables\n\nAncien paragraphe présent uniquement dans le rapport précédent.",
        text_t2="### Responsables\n\n",
    )

    assert len(changes) == 1
    assert changes[0]["diff_type"] == "removed"
    assert changes[0]["source_scope"] == "chunk"
    assert changes[0]["subsection_heading"] == "Responsables"
    assert "Ancien paragraphe" in changes[0]["source_text_t1"]


def test_compare_section_texts_calls_gpt_once_per_subsection_pair(monkeypatch) -> None:
    """Deux sous-sections appariées → deux appels GPT distincts."""
    calls: list[str] = []

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        calls.append(heading_slug)
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    md_t1 = "### Risque de marché\n\nCorps T1 A.\n\n### Risque de liquidité\n\nCorps T1 B.\n"
    md_t2 = "### Risque de marché\n\nCorps T2 A.\n\n### Risque de liquidité\n\nCorps T2 B.\n"

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    assert len(calls) == 2


def test_compare_section_texts_sends_chunked_subsection_bodies(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        captured["text_t1"] = text_t1
        captured["text_t2"] = text_t2
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    paragraph_a = (
        "Le risque de stratégie s'entend de la possibilité d'une perte financière ou d'une atteinte à la "
        "réputation attribuable à des stratégies commerciales inefficaces et à des réponses inadéquates."
    )
    paragraph_b = (
        "Le groupe Stratégies de l'organisation supervise le processus de planification stratégique et "
        "travaille avec les secteurs d'activité afin de détecter, de surveiller et d'atténuer les risques."
    )

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de stratégie\n\n{paragraph_a}\n\n{paragraph_b}",
        text_t2=f"### Risque de stratégie\n\n{paragraph_a}\n\n{paragraph_b}",
    )

    assert "[c00 | paragraph | Gestion des risques > Risque de stratégie]" in captured["text_t1"]
    assert "[c01 | paragraph | Gestion des risques > Risque de stratégie]" in captured["text_t1"]
    assert paragraph_a in captured["text_t1"]
    assert paragraph_b in captured["text_t2"]


def test_compare_section_texts_sends_tfidf_alignment_context(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        calls.append({"text_t1": text_t1, "text_t2": text_t2})
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    previous = (
        "Le cadre de gouvernance du risque stratégique prévoit une surveillance indépendante, "
        "des simulations de crise et des rapports réguliers au conseil d'administration."
    )
    added = (
        "La banque ajoute un paragraphe distinct sur l'intelligence artificielle, les modèles "
        "analytiques et la surveillance des nouveaux outils numériques."
    )

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de stratégie\n\n{previous}",
        text_t2=f"### Risque de stratégie\n\n{previous}\n\n{added}",
    )

    joined_t1 = "\n".join(call["text_t1"] for call in calls)
    joined_t2 = "\n".join(call["text_t2"] for call in calls)
    assert "[a00 | matched_strong" in joined_t1
    assert "[a01 | possible_added" in joined_t2
    assert "Meilleurs candidats T1 à vérifier" in joined_t2
    assert "[c00 | paragraph | Gestion des risques > Risque de stratégie]" in joined_t1


def test_compare_section_texts_chunk_change_carries_alignment_metadata(monkeypatch) -> None:
    paragraphs_t1 = [
        (
            "Le premier paragraphe décrit un contrôle stratégique durable, une gouvernance claire "
            "et des rapports réguliers pour former un chunk autonome."
        ),
        (
            "Le deuxième paragraphe précise que la Banque surveille les risques réglementaires "
            "au moyen de tests indépendants et de rapports au comité."
        ),
        (
            "Le troisième paragraphe décrit la mise à jour annuelle du cadre, les responsabilités "
            "des dirigeants et les mécanismes de suivi."
        ),
    ]
    paragraphs_t2 = list(paragraphs_t1)
    paragraphs_t2[1] = (
        "Le deuxième paragraphe précise que la Banque surveille les risques réglementaires "
        "au moyen de tests indépendants, d'indicateurs et de rapports au comité."
    )

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._call_structured_completion_with_correction",
        lambda *args, **kwargs: ChunkComparisonLLMResponse(
            changes=[
                {
                    "alignment_id": "a01",
                    "diff_type": "modified",
                    "text_t1": paragraphs_t1[1],
                    "text_t2": paragraphs_t2[1],
                    "change_summary": "Ajout d'indicateurs dans la surveillance.",
                }
            ]
        ),
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Risque de stratégie\n\n" + "\n\n".join(paragraphs_t1),
        text_t2="### Risque de stratégie\n\n" + "\n\n".join(paragraphs_t2),
    )

    assert len(changes) == 1
    assert changes[0]["source_scope"] == "chunk"
    assert changes[0]["alignment_id"] == "a01"
    assert changes[0]["alignment_type"] == "matched_strong"
    assert changes[0]["chunk_id_t1"] == "c01"
    assert changes[0]["chunk_id_t2"] == "c01"
    assert changes[0]["source_text_t1"] == paragraphs_t1[1]
    assert changes[0]["source_text_t2"] == paragraphs_t2[1]


def test_compare_section_texts_chunk_change_never_keeps_full_multichunk_body(monkeypatch) -> None:
    paragraphs_t1 = [
        (
            "Le premier bloc décrit une gouvernance robuste, un mandat clair et des mécanismes "
            "de supervision utilisés dans le cadre de gestion des risques."
        ),
        (
            "Le deuxième bloc décrit les contrôles de conformité, les tests indépendants et "
            "les rapports destinés aux comités de surveillance."
        ),
        (
            "Le troisième bloc décrit les responsabilités de la direction, les examens annuels "
            "et les améliorations apportées au cadre."
        ),
    ]
    paragraphs_t2 = list(paragraphs_t1)
    paragraphs_t2[1] = (
        "Le deuxième bloc décrit les contrôles de conformité, les tests indépendants, "
        "les indicateurs et les rapports destinés aux comités de surveillance."
    )
    body_t1 = "\n\n".join(paragraphs_t1)
    body_t2 = "\n\n".join(paragraphs_t2)

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._call_structured_completion_with_correction",
        lambda *args, **kwargs: ChunkComparisonLLMResponse(
            changes=[
                {
                    "alignment_id": "a01",
                    "diff_type": "modified",
                    "text_t1": body_t1,
                    "text_t2": body_t2,
                    "change_summary": "Le LLM a retourné trop large.",
                }
            ]
        ),
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de stratégie\n\n{body_t1}",
        text_t2=f"### Risque de stratégie\n\n{body_t2}",
    )

    assert len(changes) == 1
    assert changes[0]["source_scope"] == "chunk"
    assert changes[0]["alignment_id"] == "a01"
    assert changes[0]["source_text_t1"] == paragraphs_t1[1]
    assert changes[0]["source_text_t2"] == paragraphs_t2[1]
    assert changes[0]["source_text_t1"] != body_t1
    assert changes[0]["source_text_t2"] != body_t2


def test_compare_section_texts_splits_large_alignment_set_into_batches(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        calls.append({"text_t1": text_t1, "text_t2": text_t2})
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    paragraphs = [
        (
            f"Le paragraphe {index} décrit un contrôle stratégique distinct, une responsabilité de gouvernance "
            f"et un mécanisme de surveillance propre au risque de stratégie pour produire un chunk autonome."
        )
        for index in range(6)
    ]
    body = "\n\n".join(paragraphs)

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de stratégie\n\n{body}",
        text_t2=f"### Risque de stratégie\n\n{body}",
    )

    assert len(calls) == 2
    first_batch = next(call for call in calls if "[c00 |" in call["text_t1"])
    second_batch = next(call for call in calls if "[c05 |" in call["text_t1"])
    assert "[c04 |" in first_batch["text_t1"]
    assert "[c05 |" not in first_batch["text_t1"]
    assert "[c00 |" not in second_batch["text_t1"]


def test_compare_section_texts_merges_parallel_batch_results_in_source_order(monkeypatch) -> None:
    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        if "[c05 |" in text_t2:
            alignment_id = "a05"
            label = "second_batch"
            source = paragraphs[5]
        else:
            alignment_id = "a00"
            label = "first_batch"
            source = paragraphs[0]
        return [
            {
                "change_id": f"temporary_{label}",
                "section_key": section_key,
                "subsection_heading": heading_label,
                "diff_type": "modified",
                "alignment_id": alignment_id,
                "semantic_text_t1": source,
                "semantic_text_t2": source,
                "source_text_t1": source,
                "source_text_t2": source,
                "source_block_ids_t1": [],
                "source_block_ids_t2": [],
                "source_refs_t1": [],
                "source_refs_t2": [],
                "pages_t1": [],
                "pages_t2": [],
                "source_resolution_t1": "markdown",
                "source_resolution_t2": "markdown",
                "evidence_t1": {"pages": [], "snippet": source},
                "evidence_t2": {"pages": [], "snippet": source},
                "change_summary": label,
            }
        ]

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    paragraphs = [
        (
            f"Le paragraphe {index} décrit un contrôle stratégique distinct, une responsabilité de gouvernance "
            f"et un mécanisme de surveillance propre au risque de stratégie pour produire un chunk autonome."
        )
        for index in range(6)
    ]
    body = "\n\n".join(paragraphs)

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de stratégie\n\n{body}",
        text_t2=f"### Risque de stratégie\n\n{body}",
    )

    assert [change["source_text_t2"] for change in changes] == [paragraphs[0], paragraphs[5]]
    assert [change["change_id"] for change in changes] == [
        "gestion_risques_risque_de_stratégie_change_001",
        "gestion_risques_risque_de_stratégie_change_002",
    ]
    assert [change["alignment_id"] for change in changes] == ["a00", "a05"]
    assert [change["source_scope"] for change in changes] == ["chunk", "chunk"]


def test_compare_section_texts_reports_batch_id_on_batch_failure(monkeypatch) -> None:
    def fake_single_call(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    paragraph = (
        "Le risque stratégique est surveillé par un cadre de gouvernance précis avec des contrôles internes "
        "et des rapports réguliers destinés au conseil afin de former un chunk autonome."
    )

    with pytest.raises(RuntimeError, match="b00"):
        _compare_section_texts(
            client=object(),
            model="gpt-4o",
            section_key="gestion_risques",
            text_t1=f"### Risque de stratégie\n\n{paragraph}",
            text_t2=f"### Risque de stratégie\n\n{paragraph}",
        )


def test_compare_section_texts_synthetic_change_for_removed_subsection(monkeypatch) -> None:
    """Une sous-section T1 sans contrepartie T2 produit un retrait par chunk."""
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kw: [],
    )

    md_t1 = "### Risque de marché\n\nCorps T1.\n\n### Risque opérationnel\n\nSupprimé en T2.\n"
    md_t2 = "### Risque de marché\n\nCorps T2.\n"

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    removed = [c for c in changes if c["diff_type"] == "removed"]
    assert len(removed) == 1
    assert removed[0]["source_scope"] == "chunk"
    assert "Risque opérationnel" in removed[0]["change_summary"]
    assert "Supprimé en T2" in removed[0]["source_text_t1"]


def test_compare_section_texts_synthetic_change_for_added_subsection(monkeypatch) -> None:
    """Une sous-section T2 sans contrepartie T1 produit un ajout par chunk."""
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kw: [],
    )

    md_t1 = "### Risque de marché\n\nCorps T1.\n"
    md_t2 = "### Risque de marché\n\nCorps T2.\n\n### Incidence des tarifs\n\nNouveau en T2.\n"

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    added = [c for c in changes if c["diff_type"] == "added"]
    assert len(added) == 1
    assert added[0]["source_scope"] == "chunk"
    assert "Incidence des tarifs" in added[0]["change_summary"]
    assert "Nouveau en T2" in added[0]["source_text_t2"]


def test_compare_section_texts_chunks_unmatched_long_subsection(monkeypatch) -> None:
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kw: [],
    )
    paragraphs = [
        f"Le paragraphe {index} décrit un contrôle distinct, une responsabilité claire et un suivi périodique."
        for index in range(1, 4)
    ]

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Risque de marché\n\nCorps T1.",
        text_t2="### Risque de marché\n\nCorps T2.\n\n### Nouveau cadre\n\n" + "\n\n".join(paragraphs),
    )

    added = [change for change in changes if change["diff_type"] == "added"]
    assert [change["source_text_t2"] for change in added] == paragraphs
    assert all(change["source_scope"] == "chunk" for change in added)


def test_compare_section_texts_deduplicates_multiple_llm_details_for_one_alignment(monkeypatch) -> None:
    previous = "Le cadre prévoit une surveillance régulière des risques de marché et des rapports au comité."
    current = previous + " Il ajoute un indicateur de concentration."
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._call_structured_completion_with_correction",
        lambda *args, **kwargs: ChunkComparisonLLMResponse(
            changes=[
                {
                    "alignment_id": "a00",
                    "diff_type": "modified",
                    "text_t1": previous,
                    "text_t2": current,
                    "change_summary": "Ajout d'un indicateur.",
                },
                {
                    "alignment_id": "a00",
                    "diff_type": "modified",
                    "text_t1": previous,
                    "text_t2": current,
                    "change_summary": "Précision du suivi.",
                },
            ]
        ),
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=f"### Risque de marché\n\n{previous}",
        text_t2=f"### Risque de marché\n\n{current}",
    )

    assert len(changes) == 1
    assert "Ajout d'un indicateur" in changes[0]["change_summary"]
    assert "Précision du suivi" in changes[0]["change_summary"]


def test_gpt_match_orphan_headings_returns_empty_when_no_orphans() -> None:
    """Pas d'orphelins d'un côté → pas d'appel GPT, retourne []."""
    result = _gpt_match_orphan_headings(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=[],
        orphans_t2=["Incidence des tarifs douaniers"],
    )
    assert result == []


def test_orphan_match_llm_response_rejects_invalid_confidence() -> None:
    with pytest.raises(Exception, match="confidence"):
        OrphanMatchLLMResponse.model_validate(
            {
                "matches": [
                    {
                        "heading_t1": "Ancien titre",
                        "heading_t2": "Nouveau titre",
                        "confidence": "certain",
                        "reason": "x",
                    }
                ]
            }
        )


def test_gpt_match_orphan_headings_filters_low_confidence(monkeypatch) -> None:
    """Matches de confidence 'low' sont exclus du résultat."""
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._deterministic_match_orphan_headings",
        lambda *_args, **_kwargs: [],
    )

    def fake_call(client, *, model, messages, **kwargs):
        return OrphanMatchLLMResponse(
            matches=[
                {"heading_t1": "Incidence des tarifs", "heading_t2": "Incidence des tarifs douaniers", "confidence": "low", "reason": "x"},
            ]
        )
    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction", fake_call)

    result = _gpt_match_orphan_headings(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=["Incidence des tarifs"],
        orphans_t2=["Incidence des tarifs douaniers"],
    )
    assert result == []


def test_gpt_match_orphan_headings_rejects_hallucinated_headings(monkeypatch) -> None:
    """GPT invente un heading absent des listes orphelines → rejeté."""
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._deterministic_match_orphan_headings",
        lambda *_args, **_kwargs: [],
    )

    def fake_call(client, *, model, messages, **kwargs):
        return OrphanMatchLLMResponse(
            matches=[
                {"heading_t1": "Heading inventé", "heading_t2": "Incidence des tarifs douaniers", "confidence": "high", "reason": "x"},
            ]
        )
    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction", fake_call)

    result = _gpt_match_orphan_headings(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=["Incidence des tarifs"],
        orphans_t2=["Incidence des tarifs douaniers"],
    )
    assert result == []


def test_gpt_match_orphan_headings_accepts_high_confidence(monkeypatch) -> None:
    """Un match high-confidence avec headings valides est retourné."""
    def fake_call(client, *, model, messages, **kwargs):
        return OrphanMatchLLMResponse(
            matches=[
                {"heading_t1": "Incidence des tarifs", "heading_t2": "Incidence des tarifs douaniers", "confidence": "high", "reason": "précision ajoutée"},
            ]
        )
    monkeypatch.setattr("vigilance.text_analysis_pipeline._call_structured_completion_with_correction", fake_call)

    result = _gpt_match_orphan_headings(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=["Incidence des tarifs"],
        orphans_t2=["Incidence des tarifs douaniers"],
    )
    assert len(result) == 1
    assert result[0]["heading_t1"] == "Incidence des tarifs"
    assert result[0]["heading_t2"] == "Incidence des tarifs douaniers"


def test_gpt_match_orphan_headings_enforces_1_to_1(monkeypatch) -> None:
    """GPT tente d'associer le même T1 heading à deux T2 headings → seule la première paire est acceptée."""
    from vigilance.text_analysis import subsection_matching as sm

    monkeypatch.setattr(sm, "_deterministic_match_orphan_headings", lambda *_args, **_kwargs: [])

    def fake_call(client, *, model, messages, **kwargs):
        return OrphanMatchLLMResponse(
            matches=[
                {"heading_t1": "Risque de marché", "heading_t2": "Risque de marché amplifié", "confidence": "high", "reason": "a"},
                {"heading_t1": "Risque de marché", "heading_t2": "Risque de marché étendu", "confidence": "medium", "reason": "b"},
            ]
        )

    monkeypatch.setattr(sm, "_call_structured_completion_with_correction", fake_call)

    result = sm._gpt_match_orphan_headings(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=["Risque de marché"],
        orphans_t2=["Risque de marché amplifié", "Risque de marché étendu"],
    )
    assert len(result) == 1
    assert result[0]["heading_t2"] == "Risque de marché amplifié"


def test_compare_section_texts_resolves_renamed_subsection(monkeypatch) -> None:
    """Une sous-section renommée T1→T2 est exposée puis comparée."""
    single_call_slugs: list[str] = []
    single_call_labels: list[str] = []

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        single_call_slugs.append(heading_slug)
        single_call_labels.append(heading_label)
        return []

    def fake_resolve_orphans(*, client, model, section_key, orphans_t1, orphans_t2, embedding_model="text-embedding-3-small"):
        return [
            {
                "heading_t1": "Incidence des tarifs",
                "heading_t2": "Incidence des tarifs douaniers",
                "confidence": "high",
                "reason": "précision",
                "match_source": "llm",
            },
        ]

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)
    monkeypatch.setattr("vigilance.text_analysis_pipeline._resolve_orphan_subsections", fake_resolve_orphans)

    md_t1 = "### Risque de marché\n\nCorps T1.\n\n### Incidence des tarifs\n\nTexte T1.\n"
    md_t2 = "### Risque de marché\n\nCorps T2.\n\n### Incidence des tarifs douaniers\n\nTexte T2.\n"

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    # Two GPT comparison calls: one for exact match, one for renamed pair
    assert len(single_call_slugs) == 2
    # The renamed pair label uses "→" arrow notation
    renamed_labels = [lbl for lbl in single_call_labels if "→" in lbl]
    assert len(renamed_labels) == 1
    assert renamed_labels[0] == "Incidence des tarifs → Incidence des tarifs douaniers"
    renamed_changes = [change for change in changes if change["diff_type"] == "renamed"]
    assert len(renamed_changes) == 1
    assert renamed_changes[0]["source_scope"] == "heading"
    assert renamed_changes[0]["source_text_t1"] == "Incidence des tarifs"
    assert renamed_changes[0]["source_text_t2"] == "Incidence des tarifs douaniers"
    assert "Sous-section renommée" in renamed_changes[0]["change_summary"]


def test_tfidf_similarity_matrix_from_texts_matches_chunk_wrapper() -> None:
    text_a = (
        "Le groupe GRCF est responsable de la lutte contre le blanchiment d'argent "
        "et la conformité aux exigences réglementaires."
    )
    text_b = (
        "Le groupe CFGR est responsable de la lutte contre le blanchiment d'argent "
        "et la conformité aux exigences réglementaires."
    )
    chunks_a = _chunk_subsection_text(text_a, subsection_heading="A", min_chars=0)
    chunks_b = _chunk_subsection_text(text_b, subsection_heading="B", min_chars=0)
    matrix_from_texts = _tfidf_similarity_matrix_from_texts([text_a, text_b])
    matrix_from_chunks = _align_chunks_tfidf(chunks_a, chunks_b)
    matched = [alignment for alignment in matrix_from_chunks if alignment.chunk_t1 and alignment.chunk_t2]
    assert matrix_from_texts[0][1] == pytest.approx(matched[0].similarity_score, rel=1e-6)


def test_resolve_orphan_subsections_embedding_strong_matches_without_gpt(monkeypatch) -> None:
    """Embedding fort + corps similaire → match déterministe sans GPT."""
    body_t1 = (
        "Le Service de la conformité est une fonction indépendante de gestion et de surveillance "
        "du risque de conformité à l'échelle mondiale de la Banque."
    )
    body_t2 = body_t1.replace("à l'échelle mondiale", "mondiale")
    orphans_t1 = [OrphanSubsection(heading="Service conformité T1", body=body_t1)]
    orphans_t2 = [OrphanSubsection(heading="Service conformité T2", body=body_t2)]

    from vigilance.text_analysis.subsection_matching import OrphanCandidate, _shortlist_orphan_candidates

    shortlist = _shortlist_orphan_candidates(orphans_t1, orphans_t2)

    def fake_attach(**kwargs):
        enriched = [
            OrphanCandidate(
                heading_t1=item.heading_t1,
                body_t1=item.body_t1,
                heading_t2=item.heading_t2,
                body_t2=item.body_t2,
                tfidf_score=item.tfidf_score,
                heading_score=item.heading_score,
                embedding_score=0.95,
            )
            for item in shortlist
        ]
        return enriched, {}

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", fake_attach)
    gpt_called = {"value": False}

    def fail_if_called(**kwargs):
        gpt_called["value"] = True
        return []

    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._gpt_arbitrate_orphan_subsections",
        fail_if_called,
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] in {"deterministic_heading", "deterministic_embedding"}
    assert matches[0]["embedding_score"] == pytest.approx(0.95)
    assert gpt_called["value"] is False


def test_resolve_orphan_subsections_embedding_strong_match_when_llm_confirms(monkeypatch) -> None:
    body_t1 = (
        "Le Service de la conformité est une fonction indépendante de gestion et de surveillance "
        "du risque de conformité à l'échelle mondiale de la Banque."
    )
    body_t2 = body_t1.replace("à l'échelle mondiale", "mondiale")
    orphans_t1 = [OrphanSubsection(heading="Service conformité T1", body=body_t1)]
    orphans_t2 = [OrphanSubsection(heading="Service conformité T2", body=body_t2)]

    from vigilance.text_analysis.subsection_matching import OrphanCandidate, _shortlist_orphan_candidates

    shortlist = _shortlist_orphan_candidates(orphans_t1, orphans_t2)

    def fake_attach(**kwargs):
        enriched = [
            OrphanCandidate(
                heading_t1=item.heading_t1,
                body_t1=item.body_t1,
                heading_t2=item.heading_t2,
                body_t2=item.body_t2,
                tfidf_score=item.tfidf_score,
                heading_score=item.heading_score,
                embedding_score=0.95,
            )
            for item in shortlist
        ]
        return enriched, {}

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", fake_attach)
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction",
        lambda *args, **kwargs: OrphanMatchLLMResponse(
            matches=[
                {
                    "heading_t1": "Service conformité T1",
                    "heading_t2": "Service conformité T2",
                    "confidence": "high",
                    "reason": "same subsection",
                }
            ]
        ),
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] in {"deterministic_heading", "deterministic_embedding", "llm_embedding_confirmed"}
    assert matches[0]["llm_confidence"] in {None, "high"}
    assert matches[0]["embedding_score"] == pytest.approx(0.95)


def test_resolve_orphan_subsections_llm_arbitration_when_embedding_weak(monkeypatch) -> None:
    body_shared = (
        "La Banque et ses entreprises sont assujetties à une réglementation considérable "
        "et à une surveillance active des autorités de réglementation."
    )
    orphans_t1 = [OrphanSubsection(heading="Surveillance réglementaire et conformité", body=body_shared)]
    orphans_t2 = [
        OrphanSubsection(
            heading="Surveillance réglementaire et risque de conformité",
            body=body_shared + " Le cadre évolue chaque trimestre.",
        )
    ]

    from vigilance.text_analysis.subsection_matching import OrphanCandidate, _shortlist_orphan_candidates

    shortlist = _shortlist_orphan_candidates(orphans_t1, orphans_t2)

    def fake_attach(**kwargs):
        enriched = [
            OrphanCandidate(
                heading_t1=item.heading_t1,
                body_t1=item.body_t1,
                heading_t2=item.heading_t2,
                body_t2=item.body_t2,
                tfidf_score=item.tfidf_score,
                heading_score=item.heading_score,
                embedding_score=0.55,
            )
            for item in shortlist
        ]
        return enriched, {}

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", fake_attach)
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction",
        lambda *args, **kwargs: OrphanMatchLLMResponse(
            matches=[
                {
                    "heading_t1": orphans_t1[0].heading,
                    "heading_t2": orphans_t2[0].heading,
                    "confidence": "high",
                    "reason": "same topic",
                }
            ]
        ),
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] in {"deterministic_heading", "llm_embedding_confirmed"}
    if matches[0]["match_source"] == "llm_embedding_confirmed":
        assert matches[0]["llm_confidence"] == "high"


def test_compare_section_texts_orphan_match_avoids_duplicate_synthetics(monkeypatch) -> None:
    shared_body = (
        "Le groupe GRCF, anciennement le groupe Lutte mondiale contre le blanchiment d'argent, "
        "est responsable de la gouvernance des risques liés aux crimes financiers."
    )

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kwargs: [],
    )

    def fake_resolve_orphans(*, client, model, section_key, orphans_t1, orphans_t2, embedding_model="text-embedding-3-small"):
        return [
            {
                "heading_t1": "Crimes financiers, Gestion des risques (CFGR)",
                "heading_t2": "Gestion des risques liés aux crimes financiers (GRCF)",
                "confidence": "high",
                "reason": "rename",
                "match_source": "llm_embedding_confirmed",
            }
        ]

    monkeypatch.setattr("vigilance.text_analysis_pipeline._resolve_orphan_subsections", fake_resolve_orphans)

    md_t1 = (
        "### Risque de marché\n\nCorps T1.\n\n"
        "### Crimes financiers, Gestion des risques (CFGR)\n\n"
        f"{shared_body}\n"
    )
    md_t2 = (
        "### Risque de marché\n\nCorps T2.\n\n"
        "### Gestion des risques liés aux crimes financiers (GRCF)\n\n"
        f"{shared_body}\n"
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    synthetic_added = [
        change for change in changes if change["diff_type"] == "added" and change["change_summary"].startswith("Sous-section")
    ]
    synthetic_removed = [
        change for change in changes if change["diff_type"] == "removed" and change["change_summary"].startswith("Sous-section")
    ]
    assert synthetic_added == []
    assert synthetic_removed == []
    assert any(change["diff_type"] == "renamed" for change in changes)


def test_compare_section_texts_td_renamed_orphans_avoid_duplicate_synthetics(monkeypatch) -> None:
    regulatory_body = (
        "La Banque et ses entreprises sont assujetties à une réglementation considérable "
        "et à une surveillance étendue exercée par différents organismes de réglementation. "
        "Les exigences réglementaires peuvent entraîner des coûts de conformité et des mesures correctives."
    )
    financial_crime_body = (
        "Le groupe GRCF est responsable de la surveillance de la conformité en matière de LCBA, "
        "de sanctions économiques et de lutte contre le financement des activités terroristes. "
        "Il supervise les programmes de gestion du risque lié aux crimes financiers."
    )

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_texts_single_call",
        lambda **kwargs: [],
    )

    def fake_resolve_orphans(*, client, model, section_key, orphans_t1, orphans_t2, embedding_model="text-embedding-3-small"):
        return [
            {
                "heading_t1": "Surveillance réglementaire et conformité",
                "heading_t2": "Surveillance réglementaire et risque de conformité",
                "confidence": "high",
                "llm_confidence": "high",
                "reason": "same TD regulatory risk subsection",
                "match_source": "llm_embedding_confirmed",
                "tfidf_score": 0.94,
                "embedding_score": 0.97,
                "heading_score": 0.89,
            },
            {
                "heading_t1": "Crimes financiers, Gestion des risques (CFGR)",
                "heading_t2": "Gestion des risques liés aux crimes financiers (GRCF)",
                "confidence": "high",
                "llm_confidence": "high",
                "reason": "same TD financial-crime risk subsection",
                "match_source": "llm_embedding_confirmed",
                "tfidf_score": 0.91,
                "embedding_score": 0.96,
                "heading_score": 0.47,
            },
        ]

    monkeypatch.setattr("vigilance.text_analysis_pipeline._resolve_orphan_subsections", fake_resolve_orphans)

    md_t1 = (
        "### Surveillance réglementaire et conformité\n\n"
        f"{regulatory_body}\n\n"
        "### Crimes financiers, Gestion des risques (CFGR)\n\n"
        f"{financial_crime_body}\n"
    )
    md_t2 = (
        "### Surveillance réglementaire et risque de conformité\n\n"
        f"{regulatory_body}\n\n"
        "### Gestion des risques liés aux crimes financiers (GRCF)\n\n"
        f"{financial_crime_body}\n"
    )

    changes = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1=md_t1,
        text_t2=md_t2,
    )

    synthetic_added_or_removed = [
        change
        for change in changes
        if change["diff_type"] in {"added", "removed"}
        and change["change_summary"].startswith("Sous-section")
    ]
    renamed_headings = {
        (change.get("previous_subsection_heading"), change.get("current_subsection_heading"))
        for change in changes
        if change["diff_type"] == "renamed"
    }
    assert synthetic_added_or_removed == []
    assert (
        "Surveillance réglementaire et conformité",
        "Surveillance réglementaire et risque de conformité",
    ) in renamed_headings
    assert (
        "Crimes financiers, Gestion des risques (CFGR)",
        "Gestion des risques liés aux crimes financiers (GRCF)",
    ) in renamed_headings


def test_deterministic_confirm_orphan_matches_td_frauduleuses() -> None:
    from vigilance.text_analysis.subsection_matching import (
        OrphanCandidate,
        _deterministic_confirm_orphan_matches,
    )

    candidate = OrphanCandidate(
        heading_t1="Activités frauduleuses",
        body_t1="La Banque surveille les activités frauduleuses internes et externes.",
        heading_t2="Activités frauduleuses externes",
        body_t2="La Banque surveille les activités frauduleuses externes et internes.",
        tfidf_score=0.79,
        heading_score=0.83,
        embedding_score=0.55,
    )
    matches = _deterministic_confirm_orphan_matches([candidate])
    assert len(matches) == 1
    assert matches[0]["match_source"] == "deterministic_heading"


def test_resolve_orphan_subsections_gpt_failure_keeps_deterministic_matches(monkeypatch) -> None:
    body_shared = (
        "La Banque surveille les activités frauduleuses internes et externes dans l'ensemble "
        "de ses opérations bancaires et de détail."
    )
    orphans_t1 = [OrphanSubsection(heading="Activités frauduleuses", body=body_shared)]
    orphans_t2 = [OrphanSubsection(heading="Activités frauduleuses externes", body=body_shared)]

    from vigilance.text_analysis.subsection_matching import OrphanCandidate, _shortlist_orphan_candidates

    shortlist = _shortlist_orphan_candidates(orphans_t1, orphans_t2)

    def fake_attach(**kwargs):
        return [
            OrphanCandidate(
                heading_t1=item.heading_t1,
                body_t1=item.body_t1,
                heading_t2=item.heading_t2,
                body_t2=item.body_t2,
                tfidf_score=item.tfidf_score,
                heading_score=item.heading_score,
                embedding_score=0.55,
            )
            for item in shortlist
        ], {}

    def fake_gpt_failure(**kwargs):
        raise RuntimeError("gpt down")

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", fake_attach)
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._gpt_arbitrate_orphan_subsections",
        fake_gpt_failure,
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] == "deterministic_heading"


def test_resolve_orphan_subsections_ambiguous_still_calls_gpt(monkeypatch) -> None:
    body_t1 = (
        "Résolution globale des enquêtes sur le programme de LCBA-BSA aux États-Unis de la Banque "
        "avec des sanctions et des obligations de conformité."
    )
    body_t2 = (
        "Redressement du programme de LCBA-BSA aux États-Unis et du programme de LCBA à l'échelle "
        "de l'entreprise de la Banque avec un plan de remédiation."
    )
    orphans_t1 = [
        OrphanSubsection(
            heading="Résolution globale des enquêtes sur le programme de LCBA-BSA aux États-Unis de la Banque",
            body=body_t1,
        )
    ]
    orphans_t2 = [
        OrphanSubsection(
            heading="Redressement du programme de LCBA-BSA aux États-Unis et du programme de LCBA à l'échelle de l'entreprise de la Banque",
            body=body_t2,
        )
    ]

    from vigilance.text_analysis.subsection_matching import OrphanCandidate, _shortlist_orphan_candidates

    shortlist = _shortlist_orphan_candidates(orphans_t1, orphans_t2)
    gpt_called = {"value": False}

    def fake_attach(**kwargs):
        return [
            OrphanCandidate(
                heading_t1=item.heading_t1,
                body_t1=item.body_t1,
                heading_t2=item.heading_t2,
                body_t2=item.body_t2,
                tfidf_score=item.tfidf_score,
                heading_score=item.heading_score,
                embedding_score=0.40,
            )
            for item in shortlist
        ], {}

    def fake_gpt(**kwargs):
        gpt_called["value"] = True
        return [
            {
                "heading_t1": orphans_t1[0].heading,
                "heading_t2": orphans_t2[0].heading,
                "confidence": "medium",
                "llm_confidence": "medium",
                "reason": "same LCBA topic",
                "match_source": "llm_embedding_confirmed",
                "tfidf_score": round(shortlist[0].tfidf_score, 4),
                "embedding_score": 0.40,
                "heading_score": round(shortlist[0].heading_score, 4),
            }
        ]

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", fake_attach)
    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._gpt_arbitrate_orphan_subsections", fake_gpt)

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert gpt_called["value"] is True
    assert len(matches) == 1
    assert matches[0]["match_source"] == "llm_embedding_confirmed"


def test_resolve_orphan_subsections_embedding_failure_falls_back_to_llm(monkeypatch) -> None:
    orphans_t1 = [OrphanSubsection(heading="Ancien titre", body="Corps substantiel " * 20)]
    orphans_t2 = [OrphanSubsection(heading="Nouveau titre", body="Corps substantiel " * 20)]

    def exploding_attach(**kwargs):
        raise RuntimeError("embedding down")

    monkeypatch.setattr("vigilance.text_analysis.subsection_matching._attach_embedding_scores", exploding_attach)
    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction",
        lambda *args, **kwargs: OrphanMatchLLMResponse(
            matches=[
                {
                    "heading_t1": "Ancien titre",
                    "heading_t2": "Nouveau titre",
                    "confidence": "medium",
                    "reason": "fallback",
                }
            ]
        ),
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] in {"deterministic_hybrid", "llm_embedding_confirmed"}


def test_resolve_orphan_subsections_short_body_uses_title_only_fallback(monkeypatch) -> None:
    orphans_t1 = [OrphanSubsection(heading="Objectif", body="Capital disponible.")]
    orphans_t2 = [OrphanSubsection(heading="Objectif de capital", body="Capital disponible.")]

    monkeypatch.setattr(
        "vigilance.text_analysis.subsection_matching._call_structured_completion_with_correction",
        lambda *args, **kwargs: OrphanMatchLLMResponse(
            matches=[
                {
                    "heading_t1": "Objectif",
                    "heading_t2": "Objectif de capital",
                    "confidence": "medium",
                    "reason": "titre très proche",
                }
            ]
        ),
    )

    from vigilance.text_analysis.subsection_matching import _resolve_orphan_subsections as resolve_direct

    matches = resolve_direct(
        client=object(),
        model="gpt-4o",
        section_key="capital",
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    assert len(matches) == 1
    assert matches[0]["match_source"] in {"deterministic_heading", "title_only"}
    assert matches[0]["tfidf_score"] is None
    assert matches[0]["embedding_score"] is None


def test_bmo_risque_de_strategie_2024_t4_chunks_into_six() -> None:
    md_path = Path("outputs/resultats/bmo/2025_t4_vs_2024_t4/text_extraction_2024_t4.md")
    if not md_path.exists():
        pytest.skip("Artefact local BMO 2024 T4 absent.")

    section_text = _extract_section_text_from_markdown(md_path.read_text(encoding="utf-8"), "gestion_risques")
    subsections = dict(_parse_subsections(section_text))
    body = subsections["Risque de stratégie"]

    chunks = _chunk_subsection_text(
        body,
        subsection_heading="Risque de stratégie",
        section_title="Gestion des risques",
    )

    assert [chunk.chunk_id for chunk in chunks] == ["c00", "c01", "c02", "c03", "c04", "c05"]
    assert all(chunk.kind == "paragraph" for chunk in chunks)
    assert chunks[0].text.startswith("Le risque de stratégie s'entend")
    assert chunks[-1].text.startswith("Notre performance financière dépend")


def test_bmo_risque_de_strategie_tfidf_alignment_stays_local() -> None:
    previous_path = Path("outputs/resultats/bmo/2025_t4_vs_2024_t4/text_extraction_2024_t4.md")
    current_path = Path("outputs/resultats/bmo/2025_t4_vs_2024_t4/text_extraction_2025_t4.md")
    if not previous_path.exists() or not current_path.exists():
        pytest.skip("Artefacts locaux BMO T4 absents.")

    previous_section = _extract_section_text_from_markdown(previous_path.read_text(encoding="utf-8"), "gestion_risques")
    current_section = _extract_section_text_from_markdown(current_path.read_text(encoding="utf-8"), "gestion_risques")
    previous_body = dict(_parse_subsections(previous_section))["Risque de stratégie"]
    current_body = dict(_parse_subsections(current_section))["Risque de stratégie"]
    chunks_t1 = _chunk_subsection_text(
        previous_body,
        subsection_heading="Risque de stratégie",
        section_title="Gestion des risques",
    )
    chunks_t2 = _chunk_subsection_text(
        current_body,
        subsection_heading="Risque de stratégie",
        section_title="Gestion des risques",
    )

    alignments = _align_chunks_tfidf(chunks_t1, chunks_t2)
    matched_by_t2 = {
        alignment.chunk_t2.chunk_id: alignment.chunk_t1.chunk_id
        for alignment in alignments
        if alignment.chunk_t1 and alignment.chunk_t2
    }
    removed = [alignment for alignment in alignments if alignment.alignment_type == "possible_removed"]

    assert len(chunks_t1) == 6
    assert len(chunks_t2) == 5
    assert matched_by_t2["c02"] == "c02"
    assert matched_by_t2["c03"] == "c04"
    assert matched_by_t2["c04"] == "c05"
    assert len(removed) == 1
    assert removed[0].chunk_t1.chunk_id == "c03"
    assert all("Risque de stratégie" in chunk.hierarchy_path for chunk in [*chunks_t1, *chunks_t2])


def test_bnc_accord_bale_requires_semantic_services_without_legacy_fallback() -> None:
    base = Path("outputs/resultats/bnc/2025_t4_vs_2024_t4")
    previous_path = base / "text_extraction_2024_t4.md"
    current_path = base / "text_extraction_2025_t4.md"
    if not previous_path.exists() or not current_path.exists():
        pytest.skip("Artefacts locaux BNC T4 absents.")

    previous_section = _extract_section_text_from_markdown(
        previous_path.read_text(encoding="utf-8"), "gestion_capital"
    )
    current_section = _extract_section_text_from_markdown(
        current_path.read_text(encoding="utf-8"), "gestion_capital"
    )
    previous_body = dict(_parse_subsections(previous_section))["Accord de Bâle"]
    current_body = dict(_parse_subsections(current_section))["Accord de Bâle"]
    with pytest.raises(SemanticChunkingError, match="aucun fallback"):
        _chunk_subsection_text(previous_body, subsection_heading="Accord de Bâle")
    with pytest.raises(SemanticChunkingError, match="aucun fallback"):
        _chunk_subsection_text(current_body, subsection_heading="Accord de Bâle")


def test_td_future_capital_disclosures_are_not_merged_only_for_similar_boilerplate() -> None:
    """Separate TD issuances remain separate despite their similar wording."""
    heading = "Évolution future des fonds propres réglementaires"
    first = (
        "La Banque a émis des billets de fonds propres avec recours limité. "
        "Les billets portent intérêt selon les modalités annoncées."
    )
    second = (
        "La Banque a émis une autre série de billets de fonds propres avec recours limité. "
        "Cette série porte intérêt selon ses propres modalités."
    )
    chunks = _chunk_subsection_text(f"{first}\n\n{second}", subsection_heading=heading)

    assert [chunk.text for chunk in chunks] == [first, second]


# ---------------------------------------------------------------------------
# Phase 2: AMF triage — invariants stricts et erreurs explicites
# ---------------------------------------------------------------------------


from pydantic import ValidationError as _PydValidationError

from vigilance.amf_taxonomy import (
    TriageAMFCompactLLMBatch,
    TriageAMFCompactLLMResultWithIndex,
    TriageAMFBatch,
    TriageAMFLLMBatch,
    TriageAMFLLMResultWithIndex,
    TriageAMFResult,
    TriageAMFResultWithIndex,
    TriageValidationError,
    count_complete_sentences,
)
from vigilance.text_analysis_pipeline import (
    _call_structured_completion,
    _call_structured_completion_with_correction,
    _triage_section_changes,
)
from vigilance.text_analysis.triage import _deterministic_cosmetic_exclusion


def _valid_explanation() -> str:
    return (
        "Au T2 la banque introduit un nouveau modele interne pour le risque "
        "de credit. Ce changement est substantif car il modifie la base de "
        "comparaison avec le rapport precedent. Cela implique une revue de "
        "la surveillance prudentielle."
    )


def _valid_justification_oui() -> str:
    return (
        "OUI - Nouvel élément à surveiller : Oui.\n\n"
        "Sujet détecté : Méthode de calcul modifiée, exigence réglementaire, "
        "risque de crédit.\n\n"
        "Ce qui change : Le nouveau modèle interne avancé pour le risque de "
        "crédit est ajouté au T2 et n'apparaissait pas au T1. La divulgation "
        "ne présente donc plus la même approche de mesure du risque.\n\n"
        "Pertinence métier : Ce changement est pertinent pour la vigie bancaire "
        "parce qu'il touche une méthodologie prudentielle et les exigences "
        "réglementaires. Une méthode interne avancée peut modifier la lecture "
        "des actifs pondérés, du capital requis et de la comparabilité "
        "inter-pairs.\n\n"
        "Point de surveillance : Le point à retenir est que la banque présente une "
        "base méthodologique différente pour le risque de crédit, ce qui change "
        "la lecture métier de la divulgation."
    )


def _valid_justification_non() -> str:
    return (
        "NON - Nouvel élément à surveiller : Non.\n\n"
        "Sujet détecté : Mise à jour quantitative propre à la banque.\n\n"
        "Ce qui change : Le ratio CET1 existait déjà au T1 et seule sa valeur "
        "chiffrée change entre les deux trimestres. Aucun nouveau seuil ou "
        "nouvelle méthode n'est introduit dans la divulgation.\n\n"
        "Pertinence métier : Cette évolution numérique reflète l'activité "
        "normale de la banque et ne touche aucun seuil réglementaire ni "
        "méthodologie de calcul. Elle ne modifie pas la lecture prudentielle "
        "du rapport ni la comparabilité métier.\n\n"
        "Point de surveillance : Le point à retenir est que la substance de la "
        "divulgation demeure stable. Le changement correspond à une mise à "
        "jour quantitative plutôt qu'à un nouveau signal de surveillance."
    )


def _compact_reason() -> str:
    return (
        "Le rapport courant ajoute un exercice annuel de simulation de cyberattaque "
        "qui n’était pas décrit dans le rapport précédent. Cette évolution renforce "
        "la lecture des pratiques de résilience et fournit un point de comparaison "
        "concret entre les banques."
    )


# --- Invariants Pydantic ---


def test_compact_triage_accepts_two_complete_relevance_reason_sentences() -> None:
    result = TriageAMFCompactLLMResultWithIndex(
        change_index=1,
        is_relevant=True,
        themes_amf=["RISQUE_EMERGENT"],
        nouvelle_idee=True,
        relevance_reason=f"  {_compact_reason().replace(' Cette', '   Cette')}  ",
    )
    assert count_complete_sentences(result.relevance_reason) == 2
    assert "  " not in result.relevance_reason
    assert len(result.relevance_reason.split()) < 100


@pytest.mark.parametrize(
    "reason",
    [
        "Le rapport courant ajoute un nouveau contrôle de cybersécurité.",
        (
            "Le rapport courant ajoute un nouveau contrôle de cybersécurité. "
            "Cette mesure renforce le dispositif déclaré par la banque. "
            "Elle fournit aussi un nouveau point de comparaison entre les banques."
        ),
    ],
)
def test_compact_triage_rejects_other_sentence_counts(reason: str) -> None:
    with pytest.raises(_PydValidationError, match="exactement 2 phrases complètes"):
        TriageAMFCompactLLMResultWithIndex(
            change_index=1,
            is_relevant=False,
            themes_amf=[],
            nouvelle_idee=False,
            relevance_reason=reason,
        )


def test_compact_triage_rejects_incomplete_second_sentence() -> None:
    with pytest.raises(_PydValidationError, match="se terminer par une phrase complète"):
        TriageAMFCompactLLMResultWithIndex(
            change_index=1,
            is_relevant=False,
            themes_amf=[],
            nouvelle_idee=False,
            relevance_reason=(
                "Le rapport courant ajoute un nouveau contrôle de cybersécurité. "
                "Cette mesure fournit un nouveau point de comparaison entre les banques"
            ),
        )


def test_compact_triage_rejects_sentences_without_lexical_content() -> None:
    with pytest.raises(_PydValidationError, match="contenu lexical"):
        TriageAMFCompactLLMResultWithIndex(
            change_index=1,
            is_relevant=False,
            themes_amf=[],
            nouvelle_idee=False,
            relevance_reason=". .",
        )


def test_compact_triage_counts_sentence_ending_with_uppercase_label() -> None:
    result = TriageAMFCompactLLMResultWithIndex(
        change_index=1,
        is_relevant=True,
        themes_amf=["MODIFICATION_METHODOLOGIE"],
        nouvelle_idee=True,
        relevance_reason=(
            "Le rapport courant retient désormais l’approche A. "
            "Cette modification fournit une nouvelle base de comparaison des "
            "méthodes déclarées par les banques."
        ),
    )
    assert count_complete_sentences(result.relevance_reason) == 2


def test_compact_triage_ignores_abbreviations_and_decimals_when_counting_sentences() -> None:
    reason = (
        "Le cadre de Bâle 3.1, présenté p. ex. à la p. 12 par M. Dupont, "
        "est maintenant détaillé dans le rapport. "
        "2025 devient l’année de référence pour comparer son application entre les banques."
    )
    result = TriageAMFCompactLLMResultWithIndex(
        change_index=1,
        is_relevant=True,
        themes_amf=["EXIGENCES_REGLEMENTAIRES"],
        nouvelle_idee=True,
        relevance_reason=reason,
    )
    assert count_complete_sentences(result.relevance_reason) == 2


def test_compact_triage_ignores_common_french_abbreviations_inside_sentence() -> None:
    result = TriageAMFCompactLLMResultWithIndex(
        change_index=1,
        is_relevant=True,
        themes_amf=["CONTROLE_CONFORMITE"],
        nouvelle_idee=True,
        relevance_reason=(
            "Le rapport détaille plusieurs mesures, etc. afin d’encadrer le contrôle, "
            "c.-à-d. une revue annuelle documentée. Cette précision permet de "
            "comparer la fréquence des contrôles déclarés par les banques."
        ),
    )
    assert count_complete_sentences(result.relevance_reason) == 2


def test_invariant_relevant_without_themes_raises() -> None:
    with pytest.raises(_PydValidationError, match="themes_amf"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=[],
            nouvelle_idee=True,
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_invariant_relevant_with_short_explanation_raises() -> None:
    with pytest.raises(_PydValidationError, match="50"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["DIVULGATION_AJOUT"],
            nouvelle_idee=True,
            explanation="trop court",
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_data_and_third_party_cloud_themes_are_valid_amf_codes() -> None:
    triage = TriageAMFResult(
        is_relevant=True,
        themes_amf=["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"],
        impact_level="MAJEUR",
        impact_it="ELEVE",
        impact_it_justification=(
            "Éléments observés : Le rapport prévoit une migration infonuagique "
            "et une stratégie de sortie du fournisseur critique.\n\n"
            "Conséquence probable : Ces mesures nécessitent une adaptation "
            "importante des contrôles et de l'architecture IT.\n\n"
            "Limite de l'analyse : Le calendrier et le périmètre technique de "
            "la migration ne sont pas précisés."
        ),
        changement_posture="NOUVEAU_DISPOSITIF",
        justification_posture=(
            "Preuve : La banque introduit une stratégie de sortie et un nouveau "
            "contrôle contractuel pour le fournisseur critique.\n\n"
            "Effet sur la gestion du risque : Le dispositif formalise la "
            "réversibilité et renforce l'encadrement du fournisseur.\n\n"
            "Justification du statut : Le rapport présente la mesure comme "
            "planifiée, sans confirmer son déploiement complet.\n\n"
            "Justification de la confiance : Les éléments du nouveau dispositif "
            "sont décrits explicitement dans le texte."
        ),
        statut_mise_en_oeuvre="PLANIFIE",
        confiance_posture="ELEVEE",
        nouvelle_idee=True,
        action_requise="revue_prioritaire",
        explanation=_valid_explanation(),
        nouvelle_idee_justification=_valid_justification_oui(),
    )
    assert triage.themes_amf == ["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"]
    assert triage.impact_it == "ELEVE"
    assert triage.changement_posture == "NOUVEAU_DISPOSITIF"
    assert triage.statut_mise_en_oeuvre == "PLANIFIE"
    assert triage.confiance_posture == "ELEVEE"


def test_impact_it_evaluation_requires_justification() -> None:
    with pytest.raises(_PydValidationError, match="impact_it_justification"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["RISQUE_TIERS_CLOUD"],
            impact_level="MAJEUR",
            impact_it="ELEVE",
            changement_posture="RENFORCEMENT",
            nouvelle_idee=True,
            action_requise="revue_prioritaire",
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_indeterminate_it_impact_rejects_a_justification() -> None:
    with pytest.raises(_PydValidationError, match="doit être vide"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["RISQUE_DONNEES"],
            impact_it="INDETERMINE",
            impact_it_justification="Le lien IT ne peut pas être démontré dans le rapport.",
            changement_posture="INDETERMINE",
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_non(),
        )


def test_evaluated_posture_requires_justification_and_confidence() -> None:
    with pytest.raises(_PydValidationError, match="justification_posture"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["RISQUE_TIERS_CLOUD"],
            changement_posture="RENFORCEMENT",
            confiance_posture="ELEVEE",
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_non(),
        )

    with pytest.raises(_PydValidationError, match="confiance_posture"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["RISQUE_TIERS_CLOUD"],
            changement_posture="RENFORCEMENT",
            justification_posture=(
                "Preuve : La banque renforce explicitement la surveillance de "
                "ses fournisseurs critiques.\n\n"
                "Effet sur la gestion du risque : Le niveau de contrôle des "
                "tiers critiques augmente de manière identifiable.\n\n"
                "Justification du statut : Le rapport ne précise pas encore "
                "le niveau exact de déploiement.\n\n"
                "Justification de la confiance : La formulation du renforcement "
                "est explicite dans le rapport."
            ),
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_non(),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("impact_it", "ELEVE", "impact_it=INDETERMINE"),
        ("impact_it_justification", "Une justification qui ne devrait pas être présente.", "justification vide"),
        ("changement_posture", "RENFORCEMENT", "changement_posture=AUCUN"),
    ],
)
def test_irrelevant_change_rejects_it_and_posture_signals(
    field: str, value: str, error: str
) -> None:
    payload = {
        "is_relevant": False,
        "exclusion_reason": "reformulation_mineure",
        "nouvelle_idee_justification": _valid_justification_non(),
        field: value,
    }

    with pytest.raises(_PydValidationError, match=error):
        TriageAMFResult(**payload)


def test_invariant_irrelevant_with_nouvelle_idee_raises() -> None:
    with pytest.raises(_PydValidationError, match="nouvelle_idee"):
        TriageAMFResult(
            is_relevant=False,
            nouvelle_idee=True,
            exclusion_reason="reformulation_mineure",
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_invariant_irrelevant_with_majeur_impact_raises() -> None:
    with pytest.raises(_PydValidationError, match="MINEUR"):
        TriageAMFResult(
            is_relevant=False,
            impact_level="MAJEUR",
            exclusion_reason="reformulation_mineure",
            nouvelle_idee_justification=_valid_justification_non(),
        )


def test_invariant_irrelevant_without_exclusion_reason_raises() -> None:
    with pytest.raises(_PydValidationError, match="exclusion_reason"):
        TriageAMFResult(
            is_relevant=False,
            nouvelle_idee_justification=_valid_justification_non(),
        )


def test_invariant_revue_prioritaire_without_majeur_raises() -> None:
    with pytest.raises(_PydValidationError, match="revue_prioritaire"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["MODIFICATION_METHODOLOGIE"],
            impact_level="MODERE",
            nouvelle_idee=True,
            action_requise="revue_prioritaire",
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_repair_relevant_without_justification_synthesizes() -> None:
    triage = TriageAMFResult(
        is_relevant=True,
        themes_amf=["DIVULGATION_AJOUT"],
        impact_level="MINEUR",
        nouvelle_idee=True,
        explanation=_valid_explanation(),
        nouvelle_idee_justification="",
    )
    assert triage.nouvelle_idee_justification.startswith("OUI")
    assert "Nouvel élément à surveiller :" in triage.nouvelle_idee_justification


def test_repair_justification_with_single_sentence_synthesizes() -> None:
    triage = TriageAMFResult(
        is_relevant=True,
        themes_amf=["DIVULGATION_AJOUT"],
        impact_level="MINEUR",
        nouvelle_idee=True,
        explanation=_valid_explanation(),
        nouvelle_idee_justification="OUI le ratio TLAC est ajoute au tableau.",
    )
    assert "Pertinence métier :" in triage.nouvelle_idee_justification
    assert len(triage.nouvelle_idee_justification) >= 200


def test_repair_justification_without_prefix_synthesizes_structured_note() -> None:
    bad_prefix_long = (
        "Le ratio TLAC est ajoute au TABLEAU 11 absent du T1, ce qui constitue "
        "une nouveaute structurelle pour la divulgation. Cela aligne BMO sur "
        "les attentes BSIF prudentielles selon la ligne directrice canadienne. "
        "L'analyste doit considerer cette ligne comme une nouvelle exigence "
        "qui touche les ratios prudentiels (themes AMF DIVULGATION_AJOUT)."
    )
    triage = TriageAMFResult(
        is_relevant=True,
        themes_amf=["DIVULGATION_AJOUT"],
        impact_level="MINEUR",
        nouvelle_idee=True,
        explanation=_valid_explanation(),
        nouvelle_idee_justification=bad_prefix_long,
    )
    assert triage.nouvelle_idee_justification.startswith("OUI")
    assert "Point de surveillance :" in triage.nouvelle_idee_justification


def test_invariant_justification_must_start_with_non_when_not_nouvelle_idee() -> None:
    with pytest.raises(_PydValidationError, match="NON"):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["CAPITAL_REGLEMENTAIRE"],
            impact_level="MINEUR",
            nouvelle_idee=False,
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_justification_oui(),
        )


def test_invariant_irrelevant_now_requires_substantial_justification() -> None:
    """is_relevant=False exige une justification détaillée (réparée si GPT omet les rubriques)."""
    repaired = TriageAMFResult(
        is_relevant=False,
        exclusion_reason="reformulation_mineure",
        nouvelle_idee_justification="NON c'est une reformulation. Pas substantif. Trop court.",
    )
    assert repaired.nouvelle_idee_justification.startswith("NON")
    assert "Pertinence métier :" in repaired.nouvelle_idee_justification

    ok = TriageAMFResult(
        is_relevant=False,
        exclusion_reason="reformulation_mineure",
        nouvelle_idee_justification=_valid_justification_non(),
    )
    assert ok.is_relevant is False
    assert ok.nouvelle_idee_justification.startswith("NON")


def test_invariant_change_index_must_be_at_least_one() -> None:
    with pytest.raises(_PydValidationError):
        TriageAMFResultWithIndex(
            change_index=0,
            is_relevant=False,
            exclusion_reason="reformulation_mineure",
            nouvelle_idee_justification=_valid_justification_non(),
        )


# --- Helpers de mock pour l'API structured outputs ---


def _make_parsed_response(parsed_obj, *, refusal=None, finish_reason="stop"):
    """Construit une réponse OpenAI structured outputs simulée."""
    message = type(
        "FakeMessage",
        (),
        {"parsed": parsed_obj, "refusal": refusal},
    )()
    choice = type(
        "FakeChoice",
        (),
        {"message": message, "finish_reason": finish_reason},
    )()
    return type("FakeResponse", (), {"choices": [choice]})()


class _FakeStructuredCompletions:
    def __init__(self, side_effect) -> None:
        self.side_effect = side_effect
        self.call_count = 0
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        if callable(self.side_effect):
            return self.side_effect(**kwargs)
        return self.side_effect


class _FakeStructuredClient:
    def __init__(self, side_effect) -> None:
        completions = _FakeStructuredCompletions(side_effect)
        chat = type("Chat", (), {"completions": completions})()
        self.beta = type("Beta", (), {"chat": chat})()
        self._completions = completions

    @property
    def call_count(self) -> int:
        return self._completions.call_count


def _make_validation_error() -> _PydValidationError:
    try:
        TriageAMFResult(
            is_relevant=True,
            themes_amf=[],
            explanation=_valid_explanation(),
        )
    except _PydValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


# --- _call_structured_completion : pas de fallback ---


def test_call_structured_completion_raises_runtime_error_on_refusal() -> None:
    client = _FakeStructuredClient(
        _make_parsed_response(None, refusal="Je refuse pour des raisons de sécurité.")
    )

    with pytest.raises(RuntimeError, match="refused"):
        _call_structured_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
        )


def test_call_structured_completion_raises_runtime_error_on_truncation() -> None:
    client = _FakeStructuredClient(
        _make_parsed_response(None, finish_reason="length")
    )

    with pytest.raises(RuntimeError, match="truncated"):
        _call_structured_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
        )


def test_call_structured_completion_raises_runtime_error_on_empty_payload() -> None:
    client = _FakeStructuredClient(_make_parsed_response(None))

    with pytest.raises(RuntimeError, match="no parsed payload"):
        _call_structured_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
        )


# --- Retry borné avec feedback correctif ---


def test_correction_retry_succeeds_on_second_attempt() -> None:
    valid_batch = TriageAMFBatch(triages=[])
    err = _make_validation_error()
    state = {"calls": 0}

    def side_effect(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise err
        return _make_parsed_response(valid_batch)

    client = _FakeStructuredClient(side_effect)

    result = _call_structured_completion_with_correction(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        response_format=TriageAMFBatch,
        max_retries=1,
    )

    assert result is valid_batch
    assert state["calls"] == 2


def test_correction_retry_accepts_custom_validation_message() -> None:
    valid_batch = TriageAMFBatch(triages=[])
    err = _make_validation_error()
    state = {"calls": 0}

    def side_effect(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise err
        return _make_parsed_response(valid_batch)

    client = _FakeStructuredClient(side_effect)

    result = _call_structured_completion_with_correction(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        response_format=TriageAMFBatch,
        max_retries=1,
        validation_retry_message="Message correctif spécialisé.",
    )

    assert result is valid_batch
    retry_messages = client._completions.calls[1]["messages"]
    assert "Message correctif spécialisé." in retry_messages[-1]["content"]
    assert "invariants AMF" not in retry_messages[-1]["content"]


def test_correction_retry_propagates_after_exhaustion() -> None:
    err = _make_validation_error()

    def always_fail(**kwargs):
        raise err

    client = _FakeStructuredClient(always_fail)

    with pytest.raises(_PydValidationError):
        _call_structured_completion_with_correction(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
            max_retries=1,
        )

    # 2 appels = 1 initial + 1 retry
    assert client.call_count == 2


def test_correction_retry_does_not_retry_runtime_errors() -> None:
    """RuntimeError (refus, troncature) ne doit PAS déclencher de retry."""

    def fake_refusal(**kwargs):
        return _make_parsed_response(None, refusal="refus")

    client = _FakeStructuredClient(fake_refusal)

    with pytest.raises(RuntimeError, match="refused"):
        _call_structured_completion_with_correction(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
            max_retries=1,
        )

    # Un seul appel : pas de retry sur RuntimeError
    assert client.call_count == 1


def test_correction_retry_retries_transient_timeout(monkeypatch) -> None:
    valid_batch = TriageAMFBatch(triages=[])
    state = {"calls": 0}
    monkeypatch.setattr("vigilance.text_analysis_pipeline.time.sleep", lambda _seconds: None)

    def timeout_then_success(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TimeoutError("Request timed out.")
        return _make_parsed_response(valid_batch)

    client = _FakeStructuredClient(timeout_then_success)

    result = _call_structured_completion_with_correction(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        response_format=TriageAMFBatch,
        max_retries=1,
        max_transport_retries=1,
    )

    assert result is valid_batch
    assert client.call_count == 2


def test_correction_retry_exhausts_transient_timeout(monkeypatch) -> None:
    monkeypatch.setattr("vigilance.text_analysis_pipeline.time.sleep", lambda _seconds: None)

    def always_timeout(**kwargs):
        raise TimeoutError("Request timed out.")

    client = _FakeStructuredClient(always_timeout)

    with pytest.raises(TimeoutError, match="timed out"):
        _call_structured_completion_with_correction(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            response_format=TriageAMFBatch,
            max_retries=1,
            max_transport_retries=1,
        )

    assert client.call_count == 2


def test_correction_retry_retries_length_limit_once() -> None:
    valid_batch = TriageAMFBatch(triages=[])
    state = {"calls": 0}

    def length_then_success(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("Could not parse response content as the length limit was reached")
        return _make_parsed_response(valid_batch)

    client = _FakeStructuredClient(length_then_success)

    result = _call_structured_completion_with_correction(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        response_format=TriageAMFBatch,
        max_retries=1,
        max_length_retries=1,
    )

    assert result is valid_batch
    assert client.call_count == 2
    retry_messages = client._completions.calls[1]["messages"]
    assert "dépassé la limite de sortie" in retry_messages[-1]["content"]


# --- _triage_section_changes : ValidationError → TriageValidationError ---


def test_triage_section_changes_converts_validation_error_to_triage_validation_error(monkeypatch) -> None:
    err = _make_validation_error()

    def always_fail(**kwargs):
        raise err

    client = _FakeStructuredClient(always_fail)

    changes = [
        {
            "diff_type": "modified",
            "semantic_text_t1": "La banque décrit son dispositif de gouvernance du risque de crédit.",
            "semantic_text_t2": "La banque décrit un dispositif renforcé de gouvernance du risque opérationnel.",
            "source_text_t1": "La banque décrit son dispositif de gouvernance du risque de crédit.",
            "source_text_t2": "La banque décrit un dispositif renforcé de gouvernance du risque opérationnel.",
        }
    ]

    with pytest.raises(TriageValidationError) as exc_info:
        _triage_section_changes(
            client=client,
            model="gpt-4o",
            section_key="gestion_risques",
            changes=changes,
        )

    assert exc_info.value.section_key == "gestion_risques"
    assert exc_info.value.validation_error is err
    # 3 appels = 1 initial + 2 retries avant remontée
    assert client.call_count == 3
    retry_message = client._completions.calls[1]["messages"][-1]["content"]
    assert "exactement deux phrases complètes" in retry_message
    assert "description factuelle" in retry_message
    assert "analyse comparative" in retry_message


def test_triage_section_changes_length_retry_repeats_two_sentence_contract() -> None:
    valid_batch = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=False,
                themes_amf=[],
                nouvelle_idee=False,
                relevance_reason=_compact_reason(),
            )
        ]
    )
    state = {"calls": 0}

    def length_then_success(**_kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError(
                "Could not parse response content as the length limit was reached"
            )
        return _make_parsed_response(valid_batch)

    client = _FakeStructuredClient(length_then_success)
    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[
            {
                "diff_type": "added",
                "source_text_t1": "",
                "source_text_t2": "Ajout d’un exercice annuel de cyberattaque.",
            }
        ],
    )

    assert len(result) == 1
    assert client.call_count == 2
    retry_message = client._completions.calls[1]["messages"][-1]["content"]
    assert "exactement deux phrases complètes" in retry_message
    assert "interprétation comparative" in retry_message


def test_triage_section_changes_propagates_runtime_error_unwrapped() -> None:
    """Un refus modèle remonte en RuntimeError, PAS en TriageValidationError."""
    client = _FakeStructuredClient(
        _make_parsed_response(None, refusal="je refuse")
    )

    changes = [
        {
            "diff_type": "modified",
            "semantic_text_t1": "La banque souligne le risque de liquidité dans sa divulgation.",
            "semantic_text_t2": "La banque souligne le risque de marché dans sa divulgation.",
            "source_text_t1": "La banque souligne le risque de liquidité dans sa divulgation.",
            "source_text_t2": "La banque souligne le risque de marché dans sa divulgation.",
        }
    ]

    with pytest.raises(RuntimeError, match="refused"):
        _triage_section_changes(
            client=client,
            model="gpt-4o",
            section_key="gestion_risques",
            changes=changes,
        )

    assert client.call_count == 1


def test_triage_section_changes_processes_changes_one_by_one() -> None:
    def valid_response(**_kwargs):
        return _make_parsed_response(
            TriageAMFCompactLLMBatch(
                triages=[
                    TriageAMFCompactLLMResultWithIndex(
                        change_index=1,
                        is_relevant=False,
                        themes_amf=[],
                        nouvelle_idee=False,
                        relevance_reason=_compact_reason(),
                    )
                ]
            )
        )

    client = _FakeStructuredClient(valid_response)

    changes = [
        {
            "diff_type": "modified",
            "semantic_text_t1": "Ancien texte A.",
            "semantic_text_t2": "Nouveau texte A.",
        },
        {
            "diff_type": "modified",
            "semantic_text_t1": "Ancien texte B.",
            "semantic_text_t2": "Nouveau texte B.",
        },
    ]

    enriched = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    assert len(enriched) == 2
    assert client.call_count == 2
    user_prompts = [
        call["messages"][1]["content"] for call in client._completions.calls
    ]
    assert all('"change_index": 1' in prompt for prompt in user_prompts)
    assert all('"change_index": 2' not in prompt for prompt in user_prompts)
    assert all(
        call["max_completion_tokens"] == 670
        for call in client._completions.calls
    )


def test_triage_section_changes_requires_exactly_one_result_per_change() -> None:
    client = _FakeStructuredClient(
        _make_parsed_response(TriageAMFCompactLLMBatch(triages=[]))
    )

    with pytest.raises(TriageValidationError, match="exactement les change_index"):
        _triage_section_changes(
            client=client,
            model="gpt-4o",
            section_key="gestion_risques",
            changes=[
                {
                    "diff_type": "added",
                    "source_text_t2": "Nouveau contrôle contre les ransomwares.",
                }
            ],
        )


def test_triage_section_changes_batches_two_sides_of_one_semantic_distinct_decision() -> None:
    """One semantic decision remains a two-call workflow: compare, then triage."""
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=False,
                themes_amf=[],
                nouvelle_idee=False,
                relevance_reason=_compact_reason(),
            ),
            TriageAMFCompactLLMResultWithIndex(
                change_index=2,
                is_relevant=False,
                themes_amf=[],
                nouvelle_idee=False,
                relevance_reason=_compact_reason(),
            ),
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    changes = [
        {
            "diff_type": "removed",
            "semantic_alignment_group_id": "a04",
            "alignment_decision": "distinct_disclosures",
            "source_text_t1": "Exposition américaine retirée du texte.",
            "source_text_t2": "",
        },
        {
            "diff_type": "added",
            "semantic_alignment_group_id": "a04",
            "alignment_decision": "distinct_disclosures",
            "source_text_t1": "",
            "source_text_t2": "Exposition canadienne ajoutée au texte.",
        },
    ]

    enriched = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_capital",
        changes=changes,
    )

    assert len(enriched) == 2
    assert client.call_count == 1
    prompt = client._completions.calls[0]["messages"][1]["content"]
    assert '"change_index": 1' in prompt
    assert '"change_index": 2' in prompt


def test_triage_section_changes_reads_long_sources_as_full_evidence_packets() -> None:
    from vigilance.text_analysis.triage import (
        _EvidencePacketBatch,
        _EvidencePacketCoherenceCheck,
        _EvidencePacketObservation,
    )

    def valid_response(**kwargs):
        response_format = kwargs["response_format"]
        if response_format is _EvidencePacketBatch:
            return _make_parsed_response(
                _EvidencePacketBatch(
                    observations=[
                        _EvidencePacketObservation(
                            packet_index=1,
                            factual_change="Le texte courant contient une preuve complète à qualifier.",
                        )
                    ]
                )
            )
        if response_format is _EvidencePacketCoherenceCheck:
            return _make_parsed_response(
                _EvidencePacketCoherenceCheck(
                    packet_index=1,
                    verdict="supports",
                    reason="La décision proposée reste cohérente avec la preuve complète.",
                )
            )
        return _make_parsed_response(
            TriageAMFCompactLLMBatch(
                triages=[
                    TriageAMFCompactLLMResultWithIndex(
                        change_index=1,
                        is_relevant=False,
                        themes_amf=[],
                        nouvelle_idee=False,
                        relevance_reason=_compact_reason(),
                    )
                ]
            )
        )

    client = _FakeStructuredClient(valid_response)
    long_semantic = "A" * 5000
    long_source = "B" * 2000

    _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[
            {
                "diff_type": "added",
                "semantic_text_t1": "",
                "semantic_text_t2": long_semantic,
                "source_text_t1": "",
                "source_text_t2": long_source,
            }
        ],
    )

    evidence_call = client._completions.calls[0]
    evidence_prompt = evidence_call["messages"][1]["content"]
    assert evidence_call["response_format"] is _EvidencePacketBatch
    assert long_source in evidence_prompt
    assert "texte tronque pour le triage" not in evidence_prompt
    assert client._completions.calls[-1]["response_format"] is _EvidencePacketCoherenceCheck


def test_triage_section_changes_attaches_deterministic_change_segments() -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=["GOUVERNANCE_RISQUES"],
                nouvelle_idee=True,
                relevance_reason=_compact_reason(),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    changes = [
        {
            "diff_type": "modified",
            "source_text_t1": "Rapports transmis au CGRO, au CRG et au CGR.",
            "source_text_t2": "Rapports transmis au CGRO et au CGR.",
        }
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    assert result[0]["genai_triage"]["change_segments"] == [
        {"kind": "removed", "text_t1": ", au CRG", "text_t2": ""}
    ]
    prompt = "\n".join(
        str(message.get("content", ""))
        for message in client._completions.calls[0]["messages"]
    )
    assert '"exact_change_segments"' in prompt
    assert ", au CRG" in prompt
    assert "impact_it_justification" not in prompt
    assert "justification_posture" not in prompt
    assert client._completions.calls[0]["response_format"] is TriageAMFCompactLLMBatch


def test_governance_new_idea_receives_major_priority() -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=["GOUVERNANCE_RISQUES"],
                nouvelle_idee=True,
                relevance_reason=(
                    "Le rapport courant transfère au conseil d’administration "
                    "l’approbation de l’appétit pour le risque. Ce transfert "
                    "d’autorité modifie la gouvernance et permet de comparer les "
                    "responsabilités décisionnelles entre les banques."
                ),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[
            {
                "diff_type": "modified",
                "source_text_t1": (
                    "Le comité de direction approuve l’appétit pour le risque."
                ),
                "source_text_t2": (
                    "Le conseil d’administration approuve l’appétit pour le risque."
                ),
            }
        ],
    )

    triage = result[0]["genai_triage"]
    assert triage["is_relevant"] is True
    assert triage["nouvelle_idee"] is True
    assert triage["impact_level"] == "MAJEUR"
    assert triage["action_requise"] == "revue_prioritaire"


@pytest.mark.parametrize(
    ("theme", "previous", "current"),
    [
        (
            "MODIFICATION_METHODOLOGIE",
            "Le risque de crédit est mesuré selon l’approche standard.",
            "Le risque de crédit est mesuré selon un modèle interne avancé.",
        ),
        (
            "CONTROLE_CONFORMITE",
            "Le processus de clôture des alertes repose sur une validation.",
            "Le processus de clôture des alertes exige désormais deux validations.",
        ),
    ],
)
def test_real_methodology_or_process_change_receives_major_priority(
    theme: str,
    previous: str,
    current: str,
) -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=[theme],
                nouvelle_idee=True,
                relevance_reason=(
                    "Le rapport courant modifie le fonctionnement décrit dans le "
                    "rapport précédent. Cette évolution substantielle fournit un "
                    "point prioritaire de comparaison entre les banques."
                ),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[
            {
                "diff_type": "modified",
                "source_text_t1": previous,
                "source_text_t2": current,
            }
        ],
    )

    triage = result[0]["genai_triage"]
    assert triage["impact_level"] == "MAJEUR"
    assert triage["action_requise"] == "revue_prioritaire"

    prompt = client._completions.calls[0]["messages"][1]["content"]
    assert "modification réelle de méthodologie ou de processus" in prompt
    assert "Exemple 8 — changement réel de méthodologie" in prompt
    assert "Exemple 9 — modification réelle de processus" in prompt


def test_committee_rename_stays_relevant_without_becoming_a_new_idea() -> None:
    previous = (
        "Le Comité de gestion des risques (CGR) supervise le cadre de gestion "
        "intégrée des risques et présente ses conclusions chaque trimestre."
    )
    current = previous.replace("(CGR)", "(CGRI)")
    change = {
        "diff_type": "modified",
        "source_text_t1": previous,
        "source_text_t2": current,
        "change_summary": (
            "Le Comité de gestion des risques est désormais désigné par "
            "l’acronyme CGRI, sans modification de son mandat."
        ),
    }
    assert _deterministic_cosmetic_exclusion(change) is None

    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=["GOUVERNANCE_RISQUES"],
                nouvelle_idee=False,
                relevance_reason=(
                    "Le rapport courant renomme le comité par le nouvel acronyme "
                    "CGRI sans modifier son mandat. Cette désignation reste utile "
                    "pour suivre la structure de gouvernance entre les périodes."
                ),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[change],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 1
    assert triage["is_relevant"] is True
    assert triage["nouvelle_idee"] is False
    assert triage["impact_level"] == "MINEUR"
    assert triage["action_requise"] == "information"
    prompt = client._completions.calls[0]["messages"][1]["content"]
    assert "reste pertinent même si son mandat demeure identique" in prompt
    assert "simple renommage sans effet sur le mandat ne l’est pas" in prompt
    assert "Exemple 7 — comité renommé pertinent" in prompt


def test_triage_section_changes_holds_unresolved_alignment_for_analyst_review() -> None:
    """An ambiguous pairing never receives an automatic AMF priority verdict."""
    client = _FakeStructuredClient(
        lambda **_kwargs: pytest.fail("Un alignement ambigu ne doit pas atteindre le triage LLM.")
    )
    changes = [
        {
            "diff_type": "modified",
            "alignment_type": "ambiguous",
            "alignment_decision": "uncertain",
            "alignment_confidence": "low",
            "alignment_rationale": "Les candidats fournis ne permettent pas de déterminer une même divulgation.",
            "source_text_t1": "Le comité reçoit un rapport sur le risque de crédit.",
            "source_text_t2": "Le comité reçoit un rapport sur le risque de marché.",
        }
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 0
    assert triage["source"] == "alignment_review_required"
    assert triage["alignment_review_required"] is True
    assert triage["is_relevant"] is False
    assert triage["nouvelle_idee"] is False
    assert triage["change_segments"]
    assert all(segment["kind"] in {"added", "removed", "modified"} for segment in triage["change_segments"])


def test_triage_section_changes_accepts_gpt_confirmed_semantic_alignment() -> None:
    """A semantic ``same_disclosure`` decision clears the way for AMF triage."""
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=False,
                themes_amf=[],
                nouvelle_idee=False,
                relevance_reason=_compact_reason(),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    changes = [
        {
            "diff_type": "modified",
            "alignment_type": "ambiguous",
            "alignment_decision": "same_disclosure",
            "alignment_confidence": "high",
            "alignment_rationale": "Même limite prudentielle, actualisée dans le rapport courant.",
            "source_text_t1": (
                "La banque surveille le risque de crédit selon une approche "
                "interne fondée sur des revues périodiques."
            ),
            "source_text_t2": (
                "La banque surveille le risque de crédit selon une approche "
                "interne fondée sur des revues trimestrielles."
            ),
        }
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    assert client.call_count == 1
    assert result[0]["genai_triage"]["source"] != "alignment_review_required"
    prompt = client._completions.calls[0]["messages"][1]["content"]
    assert '"alignment_decision": "same_disclosure"' in prompt


def test_triage_section_changes_does_not_request_posture_or_it_impact() -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=["RISQUE_TIERS_CLOUD"],
                nouvelle_idee=True,
                relevance_reason=_compact_reason(),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    changes = [
        {
            "diff_type": "added",
            "semantic_text_t1": "",
            "semantic_text_t2": "Migration infonuagique avec stratégie de sortie.",
        }
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    prompt = "\n".join(
        str(message.get("content", ""))
        for message in client._completions.calls[0]["messages"]
    )
    assert "justification_posture" not in prompt
    assert "impact_it_justification" not in prompt
    assert "relevance_reason" in prompt
    assert "exactement deux phrases complètes" in prompt
    assert "La première décrit factuellement" in prompt
    assert "La seconde interprète" in prompt
    assert "100 à 120 mots" not in prompt
    assert result[0]["genai_triage"]["impact_it"] == "INDETERMINE"
    assert result[0]["genai_triage"]["changement_posture"] == "INDETERMINE"
    assert result[0]["genai_triage"]["statut_mise_en_oeuvre"] == "INDETERMINE"
    assert result[0]["genai_triage"]["confiance_posture"] == "INDETERMINE"


def test_normalize_themes_amf_clamps_unknown_to_emergent() -> None:
    from vigilance.text_analysis.triage import _normalize_themes_amf

    assert _normalize_themes_amf(["EXIGENCES_REGLEMENTAIRES"]) == [
        "EXIGENCES_REGLEMENTAIRES"
    ]
    assert _normalize_themes_amf(["THEME_INEXISTANT_XYZ"]) == [
        "SUJET_EMERGENT_HORS_GRILLE"
    ]
    assert _normalize_themes_amf(
        ["RISQUE_EMERGENT", "THEME_INEXISTANT_XYZ", "RISQUE_EMERGENT"]
    ) == ["RISQUE_EMERGENT", "SUJET_EMERGENT_HORS_GRILLE"]
    assert _normalize_themes_amf([]) == []


def test_triage_accepts_amf_theme_outside_candidate_shortlist(monkeypatch) -> None:
    """Un thème AMF valide hors shortlist ne fait plus planter le pipeline."""
    from vigilance.text_analysis import triage as triage_mod

    def _narrow_candidates(change, *, section_key, limit=6):
        return [
            {
                "code": "RISQUE_EMERGENT",
                "label": "Risque émergent",
                "description": "Risque émergent.",
            },
            {
                "code": "SUJET_EMERGENT_HORS_GRILLE",
                "label": "Hors grille",
                "description": "Hors grille.",
            },
        ]

    monkeypatch.setattr(triage_mod, "_candidate_themes_for_change", _narrow_candidates)

    parsed = TriageAMFCompactLLMBatch(
        triages=[
            TriageAMFCompactLLMResultWithIndex(
                change_index=1,
                is_relevant=True,
                themes_amf=["EXIGENCES_REGLEMENTAIRES"],
                nouvelle_idee=True,
                relevance_reason=_compact_reason(),
            )
        ]
    )
    client = _FakeStructuredClient(_make_parsed_response(parsed))
    changes = [
        {
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": (
                "La banque décrit une nouvelle exigence réglementaire du BSIF "
                "sur la divulgation des risques opérationnels."
            ),
        }
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        changes=changes,
    )

    triage = result[0]["genai_triage"]
    assert triage["themes_amf"] == ["EXIGENCES_REGLEMENTAIRES"]
    assert triage["is_relevant"] is True
    user_prompt = client._completions.calls[0]["messages"][1]["content"]
    assert "Taxonomie AMF autorisée" in user_prompt
    assert "EXIGENCES_REGLEMENTAIRES" in user_prompt
    assert "uniquement parmi les `candidate_themes`" not in user_prompt
