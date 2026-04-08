from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigilance.text_analysis_pipeline import (
    PDFBlock,
    ResolvedSection,
    SectionAudit,
    SemanticUnit,
    _call_json_completion,
    _allowed_target_sections,
    _build_text_extraction_markdown,
    _build_section_audit,
    _classify_block_type,
    _compare_section_texts,
    _compute_conservative_new_idea,
    _extract_audits_for_pdf,
    _extract_section_text_from_markdown,
    _is_new_major_or_allowed_moderate,
    _looks_like_footnote,
    _max_output_tokens_for_model,
    _normalize_heading,
    _pair_subsections,
    _parse_subsections,
    _resolve_sections,
    _sanitize_semantic_text,
    _section_window_for_page,
    run_text_analysis_pipeline,
)


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
    triage = {"is_relevant": True, "impact_level": "MAJEUR", "nouvelle_idee": False, "signals": {}}

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
        lambda pdf_path, bank_code: _FakeMapping(),
    )

    resolved = _resolve_sections(tmp_path / "dummy.pdf", "bnc")

    assert set(resolved) == {"gestion_capital", "gestion_risques"}


def test_keep_change_for_new_moderate_signal() -> None:
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "signals": {"regulatory_reference_added": True, "methodology_change": False},
    }

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_drop_editorial_moderate_change() -> None:
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "signals": {"regulatory_reference_added": False, "methodology_change": False},
    }

    assert _is_new_major_or_allowed_moderate(triage) is False


def test_conservative_new_idea_is_false_for_moderate_methodology_change() -> None:
    change = {
        "diff_type": "modified",
        "semantic_text_t1": "La banque améliore progressivement sa méthode de mesure des risques.",
        "semantic_text_t2": "La banque améliore sa méthode de gestion des risques selon les meilleures pratiques.",
    }
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "category": "RISQUE",
        "signals": {"regulatory_reference_added": False, "methodology_change": True},
    }

    assert _compute_conservative_new_idea(change, triage) is False


def test_conservative_new_idea_is_true_for_major_added_regulatory_change() -> None:
    change = {
        "diff_type": "added",
        "semantic_text_t1": "",
        "semantic_text_t2": "La banque introduit un nouveau dispositif de contrôle contre le crime financier.",
    }
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "category": "RISQUE",
        "signals": {"regulatory_reference_added": True, "methodology_change": False},
    }

    assert _compute_conservative_new_idea(change, triage) is True


def test_conservative_new_idea_is_false_for_modified_major_change() -> None:
    change = {
        "diff_type": "modified",
        "semantic_text_t1": "La banque surveille le risque technologique.",
        "semantic_text_t2": "La banque surveille le risque technologique et renforce ses contrôles.",
    }
    triage = {
        "is_relevant": True,
        "impact_level": "MAJEUR",
        "category": "RISQUE",
        "signals": {"regulatory_reference_added": True, "methodology_change": False},
    }

    assert _compute_conservative_new_idea(change, triage) is False


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
        "vigilance.text_analysis_pipeline._call_json_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid json from model")),
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
        lambda pdf_path, bank_code: {"gestion_risques": section},
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._extract_audits_for_pdf",
        lambda **kwargs: [audit_prev] if "prev" in str(kwargs["pdf_path"]) else [audit_curr],
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
             "semantic_text_t1": "Cosmetique avant", "semantic_text_t2": "Cosmetique apres",
             "source_text_t1": "Cosmetique avant", "source_text_t2": "Cosmetique apres",
             "source_block_ids_t1": [], "source_block_ids_t2": [], "source_refs_t1": [],
             "source_refs_t2": [], "pages_t1": [], "pages_t2": [],
             "source_resolution_t1": "markdown", "source_resolution_t2": "markdown",
             "evidence_t1": {"pages": [], "snippet": ""}, "evidence_t2": {"pages": [], "snippet": ""},
             "change_summary": "Cosmétique."},
        ],
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._triage_section_changes",
        lambda **kwargs: [
            {**kwargs["changes"][0], "genai_triage": {"is_relevant": True, "impact_level": "MODERE", "category": "STRUCTURE", "action_requise": "information", "nouvelle_idee": True, "explanation": "", "impact_description": "", "signals": {"regulatory_reference_added": False, "methodology_change": False}}},
            {**kwargs["changes"][1], "genai_triage": {"is_relevant": True, "impact_level": "MAJEUR", "category": "RISQUE", "action_requise": "escalade", "nouvelle_idee": False, "explanation": "", "impact_description": "", "signals": {"regulatory_reference_added": False, "methodology_change": False}}},
            {**kwargs["changes"][2], "genai_triage": {"is_relevant": False, "impact_level": "MINEUR", "category": "COSMETIQUE", "action_requise": "aucune", "nouvelle_idee": False, "explanation": "", "impact_description": "", "signals": {"regulatory_reference_added": False, "methodology_change": False}}},
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

    assert _classify_block_type(block, {}, [], [[0.0, 0.33, 1.0, 0.42]]) == "footnote"


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

    assert _classify_block_type(block, {}, [], [[0.0, 0.60, 1.0, 0.70]]) == "footnote"


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


def test_compare_section_texts_skips_invalid_diff_types(monkeypatch) -> None:
    def _fake_call_json_completion(client, *, model, messages, max_tokens=None):
        return {
            "changes": [
                {"diff_type": "added", "text_t1": "", "text_t2": "Nouvelle idée.", "change_summary": "Ajout."},
                {"diff_type": "invalid_type", "text_t1": "x", "text_t2": "y", "change_summary": ""},
                {"diff_type": "removed", "text_t1": "Ancienne idée.", "text_t2": "", "change_summary": "Suppression."},
            ]
        }

    monkeypatch.setattr("vigilance.text_analysis_pipeline._call_json_completion", _fake_call_json_completion)

    results = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="Ancienne idée.",
        text_t2="Nouvelle idée.",
    )

    assert len(results) == 2
    assert results[0]["diff_type"] == "added"
    assert results[1]["diff_type"] == "removed"
    assert results[0]["source_resolution_t1"] == "markdown"
    assert results[0]["source_resolution_t2"] == "markdown"


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


def test_build_text_extraction_markdown_drops_orphan_heading_without_body() -> None:
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
        lambda pdf_path, bank_code: {"gestion_risques": section},
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._extract_audits_for_pdf",
        lambda **kwargs: [audit_prev] if "prev" in str(kwargs["pdf_path"]) else [audit_curr],
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


def test_normalize_heading_strips_table_prefix_and_lowercases() -> None:
    assert _normalize_heading("T22 Mesures du risque de marché") == "mesures du risque de marché"
    assert _normalize_heading("Risque de liquidité") == "risque de liquidité"


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


def test_compare_section_texts_falls_back_to_single_call_when_no_subsections(monkeypatch) -> None:
    """Sections sans ### doivent toujours produire un seul appel GPT."""
    calls: list[str] = []

    def fake_single_call(*, client, model, section_key, heading_label, heading_slug, text_t1, text_t2, idx_offset):
        calls.append(heading_slug)
        return []

    monkeypatch.setattr("vigilance.text_analysis_pipeline._compare_texts_single_call", fake_single_call)

    _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="Texte T1 sans sous-sections.",
        text_t2="Texte T2 sans sous-sections.",
    )

    assert calls == ["full"]


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


def test_compare_section_texts_synthetic_change_for_removed_subsection(monkeypatch) -> None:
    """Une sous-section T1 sans contrepartie T2 produit un changement synthétique removed."""
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
    assert "Risque opérationnel" in removed[0]["change_summary"]
    assert "Supprimé en T2" in removed[0]["source_text_t1"]


def test_compare_section_texts_synthetic_change_for_added_subsection(monkeypatch) -> None:
    """Une sous-section T2 sans contrepartie T1 produit un changement synthétique added."""
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
    assert "Incidence des tarifs" in added[0]["change_summary"]
    assert "Nouveau en T2" in added[0]["source_text_t2"]
