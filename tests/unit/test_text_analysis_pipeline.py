from __future__ import annotations

import json
from pathlib import Path

from vigilance.text_analysis_pipeline import (
    PDFBlock,
    ResolvedSection,
    SectionAudit,
    SemanticUnit,
    _allowed_target_sections,
    _build_extraction_prompt_text,
    _build_text_extraction_markdown,
    _build_section_audit,
    _classify_block_type,
    _compute_conservative_new_idea,
    _is_new_major_or_allowed_moderate,
    _looks_like_footnote,
    _reject_semantic_unit_if_table_like,
    _resolve_sections,
    _resolve_source_block_ids,
    _sanitize_semantic_text,
    _section_window_for_page,
    run_text_analysis_pipeline,
)


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


def test_resolve_source_block_ids_prefers_explicit_ids_and_falls_back_to_similarity() -> None:
    blocks = [
        PDFBlock("p001_b001", 1, [0.1, 0.2, 0.9, 0.3], "La banque renforce sa gestion des risques géopolitiques.", 1),
        PDFBlock("p001_b002", 1, [0.1, 0.4, 0.9, 0.5], "Texte secondaire sans rapport.", 2),
    ]

    ids, resolution = _resolve_source_block_ids(
        candidate_blocks=blocks,
        provided_ids=["p001_b001"],
        reference_text="",
        semantic_text="",
    )
    assert ids == ["p001_b001"]
    assert resolution == "matched"

    ids, resolution = _resolve_source_block_ids(
        candidate_blocks=blocks,
        provided_ids=[],
        reference_text="gestion des risques géopolitiques",
        semantic_text="La banque renforce sa gestion des risques géopolitiques.",
    )
    assert ids == ["p001_b001"]
    assert resolution == "fallback"


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


def test_reject_semantic_unit_if_table_like_rejects_dense_numeric_source() -> None:
    source_block = PDFBlock(
        "p010_d001",
        10,
        [0.1, 0.3, 0.9, 0.4],
        "Billets Série 1 1 750 1 750 1 750 1 750 Total 56 890 11 614 $",
        1,
        "narrative",
        True,
        "",
    )

    reject, reason = _reject_semantic_unit_if_table_like(
        semantic_text="La banque décrit son capital.",
        source_text=source_block.text,
        evidence_snippet=source_block.text,
        source_blocks=[source_block],
        page_table_bboxes={10: []},
    )

    assert reject is True
    assert reason in {"table_like_source_text", "dense_numeric_source"}


def test_build_extraction_prompt_text_contains_red_box_and_hard_rules() -> None:
    audit = SectionAudit(
        section_key="gestion_capital",
        section_title="Gestion du capital",
        start_page=10,
        end_page=10,
        anchor_page=10,
        anchor_text="Gestion du capital",
        anchor_bbox_norm=[0.1, 0.2, 0.8, 0.25],
        included_blocks=[PDFBlock("p010_d001", 10, [0.1, 0.3, 0.9, 0.4], "La banque maintient son capital à un niveau prudent.", 1, "narrative", True, "")],
        excluded_blocks=[],
        semantic_units=[],
        table_regions=[{"table_id": "gestion_capital_p010_tbl_01", "page": 10, "bbox": [0.1, 0.5, 0.9, 0.8]}],
    )

    prompt = _build_extraction_prompt_text(audit, [10], audit.included_blocks)

    assert "RED boxes" in prompt
    assert "There may be multiple RED boxes on the same page" in prompt
    assert "table-footnote zones" in prompt
    assert "If a unit includes any table or footnote content -> REJECT the entire unit" in prompt
    assert "The target section may start or end in the middle of a page" in prompt
    assert "Ignore any visible text above or below the target section window" in prompt
    assert "Ignore only true table footnotes" in prompt
    assert "¹, ², ³, ⁴, ⁵" in prompt
    assert "s.o., n.s." in prompt
    assert "normal narrative paragraph that appears after table footnotes must still be kept" in prompt
    assert "Keep narrative prose before a table, between two tables, and after table footnotes" in prompt
    assert "footnotes: in some bank reports they are frequent, long, and may look like normal prose" in prompt
    assert "Footnotes may span multiple lines and may still be footnotes even when they are written as full sentences" in prompt
    assert "Do not reject a paragraph only because it contains several percentages, dates, ratios, or amounts" in prompt


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


def test_run_text_analysis_pipeline_writes_extraction_audits(monkeypatch, tmp_path: Path) -> None:
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
    unit_prev = SemanticUnit(
        unit_id="gestion_risques_unit_001",
        section_key="gestion_risques",
        theme="risque",
        semantic_text="La banque surveille les risques géopolitiques.",
        source_text="Texte exact T1",
        source_block_ids=["p003_b001"],
        source_resolution="matched",
        evidence_pages=[3],
        evidence_snippet="Risques géopolitiques",
    )
    unit_curr = SemanticUnit(
        unit_id="gestion_risques_unit_002",
        section_key="gestion_risques",
        theme="risque",
        semantic_text="La banque décrit une évolution des risques géopolitiques.",
        source_text="Texte exact T2",
        source_block_ids=["p004_b001"],
        source_resolution="matched",
        evidence_pages=[4],
        evidence_snippet="Risques géopolitiques",
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
        semantic_units=[unit_prev],
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
        semantic_units=[unit_curr],
    )

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._build_openai_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._resolve_sections",
        lambda pdf_path, bank_code: {"gestion_risques": section},
    )
    extraction_calls = {"count": 0}

    def _fake_extract_semantic_units_for_pdf(**kwargs):
        extraction_calls["count"] += 1
        if extraction_calls["count"] == 1:
            return {"gestion_risques": [unit_prev]}, [audit_prev]
        return {"gestion_risques": [unit_curr]}, [audit_curr]

    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._extract_semantic_units_for_pdf",
        _fake_extract_semantic_units_for_pdf,
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._compare_section_units",
        lambda **kwargs: [
            {
                "change_id": "gestion_risques_change_001",
                "section_key": "gestion_risques",
                "diff_type": "modified",
                "semantic_text_t1": unit_prev.semantic_text,
                "semantic_text_t2": unit_curr.semantic_text,
                "source_text_t1": unit_prev.source_text,
                "source_text_t2": unit_curr.source_text,
                "source_block_ids_t1": list(unit_prev.source_block_ids),
                "source_block_ids_t2": list(unit_curr.source_block_ids),
                "source_refs_t1": list(unit_prev.source_block_ids),
                "source_refs_t2": list(unit_curr.source_block_ids),
                "pages_t1": list(unit_prev.evidence_pages),
                "pages_t2": list(unit_curr.evidence_pages),
                "source_resolution_t1": unit_prev.source_resolution,
                "source_resolution_t2": unit_curr.source_resolution,
                "evidence_t1": {"pages": [3], "snippet": "Risques géopolitiques"},
                "evidence_t2": {"pages": [4], "snippet": "Risques géopolitiques"},
                "change_summary": "Evolution du risque géopolitique.",
            }
        ],
    )
    monkeypatch.setattr(
        "vigilance.text_analysis_pipeline._triage_section_changes",
        lambda **kwargs: [
            {
                "change_id": "gestion_risques_change_001",
                "section_key": "gestion_risques",
                "diff_type": "modified",
                "semantic_text_t1": unit_prev.semantic_text,
                "semantic_text_t2": unit_curr.semantic_text,
                "source_text_t1": unit_prev.source_text,
                "source_text_t2": unit_curr.source_text,
                "source_block_ids_t1": list(unit_prev.source_block_ids),
                "source_block_ids_t2": list(unit_curr.source_block_ids),
                "source_refs_t1": list(unit_prev.source_block_ids),
                "source_refs_t2": list(unit_curr.source_block_ids),
                "pages_t1": list(unit_prev.evidence_pages),
                "pages_t2": list(unit_curr.evidence_pages),
                "source_resolution_t1": unit_prev.source_resolution,
                "source_resolution_t2": unit_curr.source_resolution,
                "evidence_t1": {"pages": [3], "snippet": "Risques géopolitiques"},
                "evidence_t2": {"pages": [4], "snippet": "Risques géopolitiques"},
                "change_summary": "Evolution du risque géopolitique.",
                "genai_triage": {
                    "is_relevant": True,
                    "impact_level": "MAJEUR",
                    "category": "RISQUE",
                    "action_requise": "escalade",
                    "nouvelle_idee": False,
                    "explanation": "Changement majeur.",
                    "impact_description": "",
                    "signals": {"regulatory_reference_added": False, "methodology_change": False},
                },
            }
        ],
    )

    payload, out_path = run_text_analysis_pipeline(
        bank_code="td",
        year_current=2025,
        quarter_current="t2",
        pdf_previous=pdf_previous,
        pdf_current=pdf_current,
        out_root=tmp_path / "outputs",
        model="gpt-4o",
    )

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
