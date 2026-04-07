from __future__ import annotations

import json
from pathlib import Path

from vigilance.text_analysis_pipeline import (
    PDFBlock,
    ResolvedSection,
    SectionAudit,
    SemanticUnit,
    _build_section_audit,
    _compute_conservative_new_idea,
    _is_new_major_or_allowed_moderate,
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

    assert payload["extraction_artifact_t1"] == "text_extraction_2025_t1.json"
    assert payload["extraction_artifact_t2"] == "text_extraction_2025_t2.json"
    assert out_path.exists()

    extraction_prev = out_path.parent / "text_extraction_2025_t1.json"
    extraction_curr = out_path.parent / "text_extraction_2025_t2.json"
    assert extraction_prev.exists()
    assert extraction_curr.exists()

    prev_data = json.loads(extraction_prev.read_text())
    curr_data = json.loads(extraction_curr.read_text())
    assert prev_data["source_pdf"]["sha256"]
    assert curr_data["source_pdf"]["sha256"]
    assert prev_data["sections"][0]["semantic_units"][0]["source_block_ids"] == ["p003_b001"]
