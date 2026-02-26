"""Unit tests for GenAI executive summary parsing and fallback logic."""

from __future__ import annotations

import pytest

from app.genai_summary import (
    _build_genai_input,
    _heuristic_fallback,
    _validate_genai_response,
    generate_genai_summary,
)


def _minimal_result(**overrides) -> dict:
    base = {
        "schema_version": "comparison_canonical_v1",
        "bank_code": "bnc",
        "summary": {
            "tables_t1": 10,
            "tables_t2": 12,
            "tables_matched": 8,
            "tables_added": 2,
            "tables_removed": 1,
            "total_added_indicators": 5,
            "total_removed_indicators": 3,
            "total_renamed_indicators": 1,
            "total_footnotes_added": 2,
            "total_footnotes_removed": 0,
            "total_footnotes_modified": 1,
        },
        "table_comparisons": [
            {
                "table_id_t1": "T1",
                "table_id_t2": "T2",
                "title_t1": "Ratios reglementaires CET1",
                "title_t2": "Ratios reglementaires CET1",
                "section": "gestion_capital",
                "added_indicators": ["Nouveau ratio TLAC"],
                "removed_indicators": [],
                "renamed_indicators": [],
                "table_status": "modifie",
                "footnotes_counts": {"added": 1, "removed": 0, "modified": 0},
                "footnotes_diff": {
                    "added": [{"change_type": "new_footnote", "description": "Nouvelle note"}],
                    "removed": [],
                    "modified": [],
                    "counts": {"added": 1, "removed": 0, "modified": 0},
                },
            }
        ],
        "tables_added": [],
        "tables_removed": [],
        "meta": {},
    }
    base.update(overrides)
    return base


class TestValidateGenaiResponse:
    def test_valid_response_fr(self) -> None:
        raw = {
            "resume_executif": [
                "Nouveau ratio TLAC ajoute",
                "Changements de notes dans les tableaux de capital",
            ],
            "pertinence_globale": "ELEVEE",
            "mentions_reglementaires": ["BSIF", "TLAC"],
            "tableaux": [
                {
                    "table_uid": "T2",
                    "label_pertinence": "REGLEMENTAIRE",
                    "raison": "Nouvelle exigence de divulgation TLAC",
                    "changements_cles": ["Ajout du ratio TLAC"],
                }
            ],
        }
        result = _validate_genai_response(raw)
        assert result["pertinence_globale"] == "ELEVEE"
        assert len(result["resume_executif"]) == 2
        assert result["tableaux"][0]["label_pertinence"] == "REGLEMENTAIRE"

    def test_en_labels_mapped_to_fr(self) -> None:
        raw = {
            "executive_summary_bullets": ["English bullet"],
            "overall_relevance": "HIGH",
            "regulatory_mentions": ["OSFI"],
            "tables": [
                {
                    "table_uid": "T1",
                    "relevance_label": "REGULATORY",
                    "why": "Test",
                    "key_changes": ["change"],
                }
            ],
        }
        result = _validate_genai_response(raw)
        assert result["pertinence_globale"] == "ELEVEE"
        assert result["tableaux"][0]["label_pertinence"] == "REGLEMENTAIRE"

    def test_invalid_relevance_label_defaults(self) -> None:
        raw = {
            "resume_executif": ["Puce"],
            "pertinence_globale": "INVALIDE",
            "tableaux": [
                {
                    "table_uid": "T1",
                    "label_pertinence": "PAS_UN_LABEL",
                    "raison": "test",
                }
            ],
        }
        result = _validate_genai_response(raw)
        assert result["pertinence_globale"] == "MOYENNE"
        assert result["tableaux"][0]["label_pertinence"] == "INCONNU"

    def test_missing_fields_safe(self) -> None:
        result = _validate_genai_response({})
        assert result["resume_executif"] == []
        assert result["pertinence_globale"] == "MOYENNE"
        assert result["tableaux"] == []

    def test_non_list_tables_handled(self) -> None:
        raw = {"tableaux": "not_a_list", "resume_executif": "not_a_list"}
        result = _validate_genai_response(raw)
        assert result["tableaux"] == []
        assert result["resume_executif"] == []

    def test_truncation(self) -> None:
        raw = {
            "resume_executif": [f"Puce {i}" for i in range(20)],
            "tableaux": [
                {
                    "table_uid": "T1",
                    "label_pertinence": "RISQUE",
                    "raison": "x" * 1000,
                    "changements_cles": [f"changement_{i}" for i in range(20)],
                }
            ],
        }
        result = _validate_genai_response(raw)
        assert len(result["resume_executif"]) <= 10
        assert len(result["tableaux"][0]["raison"]) <= 500
        assert len(result["tableaux"][0]["changements_cles"]) <= 5


class TestHeuristicFallback:
    def test_basic_fallback(self) -> None:
        result = _minimal_result()
        fallback = _heuristic_fallback(result)
        assert len(fallback["resume_executif"]) > 0
        assert fallback["pertinence_globale"] in ("ELEVEE", "MOYENNE", "FAIBLE")
        assert isinstance(fallback["tableaux"], list)

    def test_empty_result_fallback(self) -> None:
        result = {"summary": {}}
        fallback = _heuristic_fallback(result)
        assert fallback["pertinence_globale"] == "FAIBLE"
        assert len(fallback["resume_executif"]) > 0

    def test_footnotes_in_fallback(self) -> None:
        result = _minimal_result()
        fallback = _heuristic_fallback(result)
        fn_bullets = [b for b in fallback["resume_executif"] if "note" in b.lower()]
        assert len(fn_bullets) >= 1


class TestBuildGenaiInput:
    def test_prompt_contains_tables(self) -> None:
        result = _minimal_result()
        prompt = _build_genai_input(result)
        assert "Ratios reglementaires CET1" in prompt
        assert "gestion_capital" in prompt

    def test_prompt_includes_footnote_info(self) -> None:
        result = _minimal_result()
        prompt = _build_genai_input(result)
        assert "Notes de bas de page" in prompt or "Notes modifiees" in prompt

    def test_max_tables_respected(self) -> None:
        comps = []
        for i in range(100):
            comps.append({
                "table_id_t1": f"T1_{i}",
                "table_id_t2": f"T2_{i}",
                "title_t1": f"Table {i}",
                "title_t2": f"Table {i}",
                "section": "gestion_capital",
                "added_indicators": ["ind"],
                "removed_indicators": [],
                "renamed_indicators": [],
                "table_status": "modifie",
                "footnotes_counts": {"added": 0, "removed": 0, "modified": 0},
            })
        result = _minimal_result(table_comparisons=comps)
        prompt = _build_genai_input(result, max_tables=5)
        assert prompt.count("--- Tableau") <= 5


class TestGenerateGenaiSummaryDisabled:
    def test_disabled_returns_heuristic(self) -> None:
        result = _minimal_result()
        summary = generate_genai_summary(result)
        assert summary["source"] == "heuristic"
        assert len(summary["resume_executif"]) > 0
