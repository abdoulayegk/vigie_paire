"""Unit tests for GenAI post-matching change classifier."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = MagicMock(content=content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def _make_comp(
    added: list[str] | None = None,
    removed: list[str] | None = None,
    renamed: list[dict[str, str]] | None = None,
    section: str = "gestion_capital",
    title: str = "Tableau 1",
) -> dict[str, Any]:
    return {
        "section": section,
        "table_title": title,
        "table_status": "modifie",
        "added_indicators": added or [],
        "removed_indicators": removed or [],
        "renamed_indicators": renamed or [],
    }


class TestSanitizeResponse:
    def test_valid_response_fr(self):
        from vigilance.genai.change_classifier import _sanitize_response

        raw = {
            "relevance": "REGLEMENTAIRE",
            "risk_level": "ELEVE",
            "confidence": 0.92,
            "justification": "Ratio CET1 ajoute.",
        }
        result = _sanitize_response(raw)
        assert result["relevance"] == "REGLEMENTAIRE"
        assert result["risk_level"] == "ELEVE"
        assert result["confidence"] == 0.92
        assert result["source"] == "gpt-4o"

    def test_en_labels_mapped_to_fr(self):
        from vigilance.genai.change_classifier import _sanitize_response

        raw = {
            "relevance": "REGULATORY",
            "risk_level": "HIGH",
            "confidence": 0.90,
        }
        result = _sanitize_response(raw)
        assert result["relevance"] == "REGLEMENTAIRE"
        assert result["risk_level"] == "ELEVE"

    def test_invalid_relevance_becomes_non_classifie(self):
        from vigilance.genai.change_classifier import _sanitize_response

        result = _sanitize_response({"relevance": "BANANA"})
        assert result["relevance"] == "NON_CLASSIFIE"

    def test_confidence_clamped(self):
        from vigilance.genai.change_classifier import _sanitize_response

        result = _sanitize_response({"confidence": 1.5, "relevance": "REGLEMENTAIRE"})
        assert result["confidence"] == 1.0

        result2 = _sanitize_response({"confidence": -0.3, "relevance": "REGLEMENTAIRE"})
        assert result2["confidence"] == 0.0

    def test_empty_input(self):
        from vigilance.genai.change_classifier import _sanitize_response

        result = _sanitize_response({})
        assert result["relevance"] == "NON_CLASSIFIE"
        assert result["risk_level"] == "MODERE"
        assert result["confidence"] == 0.5


class TestGenAIChangeClassifier:
    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_classify_reglementaire(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        gpt_response = json.dumps({
            "relevance": "REGLEMENTAIRE",
            "risk_level": "ELEVE",
            "confidence": 0.95,
            "justification": "Le ratio CET1 est une mesure d'adequation du capital Bale III.",
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _FakeResponse(gpt_response)
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        comp = _make_comp(added=["Ratio CET1", "Coussin de conservation"])
        result = classifier.classify_table_change(comp)

        assert result["relevance"] == "REGLEMENTAIRE"
        assert result["risk_level"] == "ELEVE"
        assert result["confidence"] == 0.95
        assert result["source"] == "gpt-4o"

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_classify_en_labels_mapped_to_fr(self, mock_ensure):
        """English labels from the LLM are mapped to French equivalents."""
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        gpt_response = json.dumps({
            "relevance": "REGULATORY",
            "risk_level": "HIGH",
            "confidence": 0.90,
            "justification": "English response mapped to FR.",
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _FakeResponse(gpt_response)
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        result = classifier.classify_table_change(_make_comp(added=["CET1"]))
        assert result["relevance"] == "REGLEMENTAIRE"
        assert result["risk_level"] == "ELEVE"

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_classify_non_significatif(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        gpt_response = json.dumps({
            "relevance": "NON_SIGNIFICATIF",
            "risk_level": "FAIBLE",
            "confidence": 0.88,
            "justification": "Reordonnancement cosmetique de colonnes.",
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _FakeResponse(gpt_response)
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        comp = _make_comp(renamed=[{"old_name": "Col A", "new_name": "Col B"}])
        result = classifier.classify_table_change(comp)

        assert result["relevance"] == "NON_SIGNIFICATIF"
        assert result["risk_level"] == "FAIBLE"

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_api_failure_returns_non_classifie(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        comp = _make_comp(added=["CET1 ratio"])
        result = classifier.classify_table_change(comp)

        assert result["relevance"] == "NON_CLASSIFIE"
        assert result["source"] == "fallback"
        assert classifier.stats["errors"] == 1

    def test_no_api_key_returns_non_classifie(self):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        with patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client") as m:
            m.side_effect = RuntimeError("No key")
            classifier = GenAIChangeClassifier()
            result = classifier.classify_table_change(_make_comp(added=["x"]))
            assert result["relevance"] == "NON_CLASSIFIE"

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_cet1_not_non_significatif(self, mock_ensure):
        """Un changement lie au CET1 ne doit PAS etre classifie NON_SIGNIFICATIF."""
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        gpt_response = json.dumps({
            "relevance": "REGLEMENTAIRE",
            "risk_level": "ELEVE",
            "confidence": 0.97,
            "justification": "Le CET1 est une divulgation fondamentale de Bale III.",
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _FakeResponse(gpt_response)
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        comp = _make_comp(added=["CET1 ratio"])
        result = classifier.classify_table_change(comp)

        assert result["relevance"] != "NON_SIGNIFICATIF"

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_build_compact_payload_limits_indicators(self, mock_ensure):
        """Le payload compact doit plafonner les indicateurs."""
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _FakeResponse(
            json.dumps({"relevance": "NON_SIGNIFICATIF", "risk_level": "FAIBLE", "confidence": 0.5})
        )
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        comp = _make_comp(added=[f"Ind_{i}" for i in range(100)])
        compact = classifier._build_compact_payload(comp)
        parsed = json.loads(compact)

        assert len(parsed["added_indicators"]) <= 30


class TestCircuitBreaker:
    """Verify the circuit breaker opens after consecutive failures."""

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_circuit_opens_after_threshold(self, mock_ensure):
        from vigilance.genai.change_classifier import (
            GenAIChangeClassifier,
            _CIRCUIT_BREAKER_THRESHOLD,
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        assert not classifier.circuit_open

        for i in range(_CIRCUIT_BREAKER_THRESHOLD):
            classifier.classify_table_change(_make_comp(added=[f"ind_{i}"]))

        assert classifier.circuit_open
        assert classifier.stats["errors"] == _CIRCUIT_BREAKER_THRESHOLD

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_circuit_open_returns_non_classifie_immediately(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        for _ in range(3):
            classifier.classify_table_change(_make_comp(added=["x"]))

        assert classifier.circuit_open
        calls_before = classifier.stats["calls"]

        result = classifier.classify_table_change(_make_comp(added=["CET1"]))
        assert result["relevance"] == "NON_CLASSIFIE"
        assert classifier.stats["calls"] == calls_before

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_success_resets_counter(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        mock_client = MagicMock()
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("fail")
            return _FakeResponse(json.dumps({
                "relevance": "REGLEMENTAIRE",
                "risk_level": "ELEVE",
                "confidence": 0.9,
            }))

        mock_client.chat.completions.create.side_effect = side_effect
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key")
        classifier.classify_table_change(_make_comp(added=["a"]))
        classifier.classify_table_change(_make_comp(added=["b"]))
        assert classifier._consecutive_failures == 2
        assert not classifier.circuit_open

        result = classifier.classify_table_change(_make_comp(added=["c"]))
        assert result["relevance"] == "REGLEMENTAIRE"
        assert classifier._consecutive_failures == 0


class TestBatchClassification:
    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_classify_batch_returns_ordered_results(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        responses = [
            {"relevance": "REGLEMENTAIRE", "risk_level": "ELEVE", "confidence": 0.9},
            {"relevance": "NON_SIGNIFICATIF", "risk_level": "FAIBLE", "confidence": 0.8},
            {"relevance": "STRUCTUREL", "risk_level": "MODERE", "confidence": 0.7},
        ]
        call_idx = {"n": 0}

        def side_effect(**kwargs):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return _FakeResponse(json.dumps(responses[idx % len(responses)]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key", max_workers=2)
        comps = [
            _make_comp(added=["CET1"]),
            _make_comp(renamed=[{"old_name": "A", "new_name": "B"}]),
            _make_comp(removed=["Liquidite"]),
        ]
        results = classifier.classify_batch(comps)

        assert len(results) == 3
        for r in results:
            assert r["relevance"] in {"REGLEMENTAIRE", "NON_SIGNIFICATIF", "STRUCTUREL", "NON_CLASSIFIE"}
            assert "source" in r

    def test_classify_batch_empty_list(self):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        classifier = GenAIChangeClassifier(api_key="test-key")
        assert classifier.classify_batch([]) == []

    @patch("vigilance.genai.change_classifier.GenAIChangeClassifier._ensure_client")
    def test_batch_respects_circuit_breaker(self, mock_ensure):
        from vigilance.genai.change_classifier import GenAIChangeClassifier

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")
        mock_ensure.return_value = mock_client

        classifier = GenAIChangeClassifier(api_key="test-key", max_workers=2)
        comps = [_make_comp(added=[f"ind_{i}"]) for i in range(10)]
        results = classifier.classify_batch(comps)

        assert len(results) == 10
        assert classifier.circuit_open
        non_classifie_count = sum(1 for r in results if r["relevance"] == "NON_CLASSIFIE")
        assert non_classifie_count == 10


class TestIntegrationPoint:
    """Verify the classifier is called AFTER matching, not during."""

    def test_genai_classification_position_in_runner(self):
        import ast
        from pathlib import Path

        runner_path = Path(__file__).resolve().parents[2] / "src" / "app" / "comparison_runner.py"
        source = runner_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        top_level_imports = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.add(node.module)

        assert "vigilance.genai" not in top_level_imports
        assert "GenAIChangeClassifier" in source
        assert "include_genai_classification" in source
        assert "classify_batch" in source

    def test_enrich_result_with_genai_called_in_runner(self):
        from pathlib import Path

        runner_path = Path(__file__).resolve().parents[2] / "src" / "app" / "comparison_runner.py"
        source = runner_path.read_text(encoding="utf-8")
        assert "enrich_result_with_genai" in source

    def test_genai_settings_exists_in_yaml(self):
        from pathlib import Path

        yaml_path = Path(__file__).resolve().parents[2] / "configs" / "bank_profiles.yaml"
        content = yaml_path.read_text(encoding="utf-8")
        assert "genai_settings:" in content
        assert "enable_genai_summary: true" in content


class TestExportIncludesGenAI:
    def test_csv_columns_include_genai(self):
        from vigilance.review_export import VALIDATION_CSV_COLUMNS

        assert "pertinence_genai" in VALIDATION_CSV_COLUMNS
        assert "niveau_risque_genai" in VALIDATION_CSV_COLUMNS
