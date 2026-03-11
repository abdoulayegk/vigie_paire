"""Unit tests for Vision full extractor schema validation."""

from __future__ import annotations

import pytest

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionSchemaContractError,
    _build_openai_json_schema,
    _classify_openai_error,
    _parse_vision_result,
    _validate_openai_strict_schema_contract,
)


def test_parse_vision_result_valid_with_defaults() -> None:
    raw = {
        "reasoning_scratchpad": "Test analysis",
        "indicators": [
            {"text": " Ratio CET1 ", "bbox": [0.1, 0.2, 0.4, 0.25]},
            {"text": "", "bbox": None},
            {"text": "Total", "bbox": [0.1, 0.3, 0.4, 0.34]},
        ],
        "footnotes_content": {"1": " Note A ", "2": "Note B"},
        "footnote_markers": ["1", " 2 "],
        "confidence": 0.91,
    }
    parsed = _parse_vision_result(raw)
    assert parsed is not None
    assert len(parsed.indicators) == 2
    assert parsed.indicators[0]["text"] == "Ratio CET1"
    assert parsed.indicators[0]["bbox"] == [0.1, 0.2, 0.4, 0.25]
    assert parsed.indicators[1]["text"] == "Total"
    # footnotes_content is now an ordered list of dicts with 'marker' key
    assert parsed.footnotes_content == [
        {"marker": "1", "text": "Note A"},
        {"marker": "2", "text": "Note B"},
    ]
    assert parsed.footnote_markers == ["1", "2"]
    assert parsed.appears_truncated is False
    assert parsed.estimated_content_height is None
    # New required fields
    assert parsed.table_title == ""
    assert parsed.headers == []
    assert parsed.rows == []
    assert parsed.vision_status == "ok"
    assert parsed.warnings == []


def test_parse_vision_result_valid_full_payload() -> None:
    raw = {
        "reasoning_scratchpad": "Analysis",
        "indicators": [{"text": "A", "bbox": None}],
        "footnotes_content": {"(1)": "Texte"},
        "footnote_markers": ["(1)"],
        "confidence": 0.8,
        "appears_truncated": True,
        "estimated_content_height": 72,
    }
    parsed = _parse_vision_result(raw)
    assert parsed is not None
    assert parsed.appears_truncated is True
    assert parsed.estimated_content_height == 72
    # footnotes_content is now an ordered list
    assert len(parsed.footnotes_content) == 1
    assert parsed.footnotes_content[0]["marker"] == "(1)"
    assert parsed.footnotes_content[0]["text"] == "Texte"


def test_parse_vision_result_recovers_wrapped_payload() -> None:
    raw = {
        "Reponse actuelle": {
            "reasoning_scratchpad": "Wrapped analysis",
            "table_title": "Tableau 1",
            "headers": ["Colonne 1"],
            "indicators": [{"text": "Ratio CET1", "bbox": [0.1, 0.2, 0.4, 0.25]}],
            "rows": [["Ratio CET1", "13,1 %"]],
            "footnotes_content": [{"id": "1", "text": "Note"}],
            "footnote_markers": ["1"],
            "has_hierarchy": False,
            "extraction_confidence": "high",
            "notes": "",
            "confidence": 0.93,
            "appears_truncated": False,
            "estimated_content_height": 78,
        }
    }
    parsed = _parse_vision_result(raw)
    assert parsed is not None
    assert parsed.table_title == "Tableau 1"
    assert parsed.indicators[0]["text"] == "Ratio CET1"
    assert parsed.confidence == 0.93


def test_parse_vision_result_rejects_missing_required() -> None:
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "A", "bbox": None}],
        "footnotes_content": {},
        # missing footnote_markers + confidence
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_wrong_types() -> None:
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": "not-a-list",
        "footnotes_content": [],
        "footnote_markers": {},
        "confidence": "high",
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_height_out_of_range() -> None:
    raw = {
        "reasoning_scratchpad": "test",
        "indicators": [{"text": "A", "bbox": None}],
        "footnotes_content": {},
        "footnote_markers": [],
        "confidence": 0.5,
        "estimated_content_height": 150,
    }
    assert _parse_vision_result(raw) is None


def test_openai_json_schema_contains_strict_contract() -> None:
    schema = _build_openai_json_schema()
    assert schema["type"] == "json_schema"
    block = schema["json_schema"]
    assert block["strict"] is True
    s = block["schema"]
    assert s["additionalProperties"] is False
    required = set(s["required"])
    properties = set(s["properties"].keys())
    assert required == properties
    assert "reasoning_scratchpad" in required
    assert "appears_truncated" in required
    assert "estimated_content_height" in required
    # New content fields
    assert "table_title" in required
    assert "headers" in required
    assert "rows" in required


def test_openai_schema_validator_rejects_missing_required_key() -> None:
    schema = _build_openai_json_schema()
    required = list(schema["json_schema"]["schema"]["required"])
    required.remove("appears_truncated")
    schema["json_schema"]["schema"]["required"] = required
    with pytest.raises(VisionSchemaContractError):
        _validate_openai_strict_schema_contract(schema)


def test_classify_openai_error_detects_schema_contract_invalid() -> None:
    err = RuntimeError(
        "Invalid schema for response_format 'vision_full_extraction': "
        "Missing 'appears_truncated'."
    )
    assert _classify_openai_error(err) == "schema_contract_invalid"


def test_extract_raises_schema_contract_error_once_and_circuit_breaks(
    monkeypatch,
) -> None:
    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            self.calls += 1
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "\"Invalid schema for response_format 'vision_full_extraction': "
                "In context=(), 'required' is required to be supplied and to be an "
                "array including every key in properties. Missing 'appears_truncated'.\"}}"
            )

    fake_completions = _FakeCompletions()
    fake_client = type(
        "FakeClient",
        (),
        {
            "chat": type(
                "FakeChat",
                (),
                {"completions": fake_completions},
            )()
        },
    )()

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    extractor._client = fake_client
    monkeypatch.setattr(extractor, "_ensure_client", lambda: None)

    with pytest.raises(VisionSchemaContractError):
        extractor.extract(crop_bytes=b"abc", bank_code="bnc")
    assert fake_completions.calls == 1


def test_extract_returns_result_after_successful_api_response(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": "stop",
                    },
                )()
            ]

    class _FakeCompletions:
        def __init__(self, content: str) -> None:
            self.calls = 0
            self._content = content

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            self.calls += 1
            return _FakeResponse(self._content)

    payload = {
        "reasoning_scratchpad": "Analyse",
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": [{"text": "Ratio CET1", "bbox": None}],
        "rows": [["Ratio CET1", "13,1 %"]],
        "footnotes_content": [{"id": "1", "text": "Note 1"}],
        "footnote_markers": ["1"],
        "has_hierarchy": False,
        "extraction_confidence": "high",
        "notes": "",
        "confidence": 0.93,
        "appears_truncated": False,
        "estimated_content_height": 81,
    }
    fake_completions = _FakeCompletions(__import__("json").dumps(payload))
    fake_client = type(
        "FakeClient",
        (),
        {
            "chat": type(
                "FakeChat",
                (),
                {"completions": fake_completions},
            )()
        },
    )()

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    extractor._client = fake_client
    monkeypatch.setattr(extractor, "_ensure_client", lambda: None)

    result = extractor.extract(crop_bytes=b"abc", bank_code="bnc")

    assert result is not None
    assert result.table_title == "Tableau 1"
    assert result.indicators == [{"text": "Ratio CET1", "bbox": None}]
    assert result.rows == [["Ratio CET1", "13,1 %"]]
    assert fake_completions.calls == 1
