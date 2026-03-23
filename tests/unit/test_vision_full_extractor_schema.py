"""Unit tests for Vision full extractor schema validation."""

from __future__ import annotations

from typing import Any, cast

import pytest

from vigilance.extraction.vision_full_extractor import (
    VisionFullExtractor,
    VisionSchemaContractError,
    _build_openai_json_schema,
    _classify_openai_error,
    _parse_vision_result,
    _validate_openai_strict_schema_contract,
)


def test_default_extraction_model_resolves_to_gpt54() -> None:
    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    assert extractor.model_name == "gpt-5.4"
    assert extractor.model_role == "extraction_primary"


def test_parse_vision_result_valid_with_defaults() -> None:
    raw = {
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
    assert parsed.indicators[0]["bbox"] is None
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
        "indicators": [{"text": "A", "bbox": None}],
        "footnotes_content": {},
        # missing footnote_markers + confidence
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_wrong_types() -> None:
    raw = {
        "indicators": "not-a-list",
        "footnotes_content": [],
        "footnote_markers": {},
        "confidence": "high",
    }
    assert _parse_vision_result(raw) is None


def test_parse_vision_result_rejects_height_out_of_range() -> None:
    raw = {
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
    assert "reasoning_scratchpad" not in required
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


def test_classify_openai_error_detects_invalid_request_json_body() -> None:
    err = RuntimeError(
        "Error code: 400 - {'error': {'message': "
        "\"We could not parse the JSON body of your request. "
        "(HINT: This likely means you aren't using your HTTP library correctly. "
        "The OpenAI API expects a JSON payload, but what was sent was not valid JSON.)\"}}"
    )
    assert _classify_openai_error(err) == "request_body_invalid"


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
            self.usage = type(
                "Usage",
                (),
                {
                    "prompt_tokens": 321,
                    "completion_tokens": 654,
                    "total_tokens": 975,
                },
            )()
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
            self.models: list[str] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.models.append(str(kwargs.get("model")))
            self.calls += 1
            return _FakeResponse(self._content)

    payload = {
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
    assert result.requested_max_completion_tokens == 65536
    assert result.prompt_tokens == 321
    assert result.completion_tokens == 654
    assert result.total_tokens == 975
    assert fake_completions.models == ["gpt-5.4"]
    assert fake_completions.calls == 1


def test_extract_falls_back_to_json_object_when_structured_request_body_is_rejected(
    monkeypatch,
) -> None:
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
            self.response_formats: list[dict[str, object]] = []
            self._content = content

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            self.response_formats.append(dict(kwargs.get("response_format") or {}))
            if self.calls == 1:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': "
                    "\"We could not parse the JSON body of your request. "
                    "(HINT: This likely means you aren't using your HTTP library correctly. "
                    "The OpenAI API expects a JSON payload, but what was sent was not valid JSON.)\"}}"
                )
            return _FakeResponse(self._content)

    payload = {
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": [{"text": "Ratio CET1", "bbox": None}],
        "rows": [["Ratio CET1", "13,1 %"]],
        "footnotes_content": [],
        "footnote_markers": [],
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
    assert fake_completions.calls == 2
    assert fake_completions.response_formats[0]["type"] == "json_schema"
    assert fake_completions.response_formats[1] == {"type": "json_object"}


def test_extract_with_quality_pass_keeps_same_resolved_model_for_recrop(
    monkeypatch,
) -> None:
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
        def __init__(self, contents: list[str]) -> None:
            self._contents = list(contents)
            self.models: list[str] = []
            self.max_tokens: list[int] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.models.append(str(kwargs.get("model")))
            self.max_tokens.append(int(kwargs.get("max_completion_tokens")))
            content = self._contents.pop(0)
            return _FakeResponse(content)

    first_payload = {
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": [{"text": "Ratio CET1", "bbox": None}],
        "rows": [["Ratio CET1", "13,1 %"]],
        "footnotes_content": [],
        "footnote_markers": [],
        "has_hierarchy": False,
        "extraction_confidence": "medium",
        "notes": "",
        "confidence": 0.4,
        "appears_truncated": False,
        "estimated_content_height": 60,
    }
    second_payload = dict(first_payload)
    second_payload["confidence"] = 0.96
    second_payload["extraction_confidence"] = "high"
    second_payload["estimated_content_height"] = 88

    fake_completions = _FakeCompletions(
        [__import__("json").dumps(first_payload), __import__("json").dumps(second_payload)]
    )
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

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"abc",
        bank_code="bnc",
        bbox_norm=[0.2, 0.2, 0.8, 0.8],
        vision_cfg={},
        get_recrop_fn=lambda _ext: b"recrop",
    )

    assert result is not None
    assert result.recrop_attempted is True
    assert result.recrop_used is True
    assert fake_completions.models == ["gpt-5.4", "gpt-5.4"]
    assert fake_completions.max_tokens == [65536, 65536]


def test_extract_returns_placeholder_partial_on_truncation_when_best_effort_fails(
    monkeypatch,
) -> None:
    """When finish_reason=length and best-effort salvage fails, extract() returns a minimal partial result."""
    class _FakeResponse:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": finish_reason,
                    },
                )()
            ]

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return _FakeResponse('{"table_title": "Tronque"', "length")

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

    result = extractor.extract(crop_bytes=b"abc", bank_code="bnc")

    assert result is not None
    assert result.vision_status == "partial"
    assert result.finish_reason == "length"
    assert result.appears_truncated is True
    assert result.indicators == []
    assert result.requested_max_completion_tokens == 65536
    assert len(fake_completions.calls) == 1
    assert fake_completions.calls[0]["max_completion_tokens"] == 65536


def test_extract_returns_partial_on_truncation_when_best_effort_succeeds(monkeypatch) -> None:
    """When finish_reason=length but JSON has indicators+confidence, return partial result (single call)."""
    class _FakeResponse:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": finish_reason,
                    },
                )()
            ]

    # Valid JSON with indicators and confidence so best-effort parse succeeds
    truncated_json = __import__("json").dumps({
        "table_title": "Tableau 1",
        "indicators": ["L1", "L2"],
        "confidence": 0.88,
    })

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return _FakeResponse(truncated_json, "length")

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

    result = extractor.extract(crop_bytes=b"abc", bank_code="bnc")

    assert result is not None
    assert result.vision_status == "partial"
    assert "vision_truncated" in result.warnings
    assert result.appears_truncated is True
    assert len(result.indicators) == 2
    assert result.indicators[0]["text"] == "L1"
    assert result.confidence == 0.88
    assert result.rows == []
    assert result.finish_reason == "length"
    assert result.requested_max_completion_tokens == 65536
    assert len(fake_completions.calls) == 1
    assert fake_completions.calls[0]["max_completion_tokens"] == 65536


def test_extract_retries_on_invalid_json_then_succeeds(monkeypatch) -> None:
    """First call returns invalid JSON, second (retry with json_object + repair) returns valid full; result OK."""
    class _FakeResponse:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": finish_reason,
                    },
                )()
            ]

    valid_payload = {
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": ["L1", "L2"],
        "rows": [["L1", "100"], ["L2", "200"]],
        "footnotes_content": [{"id": "1", "text": "Note"}],
        "footnote_markers": ["1"],
        "has_hierarchy": False,
        "extraction_confidence": "high",
        "notes": "",
        "confidence": 0.92,
        "appears_truncated": False,
        "estimated_content_height": None,
    }

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _FakeResponse("not valid json {", "stop")
            return _FakeResponse(
                __import__("json").dumps(valid_payload),
                "stop",
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

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False, max_retries_json=1)
    extractor._client = fake_client
    monkeypatch.setattr(extractor, "_ensure_client", lambda: None)

    result = extractor.extract(crop_bytes=b"abc", bank_code="bnc")

    assert result is not None
    assert len(fake_completions.calls) == 2
    assert fake_completions.calls[0]["max_completion_tokens"] == 65536
    assert fake_completions.calls[1]["max_completion_tokens"] == 65536
    assert result.table_title == "Tableau 1"
    assert result.rows == [["L1", "100"], ["L2", "200"]]
    assert result.vision_status == "partial"
    assert "vision_structured_output_fallback" in result.warnings
    assert result.requested_max_completion_tokens == 65536


def test_extract_with_quality_pass_retries_same_crop_at_128k_on_truncation(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": finish_reason,
                    },
                )()
            ]

    valid_payload = {
        "table_title": "Tableau 99",
        "headers": ["Indicateur", "Valeur"],
        "indicators": ["L1", "L2", "L3"],
        "rows": [["L1", "100"], ["L2", "200"], ["L3", "300"]],
        "footnotes_content": [],
        "footnote_markers": [],
        "has_hierarchy": False,
        "extraction_confidence": "high",
        "notes": "",
        "confidence": 0.97,
        "appears_truncated": False,
        "estimated_content_height": 90,
    }

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _FakeResponse('{"table_title":"Tronque"', "length")
            return _FakeResponse(__import__("json").dumps(valid_payload), "stop")

    fake_completions = _FakeCompletions()
    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("FakeChat", (), {"completions": fake_completions})()},
    )()

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    extractor._client = fake_client
    monkeypatch.setattr(extractor, "_ensure_client", lambda: None)

    recrop_calls: list[float] = []
    result = extractor.extract_with_quality_pass(
        crop_bytes=b"abc",
        bank_code="bnc",
        bbox_norm=[0.2, 0.2, 0.8, 0.8],
        vision_cfg={
            "vision_max_completion_tokens": 65536,
            "vision_max_completion_tokens_rescue_enabled": True,
            "vision_max_completion_tokens_rescue": 128000,
        },
        get_recrop_fn=lambda ext: recrop_calls.append(ext) or b"recrop",
    )

    assert result is not None
    assert [int(call["max_completion_tokens"]) for call in fake_completions.calls] == [
        65536,
        128000,
    ]
    assert result.requested_max_completion_tokens == 128000
    assert result.rescue_used is True
    assert result.recrop_attempted is False
    assert recrop_calls == []


def test_extract_with_quality_pass_recrop_truncation_triggers_128k_rescue(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type("Message", (), {"content": content})(),
                        "finish_reason": finish_reason,
                    },
                )()
            ]

    first_payload = {
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": ["L1", "L2"],
        "rows": [["L1", "100"]],
        "footnotes_content": [],
        "footnote_markers": [],
        "has_hierarchy": False,
        "extraction_confidence": "medium",
        "notes": "",
        "confidence": 0.45,
        "appears_truncated": False,
        "estimated_content_height": 62,
    }
    final_payload = {
        "table_title": "Tableau 1",
        "headers": ["Indicateur", "Valeur"],
        "indicators": ["L1", "L2", "L3"],
        "rows": [["L1", "100"], ["L2", "200"], ["L3", "300"]],
        "footnotes_content": [],
        "footnote_markers": [],
        "has_hierarchy": False,
        "extraction_confidence": "high",
        "notes": "",
        "confidence": 0.98,
        "appears_truncated": False,
        "estimated_content_height": 91,
    }

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _FakeResponse(__import__("json").dumps(first_payload), "stop")
            if len(self.calls) == 2:
                return _FakeResponse('{"table_title":"Tronque recrop"', "length")
            return _FakeResponse(__import__("json").dumps(final_payload), "stop")

    fake_completions = _FakeCompletions()
    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("FakeChat", (), {"completions": fake_completions})()},
    )()

    extractor = VisionFullExtractor(api_key="test-key", use_cache=False)
    extractor._client = fake_client
    monkeypatch.setattr(extractor, "_ensure_client", lambda: None)

    result = extractor.extract_with_quality_pass(
        crop_bytes=b"abc",
        bank_code="bnc",
        bbox_norm=[0.2, 0.2, 0.8, 0.8],
        vision_cfg={
            "vision_max_completion_tokens": 65536,
            "vision_max_completion_tokens_rescue_enabled": True,
            "vision_max_completion_tokens_rescue": 128000,
        },
        get_recrop_fn=lambda _ext: b"recrop",
    )

    assert result is not None
    assert [int(call["max_completion_tokens"]) for call in fake_completions.calls] == [
        65536,
        65536,
        128000,
    ]
    assert result.recrop_attempted is True
    assert result.recrop_used is True
    assert result.requested_max_completion_tokens == 128000
    assert result.rescue_used is True
