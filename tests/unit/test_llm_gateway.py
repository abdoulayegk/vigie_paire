"""Unit tests for the centralized LLM gateway."""

from __future__ import annotations

import pytest

from vigie.llm import (
    build_completion_kwargs,
    is_configured,
    is_reasoning_model,
    resolve_model,
    resolve_reasoning_effort,
)


def test_resolve_model_chat_aliases(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text("llm_models:\n  chat: gpt-5.4\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_MODEL_CHAT", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_DEFAULT_GENAI", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_EXTRACTION_PRIMARY", raising=False)

    assert resolve_model("chat", config_path=cfg_path) == "gpt-5.4"
    assert resolve_model("default_genai", config_path=cfg_path) == "gpt-5.4"
    assert resolve_model("extraction_primary", config_path=cfg_path) == "gpt-5.4"


def test_build_completion_kwargs_uses_reasoning_effort_for_gpt54() -> None:
    kwargs = build_completion_kwargs(model="gpt-5.4", profile="extraction")
    assert kwargs == {"reasoning_effort": "high"}
    assert "temperature" not in kwargs

    kwargs_default = build_completion_kwargs(model="gpt-5.4", profile="default")
    assert kwargs_default == {"reasoning_effort": "medium"}


def test_build_completion_kwargs_uses_temperature_for_legacy_model() -> None:
    kwargs = build_completion_kwargs(model="gpt-4o", profile="default")
    assert kwargs == {"temperature": 0.0}


def test_is_reasoning_model_detects_gpt5_family() -> None:
    assert is_reasoning_model("gpt-5.4")
    assert is_reasoning_model("gpt-5")
    assert not is_reasoning_model("gpt-4o")


def test_is_configured_openai_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_configured() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert is_configured() is True


def test_require_configured_raises_when_openai_missing(monkeypatch) -> None:
    from vigie.llm import require_configured

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY absent"):
        require_configured()


def test_is_configured_azure_requires_endpoint_and_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert is_configured() is False
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert is_configured() is True


def test_resolve_model_azure_requires_deployment(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text("llm_models:\n  chat: gpt-5.4\n", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_CHAT", raising=False)
    with pytest.raises(RuntimeError, match="deployment missing"):
        resolve_model("chat", config_path=cfg_path)


def test_resolve_model_azure_uses_deployment_env(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text("llm_models:\n  chat: gpt-5.4\n", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_CHAT", "gpt54-prod")
    assert resolve_model("chat", config_path=cfg_path) == "gpt54-prod"


def test_resolve_reasoning_effort_defaults() -> None:
    assert resolve_reasoning_effort("extraction") == "high"
    assert resolve_reasoning_effort("default") == "medium"
    assert resolve_reasoning_effort("locator") == "low"


def test_build_completion_kwargs_uses_low_for_locator() -> None:
    kwargs = build_completion_kwargs(model="gpt-5.4", profile="locator")
    assert kwargs == {"reasoning_effort": "low"}
    assert "temperature" not in kwargs
