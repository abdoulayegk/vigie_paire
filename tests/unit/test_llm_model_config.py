"""Unit tests for lightweight OpenAI model routing config."""

from __future__ import annotations

from vigie.support.config import (
    get_llm_model_config,
    get_vision_extraction_config,
    resolve_openai_model,
)


def test_llm_model_config_defaults_when_config_missing(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    cfg = get_llm_model_config(config_path=missing)
    assert cfg["extraction_primary"] == "gpt-5.4"
    assert cfg["default_genai"] == "gpt-4o"


def test_llm_model_config_defaults_when_config_path_is_none() -> None:
    cfg = get_llm_model_config(config_path=None)
    assert cfg["extraction_primary"] == "gpt-5.4"
    assert cfg["default_genai"] == "gpt-4o"


def test_resolve_openai_model_uses_yaml_when_present(tmp_path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        "llm_models:\n"
        "  extraction_primary: gpt-5.4-mini\n"
        "  default_genai: gpt-4.1\n",
        encoding="utf-8",
    )

    assert (
        resolve_openai_model("extraction_primary", config_path=cfg_path)
        == "gpt-5.4-mini"
    )
    assert resolve_openai_model("default_genai", config_path=cfg_path) == "gpt-4.1"


def test_resolve_openai_model_env_override_wins(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        "llm_models:\n"
        "  extraction_primary: gpt-5.4-mini\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_MODEL_EXTRACTION_PRIMARY", "gpt-5.4")

    assert resolve_openai_model("extraction_primary", config_path=cfg_path) == "gpt-5.4"


def test_get_vision_extraction_config_reads_64k_default_and_128k_rescue(
    tmp_path,
) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        "vision_extraction:\n"
        "  vision_max_completion_tokens: 65536\n"
        "  vision_max_completion_tokens_rescue_enabled: true\n"
        "  vision_max_completion_tokens_rescue: 128000\n",
        encoding="utf-8",
    )

    cfg = get_vision_extraction_config(config_path=cfg_path)
    assert cfg["vision_max_completion_tokens"] == 65536
    assert cfg["vision_max_completion_tokens_rescue_enabled"] is True
    assert cfg["vision_max_completion_tokens_rescue"] == 128000
