"""Configuration helpers for vigilance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vigilance.config.loader import _resolve_config_path, get_bank_cfg, load_config


_DEFAULT_OPENAI_MODELS: dict[str, str] = {
    "extraction_primary": "gpt-5.4",
    "default_genai": "gpt-4o",
}

_MODEL_ENV_OVERRIDES: dict[str, str] = {
    "extraction_primary": "OPENAI_MODEL_EXTRACTION_PRIMARY",
    "default_genai": "OPENAI_MODEL_DEFAULT_GENAI",
}


def load_bank_profiles(config_path: str | Path = "configs/bank_profiles.yaml") -> dict[str, Any]:
    """Load bank profiles map from the main YAML config."""
    cfg = load_config(config_path)
    banks = cfg.get("banks")
    if isinstance(banks, dict):
        return banks
    return {}


def get_matching_thresholds(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load optional matching thresholds from configuration.

    Supports either:
    - `matching_thresholds: {...}` at root
    - `matching: { thresholds: {...} }`

    If bank_code is provided and banks.<bank_code>.matching_overrides exists,
    those overrides are merged on top of the base thresholds (e.g. for TD).
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}

    try:
        cfg = load_config(path)
    except Exception:
        return {}

    thresholds = cfg.get("matching_thresholds")
    if isinstance(thresholds, dict):
        base = dict(thresholds)
    else:
        matching = cfg.get("matching")
        if isinstance(matching, dict):
            nested = matching.get("thresholds")
            if isinstance(nested, dict):
                base = dict(nested)
            else:
                base = {}
        else:
            base = {}

    if bank_code:
        banks = cfg.get("banks")
        if isinstance(banks, dict):
            key = str(bank_code).strip().lower()
            if key in banks:
                bank_cfg = banks[key]
                if isinstance(bank_cfg, dict):
                    overrides = bank_cfg.get("matching_overrides")
                    if isinstance(overrides, dict):
                        base = {**base, **overrides}

    # Apply indicator diff defaults (PASS 2) when keys absent
    _indicator_defaults: dict[str, Any] = {
        "indicator_hungarian_enabled": True,
        "indicator_rename_min_score": 0.86,
        "indicator_gate_min_len_ratio": 0.55,
        "indicator_gate_min_token_overlap": 1,
        "indicator_similarity_weights": {"ratio": 0.4, "token_set": 0.6},
        "neighbor_aligned_filter_enabled": True,
        "indicator_short_guard_enabled": True,
        "indicator_short_guard_max_tokens": 3,
        "indicator_short_guard_min_stable_tokens": 5,
    }
    for k, v in _indicator_defaults.items():
        if k not in base:
            base[k] = v

    # Embedding defaults (opt-in, config flag use_embeddings default false)
    _embedding_defaults: dict[str, Any] = {
        "use_embeddings": False,
        "embedding_weight_table": 0.12,
        "embedding_weight_indicator": 0.35,
        "embedding_model": "text-embedding-3-small",
    }
    for k, v in _embedding_defaults.items():
        if k not in base:
            base[k] = v

    return base


def get_vision_extraction_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load vision_extraction config with optional bank overrides.

    Global keys: enabled, bottom_extension_footnotes, run_on_all_tables,
    fallback_to_docling_on_error, save_indicators_footnotes_json.
    Per-bank overrides: footnote_marker_type, expected_markers.
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}

    try:
        cfg = load_config(path)
    except Exception:
        return {}

    global_block = cfg.get("vision_extraction")
    if not isinstance(global_block, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(global_block)

    if bank_code:
        banks = cfg.get("banks")
        if isinstance(banks, dict):
            key = str(bank_code).strip().lower()
            if key in banks:
                bank_cfg = banks[key]
                if isinstance(bank_cfg, dict):
                    bank_ve = bank_cfg.get("vision_extraction")
                    if isinstance(bank_ve, dict):
                        base = {**base, **bank_ve}

    return base


def get_llm_model_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
) -> dict[str, str]:
    """Load lightweight OpenAI model routing config."""
    path = _resolve_config_path(config_path)
    base = dict(_DEFAULT_OPENAI_MODELS)
    if not path.exists():
        return base

    try:
        cfg = load_config(path)
    except Exception:
        return base

    raw = cfg.get("llm_models")
    if not isinstance(raw, dict):
        return base

    for role in _DEFAULT_OPENAI_MODELS:
        value = raw.get(role)
        if isinstance(value, str) and value.strip():
            base[role] = value.strip()
    return base


def resolve_openai_model(
    role: str,
    config_path: str | Path = "configs/bank_profiles.yaml",
) -> str:
    """Resolve the OpenAI model for a known role with env override support."""
    key = str(role or "").strip().lower()
    if key not in _DEFAULT_OPENAI_MODELS:
        known = ", ".join(sorted(_DEFAULT_OPENAI_MODELS))
        raise ValueError(f"Unknown OpenAI model role '{role}'. Known roles: {known}")

    env_name = _MODEL_ENV_OVERRIDES.get(key)
    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()

    cfg = get_llm_model_config(config_path=config_path)
    value = cfg.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _DEFAULT_OPENAI_MODELS[key]


def get_validation_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load validation config (post-matching validators) with optional bank overrides.

    Keys: vision_pair_validation, vision_pair_confidence_min, semantic_judge_enabled,
    semantic_judge_banks, rename_validator_enabled, rename_validator_confidence_min,
    rename_validator_batch_size, rename_validator_uncertain_score_band,
    added_table_validator_enabled,
    indicator_validator_enabled, indicator_validator_use_vision,
    indicator_validator_confidence_min, indicator_validator_batch_size.

    For backward compatibility, if vision_pair_validation is absent here, falls back
    to vision_extraction.vision_pair_validation.
    """
    path = _resolve_config_path(config_path)
    cfg: dict[str, Any] = {}
    if path.exists():
        try:
            cfg = load_config(path)
        except Exception:
            pass

    global_block = cfg.get("validation")
    if not isinstance(global_block, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(global_block)

    if bank_code and isinstance(cfg.get("banks"), dict):
        key = str(bank_code).strip().lower()
        bank_cfg = cfg["banks"].get(key)
        if isinstance(bank_cfg, dict):
            bank_val = bank_cfg.get("validation")
            if isinstance(bank_val, dict):
                base = {**base, **bank_val}

    # Fallback: vision_pair_validation from vision_extraction if not in validation
    if "vision_pair_validation" not in base:
        ve = cfg.get("vision_extraction")
        if isinstance(ve, dict) and "vision_pair_validation" in ve:
            base["vision_pair_validation"] = ve["vision_pair_validation"]

    # Cross-section rescue defaults when keys absent
    if "cross_section_rescue_enabled" not in base:
        base["cross_section_rescue_enabled"] = False
    if "cross_section_rescue_rerank_min" not in base:
        base["cross_section_rescue_rerank_min"] = 0.30
    if "cross_section_rescue_vision_confidence_min" not in base:
        base["cross_section_rescue_vision_confidence_min"] = 0.85
    if "cross_section_rescue_max_candidates_per_table" not in base:
        base["cross_section_rescue_max_candidates_per_table"] = 3

    return base


def get_quality_gate_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load quality_gate config with optional bank overrides."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}

    try:
        cfg = load_config(path)
    except Exception:
        return {}

    global_block = cfg.get("quality_gate")
    if not isinstance(global_block, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(global_block)

    if bank_code:
        banks = cfg.get("banks")
        if isinstance(banks, dict):
            key = str(bank_code).strip().lower()
            if key in banks:
                bank_cfg = banks[key]
                if isinstance(bank_cfg, dict):
                    bank_qg = bank_cfg.get("quality_gate")
                    if isinstance(bank_qg, dict):
                        base = {**base, **bank_qg}

    return base


__all__ = [
    "load_config",
    "get_bank_cfg",
    "get_matching_thresholds",
    "get_llm_model_config",
    "load_bank_profiles",
    "resolve_openai_model",
    "get_vision_extraction_config",
    "get_quality_gate_config",
    "get_validation_config",
]
