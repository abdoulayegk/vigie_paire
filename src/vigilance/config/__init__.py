"""Configuration helpers for vigilance.

This module provides a YAML-based configuration system for the vigilance application.
Configuration is loaded from bank_profiles.yaml (or a custom path) with the following
layers:

- **Bank profiles**: Top-level ``banks`` mapping (bank_code -> profile dict).
- **Layered defaults**: Each config getter merges global blocks with optional
  per-bank overrides, then fills missing keys with built-in defaults.
- **Environment overrides**: LLM model selection can be overridden via env vars
  (e.g. OPENAI_MODEL_EXTRACTION_PRIMARY, OPENAI_MODEL_DEFAULT_GENAI).

Config path resolution: relative paths are resolved against the repo root
(pyproject.toml directory) when the path does not exist from the current
working directory. Absolute paths are used as-is.
"""

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
    """Load the bank profiles map from the main YAML config.

    Args:
        config_path: Path to the YAML config file. Relative paths are resolved
            against the repo root if they do not exist from cwd.

    Returns:
        Dict mapping bank_code (lowercase) to bank profile dict. Empty dict
        if config is missing, invalid, or has no ``banks`` key.
    """
    cfg = load_config(config_path)
    banks = cfg.get("banks")
    if isinstance(banks, dict):
        return banks
    return {}


def get_matching_thresholds(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load matching thresholds from configuration with bank overrides and defaults.

    Supports either:
    - ``matching_thresholds: {...}`` at root
    - ``matching: { thresholds: {...} }``

    If bank_code is provided and ``banks.<bank_code>.matching_overrides`` exists,
    those overrides are merged on top of the base thresholds.

    Indicator diff config (indicator_hungarian_enabled, indicator_rename_min_score,
    etc.) is included in the returned dict. Use this function where indicator diff
    thresholds are needed; there is no separate get_indicator_diff_config.

    Args:
        config_path: Path to the YAML config file.
        bank_code: Optional bank code for per-bank matching_overrides.

    Returns:
        Dict of matching thresholds and indicator-diff settings with defaults
        applied for missing keys (indicator_hungarian_enabled, embedding weights,
        recall-first engine params, etc.). Empty dict if config is missing or invalid.
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

    # Recall-first engine defaults
    _recall_first_defaults: dict[str, Any] = {
        "recall_first_engine_enabled": True,
        "rf_min_match_score": 0.25,
        "rf_min_match_margin": 0.04,
        "rf_strong_pair_min_score": 0.50,
        "rf_strong_pair_min_margin": 0.08,
        "rf_review_candidate_min_score": 0.15,
        "rf_max_elimination_rounds": 5,
        "rf_cross_section_rescue_min_score": 0.20,
        "rf_min_indicator_signal": 0.10,
    }
    for k, v in _recall_first_defaults.items():
        if k not in base:
            base[k] = v

    return base


def get_vision_extraction_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Load vision_extraction config with optional bank overrides.

    Args:
        config_path: Path to the YAML config file.
        bank_code: Optional bank code for per-bank vision_extraction overrides
            (e.g. footnote_marker_type, expected_markers).

    Returns:
        Dict with keys such as enabled, bottom_extension_footnotes, run_on_all_tables,
        fallback_to_docling_on_error, save_indicators_footnotes_json. Per-bank
        overrides merge on top of global vision_extraction block. Empty dict if
        config is missing or invalid.
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
    """Load OpenAI model routing config from YAML (no env overrides).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict mapping role names (e.g. extraction_primary, default_genai) to model
        identifiers. Falls back to built-in defaults if config is missing or has
        no llm_models block. Does not apply environment variable overrides; use
        resolve_openai_model for that.
    """
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
    """Resolve the OpenAI model for a known role with env override support.

    Resolution order: (1) env var override, (2) llm_models in config, (3) built-in
    default. Supported roles: extraction_primary, default_genai.

    Args:
        role: Model role (e.g. extraction_primary, default_genai). Normalized to
            lowercase. Must be a known role.
        config_path: Path to the YAML config file for llm_models block.

    Returns:
        Model identifier string (e.g. gpt-5.4, gpt-4o).

    Raises:
        ValueError: If role is not in the known roles (extraction_primary,
            default_genai).
    """
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

    Args:
        config_path: Path to the YAML config file.
        bank_code: Optional bank code for per-bank validation overrides.

    Returns:
        Dict with keys such as vision_pair_validation, vision_pair_confidence_min,
        semantic_judge_enabled, semantic_judge_banks, rename_validator_enabled,
        rename_validator_confidence_min, rename_validator_batch_size,
        rename_validator_uncertain_score_band, added_table_validator_enabled,
        indicator_validator_enabled, indicator_validator_use_vision,
        indicator_validator_confidence_min, indicator_validator_batch_size.
        For backward compatibility, if vision_pair_validation is absent in the
        validation block, falls back to vision_extraction.vision_pair_validation.
        Cross-section rescue defaults (cross_section_rescue_enabled, etc.) are
        applied when keys are absent.
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
    """Load quality_gate config with optional bank overrides.

    Args:
        config_path: Path to the YAML config file.
        bank_code: Optional bank code for per-bank quality_gate overrides.

    Returns:
        Dict merged from global quality_gate block and optional bank overrides.
        Empty dict if config is missing or invalid.
    """
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
