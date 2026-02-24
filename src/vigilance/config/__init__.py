"""Configuration helpers for vigilance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vigilance.config.loader import _resolve_config_path, get_bank_cfg, load_config


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
    }
    for k, v in _indicator_defaults.items():
        if k not in base:
            base[k] = v

    return base


__all__ = ["load_config", "get_bank_cfg", "get_matching_thresholds", "load_bank_profiles"]
