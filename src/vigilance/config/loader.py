"""YAML configuration loader for bank profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    """Return repository root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to historical layout: <repo>/src/vigilance/config/loader.py
    return current.parents[3]


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve config path robustly across different working directories."""
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path

    project_candidate = _repo_root() / config_path
    if project_candidate.exists():
        return project_candidate

    return config_path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration from *path*.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If YAML is invalid or does not produce a dictionary.
    """
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file '{config_path}': {exc}") from exc

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid config format in '{config_path}': expected mapping at root, "
            f"got {type(data).__name__}"
        )
    return data


def get_bank_cfg(cfg: dict[str, Any], bank_code: str) -> dict[str, Any]:
    """Return the configuration block for a bank code."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: expected dict, got {type(cfg).__name__}")

    banks = cfg.get("banks")
    if not isinstance(banks, dict):
        raise ValueError("Invalid config: missing 'banks' key")
    if not bank_code:
        raise ValueError("Bank code must be a non-empty string")
    if bank_code not in banks:
        available = ", ".join(sorted(banks.keys()))
        raise ValueError(
            f"Bank '{bank_code}' not found in config. Available banks: [{available}]"
        )

    bank_cfg = banks[bank_code]
    if not isinstance(bank_cfg, dict):
        raise ValueError(
            f"Invalid config for bank '{bank_code}': expected dict, got {type(bank_cfg).__name__}"
        )
    return bank_cfg
