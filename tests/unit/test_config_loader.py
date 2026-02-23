"""Tests for vigilance.config.loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from vigilance.config.loader import get_bank_cfg, load_config


@pytest.fixture()
def tmp_yaml(tmp_path: Path) -> Path:
    """Create a minimal temporary YAML config file."""
    data = {
        "version": "1.0",
        "banks": {
            "rbc": {"name": "RBC"},
        },
    }
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    return cfg_file


def test_load_config(tmp_yaml: Path) -> None:
    cfg = load_config(str(tmp_yaml))
    assert isinstance(cfg, dict)
    assert "banks" in cfg
    assert "rbc" in cfg["banks"]


def test_get_bank_cfg_known(tmp_yaml: Path) -> None:
    cfg = load_config(str(tmp_yaml))
    bank = get_bank_cfg(cfg, "rbc")
    assert isinstance(bank, dict)
    assert bank["name"] == "RBC"


def test_get_bank_cfg_unknown_raises(tmp_yaml: Path) -> None:
    cfg = load_config(str(tmp_yaml))
    with pytest.raises(ValueError, match="xxx"):
        get_bank_cfg(cfg, "xxx")


def test_get_bank_cfg_invalid_cfg() -> None:
    with pytest.raises(ValueError, match="missing 'banks' key"):
        get_bank_cfg({"no_banks": True}, "rbc")


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
