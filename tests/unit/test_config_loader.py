"""Tests for vigie.support.config.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vigie.support.config.loader import get_bank_cfg, load_config


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


def test_get_validation_config(tmp_path: Path) -> None:
    """get_validation_config loads validation section with bank override."""
    from vigie.support.config import get_validation_config

    cfg = tmp_path / "bank_profiles.yaml"
    cfg.write_text(
        yaml.dump({
            "version": "1.0",
            "validation": {
                "vision_pair_validation": True,
                "rename_validator_enabled": True,
                "rename_validator_uncertain_score_band": [0.8, 0.93],
            },
            "banks": {
                "td": {
                    "validation": {
                        "rename_validator_confidence_min": 0.9,
                        "rename_validator_uncertain_score_band": [0.82, 0.9],
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    val = get_validation_config(str(cfg), bank_code=None)
    assert val.get("vision_pair_validation") is True
    assert val.get("rename_validator_uncertain_score_band") == [0.8, 0.93]
    val_td = get_validation_config(str(cfg), bank_code="td")
    assert val_td.get("rename_validator_confidence_min") == 0.9
    assert val_td.get("rename_validator_uncertain_score_band") == [0.82, 0.9]
