from __future__ import annotations

from pathlib import Path

import yaml

from vigilance.config import get_quality_gate_config


def test_get_quality_gate_config_global(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "quality_gate": {
                    "enabled": True,
                    "duplicate_ratio_threshold": 0.2,
                },
                "banks": {"bnc": {"name": "BNC"}},
            }
        ),
        encoding="utf-8",
    )

    cfg = get_quality_gate_config(config_path=cfg_path)
    assert cfg["enabled"] is True
    assert cfg["duplicate_ratio_threshold"] == 0.2


def test_get_quality_gate_config_bank_override(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bank_profiles.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "quality_gate": {
                    "enabled": True,
                    "duplicate_ratio_threshold": 0.15,
                    "max_contaminated_titles": 2,
                },
                "banks": {
                    "rbc": {
                        "name": "RBC",
                        "quality_gate": {
                            "duplicate_ratio_threshold": 0.1,
                            "max_contaminated_titles": 1,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = get_quality_gate_config(config_path=cfg_path, bank_code="rbc")
    assert cfg["enabled"] is True
    assert cfg["duplicate_ratio_threshold"] == 0.1
    assert cfg["max_contaminated_titles"] == 1
