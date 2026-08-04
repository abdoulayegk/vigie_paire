"""Tests du validateur multiplateforme sans LLM."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from vigie.interface import validator, validator_config


def test_platform_config_base_uses_windows_appdata(monkeypatch, tmp_path: Path) -> None:
    """Utiliser APPDATA automatiquement sous Windows."""
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))

    assert validator._platform_config_base("win32") == appdata


def test_platform_config_base_uses_macos_convention() -> None:
    """Utiliser Application Support automatiquement sous macOS."""
    assert validator._platform_config_base("darwin") == (
        Path.home() / "Library" / "Application Support"
    )


def test_platform_config_base_uses_xdg_on_linux(monkeypatch, tmp_path: Path) -> None:
    """Respecter XDG_CONFIG_HOME automatiquement sous Linux."""
    xdg_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))

    assert validator._platform_config_base("linux") == xdg_home


def test_resultats_cli_has_priority_over_environment(monkeypatch, tmp_path: Path) -> None:
    """Donner la priorite au chemin explicite de la ligne de commande."""
    cli_dir = tmp_path / "cli"
    env_dir = tmp_path / "env"
    cli_dir.mkdir()
    env_dir.mkdir()
    monkeypatch.setenv("VIGIE_RESULTATS_DIR", str(env_dir))

    assert validator._resolve_resultats_dir(str(cli_dir)) == cli_dir.resolve()


def test_invalid_explicit_resultats_directory_is_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    """Refuser clairement un chemin explicite inexistant."""
    monkeypatch.delenv("VIGIE_RESULTATS_DIR", raising=False)

    with pytest.raises(ValueError, match="Dossier invalide"):
        validator._resolve_resultats_dir(str(tmp_path / "missing"))


def test_validator_mode_uses_sanitized_analyst_sidecar_name() -> None:
    """Isoler les validations par analyste sans modifier les resultats sources."""
    original_mode = validator_config.is_validator_mode()
    original_username = validator_config._USERNAME
    try:
        validator_config.set_validator_mode(True)
        validator_config.set_username("Marie / Montreal")
        assert validator_config.current_username() == "Marie_Montreal"
    finally:
        validator_config.set_username(original_username)
        validator_config.set_validator_mode(original_mode)


def test_validator_requirements_exclude_llm_and_extraction_dependencies() -> None:
    """Garder le profil validateur independant de Docling et OpenAI."""
    project_root = Path(__file__).resolve().parents[2]
    requirements = (project_root / "requirements-validateur.txt").read_text(
        encoding="utf-8"
    )
    normalized = requirements.casefold()

    assert "openai" not in normalized
    assert "docling" not in normalized


def test_validator_app_imports_without_pipeline_dependencies() -> None:
    """Garantir que Dash demarre sans OpenAI, Docling ni pile scientifique."""
    project_root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys

class BlockPipelineDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'openai', 'docling', 'scipy', 'sklearn'}:
            raise ModuleNotFoundError(f'blocked optional dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockPipelineDependencies())
from vigie.interface import validator_config
validator_config.set_validator_mode(True)
import vigie.interface.app
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
