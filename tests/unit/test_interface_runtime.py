"""Tests du point d'entrée unifié et de la revue analyste."""

from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from vigie.interface import review_runtime


def test_minimal_installation_automatically_enables_review_mode(monkeypatch) -> None:
    """Évite toute option obligatoire lorsque seules les dépendances d'interface sont présentes."""
    from vigie.interface import app as app_module

    original_mode = review_runtime.is_review_mode()
    original_analyst = review_runtime._ANALYST
    try:
        monkeypatch.delenv("VIGIE_MODE_REVUE", raising=False)
        monkeypatch.delenv("VIGIE_RESULTATS_DIR", raising=False)
        monkeypatch.delenv("VIGIE_ANALYSTE", raising=False)
        monkeypatch.setattr(app_module, "_pipeline_dependencies_available", lambda: False)
        app_module._configure_startup(
            Namespace(revue=False, resultats=None, analyste=None, port=8050),
        )
        assert review_runtime.is_review_mode() is True
        assert review_runtime.current_analyst()
    finally:
        review_runtime.set_analyst(original_analyst)
        review_runtime.set_review_mode(original_mode)


def test_review_mode_uses_sanitized_analyst_sidecar_name() -> None:
    """Isole les revues par analyste avec un nom de fichier sûr."""
    original_mode = review_runtime.is_review_mode()
    original_analyst = review_runtime._ANALYST
    try:
        review_runtime.set_review_mode(True)
        review_runtime.set_analyst("Marie / Montreal")
        assert review_runtime.current_analyst() == "Marie_Montreal"
    finally:
        review_runtime.set_analyst(original_analyst)
        review_runtime.set_review_mode(original_mode)


def test_environment_enables_review_mode(monkeypatch) -> None:
    """Active la revue lorsque l'identifiant analyste vient de l'environnement."""
    original_mode = review_runtime.is_review_mode()
    original_analyst = review_runtime._ANALYST
    try:
        monkeypatch.setenv("VIGIE_ANALYSTE", "Jean Dupont")
        monkeypatch.delenv("VIGIE_MODE_REVUE", raising=False)
        monkeypatch.delenv("VIGIE_RESULTATS_DIR", raising=False)
        review_runtime.configure_from_environment()
        assert review_runtime.is_review_mode() is True
        assert review_runtime.current_analyst() == "Jean_Dupont"
    finally:
        review_runtime.set_analyst(original_analyst)
        review_runtime.set_review_mode(original_mode)


def test_interface_requirements_exclude_llm_and_extraction_dependencies() -> None:
    """Garde le profil d'interface indépendant de Docling et OpenAI."""
    project_root = Path(__file__).resolve().parents[2]
    requirements = (project_root / "requirements-interface.txt").read_text(encoding="utf-8")
    normalized = requirements.casefold()

    assert "openai" not in normalized
    assert "docling" not in normalized
    assert "numpy" not in normalized
    assert "scipy" not in normalized
    assert "scikit-learn" not in normalized


def test_application_imports_without_pipeline_dependencies() -> None:
    """Garantit que l'interface démarre sans la pile scientifique ni les LLM."""
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
import vigie.interface.app
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["VIGIE_MODE_REVUE"] = "1"
    environment["VIGIE_RESULTATS_DIR"] = str(project_root / "outputs" / "resultats")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_application_rejects_an_invalid_results_directory(tmp_path: Path) -> None:
    """Refuse un dossier explicite invalide avant de démarrer Dash."""
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vigie.interface.app",
            "--revue",
            "--resultats",
            str(tmp_path / "absent"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Dossier de résultats introuvable" in completed.stderr
