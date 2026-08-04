"""Garde-fou : les domaines ne doivent jamais importer cli/pipelines."""

from __future__ import annotations

import ast
from pathlib import Path


_VIGIE_ROOT = Path(__file__).resolve().parents[2] / "src" / "vigie"
_DOMAIN_PACKAGES = (
    "extraction",
    "comparaison",
    "analyse_texte",
    "interface",
    "support",
)
_FORBIDDEN_PREFIXES = ("vigie.cli", "vigie.pipelines")


def _forbidden_imports(path: Path) -> list[str]:
    """Retourner les imports cli/pipelines trouves dans un fichier domaine."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(_FORBIDDEN_PREFIXES):
                    hits.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(_FORBIDDEN_PREFIXES):
                hits.append(module)
    return hits


def test_domain_packages_do_not_import_entry_layer() -> None:
    """extraction/comparaison/analyse_texte/interface/support n'importent pas cli/pipelines."""
    violations: list[str] = []
    for domain in _DOMAIN_PACKAGES:
        root = _VIGIE_ROOT / domain
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for name in _forbidden_imports(path):
                rel = path.relative_to(_VIGIE_ROOT.parent.parent)
                violations.append(f"{rel}: {name}")
    assert not violations, (
        "Imports interdits domaine -> cli/pipelines:\n" + "\n".join(violations)
    )
