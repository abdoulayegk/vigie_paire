"""Sphinx configuration for bank-peer-vigilance."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

project = "bank-peer-vigilance"
copyright = "2026"
author = "Vigie Paire"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

language = "fr"

myst_enable_extensions = ["colon_fence", "deflist"]
myst_fence_as_directive = ["mermaid"]

suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "ref.duplicate",
]
