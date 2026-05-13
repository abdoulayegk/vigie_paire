"""Configuration Sphinx de la documentation technique de Vigie de paire."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

# Pages GitHub du projet : https://<owner>.github.io/<repo>/
_base_url = (os.environ.get("SPHINX_HTML_BASEURL") or "").strip().rstrip("/")
if _base_url:
    html_baseurl = _base_url + "/"

project = "Vigie de paire"
copyright = "2026"
author = "Vigie de paire"

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
html_title = "Vigie de paire - documentation technique"
html_short_title = "Vigie de paire"
html_show_sourcelink = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
add_module_names = False
modindex_common_prefix = ["vigilance."]

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
