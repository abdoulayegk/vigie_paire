"""Localisation structurelle des tableaux via PyMuPDF4LLM + Layout."""

from __future__ import annotations

from vigie.extraction.tables_layout.table_locator import TablesLayoutLocator
from vigie.extraction.tables_layout.tables_layout_pass import run_tables_layout_pass

__all__ = [
    "TablesLayoutLocator",
    "run_tables_layout_pass",
]
