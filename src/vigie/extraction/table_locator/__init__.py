"""Selection du moteur de localisation de tableaux (Docling | tables_layout)."""

from __future__ import annotations

from .models import TableAnchor, TableLocationResult
from .selector import (
    ENGINE_DOCLING,
    ENGINE_PYMUPDF_LAYOUT,
    anchors_to_vision_items,
    get_table_locator,
    resolve_table_locator_engine,
)

__all__ = [
    "ENGINE_DOCLING",
    "ENGINE_PYMUPDF_LAYOUT",
    "TableAnchor",
    "TableLocationResult",
    "anchors_to_vision_items",
    "get_table_locator",
    "resolve_table_locator_engine",
]
