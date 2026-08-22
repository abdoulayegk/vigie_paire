"""Selection centralisee du moteur de localisation de tableaux."""

from __future__ import annotations

import logging
import os
from typing import Any

from .models import TableAnchor

logger = logging.getLogger("vigie.extraction.table_locator")

ENGINE_PYMUPDF_LAYOUT = "pymupdf_layout"
ENGINE_DOCLING = "docling"
_SUPPORTED_ENGINES = frozenset({ENGINE_PYMUPDF_LAYOUT, ENGINE_DOCLING})
_DEFAULT_ENGINE = ENGINE_PYMUPDF_LAYOUT


def resolve_table_locator_engine(explicit: str | None = None) -> str:
    """Resoudre ``TABLE_LOCATOR_ENGINE`` (argument > env > defaut).

    Args:
        explicit: Valeur forcee par l'appelant, sinon lecture env.

    Returns:
        ``pymupdf_layout`` ou ``docling``.

    Raises:
        ValueError: Si la valeur n'est pas supportée.
    """
    raw = explicit if explicit is not None else os.environ.get("TABLE_LOCATOR_ENGINE")
    if raw is None or not str(raw).strip():
        engine = _DEFAULT_ENGINE
    else:
        engine = str(raw).strip().lower()
    if engine not in _SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported TABLE_LOCATOR_ENGINE: {engine!r}. Accepted values: {sorted(_SUPPORTED_ENGINES)}")
    return engine


def get_table_locator(engine: str | None = None, **kwargs: Any) -> Any:
    """Instancier le localisateur correspondant au moteur choisi.

    Args:
        engine: Moteur explicite ; sinon resolution via env.
        **kwargs: Arguments specifiques au moteur (ex. ``converter`` pour Docling).

    Returns:
        Instance de localisateur (``TablesLayoutLocator`` ou ``DoclingTableLocator``).

    Raises:
        ValueError: Moteur inconnu.
        RuntimeError: Convertisseur Docling manquant quand requis.
    """
    resolved = resolve_table_locator_engine(engine)
    logger.info("Table locator engine: %s", resolved)

    if resolved == ENGINE_PYMUPDF_LAYOUT:
        from vigie.extraction.tables_layout.table_locator import (  # noqa: PLC0415 - import differe anti-cycle
            TablesLayoutLocator,
        )

        return TablesLayoutLocator()

    if resolved == ENGINE_DOCLING:
        converter = kwargs.get("converter")
        if converter is None:
            raise RuntimeError("Docling converter is required for TABLE_LOCATOR_ENGINE=docling")
        from vigie.extraction.docling.table_locator import (  # noqa: PLC0415 - import differe anti-cycle
            DoclingTableLocator,
        )

        return DoclingTableLocator(converter)

    raise ValueError(f"Unsupported TABLE_LOCATOR_ENGINE: {resolved}")


def anchors_to_vision_items(
    anchors: list[TableAnchor],
) -> list[tuple[int, int, list[float] | None, str, str | None]]:
    """Convertir des ancres vers le tuple historique consomme par Vision."""
    items: list[tuple[int, int, list[float] | None, str, str | None]] = []
    for idx, anchor in enumerate(anchors):
        items.append(
            (
                idx,
                int(anchor.page_number),
                list(anchor.bbox) if anchor.bbox is not None else None,
                str(anchor.table_id),
                anchor.reference_text,
            )
        )
    return items
